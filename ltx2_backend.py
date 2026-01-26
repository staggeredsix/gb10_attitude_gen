from __future__ import annotations

import inspect
import json
import logging
import os
import pathlib
import random
import threading
import time
from typing import Callable, Iterable

import cv2
import numpy as np
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw, ImageFont

LOGGER = logging.getLogger("ltx2_backend")

_PIPELINES: dict[str, object] = {}
_PIPELINE_LOCK = threading.Lock()
_PIPELINE_LOGGED = False
_PIPELINE_SOURCES: dict[str, str] = {}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        LOGGER.warning("Invalid int for %s=%s; using %s", name, value, default)
        return default


def is_local_path(model_id: str) -> bool:
    return model_id.startswith(("/", "./")) or pathlib.Path(model_id).exists()


def _resolve_local_snapshot(model_id: str) -> str | None:
    cache_root = os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("HF_HOME")
    if not cache_root:
        return None
    cache_root = os.path.expanduser(cache_root)
    repo_dir = f"models--{model_id.replace('/', '--')}"
    snapshots_dir = pathlib.Path(cache_root) / repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    candidates = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for snapshot in candidates:
        if not snapshot.is_dir():
            continue
        if (snapshot / "model_index.json").is_file():
            return str(snapshot)
    return None


def _normalize_local_snapshot(path: str) -> str:
    local_path = pathlib.Path(path).expanduser()
    if not local_path.exists():
        raise RuntimeError(f"Snapshot path does not exist: {local_path}")
    if (local_path / "model_index.json").is_file():
        return str(local_path)
    snapshots_dir = local_path / "snapshots"
    if snapshots_dir.is_dir():
        candidates = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for snapshot in candidates:
            if not snapshot.is_dir():
                continue
            if (snapshot / "model_index.json").is_file():
                return str(snapshot)
    raise RuntimeError(
        "Missing model_index.json in snapshot. "
        "Ensure you mounted the snapshot directory that contains model_index.json."
    )


def _collect_safetensors(snapshot_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(snapshot_dir.rglob("*.safetensors"))


def validate_snapshot(path: str) -> dict[str, list[str]]:
    snapshot_dir = pathlib.Path(path)
    if not snapshot_dir.is_dir():
        raise RuntimeError(f"Snapshot path does not exist or is not a directory: {snapshot_dir}")
    model_index = snapshot_dir / "model_index.json"
    if not model_index.is_file():
        raise RuntimeError(
            "Missing model_index.json in snapshot. "
            "Ensure you mounted the LTX-2 snapshot root directory."
        )
    try:
        data = json.loads(model_index.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError("model_index.json is not valid JSON") from exc
    components: list[str] = []
    if isinstance(data, dict):
        if isinstance(data.get("components"), dict):
            components = list(data["components"].keys())
        else:
            components = list(data.keys())
    LOGGER.info("Snapshot components: %s", components)

    safetensors = _collect_safetensors(snapshot_dir)
    safetensor_names = [str(path.relative_to(snapshot_dir)) for path in safetensors]
    LOGGER.info("Found safetensors: %s", safetensor_names)
    return {"safetensors": safetensor_names, "components": components}


def log_backend_configuration(model_id: str | None = None) -> None:
    resolved_model_id = model_id or os.getenv("LTX2_MODEL_ID", "Lightricks/LTX-2")
    local_files_only = _env_bool("LTX2_LOCAL_FILES_ONLY", is_local_path(resolved_model_id))
    allow_download = _env_bool("LTX2_ALLOW_DOWNLOAD", False)
    snapshot_dir = os.getenv("LTX2_SNAPSHOT_DIR", "/models/LTX-2")
    LOGGER.info(
        "LTX-2 config: model_id=%s local_files_only=%s allow_download=%s snapshot_dir=%s",
        resolved_model_id,
        local_files_only,
        allow_download,
        snapshot_dir,
    )
    LOGGER.info(
        "HF cache: HF_HOME=%s HUGGINGFACE_HUB_CACHE=%s TRANSFORMERS_CACHE=%s",
        os.getenv("HF_HOME"),
        os.getenv("HUGGINGFACE_HUB_CACHE"),
        os.getenv("TRANSFORMERS_CACHE"),
    )


def _allow_patterns() -> list[str]:
    return [
        "model_index.json",
        "*.json",
        "tokenizer/**",
        "scheduler/**",
        "transformer/**",
        "text_encoder/**",
        "vae/**",
        "*.safetensors",
    ]


def _resolve_model_source() -> tuple[str, bool, str | None]:
    model_id = os.getenv("LTX2_MODEL_ID", "Lightricks/LTX-2")
    allow_download = _env_bool("LTX2_ALLOW_DOWNLOAD", False)
    snapshot_dir = os.getenv("LTX2_SNAPSHOT_DIR", "/models/huggingface/hub/models--Lightricks--LTX-2")
    local_files_only = _env_bool("LTX2_LOCAL_FILES_ONLY", is_local_path(model_id))

    resolved_source = model_id
    snapshot_path: str | None = None
    if is_local_path(model_id):
        snapshot_path = _normalize_local_snapshot(model_id)
        LOGGER.info("Using local snapshot: %s", snapshot_path)
        validate_snapshot(snapshot_path)
        resolved_source = snapshot_path
        local_files_only = True
    elif allow_download:
        LOGGER.info("Downloading snapshot for %s to %s", model_id, snapshot_dir)
        snapshot_path = snapshot_download(
            repo_id=model_id,
            local_dir=snapshot_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=_allow_patterns(),
        )
        validate_snapshot(snapshot_path)
        resolved_source = snapshot_path
        local_files_only = True
    else:
        local_files_only = True

    return resolved_source, local_files_only, snapshot_path


def _load_pipeline(mode: str, device: str = "cuda"):
    cache_key = f"{mode}:{device}"
    with _PIPELINE_LOCK:
        if cache_key in _PIPELINES:
            return _PIPELINES[cache_key]

        dtype = torch.bfloat16

        resolved_source, local_files_only, snapshot_path = _resolve_model_source()

        LOGGER.info(
            "Loading LTX-2 pipeline: source=%s local_files_only=%s mode=%s",
            resolved_source,
            local_files_only,
            mode,
        )

        try:
            pipe = DiffusionPipeline.from_pretrained(
                resolved_source,
                local_files_only=True,
                torch_dtype=dtype,
            )
        except Exception as exc:  # noqa: BLE001
            allow_download = _env_bool("LTX2_ALLOW_DOWNLOAD", False)
            model_id = os.getenv("LTX2_MODEL_ID", "Lightricks/LTX-2")
            if not allow_download and not is_local_path(model_id):
                raise RuntimeError(
                    "LTX-2 snapshot not found in local cache. "
                    "Mount the snapshot at /models/LTX-2 or set LTX2_ALLOW_DOWNLOAD=true."
                ) from exc
            snapshot_hint = snapshot_path or (model_id if is_local_path(model_id) else _resolve_local_snapshot(model_id))
            if snapshot_hint:
                try:
                    validate_snapshot(snapshot_hint)
                except RuntimeError:
                    LOGGER.exception("Snapshot validation failed for %s", snapshot_hint)
            raise

        pipe.to(device)
        _PIPELINE_SOURCES[mode] = resolved_source
        _log_pipeline_runtime_info(pipe)
        _PIPELINES[cache_key] = pipe
        return pipe


def _resolve_prompt_length(pipe: object) -> tuple[int, str | None]:
    signature = inspect.signature(pipe.__call__)
    param_name = next(
        (name for name in ("max_sequence_length", "max_prompt_length", "max_length") if name in signature.parameters),
        None,
    )
    env_value = os.getenv("LTX2_MAX_PROMPT_LEN")
    if env_value is None:
        env_value = os.getenv("LTX2_MAX_SEQUENCE_LENGTH")
    if env_value:
        try:
            return int(env_value), param_name
        except ValueError:
            LOGGER.warning("Invalid LTX2_MAX_PROMPT_LEN=%s; ignoring.", env_value)
    if param_name is not None:
        param = signature.parameters[param_name]
        if param.default is not inspect.Parameter.empty and param.default is not None:
            return int(param.default), param_name
    tokenizer = getattr(pipe, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "model_max_length", None):
        return int(tokenizer.model_max_length), param_name
    LOGGER.warning(
        "Unable to infer prompt length from pipeline; defaulting to 128. "
        "Set LTX2_MAX_PROMPT_LEN to override."
    )
    return 128, param_name


def _log_pipeline_runtime_info(pipe: object) -> None:
    global _PIPELINE_LOGGED
    if _PIPELINE_LOGGED:
        return
    try:
        import diffusers
    except Exception:  # noqa: BLE001
        diffusers = None
    signature = inspect.signature(pipe.__call__)
    prompt_length, param_name = _resolve_prompt_length(pipe)
    LOGGER.info(
        "diffusers version: %s",
        getattr(diffusers, "__version__", "unknown"),
    )
    LOGGER.info("LTX pipeline __call__ signature: %s", signature)
    LOGGER.info(
        "Resolved max prompt length: %s (param=%s, env=LTX2_MAX_PROMPT_LEN)",
        prompt_length,
        param_name,
    )
    _PIPELINE_LOGGED = True


def warmup_pipeline(mode: str) -> dict[str, str]:
    pipe = _load_pipeline(mode)
    return {"pipeline_class": pipe.__class__.__name__}


def render_status_frame(text: str, width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), color=(10, 10, 20))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = None
    draw.multiline_text((20, height // 2 - 10), text, fill=(240, 240, 240), font=font)
    return image


def _extract_frames(result: object) -> list[Image.Image]:
    if result is None:
        return []
    if isinstance(result, np.ndarray):
        return [Image.fromarray(result)]
    if isinstance(result, list):
        return [frame if isinstance(frame, Image.Image) else Image.fromarray(frame) for frame in result]
    if isinstance(result, dict):
        for key in ("frames", "images", "videos"):
            if key in result:
                return _extract_frames(result[key])
    for key in ("frames", "images", "videos"):
        if hasattr(result, key):
            return _extract_frames(getattr(result, key))
    return []


def _prompt_drift(prompt: str) -> str:
    adjectives = [
        "iridescent",
        "fractured",
        "bioluminescent",
        "liquid",
        "ethereal",
        "cosmic",
        "surreal",
        "hypnotic",
    ]
    drift = random.sample(adjectives, k=2)
    return f"{prompt}, {', '.join(drift)}"


def _signature_requires_image(pipe: object) -> bool:
    signature = inspect.signature(pipe.__call__)
    required = {
        name
        for name, param in signature.parameters.items()
        if param.default is inspect.Parameter.empty and name not in {"self"}
    }
    return "image" in required or "init_image" in required


def _assign_image_arg(pipe: object, kwargs: dict, image: Image.Image) -> None:
    signature = inspect.signature(pipe.__call__)
    params = signature.parameters
    if "image" in params:
        kwargs["image"] = image
    elif "init_image" in params:
        kwargs["init_image"] = image


def _filter_kwargs(pipe: object, kwargs: dict) -> dict:
    signature = inspect.signature(pipe.__call__)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    allowed = set(signature.parameters.keys())
    return {key: value for key, value in kwargs.items() if key in allowed}


def _normalize_negative_prompt(negative_prompt: str | None) -> str:
    if negative_prompt is None:
        return ""
    return str(negative_prompt).strip()


def _build_ltx2_kwargs(
    pipe: object,
    *,
    prompt: str,
    negative_prompt: str | None,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: int,
    guidance_scale: float,
    num_inference_steps: int,
    generator: torch.Generator | None,
    strength: float | None = None,
    image: Image.Image | None = None,
) -> dict:
    negative_prompt = _normalize_negative_prompt(negative_prompt)
    prompt_length, param_name = _resolve_prompt_length(pipe)
    kwargs: dict[str, object] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "generator": generator,
    }
    if strength is not None:
        kwargs["strength"] = strength
    if frame_rate:
        kwargs["frame_rate"] = frame_rate
    if param_name:
        kwargs[param_name] = prompt_length
    signature = inspect.signature(pipe.__call__)
    if "padding" in signature.parameters:
        kwargs["padding"] = "max_length"
    if "truncation" in signature.parameters:
        kwargs["truncation"] = True
    if image is not None:
        _assign_image_arg(pipe, kwargs, image)
    return _filter_kwargs(pipe, {k: v for k, v in kwargs.items() if v is not None})


def _is_prompt_mismatch_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "sizes of tensors must match" in message
        or "size mismatch" in message
        or "torch.cat" in message
        or ("prompt_embeds" in message and "negative_prompt_embeds" in message)
    )


def _run_with_prompt_fallback(pipe: object, kwargs: dict) -> object:
    try:
        return pipe(**kwargs)
    except Exception as exc:  # noqa: BLE001
        if not _is_prompt_mismatch_error(exc):
            raise
        LOGGER.warning("Prompt embedding mismatch detected; retrying with negative_prompt disabled.")
        fallback_kwargs = dict(kwargs)
        if "negative_prompt" in fallback_kwargs:
            fallback_kwargs["negative_prompt"] = ""
        return pipe(**fallback_kwargs)


def _load_upscaler_pipeline(kind: str, device: str = "cuda") -> object | None:
    cache_key = f"{kind}_upscaler:{device}"
    with _PIPELINE_LOCK:
        if cache_key in _PIPELINES:
            return _PIPELINES[cache_key]

        resolved_source = _PIPELINE_SOURCES.get("text") or _PIPELINE_SOURCES.get("image")
        if resolved_source is None:
            resolved_source, _, _ = _resolve_model_source()

        LOGGER.info("Loading LTX-2 %s upscaler pipeline from %s", kind, resolved_source)
        try:
            from diffusers.pipelines.ltx import LTXLatentUpsamplePipeline
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to import LTX latent upsampler: %s", exc)
            return None

        pipe = LTXLatentUpsamplePipeline.from_pretrained(
            resolved_source,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        pipe.to(device)

        weight_env = "LTX2_SPATIAL_UPSCALER_FILE" if kind == "spatial" else "LTX2_TEMPORAL_UPSCALER_FILE"
        weight_name = os.getenv(weight_env, f"ltx-2-{kind}-upscaler-x2-1.0.safetensors")
        weight_path = pathlib.Path(weight_name)
        if not weight_path.is_absolute():
            snapshot_root = pathlib.Path(resolved_source)
            if not snapshot_root.exists():
                model_id = os.getenv("LTX2_MODEL_ID", "Lightricks/LTX-2")
                snapshot_hint = _resolve_local_snapshot(model_id)
                if snapshot_hint:
                    snapshot_root = pathlib.Path(snapshot_hint)
            weight_path = snapshot_root / weight_name
        if weight_path.is_file():
            try:
                from safetensors.torch import load_file
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("safetensors is required to load %s: %s", weight_path, exc)
            else:
                state_dict = load_file(str(weight_path))
                missing, unexpected = pipe.latent_upsampler.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    LOGGER.warning(
                        "Upscaler %s weights loaded with missing=%s unexpected=%s",
                        kind,
                        missing,
                        unexpected,
                    )
        else:
            message = f"{kind.title()} upscaler weights missing at {weight_path}"
            if _env_bool("LTX2_UPSCALER_REQUIRED", False):
                raise RuntimeError(message)
            LOGGER.warning("%s; falling back to native output.", message)
            return None

        _PIPELINES[cache_key] = pipe
        return pipe


def _apply_upscalers(
    frames: list[Image.Image],
    config,
    device: str,
) -> tuple[list[Image.Image], int, int, int]:
    preset = getattr(config, "output_preset", "native") or "native"
    width = config.width
    height = config.height
    fps = config.fps

    def run_upscaler(kind: str, target_width: int, target_height: int) -> tuple[list[Image.Image], bool]:
        upscaler = _load_upscaler_pipeline(kind, device=device)
        if upscaler is None:
            return frames, False
        result = upscaler(video=frames, width=target_width, height=target_height)
        upscaled_frames = _extract_frames(result)
        return (upscaled_frames or frames), True

    if preset in {"spatial_x2", "spatial_x2_temporal_x2"}:
        target_width = width * 2
        target_height = height * 2
        frames, applied = run_upscaler("spatial", target_width, target_height)
        if applied:
            width = target_width
            height = target_height
    if preset in {"temporal_x2", "spatial_x2_temporal_x2"}:
        before_count = len(frames)
        frames, applied = run_upscaler("temporal", width, height)
        if applied:
            after_count = len(frames)
            if before_count and after_count:
                fps = max(1, int(round(fps * after_count / before_count)))
            else:
                fps = fps * 2

    LOGGER.info(
        "Output preset=%s final_size=%sx%s final_fps=%s frames=%s",
        preset,
        width,
        height,
        fps,
        len(frames),
    )
    return frames, width, height, fps


def generate_fever_dream_frames(config, cancel_event: threading.Event) -> Iterable[Image.Image]:
    pipe = _load_pipeline("text")
    if pipe is None:
        while not cancel_event.is_set():
            yield render_status_frame("LTX-2 load failed", config.width, config.height)
            time.sleep(1.0 / max(1, config.fps))
        return
    LOGGER.info(
        "Fever Dream preset=%s native_size=%sx%s native_fps=%s",
        getattr(config, "output_preset", "native"),
        config.width,
        config.height,
        config.fps,
    )
    last_frame: Image.Image | None = None
    while not cancel_event.is_set():
        prompt = _prompt_drift(config.prompt)
        negative_prompt = config.negative_prompt
        chunk_seconds = 1.0
        num_frames = max(1, int(chunk_seconds * config.fps))
        generator = None
        if config.seed is not None:
            generator = torch.Generator(device=pipe.device).manual_seed(config.seed + int(time.time()))
        image = None
        if _signature_requires_image(pipe):
            if last_frame is None:
                LOGGER.warning("Text-to-video pipeline requires an init image; using a blank frame.")
                last_frame = render_status_frame("seed", config.width, config.height)
            image = last_frame
        filtered_kwargs = _build_ltx2_kwargs(
            pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=config.height,
            width=config.width,
            num_frames=num_frames,
            frame_rate=config.fps,
            guidance_scale=3.0 + config.dream_strength * 5.0,
            num_inference_steps=int(10 + config.motion * 10),
            generator=generator,
            image=image,
        )
        try:
            result = _run_with_prompt_fallback(pipe, filtered_kwargs)
            frames = _extract_frames(result)
            if not frames:
                raise RuntimeError("No frames returned from LTX-2")
            if getattr(config, "output_preset", "native") != "native":
                frames, _, _, _ = _apply_upscalers(frames, config, device=str(pipe.device))
            for frame in frames:
                last_frame = frame
                yield frame
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Fever Dream generation error: %s", exc)
            yield render_status_frame("Generation error", config.width, config.height)


def generate_mood_mirror_frames(
    config,
    latest_camera_state: Callable[[], tuple[np.ndarray | None, dict | None]],
    cancel_event: threading.Event,
) -> Iterable[Image.Image]:
    pipe = _load_pipeline("image")
    if pipe is None:
        while not cancel_event.is_set():
            yield render_status_frame("LTX-2 load failed", config.width, config.height)
            time.sleep(1.0 / max(1, config.fps))
        return
    LOGGER.info(
        "Mood Mirror preset=%s native_size=%sx%s native_fps=%s",
        getattr(config, "output_preset", "native"),
        config.width,
        config.height,
        config.fps,
    )
    last_frame: Image.Image | None = None
    while not cancel_event.is_set():
        camera_frame, mood_state = latest_camera_state()
        if camera_frame is None:
            yield render_status_frame("Waiting for camera feed...", config.width, config.height)
            time.sleep(1.0 / max(1, config.fps))
            continue
        prompt = config.base_prompt
        if mood_state:
            mood_prompt = mood_state.get("prompt_hint") or ""
            prompt = f"{prompt}, {mood_prompt}" if mood_prompt else prompt
        chunk_seconds = 1.0
        num_frames = max(1, int(chunk_seconds * config.fps))
        generator = None
        if config.seed is not None:
            generator = torch.Generator(device=pipe.device).manual_seed(config.seed + int(time.time()))
        init_image = Image.fromarray(cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB))
        strength = 0.2 + (1.0 - config.identity_strength) * 0.6
        filtered_kwargs = _build_ltx2_kwargs(
            pipe,
            prompt=prompt,
            negative_prompt=config.negative_prompt,
            height=config.height,
            width=config.width,
            num_frames=num_frames,
            frame_rate=config.fps,
            guidance_scale=3.0 + config.dream_strength * 4.0,
            num_inference_steps=int(10 + config.motion * 10),
            generator=generator,
            strength=strength,
            image=init_image,
        )
        try:
            result = _run_with_prompt_fallback(pipe, filtered_kwargs)
            frames = _extract_frames(result)
            if not frames:
                raise RuntimeError("No frames returned from LTX-2")
            if getattr(config, "output_preset", "native") != "native":
                frames, _, _, _ = _apply_upscalers(frames, config, device=str(pipe.device))
            for frame in frames:
                last_frame = frame
                yield frame
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Mood Mirror generation error: %s", exc)
            yield render_status_frame("Generation error", config.width, config.height)
