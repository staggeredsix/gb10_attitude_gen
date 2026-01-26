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
from diffusers import AutoPipelineForImage2Video, AutoPipelineForText2Video
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw, ImageFont

LOGGER = logging.getLogger("ltx2_backend")

_PIPELINES: dict[str, object] = {}
_PIPELINE_LOCK = threading.Lock()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


def _collect_safetensors(snapshot_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(snapshot_dir.rglob("*.safetensors"))


def validate_snapshot(path: str, variant: str) -> dict[str, list[str]]:
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
    fp4_files = [name for name in safetensor_names if "fp4" in pathlib.Path(name).name.lower()]
    LOGGER.info("Found safetensors: %s", safetensor_names)
    LOGGER.info("Found fp4 files: %s", fp4_files)
    if variant == "fp4" and not fp4_files:
        raise RuntimeError(
            "No fp4 weights found in snapshot. "
            "Expected files like <component>.fp4.safetensors."
        )
    return {"safetensors": safetensor_names, "fp4_files": fp4_files, "components": components}


def log_backend_configuration(model_id: str | None = None) -> None:
    resolved_model_id = model_id or os.getenv("LTX2_MODEL_ID", "Lightricks/LTX-2")
    variant = os.getenv("LTX2_VARIANT", "fp4")
    local_files_only = _env_bool("LTX2_LOCAL_FILES_ONLY", is_local_path(resolved_model_id))
    allow_download = _env_bool("LTX2_ALLOW_DOWNLOAD", False)
    snapshot_dir = os.getenv("LTX2_SNAPSHOT_DIR", "/models/LTX-2")
    LOGGER.info(
        "LTX-2 config: model_id=%s local_files_only=%s variant=%s allow_download=%s snapshot_dir=%s",
        resolved_model_id,
        local_files_only,
        variant,
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
        "*fp4*.safetensors",
        "*.fp4.safetensors",
    ]


def _load_pipeline(mode: str, device: str = "cuda"):
    cache_key = f"{mode}:{device}"
    with _PIPELINE_LOCK:
        if cache_key in _PIPELINES:
            return _PIPELINES[cache_key]

        model_id = os.getenv("LTX2_MODEL_ID", "Lightricks/LTX-2")
        variant = os.getenv("LTX2_VARIANT", "fp4")
        fallback_variant = os.getenv("LTX2_FALLBACK_VARIANT", "").strip() or None
        allow_download = _env_bool("LTX2_ALLOW_DOWNLOAD", False)
        snapshot_dir = os.getenv("LTX2_SNAPSHOT_DIR", "/models/LTX-2")
        local_files_only = _env_bool("LTX2_LOCAL_FILES_ONLY", is_local_path(model_id))

        pipeline_cls = AutoPipelineForImage2Video if mode == "image" else AutoPipelineForText2Video
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        resolved_source = model_id
        snapshot_path: str | None = None
        if is_local_path(model_id):
            snapshot_path = os.path.expanduser(model_id)
            LOGGER.info("Using local snapshot: %s", snapshot_path)
            validate_snapshot(snapshot_path, variant)
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
            validate_snapshot(snapshot_path, variant)
            resolved_source = snapshot_path
            local_files_only = True
        else:
            local_files_only = True

        LOGGER.info(
            "Loading LTX-2 pipeline: source=%s local_files_only=%s variant=%s mode=%s",
            resolved_source,
            local_files_only,
            variant,
            mode,
        )

        variant_arg = variant or None
        try:
            pipe = pipeline_cls.from_pretrained(
                resolved_source,
                torch_dtype=dtype,
                variant=variant_arg,
                use_safetensors=True,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
        except Exception as exc:  # noqa: BLE001
            if not allow_download and not is_local_path(model_id):
                raise RuntimeError(
                    "LTX-2 snapshot not found in local cache. "
                    "Mount the snapshot at /models/LTX-2 or set LTX2_ALLOW_DOWNLOAD=true."
                ) from exc
            if variant == "fp4":
                snapshot_hint = snapshot_path or (model_id if is_local_path(model_id) else _resolve_local_snapshot(model_id))
                fp4_files: list[str] = []
                if snapshot_hint:
                    try:
                        fp4_files = validate_snapshot(snapshot_hint, variant)["fp4_files"]
                    except RuntimeError:
                        LOGGER.exception("Snapshot validation failed for %s", snapshot_hint)
                LOGGER.error("Found fp4 files: %s", fp4_files)
                if fallback_variant:
                    LOGGER.warning(
                        "Falling back to variant %s because fp4 load failed: %s",
                        fallback_variant,
                        exc,
                    )
                    pipe = pipeline_cls.from_pretrained(
                        resolved_source,
                        torch_dtype=dtype,
                        variant=fallback_variant,
                        use_safetensors=True,
                        trust_remote_code=True,
                        local_files_only=local_files_only,
                    )
                else:
                    raise RuntimeError(
                        "Diffusers couldn’t match fp4 variant filenames. "
                        "Ensure component fp4 weights are named like <component>.fp4.safetensors "
                        "or use ltx-pipelines / rename weights."
                    ) from exc
            else:
                raise

        pipe.to(device)
        _PIPELINES[cache_key] = pipe
        return pipe


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


def generate_fever_dream_frames(config, cancel_event: threading.Event) -> Iterable[Image.Image]:
    pipe = _load_pipeline("text")
    if pipe is None:
        while not cancel_event.is_set():
            yield render_status_frame("LTX-2 load failed", config.width, config.height)
            time.sleep(1.0 / max(1, config.fps))
        return
    last_frame: Image.Image | None = None
    while not cancel_event.is_set():
        prompt = _prompt_drift(config.prompt)
        negative_prompt = config.negative_prompt
        chunk_seconds = 1.0
        num_frames = max(1, int(chunk_seconds * config.fps))
        generator = None
        if config.seed is not None:
            generator = torch.Generator(device=pipe.device).manual_seed(config.seed + int(time.time()))
        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "num_frames": num_frames,
            "height": config.height,
            "width": config.width,
            "guidance_scale": 3.0 + config.dream_strength * 5.0,
            "num_inference_steps": int(10 + config.motion * 10),
            "generator": generator,
        }
        if _signature_requires_image(pipe):
            if last_frame is None:
                LOGGER.warning("Text-to-video pipeline requires an init image; using a blank frame.")
                last_frame = render_status_frame("seed", config.width, config.height)
            _assign_image_arg(pipe, kwargs, last_frame)
        filtered_kwargs = _filter_kwargs(pipe, {k: v for k, v in kwargs.items() if v is not None})
        try:
            result = pipe(**filtered_kwargs)
            frames = _extract_frames(result)
            if not frames:
                raise RuntimeError("No frames returned from LTX-2")
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
        kwargs = {
            "prompt": prompt,
            "negative_prompt": config.negative_prompt or None,
            "num_frames": num_frames,
            "height": config.height,
            "width": config.width,
            "guidance_scale": 3.0 + config.dream_strength * 4.0,
            "num_inference_steps": int(10 + config.motion * 10),
            "generator": generator,
            "strength": strength,
        }
        _assign_image_arg(pipe, kwargs, init_image)
        filtered_kwargs = _filter_kwargs(pipe, {k: v for k, v in kwargs.items() if v is not None})
        try:
            result = pipe(**filtered_kwargs)
            frames = _extract_frames(result)
            if not frames:
                raise RuntimeError("No frames returned from LTX-2")
            for frame in frames:
                last_frame = frame
                yield frame
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Mood Mirror generation error: %s", exc)
            yield render_status_frame("Generation error", config.width, config.height)
