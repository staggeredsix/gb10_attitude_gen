from __future__ import annotations

import inspect
import logging
import os
import pathlib
import random
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Iterable

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

LOGGER = logging.getLogger("ltx2_backend")

_PIPELINES: dict[str, object] = {}
_PIPELINE_LOCK = threading.Lock()

DEFAULT_GEMMA_MODEL_ID = "google/gemma-3-12b"
DEFAULT_GEMMA_ROOT = "/models/gemma"


@dataclass(frozen=True)
class LTX2Artifacts:
    checkpoint_path: str
    gemma_root: str
    spatial_upsampler_path: str | None
    distilled_lora_path: str | None
    distilled_lora_strength: float
    loras: list[dict[str, object]]


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        LOGGER.warning("Invalid float for %s=%s; using %s", name, value, default)
        return default


def _has_preprocessor_config(root: pathlib.Path) -> bool:
    if not root.exists():
        return False
    if (root / "preprocessor_config.json").is_file():
        return True
    return any(root.rglob("preprocessor_config.json"))


def validate_gemma_root(path: str) -> tuple[bool, str]:
    if not path:
        return False, "LTX2_GEMMA_ROOT is not set."
    root = pathlib.Path(path).expanduser()
    if not root.exists():
        return False, f"Gemma root does not exist: {root}"
    config_path = root / "config.json"
    if not config_path.is_file():
        return False, f"Missing config.json under: {root}"
    if not _has_preprocessor_config(root):
        return (
            False,
            "Missing preprocessor_config.json. Re-run ./download_model.sh gemma "
            "(it must include preprocessor_config.json).",
        )
    tokenizer_candidates = [
        root / "tokenizer.json",
        root / "tokenizer.model",
        root / "tokenizer_config.json",
    ]
    if not any(candidate.is_file() for candidate in tokenizer_candidates):
        if not any(candidate.is_file() for candidate in root.glob("tokenizer.*")):
            return False, f"Missing tokenizer files under: {root}"
    return True, "ok"


def _resolve_checkpoint_path() -> str:
    env_value = os.getenv("LTX2_CHECKPOINT_PATH")
    if env_value:
        path = pathlib.Path(env_value).expanduser()
        if not path.exists():
            raise RuntimeError(f"LTX2_CHECKPOINT_PATH does not exist: {path}")
        return str(path)

    cache_roots = [
        os.getenv("HUGGINGFACE_HUB_CACHE"),
        os.getenv("HF_HOME"),
        "/models/huggingface/hub",
    ]
    for root in filter(None, cache_roots):
        root_path = pathlib.Path(root).expanduser()
        if not root_path.exists():
            continue
        repo_root = root_path / "models--Lightricks--LTX-2"
        search_root = repo_root if repo_root.exists() else root_path
        candidates = sorted(
            search_root.rglob("*fp4*.safetensors"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return str(candidates[0])

    raise RuntimeError(
        "LTX2_CHECKPOINT_PATH is not set and no fp4 checkpoint was found in cache. "
        "Set LTX2_CHECKPOINT_PATH to the fp4 checkpoint file path."
    )


def _require_env_path(name: str, *, required: bool = True) -> str | None:
    value = os.getenv(name)
    if not value:
        if required:
            raise RuntimeError(f"{name} is required but not set.")
        return None
    path = pathlib.Path(value).expanduser()
    if required and not path.exists():
        raise RuntimeError(f"{name} does not exist at: {path}")
    return str(path)


def _resolve_gemma_root() -> str:
    return os.getenv("LTX2_GEMMA_ROOT", DEFAULT_GEMMA_ROOT)


def _resolve_artifacts(output_mode: str, *, require_gemma: bool = True) -> LTX2Artifacts:
    checkpoint_path = _resolve_checkpoint_path()
    gemma_root = _resolve_gemma_root()
    gemma_ok, gemma_reason = validate_gemma_root(gemma_root)
    if require_gemma and not gemma_ok:
        raise RuntimeError(
            "Gemma is required. Set LTX2_GEMMA_ROOT to a directory created by "
            f"download_model.sh ({DEFAULT_GEMMA_MODEL_ID}). ({gemma_reason})"
        )
    spatial_upsampler_path = _require_env_path("LTX2_SPATIAL_UPSAMPLER_PATH", required=output_mode == "upscaled")
    distilled_lora_path = _require_env_path("LTX2_DISTILLED_LORA_PATH", required=False)
    distilled_lora_strength = _env_float("LTX2_DISTILLED_LORA_STRENGTH", 0.6)
    loras: list[dict[str, object]] = []

    return LTX2Artifacts(
        checkpoint_path=checkpoint_path,
        gemma_root=gemma_root,
        spatial_upsampler_path=spatial_upsampler_path,
        distilled_lora_path=distilled_lora_path,
        distilled_lora_strength=distilled_lora_strength,
        loras=loras,
    )


def log_backend_configuration(output_mode: str | None = None) -> None:
    resolved_output_mode = output_mode or os.getenv("LTX2_OUTPUT_MODE", "native")
    LOGGER.info("LTX-2 output_mode=%s", resolved_output_mode)
    try:
        checkpoint_path = _resolve_checkpoint_path()
        LOGGER.info("LTX-2 checkpoint_path=%s", checkpoint_path)
    except RuntimeError as exc:
        LOGGER.warning("LTX-2 checkpoint resolution failed: %s", exc)
    gemma_root = _resolve_gemma_root()
    gemma_ok, gemma_reason = validate_gemma_root(gemma_root)
    gemma_has_preprocessor = _has_preprocessor_config(pathlib.Path(gemma_root).expanduser())
    LOGGER.info("LTX-2 gemma_root=%s", gemma_root)
    LOGGER.info("LTX-2 gemma_preprocessor_config=%s", gemma_has_preprocessor)
    LOGGER.info("LTX-2 gemma_model_id=%s", DEFAULT_GEMMA_MODEL_ID)
    if not gemma_ok:
        LOGGER.warning(
            "Gemma is required for LTX-2 pipelines. Set LTX2_GEMMA_ROOT to the Gemma "
            "directory created by download_model.sh. (%s)",
            gemma_reason,
        )
    if resolved_output_mode == "upscaled":
        LOGGER.info("LTX-2 spatial_upsampler_path=%s", os.getenv("LTX2_SPATIAL_UPSAMPLER_PATH"))
        LOGGER.info("LTX-2 distilled_lora_path=%s", os.getenv("LTX2_DISTILLED_LORA_PATH"))
        LOGGER.info("LTX-2 distilled_lora_strength=%s", _env_float("LTX2_DISTILLED_LORA_STRENGTH", 0.6))


def _filter_kwargs_for_callable(func: Callable[..., object], kwargs: dict[str, object]) -> dict[str, object]:
    signature = inspect.signature(func)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    allowed = set(signature.parameters.keys())
    return {key: value for key, value in kwargs.items() if key in allowed}


def _instantiate_pipeline(pipe_cls: type, kwargs: dict[str, object], *, output_mode: str) -> object:
    if hasattr(pipe_cls, "from_pretrained"):
        factory = getattr(pipe_cls, "from_pretrained")
        signature = inspect.signature(factory)
    else:
        factory = pipe_cls
        signature = inspect.signature(pipe_cls)

    init_kwargs = dict(kwargs)
    params = signature.parameters

    if "loras" in params and "loras" not in init_kwargs:
        init_kwargs["loras"] = []
    if "distilled_lora" in params and output_mode == "upscaled" and not init_kwargs.get("distilled_lora"):
        raise RuntimeError(
            "Upscaled output requires a distilled LoRA. "
            "Set LTX2_DISTILLED_LORA_PATH and LTX2_DISTILLED_LORA_STRENGTH."
        )

    filtered = _filter_kwargs_for_callable(factory, init_kwargs)
    LOGGER.info("Initializing %s with keys=%s", pipe_cls.__name__, sorted(filtered.keys()))
    return factory(**filtered)


def _load_pipeline(output_mode: str, device: str = "cuda"):
    cache_key = f"{output_mode}:{device}"
    with _PIPELINE_LOCK:
        if cache_key in _PIPELINES:
            return _PIPELINES[cache_key]

        artifacts = _resolve_artifacts(output_mode)
        dtype = torch.bfloat16

        if output_mode == "upscaled":
            pipe_cls = TI2VidTwoStagesPipeline
            distilled_lora = None
            if artifacts.distilled_lora_path:
                distilled_lora = {
                    "path": artifacts.distilled_lora_path,
                    "strength": artifacts.distilled_lora_strength,
                }
            init_kwargs = {
                "checkpoint_path": artifacts.checkpoint_path,
                "gemma_root": artifacts.gemma_root,
                "spatial_upsampler_path": artifacts.spatial_upsampler_path,
                "distilled_lora": distilled_lora,
                "loras": artifacts.loras,
                "torch_dtype": dtype,
            }
        else:
            pipe_cls = TI2VidOneStagePipeline
            init_kwargs = {
                "checkpoint_path": artifacts.checkpoint_path,
                "gemma_root": artifacts.gemma_root,
                "loras": artifacts.loras,
                "torch_dtype": dtype,
            }

        LOGGER.info("Loading LTX-2 pipeline: mode=%s class=%s", output_mode, pipe_cls.__name__)
        pipe = _instantiate_pipeline(pipe_cls, init_kwargs, output_mode=output_mode)
        if hasattr(pipe, "to"):
            pipe.to(device)
        call_signature = inspect.signature(pipe.__call__)
        supports_output_path = any(name in call_signature.parameters for name in ("output_path", "output"))
        required_params = [
            name
            for name, param in call_signature.parameters.items()
            if param.default is inspect.Parameter.empty
            and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]
        LOGGER.info("LTX-2 pipeline call signature: %s", call_signature)
        LOGGER.info("LTX-2 pipeline supports output_path=%s", supports_output_path)
        LOGGER.info("LTX-2 pipeline required params=%s", required_params)

        _PIPELINES[cache_key] = pipe
        return pipe


def warmup_pipeline(output_mode: str) -> dict[str, str]:
    pipe = _load_pipeline(output_mode)
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


def _adjust_num_frames(num_frames: int) -> int:
    if num_frames < 1:
        return 1
    if (num_frames - 1) % 8 == 0:
        return num_frames
    return ((num_frames - 1) // 8 + 1) * 8 + 1


def _resolve_stage_dimensions(config) -> tuple[int, int]:
    output_mode = getattr(config, "output_mode", "native")
    width = config.width
    height = config.height
    if output_mode == "upscaled":
        stage_width = width // 2
        stage_height = height // 2
        return stage_width, stage_height
    return width, height


def _assign_first_present(params: set[str], kwargs: dict[str, object], value: object, names: list[str]) -> None:
    for name in names:
        if name in params and value is not None:
            kwargs[name] = value
            return


def _build_pipeline_kwargs(
    pipe: object,
    *,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int | None,
    output_path: str | None,
    images: list[tuple[str, int, float]] | None = None,
) -> dict[str, object]:
    signature = inspect.signature(pipe.__call__)
    params = signature.parameters
    param_names = set(params.keys())
    kwargs: dict[str, object] = {}

    if not any(name in param_names for name in ("prompt", "text")):
        raise RuntimeError("LTX-2 pipeline does not accept a prompt argument.")
    _assign_first_present(param_names, kwargs, prompt, ["prompt", "text"])
    _assign_first_present(param_names, kwargs, negative_prompt, ["negative_prompt"])
    _assign_first_present(param_names, kwargs, width, ["width"])
    _assign_first_present(param_names, kwargs, height, ["height"])
    _assign_first_present(param_names, kwargs, num_frames, ["num_frames", "video_length", "frames"])
    _assign_first_present(param_names, kwargs, fps, ["fps", "frame_rate"])
    _assign_first_present(param_names, kwargs, guidance_scale, ["guidance_scale", "cfg_scale", "cfg_guidance_scale"])
    _assign_first_present(param_names, kwargs, num_inference_steps, ["num_inference_steps", "steps"])
    if output_path and any(name in param_names for name in ("output_path", "output")):
        _assign_first_present(param_names, kwargs, output_path, ["output_path", "output"])

    images_required = "images" in params and params["images"].default is inspect.Parameter.empty
    if images is not None and "images" in param_names:
        kwargs["images"] = images
    elif images_required and "images" not in kwargs:
        kwargs["images"] = []

    needs_seed = any(name in param_names for name in ("seed", "random_seed", "generator"))
    seed_value = seed if seed is not None else (random.getrandbits(31) if needs_seed else None)
    if seed_value is not None:
        if "generator" in param_names:
            device = getattr(pipe, "device", "cuda")
            generator = torch.Generator(device=device).manual_seed(seed_value)
            kwargs["generator"] = generator
        _assign_first_present(param_names, kwargs, seed_value, ["seed", "random_seed"])

    return _filter_kwargs_for_callable(pipe.__call__, kwargs)


def _write_temp_image(frame_bgr: np.ndarray) -> str:
    temp_path = pathlib.Path(f"/tmp/img_{uuid.uuid4().hex}.png")
    cv2.imwrite(str(temp_path), frame_bgr)
    return str(temp_path)


def _yield_video_frames(video_path: str) -> Iterable[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open generated video: {video_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield Image.fromarray(frame_rgb)
    finally:
        cap.release()


def _decode_video_to_frames(video_path: str) -> list[Image.Image]:
    return list(_yield_video_frames(video_path))


def _call_ltx_pipeline(pipe: object, kwargs: dict[str, object]) -> object:
    return pipe(**kwargs)


def _extract_video_frames_from_pipeline_result(result: object) -> list[Image.Image]:
    def _as_frames(obj: object) -> list[Image.Image] | None:
        if obj is None:
            return None
        if isinstance(obj, Image.Image):
            return [obj]
        if isinstance(obj, np.ndarray):
            return [Image.fromarray(obj)]
        if isinstance(obj, (str, os.PathLike, pathlib.Path)):
            path = pathlib.Path(obj)
            if path.exists() and path.is_file():
                return _decode_video_to_frames(str(path))
            return None
        if isinstance(obj, list):
            frames: list[Image.Image] = []
            for item in obj:
                item_frames = _as_frames(item)
                if item_frames:
                    frames.extend(item_frames)
                elif isinstance(item, np.ndarray):
                    frames.append(Image.fromarray(item))
                elif isinstance(item, Image.Image):
                    frames.append(item)
            return frames or None
        if isinstance(obj, dict):
            for key in ("frames", "images", "video", "videos", "output", "result"):
                if key in obj:
                    frames = _as_frames(obj[key])
                    if frames:
                        return frames
            for value in obj.values():
                frames = _as_frames(value)
                if frames:
                    return frames
            return None
        for attr in ("frames", "images", "video", "videos", "output", "result"):
            if hasattr(obj, attr):
                frames = _as_frames(getattr(obj, attr))
                if frames:
                    return frames
        return None

    frames = _as_frames(result)
    if frames:
        return frames

    result_type = type(result).__name__
    dict_keys: list[str] = []
    attrs: list[str] = []
    if isinstance(result, dict):
        dict_keys = sorted(result.keys())
    else:
        attrs = [name for name in ("frames", "images", "video", "videos", "output", "result") if hasattr(result, name)]
    raise RuntimeError(
        f"Could not extract frames from pipeline result type={result_type} keys={dict_keys} attrs={attrs}"
    )


def _generate_video_chunk(
    pipe: object,
    *,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int | None,
    images: list[tuple[str, int, float]] | None = None,
) -> Iterable[Image.Image]:
    signature = inspect.signature(pipe.__call__)
    supports_output_path = any(name in signature.parameters for name in ("output_path", "output"))
    output_path = f"/tmp/ltx_out_{uuid.uuid4().hex}.mp4" if supports_output_path else None
    kwargs = _build_pipeline_kwargs(
        pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        output_path=output_path,
        images=images,
    )
    result = _call_ltx_pipeline(pipe, kwargs)
    if output_path and pathlib.Path(output_path).is_file():
        try:
            for frame in _yield_video_frames(output_path):
                yield frame
        finally:
            try:
                os.remove(output_path)
            except OSError:
                LOGGER.warning("Failed to remove temporary video: %s", output_path)
        return
    frames = _extract_video_frames_from_pipeline_result(result)
    for frame in frames:
        yield frame


def generate_fever_dream_frames(config, cancel_event: threading.Event) -> Iterable[Image.Image]:
    output_mode = getattr(config, "output_mode", "native")
    pipe = _load_pipeline(output_mode)
    if pipe is None:
        while not cancel_event.is_set():
            yield render_status_frame("LTX-2 load failed", config.width, config.height)
            time.sleep(1.0 / max(1, config.fps))
        return
    stage_width, stage_height = _resolve_stage_dimensions(config)
    LOGGER.info(
        "Fever Dream output_mode=%s stage_size=%sx%s final_size=%sx%s fps=%s",
        output_mode,
        stage_width,
        stage_height,
        config.width,
        config.height,
        config.fps,
    )
    while not cancel_event.is_set():
        prompt = _prompt_drift(config.prompt)
        chunk_seconds = 1.0
        num_frames = _adjust_num_frames(max(1, int(chunk_seconds * config.fps)))
        seed = None
        if config.seed is not None:
            seed = config.seed + int(time.time())
        try:
            frames = _generate_video_chunk(
                pipe,
                prompt=prompt,
                negative_prompt=getattr(config, "negative_prompt", "") or "",
                width=stage_width,
                height=stage_height,
                num_frames=num_frames,
                fps=config.fps,
                guidance_scale=3.0 + config.dream_strength * 5.0,
                num_inference_steps=int(10 + config.motion * 10),
                seed=seed,
            )
            for frame in frames:
                yield frame
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Fever Dream generation error: %s", exc)
            yield render_status_frame("Generation error", config.width, config.height)


def generate_mood_mirror_frames(
    config,
    latest_camera_state: Callable[[], tuple[np.ndarray | None, dict | None]],
    cancel_event: threading.Event,
) -> Iterable[Image.Image]:
    output_mode = getattr(config, "output_mode", "native")
    pipe = _load_pipeline(output_mode)
    if pipe is None:
        while not cancel_event.is_set():
            yield render_status_frame("LTX-2 load failed", config.width, config.height)
            time.sleep(1.0 / max(1, config.fps))
        return
    stage_width, stage_height = _resolve_stage_dimensions(config)
    LOGGER.info(
        "Mood Mirror output_mode=%s stage_size=%sx%s final_size=%sx%s fps=%s",
        output_mode,
        stage_width,
        stage_height,
        config.width,
        config.height,
        config.fps,
    )
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
        num_frames = _adjust_num_frames(max(1, int(chunk_seconds * config.fps)))
        seed = None
        if config.seed is not None:
            seed = config.seed + int(time.time())
        strength = 0.2 + (1.0 - config.identity_strength) * 0.6
        image_path = _write_temp_image(camera_frame)
        images = [(image_path, 0, strength)]
        try:
            frames = _generate_video_chunk(
                pipe,
                prompt=prompt,
                negative_prompt=getattr(config, "negative_prompt", "") or "",
                width=stage_width,
                height=stage_height,
                num_frames=num_frames,
                fps=config.fps,
                guidance_scale=3.0 + config.dream_strength * 4.0,
                num_inference_steps=int(10 + config.motion * 10),
                seed=seed,
                images=images,
            )
            for frame in frames:
                yield frame
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Mood Mirror generation error: %s", exc)
            yield render_status_frame("Generation error", config.width, config.height)
        finally:
            try:
                os.remove(image_path)
            except OSError:
                LOGGER.warning("Failed to remove temporary image: %s", image_path)
