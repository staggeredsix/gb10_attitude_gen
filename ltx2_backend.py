from __future__ import annotations

import inspect
import logging
import os
import pathlib
import random
import threading
import time
import uuid
from collections.abc import Iterable as IterableABC
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
DEFAULT_BACKEND = "pipelines"
DEFAULT_LTX2_MODEL_ID = "Lightricks/LTX-2"
DEFAULT_FP4_FILE = "ltx-2-19b-dev-fp4.safetensors"
DEFAULT_FP8_FILE = "ltx-2-19b-dev-fp8.safetensors"


@dataclass(frozen=True)
class LTX2Artifacts:
    checkpoint_path: str
    gemma_root: str
    spatial_upsampler_path: str | None
    distilled_lora_path: str | None
    distilled_lora_strength: float
    loras: list[dict[str, object]]


@dataclass(frozen=True)
class DiffusersArtifacts:
    model_id: str
    snapshot_dir: str | None
    fp4_file: str
    allow_download: bool


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        LOGGER.warning("Invalid float for %s=%s; using %s", name, value, default)
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float_optional(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        LOGGER.warning("Invalid float for %s=%s; ignoring.", name, value)
        return None


def _env_int_clamped(name: str, default: int, *, min_value: int, max_value: int) -> int:
    value = os.getenv(name)
    if value is None:
        return max(min_value, min(default, max_value))
    try:
        parsed = int(float(value))
    except ValueError:
        LOGGER.warning("Invalid int for %s=%s; using %s", name, value, default)
        parsed = default
    return max(min_value, min(parsed, max_value))


def _log_vram(prefix: str) -> None:
    if not _env_bool("LTX2_LOG_VRAM", False):
        return
    if not torch.cuda.is_available():
        return
    try:
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        LOGGER.info("VRAM %s allocated=%.1fMB reserved=%.1fMB", prefix, allocated, reserved)
    except Exception:  # noqa: BLE001
        LOGGER.exception("Failed to query VRAM stats")


def _get_backend() -> str:
    backend = os.getenv("LTX2_BACKEND", DEFAULT_BACKEND).strip().lower()
    if backend not in {"pipelines", "diffusers"}:
        LOGGER.warning("Invalid LTX2_BACKEND=%s; defaulting to %s", backend, DEFAULT_BACKEND)
        return DEFAULT_BACKEND
    return backend


def backend_requires_gemma() -> bool:
    return _get_backend() == "pipelines"


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
            LOGGER.warning(
                "LTX2_CHECKPOINT_PATH does not exist at %s; falling back to fp8 auto-discovery.",
                path,
            )
        else:
            return str(path)

    use_distilled = os.getenv("LTX2_USE_DISTILLED", "0").lower() in {"1", "true", "yes", "on"}
    fp8_file = os.getenv(
        "LTX2_FP8_FILE",
        "ltx-2-19b-distilled-fp8.safetensors" if use_distilled else DEFAULT_FP8_FILE,
    )
    snapshot_dir = os.getenv("LTX2_SNAPSHOT_DIR")
    if snapshot_dir:
        snapshot_path = pathlib.Path(snapshot_dir).expanduser()
        if snapshot_path.exists():
            snapshots_root = snapshot_path / "snapshots"
            candidate_roots: list[pathlib.Path] = []
            if snapshots_root.is_dir():
                candidate_roots.extend(
                    sorted(
                        (p for p in snapshots_root.iterdir() if p.is_dir()),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                )
            candidate_roots.append(snapshot_path)

            for root in candidate_roots:
                candidate = root / fp8_file
                if candidate.is_file():
                    return str(candidate)
        else:
            LOGGER.warning(
                "LTX2_SNAPSHOT_DIR does not exist at %s; falling back to cache search.",
                snapshot_path,
            )

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
            search_root.rglob(fp8_file),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            LOGGER.info("LTX-2 checkpoint selected=%s", candidates[0])
            return str(candidates[0])

    raise RuntimeError(
        "No fp8 checkpoint could be resolved. "
        "Set LTX2_CHECKPOINT_PATH explicitly or set LTX2_SNAPSHOT_DIR and LTX2_FP8_FILE."
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


def _resolve_diffusers_artifacts() -> DiffusersArtifacts:
    snapshot_dir = os.getenv("LTX2_SNAPSHOT_DIR")
    if snapshot_dir:
        snapshot_path = pathlib.Path(snapshot_dir).expanduser()
        if not snapshot_path.exists():
            raise RuntimeError(f"LTX2_SNAPSHOT_DIR does not exist: {snapshot_path}")
        if (snapshot_path / "model_index.json").is_file():
            snapshot_dir = str(snapshot_path)
        else:
            snapshot_dir = None
            snapshots_root = snapshot_path / "snapshots"
            if snapshots_root.is_dir():
                candidates = sorted(
                    (p for p in snapshots_root.iterdir() if p.is_dir()),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for candidate in candidates:
                    if (candidate / "model_index.json").is_file():
                        snapshot_dir = str(candidate)
                        break
            if snapshot_dir is None:
                snapshot_dir = str(snapshot_path)
    model_id = os.getenv("LTX2_MODEL_ID", DEFAULT_LTX2_MODEL_ID)
    fp4_file = os.getenv("LTX2_FP4_FILE", DEFAULT_FP4_FILE)
    allow_download = _env_bool("LTX2_ALLOW_DOWNLOAD", default=False)
    return DiffusersArtifacts(
        model_id=model_id,
        snapshot_dir=snapshot_dir,
        fp4_file=fp4_file,
        allow_download=allow_download,
    )


def log_backend_configuration(output_mode: str | None = None) -> None:
    resolved_output_mode = output_mode or os.getenv("LTX2_OUTPUT_MODE", "native")
    backend = _get_backend()
    LOGGER.info("LTX-2 backend=%s", backend)
    LOGGER.info("LTX-2 output_mode=%s", resolved_output_mode)
    if backend == "pipelines":
        enable_fp8 = os.getenv("LTX2_ENABLE_FP8", "1").lower() in {"1", "true", "yes", "on"}
        LOGGER.info("LTX2_ENABLE_FP8=%s", enable_fp8)
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
    else:
        diffusers = _resolve_diffusers_artifacts()
        LOGGER.info("LTX-2 diffusers_model_id=%s", diffusers.model_id)
        LOGGER.info("LTX-2 diffusers_snapshot_dir=%s", diffusers.snapshot_dir)
        LOGGER.info("LTX-2 diffusers_fp4_file=%s", diffusers.fp4_file)
        LOGGER.info("LTX-2 diffusers_allow_download=%s", diffusers.allow_download)


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
    if "fp8transformer" in init_kwargs and "fp8transformer" not in params:
        has_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
        if not has_kwargs:
            raise RuntimeError(
                f"{pipe_cls.__name__} does not accept fp8transformer. "
                "Update LTX-2 or disable with LTX2_ENABLE_FP8=0."
            )

    filtered = _filter_kwargs_for_callable(factory, init_kwargs)
    LOGGER.info("Initializing %s with keys=%s", pipe_cls.__name__, sorted(filtered.keys()))
    return factory(**filtered)


def _load_pipelines_pipeline(output_mode: str, device: str = "cuda") -> object:
    cache_key = f"pipelines:{output_mode}:{device}"
    with _PIPELINE_LOCK:
        if cache_key in _PIPELINES:
            return _PIPELINES[cache_key]

        artifacts = _resolve_artifacts(output_mode)
        dtype = torch.bfloat16
        enable_fp8 = os.getenv("LTX2_ENABLE_FP8", "1").lower() in {"1", "true", "yes", "on"}

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
                "fp8transformer": enable_fp8,
                "torch_dtype": dtype,
            }
        else:
            pipe_cls = TI2VidOneStagePipeline
            init_kwargs = {
                "checkpoint_path": artifacts.checkpoint_path,
                "gemma_root": artifacts.gemma_root,
                "loras": artifacts.loras,
                "fp8transformer": enable_fp8,
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


def _load_diffusers_pipeline(output_mode: str, device: str = "cuda") -> object:
    cache_key = f"diffusers:{output_mode}:{device}"
    with _PIPELINE_LOCK:
        if cache_key in _PIPELINES:
            return _PIPELINES[cache_key]

        artifacts = _resolve_diffusers_artifacts()
        try:
            from diffusers import DiffusionPipeline
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "diffusers is required for LTX2_BACKEND=diffusers. "
                "Install it with pip install diffusers."
            ) from exc

        source = artifacts.snapshot_dir or artifacts.model_id
        local_only = not artifacts.allow_download
        LOGGER.info(
            "Loading diffusers pipeline: source=%s local_only=%s fp4_file=%s",
            source,
            local_only,
            artifacts.fp4_file,
        )

        load_kwargs: dict[str, object] = {
            "torch_dtype": torch.float16,
            "local_files_only": local_only,
        }
        if os.path.isdir(source):
            load_kwargs["cache_dir"] = None
        pipe = DiffusionPipeline.from_pretrained(source, **_filter_kwargs_for_callable(DiffusionPipeline.from_pretrained, load_kwargs))

        if artifacts.fp4_file:
            fp4_path = pathlib.Path(artifacts.fp4_file)
            if not fp4_path.is_file() and artifacts.snapshot_dir:
                candidate = pathlib.Path(artifacts.snapshot_dir) / artifacts.fp4_file
                if candidate.is_file():
                    fp4_path = candidate
            if fp4_path.is_file():
                if hasattr(pipe, "load_checkpoint"):
                    LOGGER.info("Loading fp4 checkpoint via load_checkpoint: %s", fp4_path)
                    pipe.load_checkpoint(str(fp4_path))
                elif hasattr(pipe, "load_lora_weights"):
                    LOGGER.info("Loading fp4 checkpoint via load_lora_weights: %s", fp4_path)
                    pipe.load_lora_weights(str(fp4_path))
                else:
                    LOGGER.info("fp4 checkpoint available at %s (no explicit loader on pipeline)", fp4_path)
            else:
                LOGGER.warning("FP4 checkpoint not found at %s", fp4_path)

        if hasattr(pipe, "to"):
            pipe.to(device)

        _PIPELINES[cache_key] = pipe
        return pipe


def _load_pipeline(output_mode: str, device: str = "cuda") -> object:
    backend = _get_backend()
    if backend == "diffusers":
        return _load_diffusers_pipeline(output_mode, device=device)
    return _load_pipelines_pipeline(output_mode, device=device)


def warmup_pipeline(output_mode: str) -> dict[str, str]:
    pipe = _load_pipeline(output_mode)
    return {"pipeline_class": pipe.__class__.__name__, "backend": _get_backend()}


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
    if os.getenv("LTX2_REALTIME", "0").lower() in {"1", "true", "yes", "on"} and output_mode == "native":
        width = _env_int_clamped("LTX2_REALTIME_WIDTH", 640, min_value=64, max_value=4096)
        height = _env_int_clamped("LTX2_REALTIME_HEIGHT", 352, min_value=64, max_value=4096)
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
    if "images" in param_names and "images" not in kwargs:
        kwargs["images"] = images or []
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


def _build_diffusers_kwargs(
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
) -> dict[str, object]:
    signature = inspect.signature(pipe.__call__)
    params = signature.parameters
    param_names = set(params.keys())
    kwargs: dict[str, object] = {}

    _assign_first_present(param_names, kwargs, prompt, ["prompt", "text"])
    if negative_prompt:
        _assign_first_present(param_names, kwargs, negative_prompt, ["negative_prompt"])
    _assign_first_present(param_names, kwargs, width, ["width"])
    _assign_first_present(param_names, kwargs, height, ["height"])
    _assign_first_present(param_names, kwargs, num_frames, ["num_frames", "video_length", "frames"])
    _assign_first_present(param_names, kwargs, fps, ["fps", "frame_rate"])
    _assign_first_present(param_names, kwargs, guidance_scale, ["guidance_scale", "cfg_scale", "cfg_guidance_scale"])
    _assign_first_present(param_names, kwargs, num_inference_steps, ["num_inference_steps", "steps"])

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
    with torch.inference_mode():
        if os.getenv("LTX2_LOG_GRAD") == "1":
            LOGGER.info("torch.is_grad_enabled()=%s", torch.is_grad_enabled())
        return pipe(**kwargs)


def _extract_video_frames_from_pipeline_result(result: object) -> list[Image.Image]:
    if os.getenv("LTX2_LOG_PIPE_RESULT") == "1":
        try:
            if isinstance(result, tuple):
                elem_types = [type(item).__name__ for item in result]
                LOGGER.info("LTX2 pipe result=tuple types=%s", elem_types)
                for idx, item in enumerate(result):
                    if torch.is_tensor(item):
                        LOGGER.info(
                            "LTX2 pipe result[%s] tensor shape=%s dtype=%s",
                            idx,
                            tuple(item.shape),
                            item.dtype,
                        )
            elif torch.is_tensor(result):
                LOGGER.info(
                    "LTX2 pipe result tensor shape=%s dtype=%s",
                    tuple(result.shape),
                    result.dtype,
                )
            else:
                LOGGER.info("LTX2 pipe result type=%s", type(result).__name__)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to log pipeline result details")

    def _as_frames(obj: object) -> list[Image.Image] | None:
        def _tensor_to_images(tensor: torch.Tensor) -> list[Image.Image]:
            if tensor.is_cuda:
                tensor = tensor.detach().cpu()
            else:
                tensor = tensor.detach()
            if tensor.dim() == 4:
                # (F,C,H,W) or (F,H,W,C)
                if tensor.shape[1] in (1, 3, 4):
                    frames = [tensor[i] for i in range(tensor.shape[0])]
                else:
                    frames = [tensor[i] for i in range(tensor.shape[0])]
                return sum((_tensor_to_images(frame) for frame in frames), [])
            if tensor.dim() == 3:
                if tensor.shape[0] in (1, 3, 4):
                    tensor = tensor.permute(1, 2, 0)
                # else assume HWC already
            elif tensor.dim() == 2:
                tensor = tensor.unsqueeze(-1)
            elif tensor.dim() == 1:
                return []

            array = tensor
            if array.dtype.is_floating_point:
                array = array.clamp(0, 1) * 255.0
            array = array.clamp(0, 255).to(torch.uint8)
            np_img = array.numpy()
            if np_img.ndim == 3 and np_img.shape[2] == 1:
                np_img = np_img[:, :, 0]
            return [Image.fromarray(np_img)]

        if obj is None:
            return None
        if isinstance(obj, Image.Image):
            return [obj]
        if isinstance(obj, np.ndarray):
            return [Image.fromarray(obj)]
        if torch.is_tensor(obj):
            return _tensor_to_images(obj)
        if isinstance(obj, (str, os.PathLike, pathlib.Path)):
            path = pathlib.Path(obj)
            if path.exists() and path.is_file():
                return _decode_video_to_frames(str(path))
            return None
        if isinstance(obj, tuple) and obj:
            return _as_frames(obj[0])
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
        if isinstance(obj, IterableABC) and not isinstance(
            obj,
            (
                dict,
                str,
                bytes,
                bytearray,
                np.ndarray,
                Image.Image,
                os.PathLike,
                pathlib.Path,
            ),
        ):
            frames: list[Image.Image] = []
            for item in obj:
                item_frames = _as_frames(item)
                if item_frames:
                    frames.extend(item_frames)
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

    if isinstance(result, (tuple, list)) and len(result) >= 1:
        frames = _as_frames(result[0])
        if frames:
            return frames

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
    _log_vram("chunk_start")
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
    _log_vram("chunk_end")
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
    _log_vram("diffusers_chunk_end")
    frames = _extract_video_frames_from_pipeline_result(result)
    for frame in frames:
        yield frame


def _should_retry_without_negative_prompt(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        keyword in message
        for keyword in (
            "negative_prompt",
            "encoder_hidden_states",
            "size mismatch",
            "shape",
            "dimension",
        )
    )


def _generate_diffusers_chunk(
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
) -> Iterable[Image.Image]:
    _log_vram("diffusers_chunk_start")
    kwargs = _build_diffusers_kwargs(
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
    )
    try:
        result = pipe(**kwargs)
    except Exception as exc:  # noqa: BLE001
        if kwargs.get("negative_prompt") and _should_retry_without_negative_prompt(exc):
            LOGGER.warning("Retrying diffusers call without negative_prompt due to: %s", exc)
            kwargs.pop("negative_prompt", None)
            result = pipe(**kwargs)
        else:
            raise
    frames = _extract_video_frames_from_pipeline_result(result)
    for frame in frames:
        yield frame


def _get_pipelines_pipe_or_status(config) -> object | None:
    try:
        return _load_pipeline(getattr(config, "output_mode", "native"))
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to load pipelines backend: %s", exc)
        return None


def _get_diffusers_pipe_or_status(config) -> object | None:
    try:
        return _load_pipeline(getattr(config, "output_mode", "native"))
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to load diffusers backend: %s", exc)
        return None


def generate_fever_dream_frames(config, cancel_event: threading.Event) -> Iterable[Image.Image]:
    backend = _get_backend()
    output_mode = getattr(config, "output_mode", "native")
    if backend == "diffusers":
        pipe = _get_diffusers_pipe_or_status(config)
        if pipe is None:
            while not cancel_event.is_set():
                yield render_status_frame("Diffusers backend unavailable", config.width, config.height)
                time.sleep(1.0 / max(1, config.fps))
            return
        stage_width, stage_height = _resolve_stage_dimensions(config)
        LOGGER.info(
            "Fever Dream diffusers output_mode=%s stage_size=%sx%s final_size=%sx%s fps=%s",
            output_mode,
            stage_width,
            stage_height,
            config.width,
            config.height,
            config.fps,
        )
        while not cancel_event.is_set():
            prompt = _prompt_drift(config.prompt)
            chunk_seconds = _env_float("LTX2_CHUNK_SECONDS", 0.25)
            min_frames = _env_int_clamped("LTX2_MIN_FRAMES", 5, min_value=1, max_value=120)
            num_frames = _adjust_num_frames(max(min_frames, int(chunk_seconds * config.fps)))
            seed = None
            if config.seed is not None:
                seed = config.seed + int(time.time())
            try:
                realtime = os.getenv("LTX2_REALTIME", "0").lower() in {"1", "true", "yes", "on"}
                guidance_scale = 3.0 + config.dream_strength * 5.0
                if realtime:
                    guidance_scale = _env_float("LTX2_REALTIME_CFG", 1.0)
                if realtime and guidance_scale <= 1.0:
                    negative_prompt = ""
                else:
                    negative_prompt = getattr(config, "negative_prompt", "") or ""
                num_inference_steps = int(10 + config.motion * 10)
                if realtime:
                    num_inference_steps = min(num_inference_steps, _env_int_clamped("LTX2_REALTIME_STEPS", 6, min_value=1, max_value=200))
                frames = _generate_diffusers_chunk(
                    pipe,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=stage_width,
                    height=stage_height,
                    num_frames=num_frames,
                    fps=config.fps,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                )
                for frame in frames:
                    yield frame
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Fever Dream diffusers generation error: %s", exc)
                yield render_status_frame("Generation error", config.width, config.height)
        return

    pipe = _get_pipelines_pipe_or_status(config)
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
        chunk_seconds = _env_float("LTX2_CHUNK_SECONDS", 0.25)
        min_frames = _env_int_clamped("LTX2_MIN_FRAMES", 5, min_value=1, max_value=120)
        num_frames = _adjust_num_frames(max(min_frames, int(chunk_seconds * config.fps)))
        seed = None
        if config.seed is not None:
            seed = config.seed + int(time.time())
        try:
            realtime = os.getenv("LTX2_REALTIME", "0").lower() in {"1", "true", "yes", "on"}
            guidance_scale = 3.0 + config.dream_strength * 5.0
            if realtime:
                guidance_scale = _env_float("LTX2_REALTIME_CFG", 1.0)
            if realtime and guidance_scale <= 1.0:
                negative_prompt = ""
            else:
                negative_prompt = getattr(config, "negative_prompt", "") or ""
            num_inference_steps = int(10 + config.motion * 10)
            if realtime:
                num_inference_steps = min(num_inference_steps, _env_int_clamped("LTX2_REALTIME_STEPS", 6, min_value=1, max_value=200))
            frames = _generate_video_chunk(
                pipe,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=stage_width,
                height=stage_height,
                num_frames=num_frames,
                fps=config.fps,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
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
    backend = _get_backend()
    output_mode = getattr(config, "output_mode", "native")
    if backend == "diffusers":
        pipe = _get_diffusers_pipe_or_status(config)
        if pipe is None:
            while not cancel_event.is_set():
                yield render_status_frame("Diffusers backend unavailable", config.width, config.height)
                time.sleep(1.0 / max(1, config.fps))
            return
        stage_width, stage_height = _resolve_stage_dimensions(config)
        LOGGER.info(
            "Mood Mirror diffusers output_mode=%s stage_size=%sx%s final_size=%sx%s fps=%s",
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
            chunk_seconds = _env_float("LTX2_CHUNK_SECONDS", 0.25)
            min_frames = _env_int_clamped("LTX2_MIN_FRAMES", 5, min_value=1, max_value=120)
            num_frames = _adjust_num_frames(max(min_frames, int(chunk_seconds * config.fps)))
            seed = None
            if config.seed is not None:
                seed = config.seed + int(time.time())
            try:
                realtime = os.getenv("LTX2_REALTIME", "0").lower() in {"1", "true", "yes", "on"}
                guidance_scale = 3.0 + config.dream_strength * 4.0
                if realtime:
                    guidance_scale = _env_float("LTX2_REALTIME_CFG", 1.0)
                if realtime and guidance_scale <= 1.0:
                    negative_prompt = ""
                else:
                    negative_prompt = getattr(config, "negative_prompt", "") or ""
                num_inference_steps = int(10 + config.motion * 10)
                if realtime:
                    num_inference_steps = min(num_inference_steps, _env_int_clamped("LTX2_REALTIME_STEPS", 6, min_value=1, max_value=200))
                frames = _generate_diffusers_chunk(
                    pipe,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=stage_width,
                    height=stage_height,
                    num_frames=num_frames,
                    fps=config.fps,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                )
                for frame in frames:
                    yield frame
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Mood Mirror diffusers generation error: %s", exc)
                yield render_status_frame("Generation error", config.width, config.height)
        return

    pipe = _get_pipelines_pipe_or_status(config)
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
        chunk_seconds = _env_float("LTX2_CHUNK_SECONDS", 0.25)
        min_frames = _env_int_clamped("LTX2_MIN_FRAMES", 5, min_value=1, max_value=120)
        num_frames = _adjust_num_frames(max(min_frames, int(chunk_seconds * config.fps)))
        seed = None
        if config.seed is not None:
            seed = config.seed + int(time.time())
        strength = 0.2 + (1.0 - config.identity_strength) * 0.6
        image_path = _write_temp_image(camera_frame)
        images = [(image_path, 0, strength)]
        try:
            realtime = os.getenv("LTX2_REALTIME", "0").lower() in {"1", "true", "yes", "on"}
            guidance_scale = 3.0 + config.dream_strength * 4.0
            if realtime:
                guidance_scale = _env_float("LTX2_REALTIME_CFG", 1.0)
            if realtime and guidance_scale <= 1.0:
                negative_prompt = ""
            else:
                negative_prompt = getattr(config, "negative_prompt", "") or ""
            num_inference_steps = int(10 + config.motion * 10)
            if realtime:
                num_inference_steps = min(num_inference_steps, _env_int_clamped("LTX2_REALTIME_STEPS", 6, min_value=1, max_value=200))
            frames = _generate_video_chunk(
                pipe,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=stage_width,
                height=stage_height,
                num_frames=num_frames,
                fps=config.fps,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
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
