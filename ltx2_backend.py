from __future__ import annotations

import inspect
import logging
import os
import pathlib
import random
import shutil
import sys
import subprocess
import threading
import time
import uuid
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass, field
from typing import Callable, Iterable

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from settings_loader import load_settings_conf

load_settings_conf()

os.environ.setdefault("TRANSFORMERS_USE_FAST", "0")
os.environ.setdefault("HF_USE_FAST_TOKENIZERS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_GEMMA_TOKENIZER_SHIM_LOGGED = False

def _shim_gemma_tokenizer_attrs(tokenizer: object) -> None:
    global _GEMMA_TOKENIZER_SHIM_LOGGED
    shimmed: list[str] = []
    # Compatibility: some tokenizer variants lack Gemma3Processor special token attrs (boi/eoi).
    if not hasattr(tokenizer, "boi_token"):
        bos = getattr(tokenizer, "bos_token", None)
        setattr(tokenizer, "boi_token", bos or "<boi>")
        shimmed.append("boi_token")
    if not hasattr(tokenizer, "boi_token_id"):
        bos_id = getattr(tokenizer, "bos_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        setattr(tokenizer, "boi_token_id", bos_id if bos_id is not None else eos_id)
        shimmed.append("boi_token_id")
    if not hasattr(tokenizer, "eoi_token"):
        eos = getattr(tokenizer, "eos_token", None)
        setattr(tokenizer, "eoi_token", eos or "<eoi>")
        shimmed.append("eoi_token")
    if not hasattr(tokenizer, "eoi_token_id"):
        eos_id = getattr(tokenizer, "eos_token_id", None)
        bos_id = getattr(tokenizer, "bos_token_id", None)
        setattr(tokenizer, "eoi_token_id", eos_id if eos_id is not None else bos_id)
        shimmed.append("eoi_token_id")
    if shimmed and not _GEMMA_TOKENIZER_SHIM_LOGGED:
        LOGGER.info(
            "Tokenizer shim applied for %s: %s",
            tokenizer.__class__.__name__,
            ", ".join(shimmed),
        )
        _GEMMA_TOKENIZER_SHIM_LOGGED = True


try:
    from transformers import GemmaTokenizerFast  # type: ignore

    if not hasattr(GemmaTokenizerFast, "image_token_id"):
        def _image_token_id(self) -> int:  # type: ignore
            token = getattr(self, "image_token", "<image>")
            return self.convert_tokens_to_ids(token)

        GemmaTokenizerFast.image_token_id = property(_image_token_id)  # type: ignore

    _orig_init_fast = GemmaTokenizerFast.__init__
    def _init_fast(self, *args, **kwargs):  # type: ignore
        _orig_init_fast(self, *args, **kwargs)
        _shim_gemma_tokenizer_attrs(self)

    GemmaTokenizerFast.__init__ = _init_fast  # type: ignore
except Exception:
    pass

try:
    from transformers import GemmaTokenizer  # type: ignore

    _orig_init = GemmaTokenizer.__init__
    def _init(self, *args, **kwargs):  # type: ignore
        _orig_init(self, *args, **kwargs)
        _shim_gemma_tokenizer_attrs(self)

    GemmaTokenizer.__init__ = _init  # type: ignore
except Exception:
    pass

from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.ic_lora import ICLoraPipeline

LOGGER = logging.getLogger("ltx2_backend")

_PIPELINES: dict[str, object] = {}
_PIPELINE_LOCK = threading.Lock()
_CHAIN_FRAMES: dict[tuple[str, int, int], Image.Image] = {}
_CHAIN_COUNTER: dict[tuple[str, int, int], int] = {}
_CHAIN_COUNTER_LOCK = threading.Lock()
_COMMERCIAL_STATE: dict[tuple[int, int], "CommercialState"] = {}
_COMMERCIAL_LOCK = threading.Lock()
_COMMERCIAL_SEEDS: dict[tuple[int, int], int] = {}
_COMMERCIAL_SEED_LOCK = threading.Lock()
_COMMERCIAL_MP4: dict[int, tuple[str, float]] = {}
_COMMERCIAL_MP4_LOCK = threading.Lock()
_COMMERCIAL_DONE: dict[tuple[int, int], bool] = {}
_COMMERCIAL_DONE_LOCK = threading.Lock()
_CFG_STATE_LOCK = threading.Lock()
_CFG_COUNTER: dict[tuple[str, int, int], int] = {}
_COMFY_WARNED_SCHED = False
_COMFY_WARNED_VAE = False
_COMFY_WARNED_STAGE2 = False
_AUDIO_LOCK = threading.Lock()
_LATEST_AUDIO_BY_STREAM: dict[int, tuple[bytes, float]] = {}
_AUDIO_STREAM_LOCAL = threading.local()
_FFMPEG_AVAILABLE: bool | None = None

DEFAULT_GEMMA_MODEL_ID = "google/gemma-3-12b"
DEFAULT_GEMMA_ROOT = "/models/gemma"
DEFAULT_BACKEND = "pipelines"
DEFAULT_LTX2_MODEL_ID = "Lightricks/LTX-2"
DEFAULT_FP4_FILE = "ltx-2-19b-dev-fp4.safetensors"
DEFAULT_FP8_FILE = "ltx-2-19b-dev-fp8.safetensors"
DEFAULT_SPATIAL_UPSCALER_FILE = "ltx-2-spatial-upscaler-x2-1.0.safetensors"


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


@dataclass
class CommercialState:
    seed: int
    prompt: str
    negative_prompt: str
    pipeline_mode: str
    stage_width: int
    stage_height: int
    num_frames: int
    fps: int
    guidance_scale: float
    num_inference_steps: int
    chain_strength: float
    chain_frames: int
    drop_prefix: int
    output_mode: str
    reset_interval: int
    blend_frames: int
    chunk_index: int = 0
    anchor_frame: Image.Image | None = None
    last_frame: Image.Image | None = None
    last_output_path: str | None = None
    video_chunks: list[str] = field(default_factory=list)
    audio_wav_chunks: list[str] = field(default_factory=list)
    last_audio_ts: float = 0.0
    target_seconds: float = 35.0
    frames_per_chunk: int = 73
    total_chunks: int = 1
    chunks_generated: int = 0
    frames_generated: int = 0


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


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(float(value))
    except ValueError:
        LOGGER.warning("Invalid int for %s=%s; using %s", name, value, default)
        return default


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

    env_use_distilled = os.getenv("LTX2_USE_DISTILLED")
    use_distilled = env_use_distilled is None or env_use_distilled.lower() in {"1", "true", "yes", "on"}
    distilled_file = os.getenv("LTX2_DISTILLED_FP8_FILE", "ltx-2-19b-distilled-fp8.safetensors")
    fallback_file = DEFAULT_FP8_FILE
    if use_distilled:
        fp8_file = distilled_file
        if os.getenv("LTX2_FP8_FILE") and os.getenv("LTX2_FP8_FILE") != distilled_file:
            LOGGER.info("Ignoring LTX2_FP8_FILE because LTX2_USE_DISTILLED=1 (using %s).", distilled_file)
    else:
        fp8_file = os.getenv("LTX2_FP8_FILE", fallback_file)
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
                    if use_distilled:
                        LOGGER.info("Using distilled checkpoint: %s", candidate)
                    else:
                        LOGGER.info("Using checkpoint: %s", candidate)
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
            if use_distilled:
                LOGGER.info("Using distilled checkpoint: %s", candidates[0])
            else:
                LOGGER.info("Using checkpoint: %s", candidates[0])
            return str(candidates[0])

    if use_distilled:
        raise RuntimeError(
            "DistilledPipeline requires distilled checkpoint. Run download script "
            "to fetch ltx-2-19b-distilled-fp8.safetensors."
        )

    raise RuntimeError(
        "No fp8 checkpoint could be resolved. "
        "Set LTX2_CHECKPOINT_PATH explicitly or set LTX2_SNAPSHOT_DIR and LTX2_FP8_FILE."
    )


def _resolve_spatial_upsampler_path(checkpoint_path: str) -> str:
    env_value = os.getenv("LTX2_SPATIAL_UPSAMPLER_PATH")
    if env_value:
        path = pathlib.Path(env_value).expanduser()
        if path.exists():
            return str(path)
        LOGGER.warning(
            "LTX2_SPATIAL_UPSAMPLER_PATH does not exist at %s; falling back to auto-discovery.",
            path,
        )

    filename = os.getenv("LTX2_SPATIAL_UPSCALER_FILE", DEFAULT_SPATIAL_UPSCALER_FILE)
    checkpoint_parent = pathlib.Path(checkpoint_path).expanduser().parent
    candidate = checkpoint_parent / filename
    if candidate.is_file():
        return str(candidate)

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
                candidate = root / filename
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
            search_root.rglob(filename),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return str(candidates[0])

    raise RuntimeError(
        "Spatial upsampler file could not be resolved. Set LTX2_SPATIAL_UPSAMPLER_PATH "
        f"or ensure {filename} exists alongside the checkpoint or in the HuggingFace cache."
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
    spatial_upsampler_path = _resolve_spatial_upsampler_path(checkpoint_path)
    distilled_lora_path = _require_env_path("LTX2_DISTILLED_LORA_PATH", required=False)
    distilled_lora_strength = _env_float("LTX2_DISTILLED_LORA_STRENGTH", 0.6)
    loras: list[dict[str, object]] = []
    LOGGER.info("LTX-2 spatial_upsampler_path=%s", spatial_upsampler_path)

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


def _load_pipelines_pipeline(output_mode: str, device: str = "cuda", *, pipeline_variant: str | None = None) -> object:
    variant = (pipeline_variant or os.getenv("LTX2_PIPELINE_VARIANT", "distilled")).strip().lower()
    cache_key = f"pipelines:{output_mode}:{device}:{variant}"
    with _PIPELINE_LOCK:
        if cache_key in _PIPELINES:
            return _PIPELINES[cache_key]

        artifacts = _resolve_artifacts("native")
        dtype = torch.bfloat16
        enable_fp8 = os.getenv("LTX2_ENABLE_FP8", "1").lower() in {"1", "true", "yes", "on"}

        pipe_cls = DistilledPipeline
        if variant == "full":
            candidates = [
                ("ltx_pipelines.pipeline", "Pipeline"),
                ("ltx_pipelines.video", "VideoPipeline"),
                ("ltx_pipelines.standard", "StandardPipeline"),
            ]
            for module_name, class_name in candidates:
                try:
                    module = __import__(module_name, fromlist=[class_name])
                    pipe_cls = getattr(module, class_name)
                    LOGGER.info("Using full pipeline variant: %s.%s", module_name, class_name)
                    break
                except Exception:
                    continue
            if pipe_cls is DistilledPipeline:
                try:
                    import importlib
                    import pkgutil
                    import ltx_pipelines  # type: ignore
                    import inspect as _inspect
                    for mod in pkgutil.iter_modules(ltx_pipelines.__path__, ltx_pipelines.__name__ + "."):
                        try:
                            module = importlib.import_module(mod.name)
                        except Exception:
                            continue
                        for _, obj in _inspect.getmembers(module, _inspect.isclass):
                            if not obj.__name__.endswith("Pipeline"):
                                continue
                            if not obj.__module__.startswith("ltx_pipelines"):
                                continue
                            try:
                                sig = _inspect.signature(obj.__call__)
                            except Exception:
                                continue
                            params = set(sig.parameters.keys())
                            if any(name in params for name in ("num_inference_steps", "steps", "sampler_name", "sampler")):
                                pipe_cls = obj
                                LOGGER.info("Using discovered full pipeline: %s.%s", obj.__module__, obj.__name__)
                                raise StopIteration
                except StopIteration:
                    pass
                except Exception:
                    pass
            if pipe_cls is DistilledPipeline:
                LOGGER.warning("Full pipeline variant requested but not available; falling back to DistilledPipeline.")
        init_kwargs = {
            "checkpoint_path": artifacts.checkpoint_path,
            "spatial_upsampler_path": artifacts.spatial_upsampler_path,
            "gemma_root": artifacts.gemma_root,
            "loras": artifacts.loras,
            "device": device,
            "fp8transformer": enable_fp8,
        }

        if pipe_cls is DistilledPipeline:
            LOGGER.info("Using DistilledPipeline (fast inference)")
        pipe = _instantiate_pipeline(pipe_cls, init_kwargs, output_mode=output_mode)
        if _env_bool("LTX2_PERSIST_MODELS", True) and hasattr(pipe, "model_ledger"):
            _enable_model_caching(pipe.model_ledger)
            LOGGER.info("Model caching enabled; expect no checkpoint shard reloads.")
        _maybe_disable_cleanup(pipe)
        if hasattr(pipe, "to"):
            pipe.to(device)
        if hasattr(pipe, "eval"):
            pipe.eval()
        call_signature = inspect.signature(pipe.__call__)
        param_names = sorted(call_signature.parameters.keys())
        supports_output_path = any(name in call_signature.parameters for name in ("output_path", "output"))
        required_params = [
            name
            for name, param in call_signature.parameters.items()
            if param.default is inspect.Parameter.empty
            and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]
        LOGGER.info("LTX-2 pipeline call signature: %s", call_signature)
        LOGGER.info("LTX-2 pipeline params=%s", param_names)
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


def _load_pipeline(output_mode: str, device: str = "cuda", *, pipeline_variant: str | None = None) -> object:
    backend = _get_backend()
    if backend == "diffusers":
        return _load_diffusers_pipeline(output_mode, device=device)
    return _load_pipelines_pipeline(output_mode, device=device, pipeline_variant=pipeline_variant)


def _load_ic_lora_pipeline(device: str = "cuda") -> object:
    cache_key = f"pipelines:ic_lora:{device}"
    with _PIPELINE_LOCK:
        if cache_key in _PIPELINES:
            return _PIPELINES[cache_key]

        artifacts = _resolve_artifacts("native")
        enable_fp8 = os.getenv("LTX2_ENABLE_FP8", "1").lower() in {"1", "true", "yes", "on"}
        loras: list[dict[str, object]] = []
        lora_path = os.getenv("LTX2_IC_LORA_PATH")
        if lora_path:
            lora_strength = _env_float("LTX2_IC_LORA_STRENGTH", 0.8)
            loras.append({"path": lora_path, "strength": lora_strength, "ops": []})
        else:
            LOGGER.warning("ICLoraPipeline running without IC-LoRA weights; v2v control may be weak.")

        pipe_cls = ICLoraPipeline
        init_kwargs = {
            "checkpoint_path": artifacts.checkpoint_path,
            "spatial_upsampler_path": artifacts.spatial_upsampler_path,
            "gemma_root": artifacts.gemma_root,
            "loras": loras,
            "device": device,
            "fp8transformer": enable_fp8,
        }
        LOGGER.info("Loading ICLoraPipeline for v2v")
        pipe = _instantiate_pipeline(pipe_cls, init_kwargs, output_mode="native")
        if hasattr(pipe, "to"):
            pipe.to(device)
        if hasattr(pipe, "eval"):
            pipe.eval()

        _PIPELINES[cache_key] = pipe
        return pipe


def warmup_pipeline(output_mode: str) -> dict[str, str]:
    pipe = _load_pipeline(output_mode)
    return {"pipeline_class": pipe.__class__.__name__, "backend": _get_backend()}


def generate_v2v_video(
    *,
    input_video_path: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    fps: int,
    num_frames: int,
    seed: int | None,
    strength: float,
    output_path: str,
    ) -> str:
    pipe = _load_ic_lora_pipeline()
    if width % 64 != 0 or height % 64 != 0:
        raise RuntimeError("V2V requires width and height to be multiples of 64.")
    if num_frames < 1:
        raise RuntimeError("num_frames must be >= 1.")
    num_frames = normalize_num_frames(num_frames, label="v2v")
    video_conditioning = [(input_video_path, strength)]
    kwargs = _build_pipeline_kwargs(
        pipe,
        prompt=prompt,
        negative_prompt=_build_negative_prompt(prompt, negative_prompt),
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        guidance_scale=_env_float("LTX2_REALTIME_CFG", 1.0),
        num_inference_steps=0,
        seed=seed,
        output_path=None,
        images=[],
    )
    kwargs["video_conditioning"] = video_conditioning
    kwargs["frame_rate"] = float(fps)
    kwargs["tiling_config"] = None

    with torch.no_grad():
        result = pipe(**kwargs)
    _store_audio_from_result(result)
    if not isinstance(result, (tuple, list)) or len(result) < 1:
        raise RuntimeError("ICLoraPipeline returned unexpected result.")
    decoded_video = result[0]
    def _no_grad_video_iter(video_iter):
        with torch.inference_mode(False):
            with torch.no_grad():
                for chunk in video_iter:
                    yield chunk
    if hasattr(decoded_video, "__iter__") and not isinstance(decoded_video, (torch.Tensor, list)):
        decoded_video = _no_grad_video_iter(decoded_video)
    try:
        from ltx_pipelines.utils.media_io import encode_video
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("encode_video is required for v2v output.") from exc
    output_path = str(pathlib.Path(output_path))
    with torch.inference_mode(False):
        with torch.no_grad():
            encode_video(
                video=decoded_video,
                fps=fps,
                audio=result[1] if len(result) > 1 else None,
                audio_sample_rate=int(float(os.getenv("LTX2_AUDIO_SAMPLE_RATE", "48000"))),
                output_path=output_path,
                video_chunks_number=None,
            )
    return output_path


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


def _maybe_prompt_drift(prompt: str, allow_drift: bool | None = None) -> str:
    if allow_drift is None:
        allow_drift = os.getenv("LTX2_PROMPT_DRIFT", "0").strip().lower() in {"1", "true", "yes", "on"}
    if allow_drift:
        return _prompt_drift(prompt)
    return prompt


def _build_negative_prompt(prompt: str, negative_prompt: str) -> str:
    if _env_bool("LTX2_DISABLE_NEGATIVE_PROMPT", False):
        return negative_prompt.strip() if negative_prompt else ""
    env_negative = os.getenv("LTX2_NEGATIVE_PROMPT", "").strip()
    parts = [negative_prompt.strip()] if negative_prompt else []
    if env_negative:
        parts.append(env_negative)
    lowered = prompt.lower()
    if any(term in lowered for term in ("no people", "no person", "no humans", "no human", "without people", "without humans")):
        parts.append("people, person, human, face, portrait, body")
    return ", ".join([p for p in parts if p])


def _adjust_num_frames(num_frames: int) -> int:
    if num_frames < 1:
        return 1
    if (num_frames - 1) % 8 == 0:
        return num_frames
    return ((num_frames - 1) // 8 + 1) * 8 + 1


def normalize_num_frames(num_frames: int, *, label: str | None = None) -> int:
    requested = int(num_frames)
    normalized = _adjust_num_frames(requested)
    prefix = f"{label} " if label else ""
    LOGGER.info("Normalized %snum_frames: %s -> %s", prefix, requested, normalized)
    return normalized


def _round_down_to_multiple(x: int, m: int) -> int:
    return max(m, (x // m) * m)


def _round_up_to_multiple(x: int, m: int) -> int:
    return max(m, ((x + m - 1) // m) * m)


def _apply_commercial_blend(
    anchor: Image.Image | None,
    frames: list[Image.Image],
    blend_frames: int,
) -> list[Image.Image]:
    if anchor is None or blend_frames <= 0 or not frames:
        return frames
    count = min(blend_frames, len(frames))
    if count <= 0:
        return frames
    anchor_frame = anchor
    if anchor_frame.size != frames[0].size:
        anchor_frame = anchor_frame.resize(frames[0].size, Image.BICUBIC)
    for idx in range(count):
        alpha = float(idx + 1) / float(count + 1)
        try:
            frames[idx] = Image.blend(anchor_frame, frames[idx], alpha)
        except Exception:  # noqa: BLE001
            break
    return frames


def _write_commercial_mp4_from_frames(
    frames: list[Image.Image],
    fps: int,
    stream_id: int,
) -> None:
    if not frames:
        return
    temp_path = f"/tmp/ltx_commercial_tmp_{uuid.uuid4().hex}.mp4"
    writer = None
    try:
        width, height = frames[0].size
        codecs = ("mp4v", "MJPG")
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(temp_path, fourcc, float(fps), (width, height))
            if writer.isOpened():
                break
            if writer is not None:
                writer.release()
                writer = None
        if writer is None or not writer.isOpened():
            raise RuntimeError("Failed to open video writer for commercial mp4.")
        for frame in frames:
            rgb = np.array(frame)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    except Exception:  # noqa: BLE001
        LOGGER.exception("Failed to write commercial mp4 from frames.")
    finally:
        try:
            if writer is not None:
                writer.release()
        except Exception:  # noqa: BLE001
            pass
    _store_commercial_mp4(temp_path, stream_id)


def _write_commercial_mp4_to_path(
    frames: list[Image.Image],
    fps: int,
    output_path: str,
) -> None:
    if not frames:
        return
    writer = None
    try:
        width, height = frames[0].size
        codecs = ("mp4v", "MJPG")
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
            if writer.isOpened():
                break
            if writer is not None:
                writer.release()
                writer = None
        if writer is None or not writer.isOpened():
            raise RuntimeError("Failed to open video writer for commercial mp4.")
        for frame in frames:
            rgb = np.array(frame)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    except Exception:  # noqa: BLE001
        LOGGER.exception("Failed to write commercial mp4 to path.")
    finally:
        try:
            if writer is not None:
                writer.release()
        except Exception:  # noqa: BLE001
            pass


def assemble_final_mp4(
    frames: list[Image.Image],
    fps: int,
    audio_wav_path: str | None,
    out_mp4_path: str,
    *,
    video_path: str | None = None,
) -> str:
    temp_video = video_path
    if not temp_video:
        temp_video = f"/tmp/ltx_commercial_tmp_{uuid.uuid4().hex}.mp4"
        _write_commercial_mp4_to_path(frames, fps, temp_video)
    if audio_wav_path and _ffmpeg_available():
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                temp_video,
                "-i",
                audio_wav_path,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-profile:v",
                "baseline",
                "-level",
                "3.1",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                out_mp4_path,
            ]
            subprocess.run(cmd, capture_output=True, check=False, timeout=300)
            return out_mp4_path
        except Exception:
            LOGGER.exception("Failed to mux final mp4; using video-only output.")
    elif audio_wav_path and not _ffmpeg_available():
        LOGGER.warning("ffmpeg unavailable; output mp4 will be video-only.")
    if temp_video != out_mp4_path:
        try:
            os.replace(temp_video, out_mp4_path)
        except OSError:
            shutil.copyfile(temp_video, out_mp4_path)
    return out_mp4_path


def _match_exposure(anchor: Image.Image, frame: Image.Image) -> Image.Image:
    anchor_arr = np.asarray(anchor).astype(np.float32)
    frame_arr = np.asarray(frame).astype(np.float32)
    if anchor_arr.shape != frame_arr.shape:
        frame_arr = cv2.resize(frame_arr, (anchor_arr.shape[1], anchor_arr.shape[0]), interpolation=cv2.INTER_AREA)
    anchor_mean = anchor_arr.mean(axis=(0, 1), keepdims=True)
    anchor_std = anchor_arr.std(axis=(0, 1), keepdims=True) + 1e-6
    frame_mean = frame_arr.mean(axis=(0, 1), keepdims=True)
    frame_std = frame_arr.std(axis=(0, 1), keepdims=True) + 1e-6
    matched = (frame_arr - frame_mean) / frame_std * anchor_std + anchor_mean
    matched = np.clip(matched, 0, 255).astype(np.uint8)
    return Image.fromarray(matched)


def _concat_commercial_chunks(paths: list[str], output_path: str, fps: int) -> None:
    valid_paths = [p for p in paths if p and os.path.exists(p)]
    if not valid_paths:
        return
    writer = None
    try:
        first = cv2.VideoCapture(valid_paths[0])
        if not first.isOpened():
            return
        width = int(first.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(first.get(cv2.CAP_PROP_FRAME_HEIGHT))
        first.release()
        if width <= 0 or height <= 0:
            return
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
        if not writer.isOpened():
            return
        for path in valid_paths:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                cap.release()
                continue
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    writer.write(frame)
            finally:
                cap.release()
    finally:
        if writer is not None:
            writer.release()


def _finalize_commercial_with_audio(stream_id: int, video_path: str) -> str:
    audio_bytes, _ = get_latest_audio_wav(stream_id)
    if not audio_bytes:
        return video_path
    muxed_path = f"/tmp/ltx_commercial_{stream_id}_av.mp4"
    if _mux_audio_with_video(video_path, audio_bytes, muxed_path):
        return muxed_path
    return video_path


def _concat_commercial_chunks_ffmpeg(
    paths: list[str],
    output_path: str,
    fps: int,
    audio_bytes: bytes | None,
) -> bool:
    if not _ffmpeg_available():
        return False
    valid_paths = [p for p in paths if p and os.path.exists(p)]
    if not valid_paths:
        return False
    list_path = f"/tmp/ltx_commercial_concat_{uuid.uuid4().hex}.txt"
    audio_path = None
    try:
        with open(list_path, "w", encoding="utf-8") as handle:
            for path in valid_paths:
                handle.write(f"file '{path}'\n")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
        ]
        if audio_bytes:
            audio_path = f"/tmp/ltx_commercial_audio_{uuid.uuid4().hex}.wav"
            with open(audio_path, "wb") as handle:
                handle.write(audio_bytes)
            cmd.extend(["-i", audio_path])
        cmd.extend(
            [
                "-r",
                str(float(fps)),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
            ]
        )
        if audio_bytes:
            cmd.extend(["-c:a", "aac", "-shortest"])
        else:
            cmd.extend(["-an"])
        cmd.extend(["-movflags", "+faststart", output_path])
        result = subprocess.run(cmd, capture_output=True, check=False, timeout=300)
        if result.returncode != 0:
            LOGGER.warning("ffmpeg concat failed: %s", result.stderr.decode(errors="ignore").strip())
            return False
        return True
    except Exception:
        LOGGER.exception("Failed to concat commercial chunks via ffmpeg.")
        return False
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass
        if audio_path:
            try:
                os.remove(audio_path)
            except OSError:
                pass


def _concat_mp4_chunks(video_paths: list[str], out_path: str, fps: int) -> bool:
    if not _ffmpeg_available():
        _concat_commercial_chunks(video_paths, out_path, fps)
        return os.path.exists(out_path)
    valid_paths = [p for p in video_paths if p and os.path.exists(p)]
    if not valid_paths:
        return False
    list_path = f"/tmp/ltx_commercial_concat_{uuid.uuid4().hex}.txt"
    try:
        with open(list_path, "w", encoding="utf-8") as handle:
            for path in valid_paths:
                handle.write(f"file '{path}'\n")
        copy_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c",
            "copy",
            out_path,
        ]
        result = subprocess.run(copy_cmd, capture_output=True, check=False, timeout=120)
        if result.returncode == 0:
            return True
        reencode_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-r",
            str(float(fps)),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            out_path,
        ]
        result = subprocess.run(reencode_cmd, capture_output=True, check=False, timeout=300)
        if result.returncode != 0:
            LOGGER.warning("ffmpeg concat reencode failed: %s", result.stderr.decode(errors="ignore").strip())
            return False
        return True
    except Exception:
        LOGGER.exception("Failed to concat commercial chunks.")
        return False
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass


def _concat_mp4_chunks_with_audio(video_paths: list[str], out_path: str, fps: int) -> bool:
    if not _ffmpeg_available():
        return False
    valid_paths = [p for p in video_paths if p and os.path.exists(p)]
    if not valid_paths:
        return False
    list_path = f"/tmp/ltx_commercial_concat_{uuid.uuid4().hex}.txt"
    try:
        with open(list_path, "w", encoding="utf-8") as handle:
            for path in valid_paths:
                handle.write(f"file '{path}'\n")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-r",
            str(float(fps)),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-shortest",
            "-movflags",
            "+faststart",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, check=False, timeout=300)
        if result.returncode != 0:
            LOGGER.warning("ffmpeg concat av failed: %s", result.stderr.decode(errors="ignore").strip())
            return False
        return True
    except Exception:
        LOGGER.exception("Failed to concat commercial chunks with audio.")
        return False
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass


def _extract_wav_from_mp4(video_path: str, sample_rate: int) -> bytes | None:
    if not _ffmpeg_available():
        return None
    temp_path = f"/tmp/ltx_commercial_audio_{uuid.uuid4().hex}.wav"
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(int(sample_rate)),
            "-ac",
            "2",
            temp_path,
        ]
        result = subprocess.run(cmd, capture_output=True, check=False, timeout=120)
        if result.returncode != 0:
            LOGGER.warning("ffmpeg audio extract failed: %s", result.stderr.decode(errors="ignore").strip())
            return None
        with open(temp_path, "rb") as handle:
            return handle.read()
    except Exception:
        LOGGER.exception("Failed to extract wav from mp4.")
        return None
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def _requires_64_multiple(pipe: object, output_mode: str) -> bool:
    if output_mode == "upscaled":
        return True
    if "Distilled" in pipe.__class__.__name__:
        return True
    return os.getenv("LTX2_USE_DISTILLED", "0").strip().lower() in {"1", "true", "yes", "on"}


def _adjust_resolution_for_two_stage(
    pipe: object,
    output_mode: str,
    width: int,
    height: int,
) -> tuple[int, int]:
    if not _requires_64_multiple(pipe, output_mode):
        return width, height
    new_width = _round_down_to_multiple(width, 64)
    new_height = _round_down_to_multiple(height, 64)
    if (new_width, new_height) != (width, height):
        LOGGER.info(
            "Adjusted resolution for two-stage: %sx%s -> %sx%s",
            width,
            height,
            new_width,
            new_height,
        )
    return new_width, new_height


def _resolve_stage_dimensions(config) -> tuple[int, int]:
    output_mode = getattr(config, "output_mode", "native")
    width = config.width
    height = config.height
    if os.getenv("LTX2_REALTIME", "0").lower() in {"1", "true", "yes", "on"} and output_mode == "native":
        width = _env_int_clamped("LTX2_REALTIME_WIDTH", 640, min_value=64, max_value=4096)
        height = _env_int_clamped("LTX2_REALTIME_HEIGHT", 384, min_value=64, max_value=4096)
    if output_mode == "upscaled":
        stage_width = width // 2
        stage_height = height // 2
    return stage_width, stage_height


def _resolve_commercial_stage_dimensions(config) -> tuple[int, int]:
    output_mode = getattr(config, "output_mode", "native")
    width = config.width
    height = config.height
    if output_mode == "upscaled":
        return width // 2, height // 2
    return width, height


def _resolve_commercial_chunk_settings(config) -> tuple[int, int, int]:
    target_seconds = os.getenv("LTX2_TARGET_CHUNK_SECONDS")
    if target_seconds:
        try:
            chunk_seconds = float(target_seconds)
        except ValueError:
            LOGGER.warning("Invalid LTX2_TARGET_CHUNK_SECONDS=%s; using LTX2_CHUNK_SECONDS.", target_seconds)
            chunk_seconds = _env_float("LTX2_CHUNK_SECONDS", 1.0)
    else:
        chunk_seconds = _env_float("LTX2_CHUNK_SECONDS", 1.0)
    min_frames = _env_int_clamped("LTX2_MIN_FRAMES", 9, min_value=1, max_value=120)
    num_frames = _adjust_num_frames(max(min_frames, int(chunk_seconds * config.fps)))
    stage_width, stage_height = _resolve_commercial_stage_dimensions(config)
    return stage_width, stage_height, num_frames
    return width, height


def _assign_first_present(params: set[str], kwargs: dict[str, object], value: object, names: list[str]) -> None:
    for name in names:
        if name in params and value is not None:
            kwargs[name] = value
            return


def _comfy_enabled() -> bool:
    return _env_bool("LTX2_COMFY_PRESET", False)


def _parse_manual_sigmas_env() -> list[float] | None:
    raw = os.getenv("LTX2_STAGE2_MANUAL_SIGMAS", "")
    if not raw.strip():
        return None
    values: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            LOGGER.warning("Invalid sigma value in LTX2_STAGE2_MANUAL_SIGMAS: %s", part)
            return None
    return values or None


def _apply_comfy_pipeline_overrides(
    *,
    pipe: object,
    kwargs: dict[str, object],
    stage: str,
) -> None:
    if not _comfy_enabled():
        return
    signature = inspect.signature(pipe.__call__)
    param_names = set(signature.parameters.keys())
    sampler_name = os.getenv("LTX2_SAMPLER_NAME", "euler_ancestral")
    _assign_first_present(param_names, kwargs, sampler_name, ["sampler_name", "sampler", "sampler_type"])
    if stage == "stage1":
        stage1_steps = _env_int("LTX2_STAGE1_STEPS", 20)
        stage1_cfg = _env_float("LTX2_STAGE1_CFG", 4.0)
        _assign_first_present(param_names, kwargs, stage1_cfg, ["guidance_scale", "cfg_scale", "cfg_guidance_scale"])
        _assign_first_present(param_names, kwargs, stage1_steps, ["num_inference_steps", "steps"])
    sched_values = {
        "LTX2_SCHED_MAX_SHIFT": _env_float("LTX2_SCHED_MAX_SHIFT", 2.05),
        "LTX2_SCHED_BASE_SHIFT": _env_float("LTX2_SCHED_BASE_SHIFT", 0.95),
        "LTX2_SCHED_TERMINAL": _env_float("LTX2_SCHED_TERMINAL", 0.1),
        "LTX2_SCHED_STRETCH": _env_float("LTX2_SCHED_STRETCH", 1.0),
    }
    sched_param_names = [
        ("scheduler_max_shift", "LTX2_SCHED_MAX_SHIFT"),
        ("max_shift", "LTX2_SCHED_MAX_SHIFT"),
        ("scheduler_base_shift", "LTX2_SCHED_BASE_SHIFT"),
        ("base_shift", "LTX2_SCHED_BASE_SHIFT"),
        ("scheduler_terminal", "LTX2_SCHED_TERMINAL"),
        ("terminal", "LTX2_SCHED_TERMINAL"),
        ("scheduler_stretch", "LTX2_SCHED_STRETCH"),
        ("stretch", "LTX2_SCHED_STRETCH"),
    ]
    supported_sched = False
    for param, env_key in sched_param_names:
        if param in param_names:
            kwargs[param] = sched_values[env_key]
            supported_sched = True
    global _COMFY_WARNED_SCHED
    if not supported_sched and not _COMFY_WARNED_SCHED:
        LOGGER.warning("Comfy preset: scheduler shift params not supported by pipeline.")
        _COMFY_WARNED_SCHED = True
    if _env_bool("LTX2_VAE_DECODE_TILED", False):
        tile_size = _env_int_clamped("LTX2_VAE_TILE_SIZE", 512, min_value=64, max_value=4096)
        overlap = _env_int_clamped("LTX2_VAE_OVERLAP", 64, min_value=0, max_value=1024)
        temporal_size = _env_int_clamped("LTX2_VAE_TEMPORAL_SIZE", 4096, min_value=64, max_value=16384)
        temporal_overlap = _env_int_clamped("LTX2_VAE_TEMPORAL_OVERLAP", 8, min_value=0, max_value=1024)
        vae_param_names = [
            "vae_decode_tiled",
            "decode_tiled",
            "tiled_decode",
            "vae_tile_size",
            "tile_size",
            "vae_tile_overlap",
            "tile_overlap",
            "vae_temporal_tile_size",
            "temporal_tile_size",
            "vae_temporal_overlap",
            "temporal_tile_overlap",
        ]
        supported_vae = any(name in param_names for name in vae_param_names)
        if "vae_decode_tiled" in param_names:
            kwargs["vae_decode_tiled"] = True
        if "decode_tiled" in param_names:
            kwargs["decode_tiled"] = True
        if "tiled_decode" in param_names:
            kwargs["tiled_decode"] = True
        if "vae_tile_size" in param_names:
            kwargs["vae_tile_size"] = tile_size
        if "tile_size" in param_names:
            kwargs["tile_size"] = tile_size
        if "vae_tile_overlap" in param_names:
            kwargs["vae_tile_overlap"] = overlap
        if "tile_overlap" in param_names:
            kwargs["tile_overlap"] = overlap
        if "vae_temporal_tile_size" in param_names:
            kwargs["vae_temporal_tile_size"] = temporal_size
        if "temporal_tile_size" in param_names:
            kwargs["temporal_tile_size"] = temporal_size
        if "vae_temporal_overlap" in param_names:
            kwargs["vae_temporal_overlap"] = temporal_overlap
        if "temporal_tile_overlap" in param_names:
            kwargs["temporal_tile_overlap"] = temporal_overlap
        global _COMFY_WARNED_VAE
        if not supported_vae and not _COMFY_WARNED_VAE:
            LOGGER.warning("Comfy preset: VAE tiled decode params not supported by pipeline.")
            _COMFY_WARNED_VAE = True


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
    apply_comfy_overrides: bool = True,
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

    if apply_comfy_overrides:
        _apply_comfy_pipeline_overrides(pipe=pipe, kwargs=kwargs, stage="stage1")

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

    if _comfy_enabled():
        _apply_comfy_pipeline_overrides(pipe=pipe, kwargs=kwargs, stage="stage1")

    return _filter_kwargs_for_callable(pipe.__call__, kwargs)


def _write_temp_image(frame_bgr: np.ndarray) -> str:
    temp_path = pathlib.Path(f"/tmp/img_{uuid.uuid4().hex}.png")
    cv2.imwrite(str(temp_path), frame_bgr)
    return str(temp_path)


def _write_temp_pil(image: Image.Image) -> str:
    rgb = np.array(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return _write_temp_image(bgr)


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


def _enable_model_caching(ledger: object) -> None:
    if getattr(ledger, "_ltx2_cache_enabled", False):
        return
    cache: dict[str, object] = {}
    setattr(ledger, "_ltx2_cache_enabled", True)
    setattr(ledger, "_ltx2_cached_models", cache)
    for name in (
        "transformer",
        "video_encoder",
        "video_decoder",
        "text_encoder",
        "audio_decoder",
        "vocoder",
        "spatial_upsampler",
    ):
        if not hasattr(ledger, name):
            continue
        orig = getattr(ledger, name)

        def cached_call(*args, _orig=orig, _name=name, **kwargs):
            if _name not in cache:
                cache[_name] = _orig(*args, **kwargs)
            return cache[_name]

        setattr(ledger, name, cached_call)


def _maybe_disable_cleanup(pipe: object) -> None:
    if not _env_bool("LTX2_SKIP_CLEANUP", True):
        return
    module = sys.modules.get(pipe.__class__.__module__)
    if not module or not hasattr(module, "cleanup_memory"):
        return

    def _noop_cleanup_memory() -> None:
        return None

    setattr(module, "cleanup_memory", _noop_cleanup_memory)
    LOGGER.info("LTX2 cleanup_memory disabled via LTX2_SKIP_CLEANUP=1")


def _use_inference_mode() -> bool:
    return os.getenv("LTX2_USE_INFERENCE_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}


def set_audio_stream_id(stream_id: int | None) -> None:
    if stream_id is None:
        if hasattr(_AUDIO_STREAM_LOCAL, "stream_id"):
            delattr(_AUDIO_STREAM_LOCAL, "stream_id")
        return
    _AUDIO_STREAM_LOCAL.stream_id = int(stream_id)


def _current_audio_stream_id() -> int:
    return int(getattr(_AUDIO_STREAM_LOCAL, "stream_id", 0))


def _store_audio_from_result(result: object) -> None:
    if not isinstance(result, (tuple, list)) or len(result) < 2:
        return
    audio = result[1]
    if not torch.is_tensor(audio):
        return
    try:
        import io
        import wave
        audio_tensor = audio.detach()
        if audio_tensor.is_cuda:
            audio_tensor = audio_tensor.cpu()
        audio_tensor = audio_tensor.float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)
        pcm16 = (audio_tensor * 32767.0).to(torch.int16).cpu().numpy()
        sample_rate = int(float(os.getenv("LTX2_AUDIO_SAMPLE_RATE", "48000")))
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(pcm16.shape[0])
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.T.tobytes())
        data = buffer.getvalue()
        with _AUDIO_LOCK:
            stream_id = _current_audio_stream_id()
            _LATEST_AUDIO_BY_STREAM[stream_id] = (data, time.time())
    except Exception:  # noqa: BLE001
        LOGGER.exception("Failed to serialize audio output")


def get_latest_audio_wav(stream_id: int = 0) -> tuple[bytes | None, float]:
    with _AUDIO_LOCK:
        data = _LATEST_AUDIO_BY_STREAM.get(stream_id)
        if not data:
            return None, 0.0
        return data


def _ffmpeg_available() -> bool:
    global _FFMPEG_AVAILABLE
    if _FFMPEG_AVAILABLE is not None:
        return _FFMPEG_AVAILABLE
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, check=False, timeout=2)
        _FFMPEG_AVAILABLE = result.returncode == 0
    except Exception:
        _FFMPEG_AVAILABLE = False
    return _FFMPEG_AVAILABLE


def _mux_audio_with_video(video_path: str, audio_bytes: bytes, output_path: str) -> bool:
    if not _ffmpeg_available():
        return False
    audio_path = f"/tmp/ltx_commercial_audio_{uuid.uuid4().hex}.wav"
    try:
        with open(audio_path, "wb") as handle:
            handle.write(audio_bytes)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, check=False, timeout=60)
        if result.returncode != 0:
            LOGGER.warning("ffmpeg mux failed: %s", result.stderr.decode(errors="ignore").strip())
            return False
        return True
    except Exception:
        LOGGER.exception("Failed to mux audio with video via ffmpeg.")
        return False
    finally:
        try:
            os.remove(audio_path)
        except OSError:
            pass


def _concat_wav_paths(chunks: list[str]) -> str | None:
    valid_paths = [p for p in chunks if p and os.path.exists(p)]
    if not valid_paths:
        return None
    try:
        import wave
        output_path = f"/tmp/ltx_commercial_concat_{uuid.uuid4().hex}.wav"
        with wave.open(output_path, "wb") as out_wav:
            for path in valid_paths:
                with wave.open(path, "rb") as in_wav:
                    params = in_wav.getparams()
                    if out_wav.getnframes() == 0:
                        out_wav.setparams(params)
                    elif out_wav.getparams() != params:
                        LOGGER.warning("Commercial audio chunk params mismatch; skipping chunk.")
                        continue
                    out_wav.writeframes(in_wav.readframes(in_wav.getnframes()))
        return output_path
    except Exception:
        LOGGER.exception("Failed to concat commercial audio.")
        return None


def _trim_or_pad_wav_to_frames(
    wav_bytes: bytes,
    *,
    fps: int,
    frames: int,
    sample_rate: int,
) -> bytes | None:
    import io
    import wave
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as in_wav:
            params = in_wav.getparams()
            channels = params.nchannels
            sampwidth = params.sampwidth
            framerate = params.framerate
            if framerate <= 0:
                framerate = sample_rate
            raw = in_wav.readframes(in_wav.getnframes())
        bytes_per_frame = channels * sampwidth
        if bytes_per_frame <= 0:
            return None
        target_frames = int(round(frames * framerate / max(1, fps)))
        target_bytes = target_frames * bytes_per_frame
        current_bytes = len(raw)
        applied = "none"
        if current_bytes > target_bytes:
            raw = raw[:target_bytes]
            applied = "trim"
        elif current_bytes < target_bytes:
            raw = raw + (b"\x00" * (target_bytes - current_bytes))
            applied = "pad"
        out_buf = io.BytesIO()
        with wave.open(out_buf, "wb") as out_wav:
            out_wav.setnchannels(channels)
            out_wav.setsampwidth(sampwidth)
            out_wav.setframerate(framerate)
            out_wav.writeframes(raw)
        duration = target_frames / max(1, framerate)
        LOGGER.info(
            "Commercial audio normalize: frames=%s fps=%s wav_seconds=%.3f expected_seconds=%.3f applied=%s",
            frames,
            fps,
            current_bytes / bytes_per_frame / max(1, framerate),
            duration,
            applied,
        )
        return out_buf.getvalue()
    except Exception:
        LOGGER.exception("Failed to trim/pad commercial wav.")
        return None


def _call_ltx_pipeline(pipe: object, kwargs: dict[str, object]) -> object:
    context = torch.inference_mode() if _use_inference_mode() else torch.no_grad()
    with context:
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


def _extract_latents_from_pipeline_result(result: object) -> object | None:
    if isinstance(result, dict):
        for key in ("latents", "video_latents", "samples", "sample", "latent"):
            if key in result:
                return result[key]
    if isinstance(result, (tuple, list)):
        for item in result:
            latents = _extract_latents_from_pipeline_result(item)
            if latents is not None:
                return latents
    for attr in ("latents", "video_latents", "samples", "sample", "latent"):
        if hasattr(result, attr):
            return getattr(result, attr)
    return None


def _generate_video_chunk(
    pipe: object,
    *,
    output_mode: str,
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
    width, height = _adjust_resolution_for_two_stage(pipe, output_mode, width, height)
    comfy_enabled = _comfy_enabled()
    stage2_enabled = comfy_enabled and _env_bool("LTX2_STAGE2_ENABLE", True)
    stage2_sigmas = _parse_manual_sigmas_env() if stage2_enabled else None
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
    frames = int(kwargs.get("num_frames") or 0)
    frames = max(frames, 9)
    frames = _adjust_num_frames(frames)
    if (frames - 1) % 8 != 0:
        LOGGER.warning("Invalid num_frames=%s; forcing 73", frames)
        frames = 73
    kwargs["num_frames"] = frames
    result = _call_ltx_pipeline(pipe, kwargs)
    _store_audio_from_result(result)
    _log_vram("chunk_end")

    if stage2_enabled and stage2_sigmas:
        sigmas_param_names = {"sigmas", "manual_sigmas", "sigmas_list", "sigma_schedule"}
        latent_param_names = {"latents", "init_latents", "video_latents", "samples", "sample"}
        param_names = set(signature.parameters.keys())
        supports_sigmas = any(name in param_names for name in sigmas_param_names)
        supports_latents = any(name in param_names for name in latent_param_names)
        latents = _extract_latents_from_pipeline_result(result) if supports_latents else None
        if supports_sigmas and latents is not None:
            stage2_kwargs = _build_pipeline_kwargs(
                pipe,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps,
                guidance_scale=_env_float("LTX2_STAGE2_CFG", 1.0),
                num_inference_steps=len(stage2_sigmas),
                seed=seed,
                output_path=output_path,
                images=images,
                apply_comfy_overrides=False,
            )
            if "sigmas" in param_names:
                stage2_kwargs["sigmas"] = stage2_sigmas
            elif "manual_sigmas" in param_names:
                stage2_kwargs["manual_sigmas"] = stage2_sigmas
            elif "sigmas_list" in param_names:
                stage2_kwargs["sigmas_list"] = stage2_sigmas
            elif "sigma_schedule" in param_names:
                stage2_kwargs["sigma_schedule"] = stage2_sigmas
            if "latents" in param_names:
                stage2_kwargs["latents"] = latents
            elif "init_latents" in param_names:
                stage2_kwargs["init_latents"] = latents
            elif "video_latents" in param_names:
                stage2_kwargs["video_latents"] = latents
            elif "samples" in param_names:
                stage2_kwargs["samples"] = latents
            elif "sample" in param_names:
                stage2_kwargs["sample"] = latents
            _apply_comfy_pipeline_overrides(pipe=pipe, kwargs=stage2_kwargs, stage="stage2")
            result = _call_ltx_pipeline(pipe, stage2_kwargs)
            _store_audio_from_result(result)
        else:
            global _COMFY_WARNED_STAGE2
            if not _COMFY_WARNED_STAGE2:
                LOGGER.warning("Comfy preset: stage2 refine not supported by pipeline or latents unavailable.")
                _COMFY_WARNED_STAGE2 = True

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
    with torch.no_grad():
        frames = _extract_video_frames_from_pipeline_result(result)
    for frame in frames:
        yield frame


def _generate_commercial_video_chunk(
    pipe: object,
    *,
    stream_id: int,
    chunk_index: int,
    output_mode: str,
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
    apply_comfy_overrides: bool = True,
) -> tuple[list[Image.Image], str | None]:
    _log_vram("commercial_chunk_start")
    if num_frames <= 0:
        raise ValueError(f"Commercial mode: invalid num_frames={num_frames}")
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(
            f"Commercial mode: invalid fps/size fps={fps} width={width} height={height}"
        )
    signature = inspect.signature(pipe.__call__)
    supports_output_path = any(name in signature.parameters for name in ("output_path", "output"))
    output_path = f"/tmp/ltx_commercial_tmp_{uuid.uuid4().hex}.mp4" if supports_output_path else None
    width, height = _adjust_resolution_for_two_stage(pipe, output_mode, width, height)
    comfy_enabled = _comfy_enabled()
    stage2_enabled = comfy_enabled and _env_bool("LTX2_STAGE2_ENABLE", True)
    stage2_sigmas = _parse_manual_sigmas_env() if stage2_enabled else None
    LOGGER.info(
        "Commercial mode preflight: stream_id=%s chunk_index=%s frames_this_chunk=%s fps=%s width=%s height=%s seed=%s",
        stream_id,
        chunk_index,
        num_frames,
        fps,
        width,
        height,
        seed,
    )
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
        apply_comfy_overrides=apply_comfy_overrides,
    )
    frames = int(kwargs.get("num_frames") or 0)
    frames = max(frames, 9)
    frames = _adjust_num_frames(frames)
    if (frames - 1) % 8 != 0:
        LOGGER.warning("Invalid num_frames=%s; forcing 73", frames)
        frames = 73
    kwargs["num_frames"] = frames
    sampler = kwargs.get("sampler_name") or kwargs.get("sampler") or kwargs.get("scheduler")
    cfg_value = kwargs.get("cfg_guidance_scale") or kwargs.get("cfg_scale") or kwargs.get("guidance_scale") or guidance_scale
    steps_value = kwargs.get("num_inference_steps") or kwargs.get("steps")
    fps_value = kwargs.get("frame_rate") or kwargs.get("fps")
    LOGGER.info(
        "Commercial mode kwargs: steps=%s cfg=%.3f sampler=%s num_frames=%s fps=%s",
        steps_value,
        cfg_value,
        sampler,
        kwargs.get("num_frames"),
        fps_value,
    )
    if steps_value is None:
        LOGGER.warning("Commercial mode pipeline does not accept steps/num_inference_steps.")
    if sampler is None:
        LOGGER.warning("Commercial mode pipeline does not accept sampler/scheduler params.")
    result = _call_ltx_pipeline(pipe, kwargs)
    _store_audio_from_result(result)
    _log_vram("commercial_chunk_end")

    if stage2_enabled and stage2_sigmas:
        sigmas_param_names = {"sigmas", "manual_sigmas", "sigmas_list", "sigma_schedule"}
        latent_param_names = {"latents", "init_latents", "video_latents", "samples", "sample"}
        param_names = set(signature.parameters.keys())
        supports_sigmas = any(name in param_names for name in sigmas_param_names)
        supports_latents = any(name in param_names for name in latent_param_names)
        latents = _extract_latents_from_pipeline_result(result) if supports_latents else None
        if supports_sigmas and latents is not None:
            stage2_kwargs = _build_pipeline_kwargs(
                pipe,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps,
                guidance_scale=_env_float("LTX2_STAGE2_CFG", 1.0),
                num_inference_steps=len(stage2_sigmas),
                seed=seed,
                output_path=output_path,
                images=images,
                apply_comfy_overrides=False,
            )
            if "sigmas" in param_names:
                stage2_kwargs["sigmas"] = stage2_sigmas
            elif "manual_sigmas" in param_names:
                stage2_kwargs["manual_sigmas"] = stage2_sigmas
            elif "sigmas_list" in param_names:
                stage2_kwargs["sigmas_list"] = stage2_sigmas
            elif "sigma_schedule" in param_names:
                stage2_kwargs["sigma_schedule"] = stage2_sigmas
            if "latents" in param_names:
                stage2_kwargs["latents"] = latents
            elif "init_latents" in param_names:
                stage2_kwargs["init_latents"] = latents
            elif "video_latents" in param_names:
                stage2_kwargs["video_latents"] = latents
            elif "samples" in param_names:
                stage2_kwargs["samples"] = latents
            elif "sample" in param_names:
                stage2_kwargs["sample"] = latents
            _apply_comfy_pipeline_overrides(pipe=pipe, kwargs=stage2_kwargs, stage="stage2")
            result = _call_ltx_pipeline(pipe, stage2_kwargs)
            _store_audio_from_result(result)
        else:
            global _COMFY_WARNED_STAGE2
            if not _COMFY_WARNED_STAGE2:
                LOGGER.warning("Comfy preset: stage2 refine not supported by pipeline or latents unavailable.")
                _COMFY_WARNED_STAGE2 = True

    if output_path and pathlib.Path(output_path).is_file():
        frames = list(_yield_video_frames(output_path))
        return frames, output_path
    with torch.no_grad():
        frames = _extract_video_frames_from_pipeline_result(result)
    return frames, None


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
    output_mode: str,
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
    width, height = _adjust_resolution_for_two_stage(pipe, output_mode, width, height)
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
    if _comfy_enabled() and _env_bool("LTX2_STAGE2_ENABLE", True):
        global _COMFY_WARNED_STAGE2
        if not _COMFY_WARNED_STAGE2:
            LOGGER.warning("Comfy preset: stage2 refine not supported on diffusers backend.")
            _COMFY_WARNED_STAGE2 = True
    try:
        result = pipe(**kwargs)
    except Exception as exc:  # noqa: BLE001
        if kwargs.get("negative_prompt") and _should_retry_without_negative_prompt(exc):
            LOGGER.warning("Retrying diffusers call without negative_prompt due to: %s", exc)
            kwargs.pop("negative_prompt", None)
            result = pipe(**kwargs)
        else:
            raise
    _store_audio_from_result(result)
    with torch.no_grad():
        frames = _extract_video_frames_from_pipeline_result(result)
    for frame in frames:
        yield frame


def _get_pipelines_pipe_or_status(config, *, pipeline_variant: str | None = None) -> object | None:
    try:
        return _load_pipeline(getattr(config, "output_mode", "native"), pipeline_variant=pipeline_variant)
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
    while not cancel_event.is_set():
        for frame in generate_fever_dream_chunk(config, cancel_event):
            yield frame


def _chain_key(mode: str, cancel_event: threading.Event, stream_id: int) -> tuple[str, int, int]:
    return (mode, id(cancel_event), int(stream_id))


def _get_chain_frame(mode: str, cancel_event: threading.Event, stream_id: int) -> Image.Image | None:
    return _CHAIN_FRAMES.get(_chain_key(mode, cancel_event, stream_id))


def _set_chain_frame(mode: str, cancel_event: threading.Event, frame: Image.Image | None, stream_id: int) -> None:
    key = _chain_key(mode, cancel_event, stream_id)
    if frame is None:
        _CHAIN_FRAMES.pop(key, None)
    else:
        _CHAIN_FRAMES[key] = frame


def _should_reset_chain(mode: str, cancel_event: threading.Event, stream_id: int, reset_interval: int) -> bool:
    if reset_interval <= 0:
        return False
    key = _chain_key(mode, cancel_event, stream_id)
    with _CHAIN_COUNTER_LOCK:
        count = _CHAIN_COUNTER.get(key, 0) + 1
        _CHAIN_COUNTER[key] = count
    return count % reset_interval == 0


def _commercial_key(cancel_event: threading.Event, stream_id: int) -> tuple[int, int]:
    return (id(cancel_event), int(stream_id))


def _store_commercial_mp4(temp_path: str | None, stream_id: int) -> None:
    if not temp_path:
        return
    path = pathlib.Path(temp_path)
    if not path.is_file():
        return
    stable_path = pathlib.Path(f"/tmp/ltx_commercial_{stream_id}.mp4")
    try:
        os.replace(path, stable_path)
    except OSError:
        LOGGER.exception("Failed to store commercial mp4: %s", path)
        return
    with _COMMERCIAL_MP4_LOCK:
        _COMMERCIAL_MP4[int(stream_id)] = (str(stable_path), time.time())


def _store_commercial_chunk_path(
    temp_path: str | None,
    stream_id: int,
    chunk_index: int,
    state: CommercialState,
) -> str | None:
    if not temp_path:
        return None
    path = pathlib.Path(temp_path)
    if not path.is_file():
        return None
    stable_path = pathlib.Path(f"/tmp/ltx_commercial_{stream_id}_chunk{chunk_index}.mp4")
    try:
        os.replace(path, stable_path)
    except OSError:
        LOGGER.exception("Failed to store commercial chunk mp4: %s", path)
        return None
    try:
        size = stable_path.stat().st_size
        LOGGER.info("Commercial chunk stored: stream_id=%s chunk_index=%s bytes=%s", stream_id, chunk_index, size)
    except OSError:
        LOGGER.warning("Commercial chunk size unavailable: %s", stable_path)
    while len(state.video_chunks) < chunk_index:
        state.video_chunks.append("")
    state.video_chunks[chunk_index - 1] = str(stable_path)
    state.last_output_path = str(stable_path)
    return str(stable_path)


def _store_commercial_audio_chunk(
    stream_id: int,
    chunk_index: int,
    audio_bytes: bytes,
    state: CommercialState,
) -> str | None:
    if not audio_bytes:
        return None
    temp_path = f"/tmp/ltx_commercial_audio_{stream_id}_chunk{chunk_index}_{uuid.uuid4().hex}.wav"
    stable_path = f"/tmp/ltx_commercial_audio_{stream_id}_chunk{chunk_index}.wav"
    try:
        with open(temp_path, "wb") as handle:
            handle.write(audio_bytes)
        os.replace(temp_path, stable_path)
    except OSError:
        LOGGER.exception("Failed to store commercial audio chunk: stream_id=%s chunk_index=%s", stream_id, chunk_index)
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        return None
    while len(state.audio_wav_chunks) < chunk_index:
        state.audio_wav_chunks.append("")
    state.audio_wav_chunks[chunk_index - 1] = stable_path
    return stable_path


def get_latest_commercial_mp4(stream_id: int = 0) -> tuple[str | None, float]:
    with _COMMERCIAL_MP4_LOCK:
        data = _COMMERCIAL_MP4.get(int(stream_id))
    if not data:
        return None, 0.0
    path, updated_at = data
    if not path or not os.path.exists(path):
        return None, 0.0
    return path, updated_at


def get_commercial_chunk_paths(cancel_event: threading.Event, stream_id: int = 0) -> list[str]:
    key = _commercial_key(cancel_event, stream_id)
    with _COMMERCIAL_LOCK:
        state = _COMMERCIAL_STATE.get(key)
        if not state:
            return []
        return list(state.video_chunks or [])


def is_commercial_done(cancel_event: threading.Event, stream_id: int) -> bool:
    key = _commercial_key(cancel_event, stream_id)
    with _COMMERCIAL_DONE_LOCK:
        return bool(_COMMERCIAL_DONE.get(key, False))


def get_commercial_progress(cancel_event: threading.Event, stream_id: int) -> tuple[int, int]:
    key = _commercial_key(cancel_event, stream_id)
    with _COMMERCIAL_LOCK:
        state = _COMMERCIAL_STATE.get(key)
        if not state:
            return 0, 0
        total_frames = int(state.frames_per_chunk * state.total_chunks)
        return int(state.frames_generated), total_frames


def get_commercial_chunk_counts(cancel_event: threading.Event, stream_id: int) -> tuple[int, int]:
    key = _commercial_key(cancel_event, stream_id)
    with _COMMERCIAL_LOCK:
        state = _COMMERCIAL_STATE.get(key)
        if not state:
            return 0, 0
        return len([p for p in (state.video_chunks or []) if p]), int(state.total_chunks)


def _mark_commercial_done(cancel_event: threading.Event, stream_id: int) -> None:
    key = _commercial_key(cancel_event, stream_id)
    with _COMMERCIAL_DONE_LOCK:
        _COMMERCIAL_DONE[key] = True


def stop_commercial(cancel_event: threading.Event, stream_id: int) -> None:
    _mark_commercial_done(cancel_event, stream_id)


def reset_commercial_state(cancel_event: threading.Event, stream_id: int) -> None:
    key = _commercial_key(cancel_event, stream_id)
    with _COMMERCIAL_DONE_LOCK:
        _COMMERCIAL_DONE.pop(key, None)
    with _COMMERCIAL_LOCK:
        state = _COMMERCIAL_STATE.pop(key, None)
        if state:
            for path in state.video_chunks or []:
                if path:
                    try:
                        os.remove(path)
                    except OSError:
                        LOGGER.warning("Failed to remove commercial chunk: %s", path)
            for audio_path in state.audio_wav_chunks or []:
                if audio_path:
                    try:
                        os.remove(audio_path)
                    except OSError:
                        LOGGER.warning("Failed to remove commercial audio: %s", audio_path)
            if state.last_output_path:
                try:
                    os.remove(state.last_output_path)
                except OSError:
                    LOGGER.warning("Failed to remove commercial video: %s", state.last_output_path)
    with _COMMERCIAL_SEED_LOCK:
        _COMMERCIAL_SEEDS.pop(key, None)
    _set_chain_frame("commercial_lock", cancel_event, None, stream_id)
    with _COMMERCIAL_MP4_LOCK:
        data = _COMMERCIAL_MP4.pop(int(stream_id), None)
    if data:
        try:
            os.remove(data[0])
        except OSError:
            LOGGER.warning("Failed to remove commercial final mp4: %s", data[0])
    av_path = f"/tmp/ltx_commercial_{stream_id}_av.mp4"
    try:
        if os.path.exists(av_path):
            os.remove(av_path)
    except OSError:
        LOGGER.warning("Failed to remove commercial AV mp4: %s", av_path)


def _get_or_create_commercial_seed(cancel_event: threading.Event, stream_id: int) -> int:
    key = _commercial_key(cancel_event, stream_id)
    with _COMMERCIAL_SEED_LOCK:
        seed = _COMMERCIAL_SEEDS.get(key)
        if seed is None:
            seed = random.SystemRandom().randint(0, 2**31 - 1)
            _COMMERCIAL_SEEDS[key] = seed
    return seed


def _get_or_create_commercial_state(
    config,
    cancel_event: threading.Event,
    stream_id: int,
) -> CommercialState:
    key = _commercial_key(cancel_event, stream_id)
    with _COMMERCIAL_LOCK:
        state = _COMMERCIAL_STATE.get(key)
        if state is not None:
            return state
        stage_width, stage_height, _ = _resolve_commercial_chunk_settings(config)
        prompt = getattr(config, "prompt", "")
        negative_prompt = _build_negative_prompt(prompt, getattr(config, "negative_prompt", "") or "")
        seed = config.seed if getattr(config, "seed", None) is not None else _get_or_create_commercial_seed(cancel_event, stream_id)
        guidance_scale = _env_float("LTX2_COMMERCIAL_CFG", 4.0)
        num_inference_steps = _env_int_clamped("LTX2_COMMERCIAL_STEPS", 20, min_value=1, max_value=200)
        chain_strength = _env_float("LTX2_COMMERCIAL_CHAIN_STRENGTH", 0.20)
        chain_frames = _env_int_clamped("LTX2_COMMERCIAL_CHAIN_FRAMES", 1, min_value=1, max_value=8)
        drop_prefix = _clamp_env_int("LTX2_DROP_PREFIX_FRAMES", 0, min_value=0, max_value=8)
        reset_interval = _env_int_clamped("LTX2_COMMERCIAL_RESET_INTERVAL_CHUNKS", 8, min_value=0, max_value=10000)
        blend_frames = _env_int_clamped("LTX2_COMMERCIAL_BLEND_FRAMES", 3, min_value=0, max_value=16)
        target_seconds = max(0.0, _env_float("LTX2_COMMERCIAL_TARGET_SECONDS", 35.0))
        frames_per_chunk = _env_int_clamped("LTX2_COMMERCIAL_FRAMES_PER_CHUNK", 73, min_value=9, max_value=513)
        frames_per_chunk = normalize_num_frames(frames_per_chunk, label="commercial_frames_per_chunk")
        if (frames_per_chunk - 1) % 8 != 0:
            LOGGER.warning("Invalid commercial frames_per_chunk=%s; forcing 73", frames_per_chunk)
            frames_per_chunk = 73
        fps = int(getattr(config, "fps", 24))
        target_frames = int(round(target_seconds * fps)) if target_seconds > 0 else frames_per_chunk
        num_chunks = max(1, int(round(target_frames / float(frames_per_chunk))))
        state = CommercialState(
            seed=seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            pipeline_mode=os.getenv("LTX2_COMMERCIAL_PIPELINE", "distilled").strip().lower() or "distilled",
            stage_width=stage_width,
            stage_height=stage_height,
            num_frames=frames_per_chunk,
            fps=fps,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            chain_strength=chain_strength,
            chain_frames=chain_frames,
            drop_prefix=drop_prefix,
            output_mode=getattr(config, "output_mode", "native"),
            reset_interval=reset_interval,
            blend_frames=blend_frames,
            target_seconds=target_seconds,
            frames_per_chunk=frames_per_chunk,
            total_chunks=num_chunks,
        )
        state.video_chunks = []
        state.audio_wav_chunks = []
        _COMMERCIAL_STATE[key] = state
        LOGGER.info(
            "Commercial mode start: stream_id=%s generation_mode=commercial_lock seed=%s steps=%s cfg=%.3f reset_interval=%s target_seconds=%.1f effective_seconds=%.2f frames_per_chunk=%s total_chunks=%s",
            stream_id,
            state.seed,
            state.num_inference_steps,
            state.guidance_scale,
            state.reset_interval,
            state.target_seconds,
            (state.frames_per_chunk * state.total_chunks) / max(1, state.fps),
            state.frames_per_chunk,
            state.total_chunks,
        )
        return state


def _resolve_chunk_settings(config) -> tuple[int, int, int]:
    explicit_frames = os.getenv("LTX2_NUM_FRAMES")
    if explicit_frames:
        try:
            requested = int(float(explicit_frames))
        except ValueError:
            LOGGER.warning("Invalid LTX2_NUM_FRAMES=%s; ignoring.", explicit_frames)
            requested = 0
        if requested > 0:
            num_frames = normalize_num_frames(requested, label="chunk")
            stage_width, stage_height = _resolve_stage_dimensions(config)
            LOGGER.info("Chunk frames override: LTX2_NUM_FRAMES=%s -> %s frames", requested, num_frames)
            return stage_width, stage_height, num_frames

    target_seconds = os.getenv("LTX2_TARGET_CHUNK_SECONDS")
    if target_seconds:
        try:
            chunk_seconds = float(target_seconds)
        except ValueError:
            LOGGER.warning("Invalid LTX2_TARGET_CHUNK_SECONDS=%s; using LTX2_CHUNK_SECONDS.", target_seconds)
            chunk_seconds = _env_float("LTX2_CHUNK_SECONDS", 1.0)
    else:
        chunk_seconds = _env_float("LTX2_CHUNK_SECONDS", 1.0)
    min_frames = _env_int_clamped("LTX2_MIN_FRAMES", 9, min_value=1, max_value=120)
    num_frames = normalize_num_frames(max(min_frames, int(chunk_seconds * config.fps)), label="chunk")
    stage_width, stage_height = _resolve_stage_dimensions(config)
    return stage_width, stage_height, num_frames


def _adaptive_cfg(
    mode: str,
    cancel_event: threading.Event,
    stream_id: int | None,
    *,
    base_cfg: float | None = None,
) -> float:
    resolved_base_cfg = base_cfg if base_cfg is not None else _env_float("LTX2_REALTIME_CFG", 1.0)
    boost_cfg = _env_float("LTX2_CFG_BOOST", resolved_base_cfg)
    every = _env_int_clamped("LTX2_CFG_BOOST_EVERY", 0, min_value=0, max_value=1000)
    if every <= 0 or boost_cfg <= resolved_base_cfg:
        return resolved_base_cfg
    sid = stream_id or 0
    key = (mode, id(cancel_event), sid)
    with _CFG_STATE_LOCK:
        count = _CFG_COUNTER.get(key, 0) + 1
        _CFG_COUNTER[key] = count
    if count % every == 0:
        return boost_cfg
    return resolved_base_cfg


def _resolve_prompt_drift(config, *, quality_lock: bool) -> bool:
    if quality_lock:
        return False
    if hasattr(config, "prompt_drift"):
        return bool(getattr(config, "prompt_drift"))
    return _env_bool("LTX2_PROMPT_DRIFT", False)


def _resolve_chain_settings(config) -> tuple[bool, float, int, int]:
    chain_enabled = _env_bool("LTX2_CHAINING", True)
    chain_strength = _env_float("LTX2_CHAIN_STRENGTH", 0.35)
    chain_frames = _clamp_env_int("LTX2_CHAIN_FRAMES", 3, min_value=1, max_value=8)
    drop_prefix = _clamp_env_int("LTX2_DROP_PREFIX_FRAMES", 0, min_value=0, max_value=8)
    if hasattr(config, "quality_lock"):
        quality_lock = bool(getattr(config, "quality_lock"))
        if quality_lock:
            chain_enabled = True
            chain_strength = float(getattr(config, "quality_lock_strength", chain_strength))
            chain_frames = int(getattr(config, "quality_lock_frames", chain_frames))
            drop_prefix = int(getattr(config, "drop_prefix_frames", drop_prefix))
        else:
            chain_enabled = False
            drop_prefix = 0
    return chain_enabled, chain_strength, chain_frames, drop_prefix


def _clamp_env_int(name: str, default: int, *, min_value: int, max_value: int) -> int:
    value = os.getenv(name)
    if value is None:
        return max(min_value, min(default, max_value))
    try:
        parsed = int(float(value))
    except ValueError:
        LOGGER.warning("Invalid int for %s=%s; using %s", name, value, default)
        parsed = default
    return max(min_value, min(parsed, max_value))


def generate_fever_dream_chunk(config, cancel_event: threading.Event) -> list[Image.Image]:
    backend = _get_backend()
    output_mode = getattr(config, "output_mode", "native")
    stream_id = _current_audio_stream_id()
    quality_lock = bool(getattr(config, "quality_lock", False))
    prompt_drift_enabled = _resolve_prompt_drift(config, quality_lock=quality_lock)
    realtime_cfg = getattr(config, "prompt_strength", None)
    if realtime_cfg is None:
        realtime_cfg = _env_float("LTX2_REALTIME_CFG", 1.0)
    steps_cap = getattr(config, "quality_steps", None)
    if steps_cap is None:
        steps_cap = _env_int_clamped("LTX2_REALTIME_STEPS", 6, min_value=1, max_value=200)
    else:
        steps_cap = int(steps_cap)
    reset_interval = _env_int_clamped("LTX2_CHAIN_RESET_INTERVAL", 0, min_value=0, max_value=10000)
    if _should_reset_chain("fever", cancel_event, stream_id, reset_interval):
        _set_chain_frame("fever", cancel_event, None, stream_id)
    if backend == "diffusers":
        pipe = _get_diffusers_pipe_or_status(config)
        if pipe is None:
            return [render_status_frame("Diffusers backend unavailable", config.width, config.height)]
        stage_width, stage_height, num_frames = _resolve_chunk_settings(config)
        prompt = _maybe_prompt_drift(config.prompt, allow_drift=prompt_drift_enabled)
        seed = None
        if config.seed is not None:
            seed = config.seed + int(time.time())
        try:
            realtime = os.getenv("LTX2_REALTIME", "0").lower() in {"1", "true", "yes", "on"}
            guidance_scale = 3.0 + config.dream_strength * 5.0
            if realtime:
                guidance_scale = _adaptive_cfg("fever", cancel_event, stream_id, base_cfg=realtime_cfg)
            negative_prompt = getattr(config, "negative_prompt", "") or ""
            negative_prompt = _build_negative_prompt(prompt, negative_prompt)
            num_inference_steps = int(10 + config.motion * 10)
            if realtime:
                num_inference_steps = min(num_inference_steps, steps_cap)
            frames = list(
                _generate_diffusers_chunk(
                    pipe,
                    output_mode=output_mode,
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
            )
            if frames:
                _set_chain_frame("fever", cancel_event, frames[-1], stream_id)
            return frames
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Continuity Studio diffusers generation error: %s", exc)
            return [render_status_frame("Generation error", config.width, config.height)]

    pipeline_variant = os.getenv("LTX2_COMMERCIAL_PIPELINE_VARIANT", "").strip().lower()
    if chunk_index == 1:
        LOGGER.info("Commercial pipeline variant=%s", pipeline_variant or "default")
    pipe = _get_pipelines_pipe_or_status(config, pipeline_variant=pipeline_variant or None)
    if pipe is None:
        return [render_status_frame("LTX-2 load failed", config.width, config.height)]
    stage_width, stage_height, num_frames = _resolve_chunk_settings(config)
    prompt = _maybe_prompt_drift(config.prompt, allow_drift=prompt_drift_enabled)
    seed = None
    if config.seed is not None:
        seed = config.seed + int(time.time())
    images: list[tuple[str, int, float]] | None = None
    temp_paths: list[str] = []
    chain_enabled, chain_strength, chain_frames, drop_prefix = _resolve_chain_settings(config)
    if os.getenv("LTX2_LOG_CHAINING") == "1":
        LOGGER.info(
            "Chaining: enabled=%s strength=%.3f frames=%s drop_prefix=%s",
            chain_enabled,
            chain_strength,
            chain_frames,
            drop_prefix,
        )
    try:
        if chain_enabled:
            carry = _get_chain_frame("fever", cancel_event, stream_id)
            if carry is not None:
                carry_path = _write_temp_pil(carry)
                temp_paths.append(carry_path)
                images = [(carry_path, i, chain_strength) for i in range(chain_frames)]
        realtime = os.getenv("LTX2_REALTIME", "0").lower() in {"1", "true", "yes", "on"}
        guidance_scale = 3.0 + config.dream_strength * 5.0
        if realtime:
            guidance_scale = _adaptive_cfg("fever", cancel_event, stream_id, base_cfg=realtime_cfg)
        negative_prompt = getattr(config, "negative_prompt", "") or ""
        negative_prompt = _build_negative_prompt(prompt, negative_prompt)
        num_inference_steps = int(10 + config.motion * 10)
        if realtime:
            num_inference_steps = min(num_inference_steps, steps_cap)
        frames = list(
            _generate_video_chunk(
                pipe,
                output_mode=output_mode,
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
        )
        if drop_prefix > 0 and len(frames) > drop_prefix:
            frames = frames[drop_prefix:]
        if frames:
            _set_chain_frame("fever", cancel_event, frames[-1], stream_id)
        return frames
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Continuity Studio generation error: %s", exc)
        return [render_status_frame("Generation error", config.width, config.height)]
    finally:
        for path in temp_paths:
            try:
                os.remove(path)
            except OSError:
                LOGGER.warning("Failed to remove temporary image: %s", path)


def generate_commercial_lock_chunk(config, cancel_event: threading.Event) -> list[Image.Image]:
    backend = _get_backend()
    stream_id = _current_audio_stream_id()
    if is_commercial_done(cancel_event, stream_id):
        return []
    state = _get_or_create_commercial_state(config, cancel_event, stream_id)
    env_pipeline = os.getenv("LTX2_COMMERCIAL_PIPELINE")
    if env_pipeline:
        state.pipeline_mode = env_pipeline.strip().lower() or state.pipeline_mode
    commercial_pipeline = (state.pipeline_mode or "distilled").strip().lower()
    if commercial_pipeline not in {"distilled", "comfy_equivalent"}:
        LOGGER.warning("Unknown LTX2_COMMERCIAL_PIPELINE=%s; defaulting to distilled", commercial_pipeline)
        commercial_pipeline = "distilled"
        state.pipeline_mode = "distilled"
    if state.chunk_index == 0:
        LOGGER.info("Commercial pipeline mode: %s", commercial_pipeline)
    use_comfy = _env_bool("LTX2_COMMERCIAL_USE_COMFY_PRESET", True)
    exposure_lock = _env_bool("LTX2_COMMERCIAL_EXPOSURE_LOCK", True)
    env_overrides: dict[str, str | None] = {}

    def _set_env(key: str, value: str | None) -> None:
        if key not in env_overrides:
            env_overrides[key] = os.getenv(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    if use_comfy:
        _set_env("LTX2_COMFY_PRESET", "1")
        _set_env("LTX2_SAMPLER_NAME", os.getenv("LTX2_COMMERCIAL_SAMPLER", "euler_ancestral"))
        _set_env("LTX2_STAGE2_ENABLE", os.getenv("LTX2_COMMERCIAL_STAGE2_ENABLE", "1"))
        _set_env("LTX2_STAGE2_CFG", os.getenv("LTX2_STAGE2_CFG", "1.0"))
        _set_env("LTX2_STAGE2_MANUAL_SIGMAS", os.getenv("LTX2_STAGE2_MANUAL_SIGMAS", "0.909375,0.725,0.421875,0.0"))
    if _env_bool("LTX2_COMMERCIAL_DISABLE_PROMPT_DRIFT", True):
        _set_env("LTX2_PROMPT_DRIFT", "0")
    if _env_bool("LTX2_COMMERCIAL_USE_DEV_CHECKPOINT", False):
        _set_env("LTX2_USE_DISTILLED", "0")
        _set_env("LTX2_FP8_FILE", os.getenv("LTX2_COMMERCIAL_DEV_FP8_FILE", "ltx-2-19b-dev-fp8.safetensors"))
    if state.chunks_generated >= state.total_chunks:
        _mark_commercial_done(cancel_event, stream_id)
        LOGGER.info(
            "Commercial mode complete: stream_id=%s generation_mode=commercial_lock target_seconds=%.1f effective_seconds=%.2f",
            stream_id,
            state.target_seconds,
            (state.frames_per_chunk * state.total_chunks) / max(1, state.fps),
        )
        return []
    chunk_frames = state.frames_per_chunk
    chunk_frames = max(chunk_frames, 9)
    chunk_frames = normalize_num_frames(chunk_frames, label="commercial_chunk")
    if (chunk_frames - 1) % 8 != 0:
        LOGGER.warning("Invalid num_frames=%s; forcing 73", chunk_frames)
        chunk_frames = 73
    with _COMMERCIAL_LOCK:
        state.chunk_index += 1
        chunk_index = state.chunk_index
        reset_occurred = state.reset_interval > 0 and chunk_index > 0 and chunk_index % state.reset_interval == 0
    LOGGER.info(
        "Commercial mode chunk: stream_id=%s chunk_index=%s frames_this_chunk=%s fps=%s seed=%s",
        stream_id,
        chunk_index,
        chunk_frames,
        state.fps,
        state.seed,
    )
    last_frame_for_blend = state.last_frame
    if reset_occurred:
        _set_chain_frame("commercial_lock", cancel_event, None, stream_id)
        LOGGER.info(
            "Commercial mode reset: stream_id=%s generation_mode=commercial_lock seed=%s steps=%s cfg=%.3f reset_interval=%s chunk_index=%s reset=%s",
            stream_id,
            state.seed,
            state.num_inference_steps,
            state.guidance_scale,
            state.reset_interval,
            chunk_index,
            True,
        )

    if commercial_pipeline == "comfy_equivalent" and backend == "diffusers":
        return [render_status_frame("Comfy-equivalent requires pipelines backend", config.width, config.height)]

    if backend == "diffusers":
        pipe = _get_diffusers_pipe_or_status(config)
        if pipe is None:
            return [render_status_frame("Diffusers backend unavailable", config.width, config.height)]
        try:
            frames = list(
                _generate_diffusers_chunk(
                    pipe,
                    output_mode=state.output_mode,
                    prompt=state.prompt,
                    negative_prompt=state.negative_prompt,
                    width=state.stage_width,
                    height=state.stage_height,
                    num_frames=chunk_frames,
                    fps=state.fps,
                    guidance_scale=state.guidance_scale,
                    num_inference_steps=state.num_inference_steps,
                    seed=state.seed,
                )
            )
            if state.drop_prefix > 0 and len(frames) > state.drop_prefix:
                frames = frames[state.drop_prefix:]
            if reset_occurred:
                frames = _apply_commercial_blend(last_frame_for_blend, frames, state.blend_frames)
            if frames:
                if state.anchor_frame is None:
                    state.anchor_frame = frames[0].copy()
                if exposure_lock and state.anchor_frame is not None:
                    frames = [_match_exposure(state.anchor_frame, frame) for frame in frames]
            if frames:
                _set_chain_frame("commercial_lock", cancel_event, frames[-1], stream_id)
                state.last_frame = frames[-1].copy()
                state.frames_generated += len(frames)
                chunk_path = f"/tmp/ltx_commercial_chunk_{stream_id}_{chunk_index}.mp4"
                _write_commercial_mp4_to_path(frames, state.fps, chunk_path)
                try:
                    size = os.path.getsize(chunk_path)
                    LOGGER.info("Commercial chunk stored: stream_id=%s chunk_index=%s bytes=%s", stream_id, chunk_index, size)
                except OSError:
                    LOGGER.warning("Commercial chunk size unavailable: %s", chunk_path)
                state.video_chunks.append(chunk_path)
                state.last_output_path = chunk_path
                audio_bytes, audio_ts = get_latest_audio_wav(stream_id)
                if audio_bytes and audio_ts > state.last_audio_ts:
                    if _store_commercial_audio_chunk(stream_id, chunk_index, audio_bytes, state):
                        state.last_audio_ts = audio_ts
            state.chunks_generated += 1
            if state.chunks_generated >= state.total_chunks:
                _mark_commercial_done(cancel_event, stream_id)
                final_path = f"/tmp/ltx_commercial_{stream_id}.mp4"
                video_concat = f"/tmp/ltx_commercial_{stream_id}_video.mp4"
                ordered_chunks = [p for p in state.video_chunks if p and os.path.exists(p)]
                if ordered_chunks:
                    _concat_mp4_chunks(ordered_chunks, video_concat, state.fps)
                    audio_path = _concat_wav_paths(state.audio_wav_chunks)
                    final_path = assemble_final_mp4(
                        [],
                        state.fps,
                        audio_path,
                        final_path,
                        video_path=video_concat,
                    )
                    _store_commercial_mp4(final_path, stream_id)
                    for path in ordered_chunks:
                        try:
                            os.remove(path)
                        except OSError:
                            pass
                    state.video_chunks.clear()
                    for audio_chunk in state.audio_wav_chunks:
                        if audio_chunk:
                            try:
                                os.remove(audio_chunk)
                            except OSError:
                                pass
                    state.audio_wav_chunks.clear()
                    if audio_path:
                        try:
                            os.remove(audio_path)
                        except OSError:
                            pass
                    try:
                        if os.path.exists(video_concat):
                            os.remove(video_concat)
                    except OSError:
                        pass
                else:
                    LOGGER.warning("Commercial finalization missing chunk files; skipping final mp4.")
            return frames
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Continuity Studio diffusers generation error: %s", exc)
            return [render_status_frame("Generation error", config.width, config.height)]

    pipe = None
    if commercial_pipeline != "comfy_equivalent":
        pipe = _get_pipelines_pipe_or_status(config)
        if pipe is None:
            return [render_status_frame("LTX-2 load failed", config.width, config.height)]
    if chunk_index == 1:
        try:
            checkpoint_path = _resolve_checkpoint_path()
        except Exception as exc:  # noqa: BLE001
            checkpoint_path = f"unresolved ({exc})"
        LOGGER.info(
            "Commercial mode checkpoint: stream_id=%s checkpoint_path=%s",
            stream_id,
            checkpoint_path,
        )
    images: list[tuple[str, int, float]] | None = None
    temp_paths: list[str] = []
    try:
        if reset_occurred and state.anchor_frame is not None:
            anchor_path = _write_temp_pil(state.anchor_frame)
            temp_paths.append(anchor_path)
            images = [(anchor_path, i, state.chain_strength) for i in range(state.chain_frames)]
        else:
            carry = _get_chain_frame("commercial_lock", cancel_event, stream_id)
            if carry is not None:
                carry_path = _write_temp_pil(carry)
                temp_paths.append(carry_path)
                images = [(carry_path, i, state.chain_strength) for i in range(state.chain_frames)]
        if use_comfy:
            os.environ["LTX2_COMFY_PRESET"] = "1"
        if commercial_pipeline == "comfy_equivalent":
            try:
                from comfy_equivalent import render_comfy_equivalent_mp4  # type: ignore
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Comfy-equivalent module import failed: %s", exc)
                return [render_status_frame("Comfy-equivalent unavailable", config.width, config.height)]

            try:
                artifacts = _resolve_artifacts(state.output_mode)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Comfy-equivalent artifacts resolve failed: %s", exc)
                return [render_status_frame("Model assets missing", config.width, config.height)]

            try:
                checkpoint_path = _resolve_checkpoint_path()
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Comfy-equivalent checkpoint resolve failed: %s", exc)
                return [render_status_frame("Checkpoint missing", config.width, config.height)]

            chunk_path = f"/tmp/ltx_commercial_tmp_{uuid.uuid4().hex}.mp4"
            output_path = render_comfy_equivalent_mp4(
                prompt=state.prompt,
                negative_prompt=state.negative_prompt,
                width=int(getattr(config, "width", state.stage_width)),
                height=int(getattr(config, "height", state.stage_height)),
                fps=state.fps,
                num_frames=chunk_frames,
                ckpt_path=str(checkpoint_path),
                gemma_root=str(artifacts.gemma_root),
                distilled_lora_path=str(artifacts.distilled_lora_path or ""),
                distilled_lora_strength=float(getattr(artifacts, "distilled_lora_strength", 1.0)),
                spatial_upscaler_path=str(artifacts.spatial_upsampler_path or ""),
                out_mp4_path=chunk_path,
            )
            frames = _decode_video_to_frames(output_path)
        else:
            frames, output_path = _generate_commercial_video_chunk(
                pipe,
                stream_id=stream_id,
                chunk_index=chunk_index,
                output_mode=state.output_mode,
                prompt=state.prompt,
                negative_prompt=state.negative_prompt,
                width=state.stage_width,
                height=state.stage_height,
                num_frames=chunk_frames,
                fps=state.fps,
                guidance_scale=state.guidance_scale,
                num_inference_steps=state.num_inference_steps,
                seed=state.seed,
                images=images,
                apply_comfy_overrides=use_comfy,
            )
        if output_path:
            stored_path = _store_commercial_chunk_path(output_path, stream_id, chunk_index, state)
            if stored_path is None:
                output_path = None
        if not output_path:
            chunk_path = f"/tmp/ltx_commercial_chunk_{stream_id}_{chunk_index}.mp4"
            _write_commercial_mp4_to_path(frames, state.fps, chunk_path)
            try:
                size = os.path.getsize(chunk_path)
                LOGGER.info("Commercial chunk stored: stream_id=%s chunk_index=%s bytes=%s", stream_id, chunk_index, size)
            except OSError:
                LOGGER.warning("Commercial chunk size unavailable: %s", chunk_path)
            state.video_chunks.append(chunk_path)
            state.last_output_path = chunk_path
        if state.drop_prefix > 0 and len(frames) > state.drop_prefix:
            frames = frames[state.drop_prefix:]
        if reset_occurred:
            frames = _apply_commercial_blend(last_frame_for_blend, frames, state.blend_frames)
        if frames:
            if state.anchor_frame is None:
                state.anchor_frame = frames[0].copy()
            if exposure_lock and state.anchor_frame is not None:
                frames = [_match_exposure(state.anchor_frame, frame) for frame in frames]
        if frames:
            _set_chain_frame("commercial_lock", cancel_event, frames[-1], stream_id)
            state.last_frame = frames[-1].copy()
            state.frames_generated += len(frames)
            sample_rate = int(float(os.getenv("LTX2_AUDIO_SAMPLE_RATE", "48000")))
            if commercial_pipeline == "comfy_equivalent":
                if state.last_output_path:
                    audio_bytes = _extract_wav_from_mp4(state.last_output_path, sample_rate)
                    if audio_bytes:
                        _store_commercial_audio_chunk(stream_id, chunk_index, audio_bytes, state)
            else:
                audio_bytes, audio_ts = get_latest_audio_wav(stream_id)
                if audio_bytes and audio_ts > state.last_audio_ts:
                    if _store_commercial_audio_chunk(stream_id, chunk_index, audio_bytes, state):
                        state.last_audio_ts = audio_ts
        state.chunks_generated += 1
        if state.chunks_generated >= state.total_chunks:
            _mark_commercial_done(cancel_event, stream_id)
            final_path = f"/tmp/ltx_commercial_{stream_id}.mp4"
            video_concat = f"/tmp/ltx_commercial_{stream_id}_video.mp4"
            audio_path = None
            ordered_chunks = [p for p in state.video_chunks if p and os.path.exists(p)]
            if ordered_chunks:
                if commercial_pipeline == "comfy_equivalent" and not state.audio_wav_chunks:
                    if _concat_mp4_chunks_with_audio(ordered_chunks, final_path, state.fps):
                        _store_commercial_mp4(final_path, stream_id)
                    else:
                        _concat_mp4_chunks(ordered_chunks, video_concat, state.fps)
                        _store_commercial_mp4(video_concat, stream_id)
                else:
                    _concat_mp4_chunks(ordered_chunks, video_concat, state.fps)
                    audio_path = _concat_wav_paths(state.audio_wav_chunks)
                    final_path = assemble_final_mp4(
                        [],
                        state.fps,
                        audio_path,
                        final_path,
                        video_path=video_concat,
                    )
                    _store_commercial_mp4(final_path, stream_id)
                for path in ordered_chunks:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                state.video_chunks.clear()
                for audio_chunk in state.audio_wav_chunks:
                    if audio_chunk:
                        try:
                            os.remove(audio_chunk)
                        except OSError:
                            pass
                state.audio_wav_chunks.clear()
                if audio_path:
                    try:
                        os.remove(audio_path)
                    except OSError:
                        pass
                try:
                    if os.path.exists(video_concat):
                        os.remove(video_concat)
                except OSError:
                    pass
            else:
                LOGGER.warning("Commercial finalization missing chunk files; skipping final mp4.")
        return frames
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Continuity Studio generation error: %s", exc)
        return [render_status_frame("Generation error", config.width, config.height)]
    finally:
        for key, prior in env_overrides.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior
        for path in temp_paths:
            try:
                os.remove(path)
            except OSError:
                LOGGER.warning("Failed to remove temporary image: %s", path)

def generate_mood_mirror_frames(
    config,
    latest_camera_state: Callable[[], tuple[np.ndarray | None, dict | None]],
    cancel_event: threading.Event,
) -> Iterable[Image.Image]:
    while not cancel_event.is_set():
        for frame in generate_mood_mirror_chunk(config, latest_camera_state, cancel_event):
            yield frame


def generate_mood_mirror_chunk(
    config,
    latest_camera_state: Callable[[], tuple[np.ndarray | None, dict | None]],
    cancel_event: threading.Event,
) -> list[Image.Image]:
    backend = _get_backend()
    output_mode = getattr(config, "output_mode", "native")
    stream_id = _current_audio_stream_id()
    quality_lock = bool(getattr(config, "quality_lock", False))
    prompt_drift_enabled = _resolve_prompt_drift(config, quality_lock=quality_lock)
    realtime_cfg = getattr(config, "prompt_strength", None)
    if realtime_cfg is None:
        realtime_cfg = _env_float("LTX2_REALTIME_CFG", 1.0)
    steps_cap = getattr(config, "quality_steps", None)
    if steps_cap is None:
        steps_cap = _env_int_clamped("LTX2_REALTIME_STEPS", 6, min_value=1, max_value=200)
    else:
        steps_cap = int(steps_cap)
    reset_interval = _env_int_clamped("LTX2_CHAIN_RESET_INTERVAL", 0, min_value=0, max_value=10000)
    if _should_reset_chain("mood", cancel_event, stream_id, reset_interval):
        _set_chain_frame("mood", cancel_event, None, stream_id)
    if backend == "diffusers":
        pipe = _get_diffusers_pipe_or_status(config)
        if pipe is None:
            return [render_status_frame("Diffusers backend unavailable", config.width, config.height)]
        stage_width, stage_height, num_frames = _resolve_chunk_settings(config)
        camera_frame, mood_state = latest_camera_state()
        if camera_frame is None:
            return [render_status_frame("Waiting for camera feed...", config.width, config.height)]
        prompt = config.base_prompt
        if mood_state:
            mood_prompt = mood_state.get("prompt_hint") or ""
            prompt = f"{prompt}, {mood_prompt}" if mood_prompt else prompt
        prompt = _maybe_prompt_drift(prompt, allow_drift=prompt_drift_enabled)
        seed = None
        if config.seed is not None:
            seed = config.seed + int(time.time())
        try:
            realtime = os.getenv("LTX2_REALTIME", "0").lower() in {"1", "true", "yes", "on"}
            guidance_scale = 3.0 + config.dream_strength * 4.0
            if realtime:
                guidance_scale = _adaptive_cfg("mood", cancel_event, stream_id, base_cfg=realtime_cfg)
            negative_prompt = getattr(config, "negative_prompt", "") or ""
            negative_prompt = _build_negative_prompt(prompt, negative_prompt)
            num_inference_steps = int(10 + config.motion * 10)
            if realtime:
                num_inference_steps = min(num_inference_steps, steps_cap)
            frames = list(
                _generate_diffusers_chunk(
                    pipe,
                    output_mode=output_mode,
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
            )
            if frames:
                _set_chain_frame("mood", cancel_event, frames[-1], stream_id)
            return frames
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Mood Mirror diffusers generation error: %s", exc)
            return [render_status_frame("Generation error", config.width, config.height)]

    pipe = _get_pipelines_pipe_or_status(config)
    if pipe is None:
        return [render_status_frame("LTX-2 load failed", config.width, config.height)]
    stage_width, stage_height, num_frames = _resolve_chunk_settings(config)
    camera_frame, mood_state = latest_camera_state()
    if camera_frame is None:
        return [render_status_frame("Waiting for camera feed...", config.width, config.height)]
    prompt = config.base_prompt
    if mood_state:
        mood_prompt = mood_state.get("prompt_hint") or ""
        prompt = f"{prompt}, {mood_prompt}" if mood_prompt else prompt
    prompt = _maybe_prompt_drift(prompt, allow_drift=prompt_drift_enabled)
    seed = None
    if config.seed is not None:
        seed = config.seed + int(time.time())
    strength = 0.2 + (1.0 - config.identity_strength) * 0.6
    temp_paths: list[str] = []
    images: list[tuple[str, int, float]] = []
    chain_enabled, chain_strength, chain_frames, drop_prefix = _resolve_chain_settings(config)
    if os.getenv("LTX2_LOG_CHAINING") == "1":
        LOGGER.info(
            "Chaining: enabled=%s strength=%.3f frames=%s drop_prefix=%s",
            chain_enabled,
            chain_strength,
            chain_frames,
            drop_prefix,
        )
    try:
        image_path = _write_temp_image(camera_frame)
        temp_paths.append(image_path)
        images.append((image_path, 0, strength))
        if chain_enabled:
            carry = _get_chain_frame("mood", cancel_event, stream_id)
            if carry is not None:
                carry_path = _write_temp_pil(carry)
                temp_paths.append(carry_path)
                images.extend((carry_path, i, chain_strength) for i in range(chain_frames))
        realtime = os.getenv("LTX2_REALTIME", "0").lower() in {"1", "true", "yes", "on"}
        guidance_scale = 3.0 + config.dream_strength * 4.0
        if realtime:
            guidance_scale = _adaptive_cfg("mood", cancel_event, stream_id, base_cfg=realtime_cfg)
        negative_prompt = getattr(config, "negative_prompt", "") or ""
        negative_prompt = _build_negative_prompt(prompt, negative_prompt)
        num_inference_steps = int(10 + config.motion * 10)
        if realtime:
            num_inference_steps = min(num_inference_steps, steps_cap)
        frames = list(
            _generate_video_chunk(
                pipe,
                output_mode=output_mode,
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
        )
        if drop_prefix > 0 and len(frames) > drop_prefix:
            frames = frames[drop_prefix:]
        if frames:
            _set_chain_frame("mood", cancel_event, frames[-1], stream_id)
        return frames
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Mood Mirror generation error: %s", exc)
        return [render_status_frame("Generation error", config.width, config.height)]
    finally:
        for path in temp_paths:
            try:
                os.remove(path)
            except OSError:
                LOGGER.warning("Failed to remove temporary image: %s", path)
