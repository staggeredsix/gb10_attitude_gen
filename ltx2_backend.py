from __future__ import annotations

import inspect
import logging
import os
import pathlib
import random
import sys
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
_CHAIN_FRAMES: dict[tuple[str, int], Image.Image] = {}
_CFG_STATE_LOCK = threading.Lock()
_CFG_COUNTER: dict[tuple[str, int, int], int] = {}
_AUDIO_LOCK = threading.Lock()
_LATEST_AUDIO_BY_STREAM: dict[int, tuple[bytes, float]] = {}
_AUDIO_STREAM_LOCAL = threading.local()

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


def _load_pipelines_pipeline(output_mode: str, device: str = "cuda") -> object:
    cache_key = f"pipelines:{output_mode}:{device}"
    with _PIPELINE_LOCK:
        if cache_key in _PIPELINES:
            return _PIPELINES[cache_key]

        artifacts = _resolve_artifacts("native")
        dtype = torch.bfloat16
        enable_fp8 = os.getenv("LTX2_ENABLE_FP8", "1").lower() in {"1", "true", "yes", "on"}

        pipe_cls = DistilledPipeline
        init_kwargs = {
            "checkpoint_path": artifacts.checkpoint_path,
            "spatial_upsampler_path": artifacts.spatial_upsampler_path,
            "gemma_root": artifacts.gemma_root,
            "loras": artifacts.loras,
            "device": device,
            "fp8transformer": enable_fp8,
        }

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
    num_frames = _adjust_num_frames(num_frames)
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


def _round_down_to_multiple(x: int, m: int) -> int:
    return max(m, (x // m) * m)


def _round_up_to_multiple(x: int, m: int) -> int:
    return max(m, ((x + m - 1) // m) * m)


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
    is_distilled = isinstance(pipe, DistilledPipeline)

    if not any(name in param_names for name in ("prompt", "text")):
        raise RuntimeError("LTX-2 pipeline does not accept a prompt argument.")
    _assign_first_present(param_names, kwargs, prompt, ["prompt", "text"])
    _assign_first_present(param_names, kwargs, negative_prompt, ["negative_prompt"])
    _assign_first_present(param_names, kwargs, width, ["width"])
    _assign_first_present(param_names, kwargs, height, ["height"])
    _assign_first_present(param_names, kwargs, num_frames, ["num_frames", "video_length", "frames"])
    _assign_first_present(param_names, kwargs, fps, ["fps", "frame_rate"])
    _assign_first_present(param_names, kwargs, guidance_scale, ["guidance_scale", "cfg_scale", "cfg_guidance_scale"])
    if not is_distilled:
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
    _store_audio_from_result(result)
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
    with torch.no_grad():
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
    while not cancel_event.is_set():
        for frame in generate_fever_dream_chunk(config, cancel_event):
            yield frame


def _chain_key(mode: str, cancel_event: threading.Event) -> tuple[str, int]:
    return (mode, id(cancel_event))


def _get_chain_frame(mode: str, cancel_event: threading.Event) -> Image.Image | None:
    return _CHAIN_FRAMES.get(_chain_key(mode, cancel_event))


def _set_chain_frame(mode: str, cancel_event: threading.Event, frame: Image.Image | None) -> None:
    key = _chain_key(mode, cancel_event)
    if frame is None:
        _CHAIN_FRAMES.pop(key, None)
    else:
        _CHAIN_FRAMES[key] = frame


def _resolve_chunk_settings(config) -> tuple[int, int, int]:
    explicit_frames = os.getenv("LTX2_NUM_FRAMES")
    if explicit_frames:
        try:
            requested = int(float(explicit_frames))
        except ValueError:
            LOGGER.warning("Invalid LTX2_NUM_FRAMES=%s; ignoring.", explicit_frames)
            requested = 0
        if requested > 0:
            num_frames = _adjust_num_frames(requested)
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
    num_frames = _adjust_num_frames(max(min_frames, int(chunk_seconds * config.fps)))
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
                _set_chain_frame("fever", cancel_event, frames[-1])
            return frames
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Fever Dream diffusers generation error: %s", exc)
            return [render_status_frame("Generation error", config.width, config.height)]

    pipe = _get_pipelines_pipe_or_status(config)
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
            carry = _get_chain_frame("fever", cancel_event)
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
            _set_chain_frame("fever", cancel_event, frames[-1])
        return frames
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Fever Dream generation error: %s", exc)
        return [render_status_frame("Generation error", config.width, config.height)]
    finally:
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
                _set_chain_frame("mood", cancel_event, frames[-1])
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
            carry = _get_chain_frame("mood", cancel_event)
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
            _set_chain_frame("mood", cancel_event, frames[-1])
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
