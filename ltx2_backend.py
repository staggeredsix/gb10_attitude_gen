from __future__ import annotations

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
from PIL import Image, ImageDraw, ImageFont

LOGGER = logging.getLogger("ltx2_backend")

_PIPELINE = None
_PIPELINE_LOCK = threading.Lock()


def _resolve_local_snapshot(model_id: str, filename: str) -> str | None:
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
        if (snapshot / filename).is_file() and (snapshot / "model_index.json").is_file():
            return str(snapshot)
    return None


def _load_pipeline(device: str = "cuda"):
    global _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is not None:
            return _PIPELINE
        model_id = os.getenv("LTX2_MODEL_ID", "Lightricks/LTX-2")
        variant = os.getenv("LTX2_VARIANT", "fp4")
        model_file = os.getenv("LTX2_MODEL_FILE", "ltx-2-19b-dev-fp4.safetensors")
        local_path = os.getenv("LTX2_MODEL_PATH")
        if not local_path:
            local_path = _resolve_local_snapshot(model_id, model_file)
        model_source = local_path or model_id
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        LOGGER.info("Loading LTX-2 pipeline from %s (variant=%s)", model_source, variant)
        try:
            pipe = DiffusionPipeline.from_pretrained(
                model_source,
                torch_dtype=dtype,
                variant=variant,
                use_safetensors=True,
                trust_remote_code=True,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to load LTX-2 with variant %s: %s", variant, exc)
            try:
                pipe = DiffusionPipeline.from_pretrained(
                    model_source,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    trust_remote_code=True,
                )
            except Exception as inner_exc:  # noqa: BLE001
                LOGGER.error("Unable to initialize LTX-2: %s", inner_exc)
                _PIPELINE = None
                return None
        pipe.to(device)
        _PIPELINE = pipe
        return _PIPELINE


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


def generate_fever_dream_frames(config, cancel_event: threading.Event) -> Iterable[Image.Image]:
    pipe = _load_pipeline()
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
        if last_frame is not None:
            kwargs["image"] = last_frame
        try:
            result = pipe(**{k: v for k, v in kwargs.items() if v is not None})
            frames = _extract_frames(result)
            if not frames:
                raise RuntimeError("No frames returned from LTX-2")
            for frame in frames:
                last_frame = frame
                yield frame
        except TypeError:
            kwargs.pop("image", None)
            result = pipe(**{k: v for k, v in kwargs.items() if v is not None})
            frames = _extract_frames(result)
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
    pipe = _load_pipeline()
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
            "image": init_image,
            "strength": strength,
        }
        if last_frame is not None:
            kwargs["init_image"] = last_frame
        try:
            result = pipe(**{k: v for k, v in kwargs.items() if v is not None})
            frames = _extract_frames(result)
            if not frames:
                raise RuntimeError("No frames returned from LTX-2")
            for frame in frames:
                last_frame = frame
                yield frame
        except TypeError:
            kwargs.pop("init_image", None)
            kwargs.pop("strength", None)
            result = pipe(**{k: v for k, v in kwargs.items() if v is not None})
            frames = _extract_frames(result)
            for frame in frames:
                last_frame = frame
                yield frame
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Mood Mirror generation error: %s", exc)
            yield render_status_frame("Generation error", config.width, config.height)
