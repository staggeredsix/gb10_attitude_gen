from __future__ import annotations

import argparse
import logging
import os
import shutil
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.background import BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image

from settings_loader import load_settings_conf

load_settings_conf()

from ltx2_backend import (
    DEFAULT_GEMMA_ROOT,
    backend_requires_gemma,
    generate_fever_dream_chunk,
    generate_commercial_lock_chunk,
    is_commercial_done,
    get_commercial_progress,
    get_commercial_chunk_paths,
    get_commercial_chunk_counts,
    reset_commercial_state,
    stop_commercial,
    generate_mood_mirror_chunk,
    generate_v2v_video,
    get_latest_audio_wav,
    get_latest_commercial_mp4,
    normalize_num_frames,
    set_audio_stream_id,
    log_backend_configuration,
    render_status_frame,
    validate_gemma_root,
    warmup_pipeline,
)

LOGGER = logging.getLogger("ltx2_app")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        LOGGER.warning("Invalid int for %s=%s; using %s", name, value, default)
        return default


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value is not None else default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_fps(name: str, default: int) -> int:
    value = _env_int(name, default)
    return max(1, min(value, 60))


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


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        LOGGER.warning("Invalid float for %s=%s; using %s", name, value, default)
        return default


def _find_blocked_terms(text: str) -> list[str]:
    if not text:
        return []
    lowered = text.lower()
    blocked = [
        # minors / youth
        "child",
        "children",
        "kid",
        "kids",
        "teen",
        "teenager",
        "young boy",
        "young girl",
        "schoolgirl",
        "schoolboy",
        "underage",
        "minor",
        "preteen",
        "toddler",
        "infant",
        "baby",
        "loli",
        "lolita",
        # nudity / sexual content
        "nude",
        "nudity",
        "naked",
        "topless",
        "breast",
        "breasts",
        "boob",
        "boobs",
        "nipples",
        "areola",
        "lingerie",
        "bikini",
        "swimsuit",
        "thong",
        "genitals",
        "vagina",
        "penis",
        "porn",
        "porno",
        "nsfw",
        "explicit",
        "erotic",
        "sex",
        "sexual",
        "fetish",
    ]
    hits = [term for term in blocked if term in lowered]
    return hits


DEFAULT_WIDTH = _env_int("LTX2_NATIVE_WIDTH", 1280)
DEFAULT_HEIGHT = _env_int("LTX2_NATIVE_HEIGHT", 736)
DEFAULT_FPS = _env_int("LTX2_NATIVE_FPS", 24)
DEFAULT_STREAMS = _env_int("LTX2_STREAMS", 2)
DEFAULT_OUTPUT_MODE = _env_str("LTX2_OUTPUT_MODE", "native")
if DEFAULT_OUTPUT_MODE not in {"native", "upscaled"}:
    LOGGER.warning("Invalid LTX2_OUTPUT_MODE=%s; using native.", DEFAULT_OUTPUT_MODE)
    DEFAULT_OUTPUT_MODE = "native"


class RunConfig(BaseModel):
    mode: Literal["fever", "mood", "commercial_lock"] = "fever"
    generation_mode: str | None = None
    prompt: str = Field("surreal dreamscape, liquid light, ethereal forms", min_length=1)
    negative_prompt: str = ""
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: int = DEFAULT_FPS
    streams: int = DEFAULT_STREAMS
    seed: int | None = None
    dream_strength: float = 0.0
    motion: float = 0.6
    prompt_strength: float = 1.0
    quality_steps: int = 8
    quality_lock: bool = False
    quality_lock_strength: float = 0.35
    quality_lock_frames: int = 3
    drop_prefix_frames: int = 0
    prompt_drift: bool = False
    base_prompt: str = Field("portrait, cinematic lighting", min_length=1)
    identity_strength: float = 0.7
    output_mode: Literal["native", "upscaled"] = DEFAULT_OUTPUT_MODE


class SettingsUpdate(BaseModel):
    settings: dict[str, float | int | str | None]


@dataclass
class StreamState:
    queue: queue.Queue[bytes]
    thread: threading.Thread
    last_frame: bytes | None = None
    status: str = "starting"
    stream_fps: int = 30
    buffer_max: int = 0
    last_gen_seconds: float = 0.0
    last_chunk_playback_seconds: float = 0.0


class StreamManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._streams: dict[int, StreamState] = {}

    def restart(self, config: RunConfig, latest_camera_state: Callable[[], tuple[np.ndarray | None, dict | None]]) -> None:
        with self._lock:
            self._cancel_event.set()
            for stream in self._streams.values():
                if stream.thread.is_alive():
                    stream.thread.join(timeout=1.0)
            self._streams = {}
            self._cancel_event = threading.Event()
            playback_fps = _env_fps("LTX2_PLAYBACK_FPS", config.fps or 24)
            buffer_seconds = _env_float("LTX2_BUFFER_SECONDS", 4.0)
            maxsize = int(playback_fps * buffer_seconds)
            maxsize = max(60, min(maxsize, 2000))
            for stream_id in range(config.streams):
                stream_queue: queue.Queue[bytes] = queue.Queue(maxsize=maxsize)
                stream = StreamState(
                    queue=stream_queue,
                    thread=threading.Thread(),
                    stream_fps=playback_fps,
                    buffer_max=maxsize,
                )
                thread = threading.Thread(
                    target=self._run_stream,
                    args=(stream_id, config, stream, latest_camera_state, self._cancel_event),
                    daemon=True,
                )
                stream.thread = thread
                self._streams[stream_id] = stream
                thread.start()

    def get_stream(self, stream_id: int) -> StreamState:
        with self._lock:
            stream = self._streams.get(stream_id)
            if stream is None:
                raise KeyError
            return stream

    def _run_stream(
        self,
        stream_id: int,
        config: RunConfig,
        stream: StreamState,
        latest_camera_state: Callable[[], tuple[np.ndarray | None, dict | None]],
        cancel_event: threading.Event,
    ) -> None:
        status_text = "Starting LTX-2 stream..."
        stream.last_frame = _encode_frame(render_status_frame(status_text, current_config.width, current_config.height))
        _queue_frame(stream.queue, stream.last_frame, drop_old=True)
        try:
            while not cancel_event.is_set() and not inference_enabled.is_set():
                status_frame = render_status_frame("Waiting for POST /api/config", current_config.width, current_config.height)
                stream.last_frame = _encode_frame(status_frame)
                _queue_frame(stream.queue, stream.last_frame, drop_old=True)
                time.sleep(1.0 / max(1, current_config.fps))
            if cancel_event.is_set():
                return
            playback_fps = stream.stream_fps
            buffer_seconds = _env_float("LTX2_BUFFER_SECONDS", 4.0)
            low_water_seconds = _env_float("LTX2_LOW_WATER_SECONDS", 1.0)
            drop_old = _env_bool("LTX2_DROP_OLD_FRAMES", True)
            while not cancel_event.is_set():
                cfg = current_config
                if cfg.mode == "commercial_lock" and is_commercial_done(cancel_event, stream_id):
                    time.sleep(0.25)
                    continue
                if cfg.mode != "commercial_lock":
                    queued_seconds = stream.queue.qsize() / max(1, playback_fps)
                    if queued_seconds >= buffer_seconds:
                        time.sleep(0.01)
                        continue
                    dynamic_low_water = max(low_water_seconds, stream.last_gen_seconds)
                    if queued_seconds > dynamic_low_water and not stream.queue.empty():
                        time.sleep(0.005)
                        continue
                start_time = time.time()
                set_audio_stream_id(stream_id)
                if cfg.mode == "fever":
                    frames = generate_fever_dream_chunk(cfg, cancel_event)
                elif cfg.mode == "commercial_lock":
                    frames = generate_commercial_lock_chunk(cfg, cancel_event)
                    drop_old = True
                else:
                    frames = generate_mood_mirror_chunk(cfg, latest_camera_state, cancel_event)
                set_audio_stream_id(None)
                stream.last_gen_seconds = max(0.0, time.time() - start_time)
                stream.last_chunk_playback_seconds = len(frames) / max(1, playback_fps)
                for frame in frames:
                    if cancel_event.is_set():
                        break
                    stream.last_frame = _encode_frame(frame)
                    _queue_frame(stream.queue, stream.last_frame, drop_old=drop_old)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Stream %s crashed: %s", stream_id, exc)
            error_frame = render_status_frame(f"Stream error: {exc}", config.width, config.height)
            stream.last_frame = _encode_frame(error_frame)
            _queue_frame(stream.queue, stream.last_frame, drop_old=True)


def _queue_frame(q: queue.Queue[bytes], frame: bytes, *, drop_old: bool) -> None:
    try:
        if q.full():
            if drop_old:
                q.get_nowait()
            else:
                return
        q.put_nowait(frame)
    except queue.Full:
        return


def _encode_frame(frame: Image.Image | np.ndarray) -> bytes:
    if isinstance(frame, Image.Image):
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not success:
        raise RuntimeError("Failed to encode frame")
    return buffer.tobytes()


def _validate_config(config: RunConfig) -> RunConfig:
    for label, value in (
        ("prompt", config.prompt),
        ("negative_prompt", config.negative_prompt),
        ("base_prompt", config.base_prompt),
    ):
        hits = _find_blocked_terms(value)
        if hits:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Disallowed content in {label}. "
                    "This app blocks minors/children and NSFW/sexual content."
                ),
            )
    if config.width % 32 != 0 or config.height % 32 != 0:
        raise HTTPException(status_code=400, detail="Width and height must be multiples of 32.")
    if config.output_mode == "upscaled" and (config.width % 64 != 0 or config.height % 64 != 0):
        raise HTTPException(
            status_code=400,
            detail="Upscaled output requires width and height to be multiples of 64.",
        )
    if config.output_mode == "native":
        max_native_w = _env_int_clamped("LTX2_NATIVE_MAX_WIDTH", 1280, min_value=64, max_value=8192)
        max_native_h = _env_int_clamped("LTX2_NATIVE_MAX_HEIGHT", 736, min_value=64, max_value=8192)
        if config.width > max_native_w or config.height > max_native_h:
            new_w = min(config.width, max_native_w)
            new_h = min(config.height, max_native_h)
            new_w = _round_down_to_multiple(new_w, 32)
            new_h = _round_down_to_multiple(new_h, 32)
            LOGGER.warning(
                "Native resolution capped: %sx%s -> %sx%s (LTX2_NATIVE_MAX_WIDTH/HEIGHT)",
                config.width,
                config.height,
                new_w,
                new_h,
            )
            config.width = new_w
            config.height = new_h
    if config.output_mode == "upscaled":
        max_up_w = _env_int_clamped("LTX2_UPSCALED_MAX_WIDTH", 4096, min_value=64, max_value=16384)
        max_up_h = _env_int_clamped("LTX2_UPSCALED_MAX_HEIGHT", 2304, min_value=64, max_value=16384)
        if config.width > max_up_w or config.height > max_up_h:
            new_w = min(config.width, max_up_w)
            new_h = min(config.height, max_up_h)
            new_w = _round_down_to_multiple(new_w, 64)
            new_h = _round_down_to_multiple(new_h, 64)
            LOGGER.warning(
                "Upscaled resolution capped: %sx%s -> %sx%s (LTX2_UPSCALED_MAX_WIDTH/HEIGHT)",
                config.width,
                config.height,
                new_w,
                new_h,
            )
            config.width = new_w
            config.height = new_h
    config.streams = max(1, min(config.streams, 16))
    if os.getenv("LTX2_BACKEND", "pipelines").strip().lower() == "pipelines" and config.streams > 1:
        LOGGER.warning(
            "Pipelines backend with streams=%s can cause repeated model loads/VRAM churn. "
            "Consider setting LTX2_STREAMS=1 for fp8.",
            config.streams,
        )
    config.fps = max(1, min(config.fps, 60))
    config.dream_strength = float(np.clip(config.dream_strength, 0.0, 1.0))
    config.motion = float(np.clip(config.motion, 0.0, 1.0))
    config.prompt_strength = float(np.clip(config.prompt_strength, 0.0, 2.0))
    config.quality_steps = int(max(1, min(config.quality_steps, 60)))
    config.quality_lock_strength = float(np.clip(config.quality_lock_strength, 0.0, 1.0))
    config.quality_lock_frames = int(max(1, min(config.quality_lock_frames, 8)))
    config.drop_prefix_frames = int(max(0, min(config.drop_prefix_frames, 8)))
    config.identity_strength = float(np.clip(config.identity_strength, 0.0, 1.0))
    return config


def _validate_prompt_value(prompt: str) -> None:
    hits = _find_blocked_terms(prompt)
    if hits:
        raise HTTPException(
            status_code=400,
            detail="Disallowed content in prompt. This app blocks minors/children and NSFW/sexual content.",
        )


def _round_down_to_multiple(x: int, m: int) -> int:
    return max(m, (x // m) * m)


def _adjust_v2v_frames(num_frames: int) -> int:
    if num_frames < 1:
        return 1
    k = int(round((num_frames - 1) / 8))
    return max(1, 1 + 8 * k)


def _round_down_v2v_frames(num_frames: int) -> int:
    if num_frames < 1:
        return 1
    k = (num_frames - 1) // 8
    return max(1, 1 + 8 * k)


def _transcode_video(
    input_path: str,
    output_path: str,
    *,
    width: int,
    height: int,
    fps: int,
    max_frames: int | None = None,
) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Failed to open uploaded video.")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = fps if fps and fps > 0 else int(source_fps) if source_fps else current_config.fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(target_fps), (width, height))
    if not writer.isOpened():
        cap.release()
        raise HTTPException(status_code=500, detail="Failed to initialize video writer.")
    frames_written = 0
    last_frame = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frames_written >= max_frames:
                break
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(resized)
            frames_written += 1
            last_frame = resized
        if max_frames is not None and frames_written < max_frames and last_frame is not None:
            while frames_written < max_frames:
                writer.write(last_frame)
                frames_written += 1
    finally:
        cap.release()
        writer.release()


def _concat_videos(video_paths: list[str], output_path: str, *, fps: int, width: int, height: int) -> None:
    if not video_paths:
        raise HTTPException(status_code=500, detail="No video chunks to concatenate.")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise HTTPException(status_code=500, detail="Failed to initialize video writer for concatenation.")
    try:
        for path in video_paths:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                cap.release()
                raise HTTPException(status_code=500, detail=f"Failed to open chunk: {path}")
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
        writer.release()


def _should_restart_streams(old: RunConfig, new: RunConfig) -> bool:
    if old.mode != new.mode:
        return True
    if old.width != new.width or old.height != new.height:
        return True
    if old.fps != new.fps:
        return True
    if old.streams != new.streams:
        return True
    if old.output_mode != new.output_mode:
        return True
    return False


def _estimate_mood(frame_bgr: np.ndarray) -> dict[str, Any]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2].mean() / 255.0
    saturation = hsv[:, :, 1].mean() / 255.0
    valence = float(np.clip((brightness - 0.5) * 2.0, -1.0, 1.0))
    arousal = float(np.clip(0.3 + saturation * 0.7, 0.0, 1.0))
    labels = []
    if brightness > 0.65:
        labels.append("bright")
    if brightness < 0.35:
        labels.append("dim")
    if saturation > 0.6:
        labels.append("vivid")
    face_count = 0
    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(frame_bgr, scaleFactor=1.1, minNeighbors=5)
        face_count = len(faces)
    except Exception:  # noqa: BLE001
        face_count = 0
    if face_count:
        labels.append("face")
    return {"valence": valence, "arousal": arousal, "labels": labels}


def _mood_to_prompt(mood: dict[str, Any] | None) -> str:
    if not mood:
        return "balanced tone, cinematic portrait"
    valence = mood.get("valence", 0.0)
    arousal = mood.get("arousal", 0.5)
    if valence >= 0.3:
        valence_words = "warm, uplifting, golden light"
    elif valence <= -0.3:
        valence_words = "cool, moody, nocturnal tones"
    else:
        valence_words = "neutral, balanced lighting"
    if arousal >= 0.65:
        arousal_words = "dynamic motion blur, energetic atmosphere"
    elif arousal <= 0.35:
        arousal_words = "calm, serene, soft gradients"
    else:
        arousal_words = "gentle movement, steady ambience"
    return f"{valence_words}, {arousal_words}"


app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

stream_manager = StreamManager()
current_config = RunConfig()
current_streams_count = current_config.streams
latest_camera_frame: np.ndarray | None = None
latest_mood: dict[str, Any] | None = None
latest_camera_lock = threading.Lock()
inference_enabled = threading.Event()
health_status: dict[str, Any] = {"pipeline_loaded": False, "errors": {}, "pipelines": {}}
_settings_mtime: float | None = None


def _settings_path() -> str:
    return os.path.join(os.path.dirname(__file__), "settings.conf")


def _settings_defaults_path() -> str:
    return os.path.join(os.path.dirname(__file__), "settings.defaults.conf")


def _ensure_settings_defaults() -> None:
    defaults_path = _settings_defaults_path()
    if os.path.exists(defaults_path):
        return
    settings_path = _settings_path()
    if not os.path.exists(settings_path):
        return
    try:
        shutil.copyfile(settings_path, defaults_path)
    except OSError:
        LOGGER.exception("Failed to snapshot settings defaults")


def _parse_settings_lines(lines: list[str]) -> tuple[dict[str, str], dict[str, int], int | None]:
    settings: dict[str, str] = {}
    locations: dict[str, int] = {}
    current_section: str | None = None
    inference_section_idx: int | None = None
    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip().lower() or None
            if current_section == "inference":
                inference_section_idx = idx
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if current_section == "inference" and key.lower() == "steps":
            settings["inference.steps"] = value
            locations["inference.steps"] = idx
            continue
        if current_section is None:
            settings[key] = value
            locations[key] = idx
    return settings, locations, inference_section_idx


def _apply_settings_updates(updates: dict[str, str]) -> None:
    settings_path = _settings_path()
    lines = []
    if os.path.exists(settings_path):
        lines = Path(settings_path).read_text(encoding="utf-8").splitlines()
    settings, locations, inference_section_idx = _parse_settings_lines(lines)
    updated_lines = list(lines)
    pending_new: list[str] = []
    pending_inference: list[str] = []
    for key, value in updates.items():
        value_text = "" if value is None else str(value)
        if key == "inference.steps":
            if key in locations:
                updated_lines[locations[key]] = f"steps={value_text}"
            else:
                pending_inference.append(f"steps={value_text}")
            continue
        if key in locations:
            updated_lines[locations[key]] = f"{key}={value_text}"
        else:
            pending_new.append(f"{key}={value_text}")
    if pending_inference:
        if inference_section_idx is None:
            updated_lines.extend(["", "[inference]"])
            updated_lines.extend(pending_inference)
        else:
            insert_at = inference_section_idx + 1
            for line in pending_inference:
                updated_lines.insert(insert_at, line)
                insert_at += 1
    if pending_new:
        if updated_lines and updated_lines[-1].strip():
            updated_lines.append("")
        updated_lines.extend(pending_new)
    Path(settings_path).write_text("\n".join(updated_lines) + "\n", encoding="utf-8")

def _start_settings_watcher() -> None:
    def _watch() -> None:
        global _settings_mtime
        poll_seconds = _env_float("LTX2_SETTINGS_POLL_SECONDS", 2.0)
        settings_path = _settings_path()
        while True:
            try:
                mtime = os.path.getmtime(settings_path)
                if _settings_mtime is None:
                    _settings_mtime = mtime
                elif mtime != _settings_mtime:
                    _settings_mtime = mtime
                    load_settings_conf(override=True)
                    if inference_enabled.is_set():
                        stream_manager.restart(current_config, _get_latest_camera_state)
                        LOGGER.info("settings.conf changed; streams restarted.")
                    else:
                        LOGGER.info("settings.conf changed; reloaded.")
            except FileNotFoundError:
                _settings_mtime = None
            except Exception:  # noqa: BLE001
                LOGGER.exception("settings.conf watcher error")
            time.sleep(max(0.2, poll_seconds))

    thread = threading.Thread(target=_watch, name="settings-watcher", daemon=True)
    thread.start()


def _gemma_status() -> dict[str, Any]:
    gemma_required = backend_requires_gemma()
    gemma_root = os.getenv("LTX2_GEMMA_ROOT", DEFAULT_GEMMA_ROOT)
    if not gemma_required:
        return {
            "gemma_root": gemma_root,
            "gemma_ok": True,
            "gemma_required": False,
            "gemma_reason": "Gemma not required for diffusers backend.",
        }
    gemma_ok, gemma_reason = validate_gemma_root(gemma_root)
    return {
        "gemma_root": gemma_root,
        "gemma_ok": gemma_ok,
        "gemma_required": True,
        "gemma_reason": gemma_reason,
    }


@app.on_event("startup")
def _startup() -> None:
    log_backend_configuration(current_config.output_mode)
    _ensure_settings_defaults()
    health_status.update(_gemma_status())
    health_status["autostart"] = _env_bool("LTX2_AUTOSTART", False)
    health_status["inference_enabled"] = inference_enabled.is_set()
    _start_settings_watcher()
    if not health_status.get("gemma_ok", False):
        health_status["errors"]["gemma"] = health_status.get("gemma_reason")
    if health_status["autostart"]:
        try:
            info = warmup_pipeline(current_config.output_mode)
            health_status["pipeline_loaded"] = True
            health_status["pipelines"][current_config.output_mode] = info
            inference_enabled.set()
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("LTX-2 warmup failed: %s", exc)
            health_status["errors"][current_config.output_mode] = str(exc)
    else:
        LOGGER.info("Autostart disabled; waiting for POST /api/config.")
    stream_manager.restart(current_config, _get_latest_camera_state)
    global current_streams_count
    current_streams_count = current_config.streams


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/config")
def get_config() -> RunConfig:
    return current_config


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    settings_path = _settings_path()
    if not os.path.exists(settings_path):
        return {"settings": []}
    lines = Path(settings_path).read_text(encoding="utf-8").splitlines()
    settings, _, _ = _parse_settings_lines(lines)
    items = []
    for key in sorted(settings.keys()):
        value = settings[key]
        items.append({"key": key, "value": value, "empty_allowed": value == ""})
    return {"settings": items}


@app.post("/api/settings")
def set_settings(payload: SettingsUpdate) -> dict[str, Any]:
    updates: dict[str, str] = {}
    for key, value in payload.settings.items():
        if value is None:
            updates[key] = ""
        else:
            updates[key] = str(value)
    _apply_settings_updates(updates)
    load_settings_conf(override=True)
    if inference_enabled.is_set():
        stream_manager.restart(current_config, _get_latest_camera_state)
    return {"ok": True}


@app.post("/api/settings/reset")
def reset_settings() -> dict[str, Any]:
    _ensure_settings_defaults()
    defaults_path = _settings_defaults_path()
    settings_path = _settings_path()
    if not os.path.exists(defaults_path):
        raise HTTPException(status_code=500, detail="Defaults snapshot missing.")
    shutil.copyfile(defaults_path, settings_path)
    load_settings_conf(override=True)
    if inference_enabled.is_set():
        stream_manager.restart(current_config, _get_latest_camera_state)
    return {"ok": True}


@app.post("/api/config")
async def set_config(request: Request) -> RunConfig:
    payload = await request.json()
    if "generation_mode" in payload and "mode" not in payload:
        payload["mode"] = payload["generation_mode"]
    config = RunConfig(**payload)
    config = _validate_config(config)
    gemma_status = _gemma_status()
    if gemma_status.get("gemma_required") and not gemma_status.get("gemma_ok", False):
        LOGGER.warning("Gemma missing; streams will show status frames. (%s)", gemma_status.get("gemma_reason"))
    global current_config
    restart_streams = _should_restart_streams(current_config, config)
    current_config = config
    global current_streams_count
    current_streams_count = config.streams
    LOGGER.info("Updated config output_mode=%s", current_config.output_mode)
    if not inference_enabled.is_set():
        health_status["inference_enabled"] = inference_enabled.is_set()
    if not health_status.get("pipeline_loaded", False):
        try:
            info = warmup_pipeline(current_config.output_mode)
            health_status["pipeline_loaded"] = True
            health_status["pipelines"][current_config.output_mode] = info
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("LTX-2 warmup failed: %s", exc)
            health_status["errors"][current_config.output_mode] = str(exc)
    if restart_streams:
        stream_manager.restart(current_config, _get_latest_camera_state)
    return current_config


@app.post("/api/start")
def start_inference() -> dict[str, Any]:
    if not inference_enabled.is_set():
        inference_enabled.set()
    health_status["inference_enabled"] = inference_enabled.is_set()
    if not health_status.get("pipeline_loaded", False):
        try:
            info = warmup_pipeline(current_config.output_mode)
            health_status["pipeline_loaded"] = True
            health_status["pipelines"][current_config.output_mode] = info
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("LTX-2 warmup failed: %s", exc)
            health_status["errors"][current_config.output_mode] = str(exc)
    return {"ok": True, "inference_enabled": inference_enabled.is_set()}


@app.post("/api/stop")
def stop_inference() -> dict[str, Any]:
    if inference_enabled.is_set():
        inference_enabled.clear()
    health_status["inference_enabled"] = inference_enabled.is_set()
    stream_manager.restart(current_config, _get_latest_camera_state)
    return {"ok": True, "inference_enabled": inference_enabled.is_set()}


@app.post("/api/v2v")
async def v2v_endpoint(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    strength: float = Form(0.7),
    width: int | None = Form(None),
    height: int | None = Form(None),
    fps: int | None = Form(None),
    seconds: float | None = Form(4.0),
    num_frames: int | None = Form(None),
    seed: int | None = Form(None),
) -> FileResponse:
    _validate_prompt_value(prompt)
    w = width or current_config.width
    h = height or current_config.height
    f = fps or current_config.fps
    if w % 64 != 0 or h % 64 != 0:
        adjusted_w = _round_down_to_multiple(w, 64)
        adjusted_h = _round_down_to_multiple(h, 64)
        LOGGER.info("V2V adjusted resolution: %sx%s -> %sx%s", w, h, adjusted_w, adjusted_h)
        w, h = adjusted_w, adjusted_h
    if w < 64 or h < 64:
        raise HTTPException(status_code=400, detail="V2V resolution too small after adjustment.")
    if num_frames is None:
        num_frames = int(max(1, float(seconds or 0.0) * f))
    if num_frames < 1:
        raise HTTPException(status_code=400, detail="Invalid num_frames.")
    max_frames_env = int(os.getenv("LTX2_V2V_MAX_FRAMES", "257"))
    total_max_frames_env = int(os.getenv("LTX2_V2V_TOTAL_MAX_FRAMES", "8640"))
    max_chunk_frames = _round_down_v2v_frames(max_frames_env) if max_frames_env > 0 else None
    adjusted_frames = normalize_num_frames(num_frames, label="v2v_request")
    if total_max_frames_env > 0 and adjusted_frames > total_max_frames_env:
        max_seconds = total_max_frames_env / float(f or 1)
        raise HTTPException(
            status_code=400,
            detail=f"Requested V2V duration too long. Max ~{max_seconds:.1f}s at {f} fps.",
        )
    if adjusted_frames != num_frames:
        LOGGER.info("V2V num_frames adjusted: %s -> %s", num_frames, adjusted_frames)
    num_frames = adjusted_frames
    input_path = f"/tmp/v2v_{uuid.uuid4().hex}.mp4"
    resized_path = f"/tmp/v2v_resized_{uuid.uuid4().hex}.mp4"
    output_path = f"/tmp/v2v_out_{uuid.uuid4().hex}.mp4"
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    with open(input_path, "wb") as fobj:
        fobj.write(data)

    def _cleanup(paths: list[str]) -> None:
        for path in paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                continue
            except OSError:
                LOGGER.warning("Failed to remove temp file: %s", path)

    temp_paths: list[str] = [input_path, resized_path, output_path]
    if not max_chunk_frames or num_frames <= max_chunk_frames:
        _transcode_video(input_path, resized_path, width=w, height=h, fps=f, max_frames=num_frames)
        generate_v2v_video(
            input_video_path=resized_path,
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            width=w,
            height=h,
            fps=f,
            num_frames=num_frames,
            seed=seed,
            strength=float(strength),
            output_path=output_path,
        )
    else:
        remaining = num_frames
        chunk_outputs: list[str] = []
        prev_source = input_path
        while remaining > 0:
            chunk_frames = min(max_chunk_frames, remaining)
            chunk_frames = _round_down_v2v_frames(chunk_frames)
            if chunk_frames < 1:
                break
            cond_path = f"/tmp/v2v_cond_{uuid.uuid4().hex}.mp4"
            chunk_path = f"/tmp/v2v_chunk_{uuid.uuid4().hex}.mp4"
            temp_paths.extend([cond_path, chunk_path])
            _transcode_video(prev_source, cond_path, width=w, height=h, fps=f, max_frames=chunk_frames)
            generate_v2v_video(
                input_video_path=cond_path,
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                width=w,
                height=h,
                fps=f,
                num_frames=chunk_frames,
                seed=seed,
                strength=float(strength),
                output_path=chunk_path,
            )
            chunk_outputs.append(chunk_path)
            prev_source = chunk_path
            remaining -= chunk_frames
        if not chunk_outputs:
            raise HTTPException(status_code=500, detail="Failed to generate stitched chunks.")
        _concat_videos(chunk_outputs, output_path, fps=f, width=w, height=h)
    tasks = BackgroundTasks()
    tasks.add_task(_cleanup, temp_paths)
    return FileResponse(output_path, media_type="video/mp4", background=tasks)


def _get_latest_camera_state() -> tuple[np.ndarray | None, dict | None]:
    with latest_camera_lock:
        return latest_camera_frame, latest_mood


@app.post("/api/mood/frame")
async def ingest_mood_frame(request: Request) -> JSONResponse:
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="Empty frame data")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid JPEG data")
    mood = _estimate_mood(image)
    mood["prompt_hint"] = _mood_to_prompt(mood)
    with latest_camera_lock:
        global latest_camera_frame, latest_mood
        latest_camera_frame = image
        latest_mood = mood
    return JSONResponse(mood)


@app.get("/api/mood/prompt")
def mood_prompt() -> dict[str, str]:
    with latest_camera_lock:
        prompt = _mood_to_prompt(latest_mood)
    return {"prompt": prompt}


@app.get("/healthz")
def healthz(deep: bool = False) -> dict[str, Any]:
    if not deep:
        health_status.update(_gemma_status())
        if not health_status.get("gemma_ok", False):
            health_status["errors"]["gemma"] = health_status.get("gemma_reason")
        health_status["autostart"] = _env_bool("LTX2_AUTOSTART", False)
        health_status["inference_enabled"] = inference_enabled.is_set()
        return health_status
    status = {"pipeline_loaded": False, "pipelines": {}, "errors": {}}
    status.update(_gemma_status())
    status["autostart"] = _env_bool("LTX2_AUTOSTART", False)
    status["inference_enabled"] = inference_enabled.is_set()
    if not status.get("gemma_ok", False):
        status["errors"]["gemma"] = status.get("gemma_reason")
    if not status["inference_enabled"]:
        return status
    try:
        info = warmup_pipeline(current_config.output_mode)
        status["pipelines"][current_config.output_mode] = info
        status["pipeline_loaded"] = True
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Health check warmup failed for %s: %s", current_config.output_mode, exc)
        status["errors"][current_config.output_mode] = str(exc)
    return status


@app.get("/stream/{stream_id}.mjpg")
def stream(stream_id: int) -> StreamingResponse:
    try:
        stream_state = stream_manager.get_stream(stream_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown stream id") from exc

    boundary = "frame"

    def generator() -> bytes:
        interval = 1.0 / max(1, stream_state.stream_fps)
        next_frame_time = time.monotonic()
        while True:
            frame = stream_state.last_frame
            try:
                frame = stream_state.queue.get(timeout=interval * 0.5)
                stream_state.last_frame = frame
            except queue.Empty:
                frame = stream_state.last_frame
            if frame is None:
                placeholder = render_status_frame("Waiting for frames...", current_config.width, current_config.height)
                frame = _encode_frame(placeholder)
                stream_state.last_frame = frame
            yield (
                b"--" + boundary.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
            )
            next_frame_time += interval
            now = time.monotonic()
            if next_frame_time < now - interval:
                next_frame_time = now + interval
            sleep_for = next_frame_time - now
            if sleep_for > 0:
                time.sleep(sleep_for)

    return StreamingResponse(generator(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")


@app.get("/api/commercial/latest.mp4")
def commercial_latest(stream: int = 0) -> FileResponse:
    path, updated_at = get_latest_commercial_mp4(stream)
    if not path:
        raise HTTPException(status_code=404, detail="No commercial video available")
    try:
        if os.path.getsize(path) < 1024:
            raise HTTPException(status_code=404, detail="Commercial video not ready")
    except OSError:
        raise HTTPException(status_code=404, detail="Commercial video not ready")
    headers = {"Cache-Control": "no-cache", "X-Updated-At": str(updated_at)}
    return FileResponse(path, media_type="video/mp4", headers=headers)


@app.get("/api/commercial/status")
def commercial_status(stream: int = 0) -> dict[str, Any]:
    path, updated_at = get_latest_commercial_mp4(stream)
    done = is_commercial_done(stream_manager._cancel_event, stream)
    frames_done, frames_total = get_commercial_progress(stream_manager._cancel_event, stream)
    chunks_done, chunks_total = get_commercial_chunk_counts(stream_manager._cancel_event, stream)
    return {
        "available": path is not None,
        "updated_at": updated_at,
        "done": done,
        "frames_done": frames_done,
        "frames_total": frames_total,
        "chunks_done": chunks_done,
        "chunks_total": chunks_total,
    }


@app.post("/api/commercial/start")
def commercial_start(stream: int = 0) -> dict[str, Any]:
    reset_commercial_state(stream_manager._cancel_event, stream)
    return {"ok": True}


@app.post("/api/commercial/stop")
def commercial_stop(stream: int = 0) -> dict[str, Any]:
    stop_commercial(stream_manager._cancel_event, stream)
    return {"ok": True}


@app.get("/api/commercial/chunks")
def commercial_chunks(stream: int = 0) -> dict[str, Any]:
    if not is_commercial_done(stream_manager._cancel_event, stream):
        return {"chunks": []}
    paths = get_commercial_chunk_paths(stream_manager._cancel_event, stream)
    indices = [idx for idx, path in enumerate(paths) if path]
    return {"chunks": indices}


@app.get("/api/commercial/chunk/{chunk_idx}.mp4")
def commercial_chunk(chunk_idx: int, stream: int = 0) -> FileResponse:
    if not is_commercial_done(stream_manager._cancel_event, stream):
        raise HTTPException(status_code=404, detail="Commercial video not ready")
    paths = get_commercial_chunk_paths(stream_manager._cancel_event, stream)
    if chunk_idx < 0 or chunk_idx >= len(paths) or not paths[chunk_idx]:
        raise HTTPException(status_code=404, detail="Chunk not found")
    try:
        if os.path.getsize(paths[chunk_idx]) < 1024:
            raise HTTPException(status_code=404, detail="Chunk not ready")
    except OSError:
        raise HTTPException(status_code=404, detail="Chunk not ready")
    headers = {"Cache-Control": "no-cache"}
    return FileResponse(paths[chunk_idx], media_type="video/mp4", headers=headers)


@app.get("/audio/status")
def audio_status(stream: int = 0) -> dict[str, Any]:
    data, ts = get_latest_audio_wav(stream)
    return {"available": data is not None, "updated_at": ts}


@app.get("/audio/latest.wav")
def audio_latest(stream: int = 0) -> StreamingResponse:
    data, _ = get_latest_audio_wav(stream)
    if not data:
        raise HTTPException(status_code=404, detail="No audio available")
    return StreamingResponse(iter([data]), media_type="audio/wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuity Studio + Mood Mirror server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="[%(asctime)s] %(levelname)s - %(message)s")
    import uvicorn

    uvicorn.run("app:app", host=args.host, port=args.port, log_level=args.log_level)
