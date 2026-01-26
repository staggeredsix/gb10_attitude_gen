from __future__ import annotations

import argparse
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image

from ltx2_backend import (
    generate_fever_dream_frames,
    generate_mood_mirror_frames,
    log_backend_configuration,
    render_status_frame,
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


DEFAULT_WIDTH = _env_int("LTX2_NATIVE_WIDTH", 1280)
DEFAULT_HEIGHT = _env_int("LTX2_NATIVE_HEIGHT", 736)
DEFAULT_FPS = _env_int("LTX2_NATIVE_FPS", 24)
DEFAULT_OUTPUT_PRESET = _env_str("LTX2_OUTPUT_PRESET", "native")
if DEFAULT_OUTPUT_PRESET not in {"native", "spatial_x2", "temporal_x2", "spatial_x2_temporal_x2"}:
    LOGGER.warning("Invalid LTX2_OUTPUT_PRESET=%s; using native.", DEFAULT_OUTPUT_PRESET)
    DEFAULT_OUTPUT_PRESET = "native"


class RunConfig(BaseModel):
    mode: Literal["fever", "mood"] = "fever"
    prompt: str = Field("surreal dreamscape, liquid light, ethereal forms", min_length=1)
    negative_prompt: str = ""
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: int = DEFAULT_FPS
    streams: int = 2
    seed: int | None = None
    dream_strength: float = 0.7
    motion: float = 0.6
    base_prompt: str = Field("portrait, cinematic lighting", min_length=1)
    identity_strength: float = 0.7
    output_preset: Literal["native", "spatial_x2", "temporal_x2", "spatial_x2_temporal_x2"] = (
        DEFAULT_OUTPUT_PRESET
    )


@dataclass
class StreamState:
    queue: queue.Queue[bytes]
    thread: threading.Thread
    last_frame: bytes | None = None
    status: str = "starting"


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
            for stream_id in range(config.streams):
                stream_queue: queue.Queue[bytes] = queue.Queue(maxsize=2)
                stream = StreamState(queue=stream_queue, thread=threading.Thread())
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
        stream.last_frame = _encode_frame(render_status_frame(status_text, config.width, config.height))
        _queue_frame(stream.queue, stream.last_frame)
        try:
            if config.mode == "fever":
                frame_iter = generate_fever_dream_frames(config, cancel_event)
            else:
                frame_iter = generate_mood_mirror_frames(config, latest_camera_state, cancel_event)
            for frame in frame_iter:
                if cancel_event.is_set():
                    break
                stream.last_frame = _encode_frame(frame)
                _queue_frame(stream.queue, stream.last_frame)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Stream %s crashed: %s", stream_id, exc)
            error_frame = render_status_frame(f"Stream error: {exc}", config.width, config.height)
            stream.last_frame = _encode_frame(error_frame)
            _queue_frame(stream.queue, stream.last_frame)


def _queue_frame(q: queue.Queue[bytes], frame: bytes) -> None:
    try:
        if q.full():
            q.get_nowait()
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
    if config.width % 32 != 0 or config.height % 32 != 0:
        raise HTTPException(status_code=400, detail="Width and height must be multiples of 32.")
    config.streams = max(1, min(config.streams, 16))
    config.fps = max(1, min(config.fps, 60))
    config.dream_strength = float(np.clip(config.dream_strength, 0.0, 1.0))
    config.motion = float(np.clip(config.motion, 0.0, 1.0))
    config.identity_strength = float(np.clip(config.identity_strength, 0.0, 1.0))
    return config


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
latest_camera_frame: np.ndarray | None = None
latest_mood: dict[str, Any] | None = None
latest_camera_lock = threading.Lock()
health_status: dict[str, Any] = {"pipeline_loaded": False, "errors": {}, "pipelines": {}}


@app.on_event("startup")
def _startup() -> None:
    log_backend_configuration()
    try:
        info = warmup_pipeline("text")
        health_status["pipeline_loaded"] = True
        health_status["pipelines"]["text"] = info
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("LTX-2 warmup failed: %s", exc)
        health_status["errors"]["text"] = str(exc)
    stream_manager.restart(current_config, _get_latest_camera_state)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/config")
def get_config() -> RunConfig:
    return current_config


@app.post("/api/config")
async def set_config(request: Request) -> RunConfig:
    payload = await request.json()
    config = RunConfig(**payload)
    config = _validate_config(config)
    global current_config
    current_config = config
    stream_manager.restart(current_config, _get_latest_camera_state)
    return current_config


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
        return health_status
    status = {"pipeline_loaded": False, "pipelines": {}, "errors": {}}
    for mode in ("text", "image"):
        try:
            info = warmup_pipeline(mode)
            status["pipelines"][mode] = info
            status["pipeline_loaded"] = True
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Health check warmup failed for %s: %s", mode, exc)
            status["errors"][mode] = str(exc)
    return status


@app.get("/stream/{stream_id}.mjpg")
def stream(stream_id: int) -> StreamingResponse:
    try:
        stream_state = stream_manager.get_stream(stream_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown stream id") from exc

    boundary = "frame"

    def generator() -> bytes:
        while True:
            frame = stream_state.last_frame
            try:
                frame = stream_state.queue.get(timeout=1.0)
                stream_state.last_frame = frame
            except queue.Empty:
                if frame is None:
                    placeholder = render_status_frame("Waiting for frames...", current_config.width, current_config.height)
                    frame = _encode_frame(placeholder)
                    stream_state.last_frame = frame
            if frame is None:
                continue
            yield (
                b"--" + boundary.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
            )

    return StreamingResponse(generator(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LTX-2 Fever Dream + Mood Mirror server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="[%(asctime)s] %(levelname)s - %(message)s")
    import uvicorn

    uvicorn.run("app:app", host=args.host, port=args.port, log_level=args.log_level)
