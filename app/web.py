"""Web UI server for the AI Mood Mirror pipeline."""
from __future__ import annotations

import base64
import logging
import sys
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from .config import AppConfig, load_config, parse_args
from .emotion_classifier import EmotionClassifier
from .face_detection import FaceDetector
from .image_generator import ImageGenerator
from .prompt_builder import STYLE_MAP, build_prompt

LOGGER = logging.getLogger(__name__)


@dataclass
class SessionState:
    """Track generation state for a websocket client."""

    last_emotion: Optional[str] = None
    last_gen_time: float = 0.0
    generated_img: Optional[np.ndarray] = None
    has_pending_image: bool = False


class InferencePipeline:
    """Run detection, classification, and generation for incoming frames."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.detector = FaceDetector(min_confidence=config.detection_confidence)
        self.classifier = EmotionClassifier(config.emotion_model, config.device)
        self.generator = ImageGenerator(config.diffusion_model, config.device)

    @staticmethod
    def _extract_face(frame: cv2.typing.MatLike, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None
        return face_crop

    def process(
        self, frame: cv2.typing.MatLike, style_key: Optional[str], state: SessionState
    ) -> tuple[Optional[str], Optional[np.ndarray], bool]:
        """Process a frame and update session state as needed."""

        boxes = self.detector.detect(frame)
        emotion: Optional[str] = None

        if boxes:
            box = boxes[0]
            face = self._extract_face(frame, box.x1, box.y1, box.x2, box.y2)
            if face is not None:
                emotion = self.classifier.classify(face)
        else:
            LOGGER.debug("No face detected in incoming frame")

        now = time.time()
        should_generate = (
            emotion
            and (
                emotion != state.last_emotion
                or now - state.last_gen_time > self.config.generation_interval
            )
        )

        if should_generate:
            prompt = build_prompt(emotion, style_key)
            generated = self.generator.generate(prompt)
            if generated is not None:
                state.generated_img = generated
                state.last_emotion = emotion
                state.last_gen_time = now
                state.has_pending_image = True

        return emotion, state.generated_img, state.has_pending_image


def _decode_frame(data_url: str) -> Optional[np.ndarray]:
    try:
        encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
        data = base64.b64decode(encoded)
        np_data = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to decode frame: %s", exc)
        return None


def _encode_image_b64(image: np.ndarray) -> Optional[str]:
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        return None
    return base64.b64encode(buffer).decode("ascii")


def _build_html() -> str:
    options = "\n".join(
        f'<option value="{key}">{key.title()}</option>' for key in STYLE_MAP.keys()
    )
    return f"""
    <!doctype html>
    <html lang=\"en\">
    <head>
        <meta charset=\"utf-8\" />
        <title>AI Mood Mirror - Web</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .row {{ display: flex; gap: 20px; align-items: center; flex-wrap: wrap; }}
            video, img {{ border: 1px solid #ccc; max-width: 420px; height: auto; }}
            #status {{ margin-top: 10px; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h2>AI Mood Mirror</h2>
        <div class=\"row\">
            <div>
                <p>Webcam feed</p>
                <video id=\"video\" autoplay playsinline width=\"420\" height=\"315\"></video>
                <div>
                    <label for=\"style\">Style template:</label>
                    <select id=\"style\">{options}</select>
                </div>
                <p id=\"status\">Connecting...</p>
            </div>
            <div>
                <p>AI Mood Portrait</p>
                <img id=\"portrait\" width=\"420\" height=\"315\" alt=\"Generated portrait will appear here\" />
            </div>
        </div>
        <canvas id=\"canvas\" width=\"320\" height=\"240\" style=\"display:none;\"></canvas>
        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            const statusEl = document.getElementById('status');
            const portrait = document.getElementById('portrait');
            const styleSelect = document.getElementById('style');

            async function startCamera() {{
                try {{
                    const stream = await navigator.mediaDevices.getUserMedia({{ video: true, audio: false }});
                    video.srcObject = stream;
                }} catch (err) {{
                    statusEl.textContent = `Camera error: ${err}`;
                    throw err;
                }}
            }}

            function openSocket() {{
                const socket = new WebSocket(`ws://${{location.host}}/ws`);
                let ready = false;

                socket.onopen = () => {{
                    statusEl.textContent = 'Connected';
                    ready = true;
                }};

                socket.onclose = () => {{
                    ready = false;
                    statusEl.textContent = 'Disconnected';
                    setTimeout(openSocket, 1000);
                }};

                socket.onerror = (err) => {{
                    console.error('WebSocket error', err);
                }};

                socket.onmessage = (event) => {{
                    const payload = JSON.parse(event.data);
                    if (payload.emotion) {{
                        statusEl.textContent = `Emotion: ${payload.emotion}`;
                    }} else {{
                        statusEl.textContent = 'No face detected';
                    }}
                    if (payload.generated_image) {{
                        portrait.src = `data:image/jpeg;base64,${payload.generated_image}`;
                    }}
                }};

                function pushFrame() {{
                    if (!ready) {{
                        return;
                    }}
                    try {{
                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                        const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
                        socket.send(JSON.stringify({{ frame: dataUrl, style: styleSelect.value }}));
                    }} catch (err) {{
                        console.error('Frame send failed', err);
                    }}
                }}

                setInterval(pushFrame, 500);
            }}

            startCamera().then(openSocket).catch(() => {{}});
        </script>
    </body>
    </html>
    """


def create_app(config: AppConfig) -> FastAPI:
    """Create the FastAPI application with configured routes."""

    app = FastAPI()
    pipeline = InferencePipeline(config)

    @app.get("/")
    async def index() -> HTMLResponse:  # noqa: D401
        """Serve the basic HTML UI."""

        return HTMLResponse(content=_build_html(), media_type="text/html")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        state = SessionState()
        LOGGER.info("Client connected")
        try:
            while True:
                data = await websocket.receive_json()
                frame_data = data.get("frame")
                style_key = data.get("style")
                if not frame_data:
                    continue

                frame = _decode_frame(frame_data)
                if frame is None:
                    await websocket.send_json({"error": "invalid_frame"})
                    continue

                emotion, generated, has_pending = pipeline.process(frame, style_key, state)
                response: dict[str, Optional[str]] = {
                    "emotion": emotion,
                    "style": style_key,
                }
                if generated is not None and has_pending:
                    encoded = _encode_image_b64(generated)
                    if encoded:
                        response["generated_image"] = encoded
                        state.has_pending_image = False
                await websocket.send_json(response)
        except WebSocketDisconnect:
            LOGGER.info("Client disconnected")
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("WebSocket error: %s", exc)
            await websocket.close()

    return app


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    args = parse_args(argv)
    config = load_config(args)
    app = create_app(config)
    LOGGER.info("Starting web server on %s:%s", config.server_host, config.server_port)
    uvicorn.run(app, host=config.server_host, port=config.server_port)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
