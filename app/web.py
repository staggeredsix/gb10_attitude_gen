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
            :root {{
                --nvidia-green: #76b900;
                --nvidia-dark: #0a0f0d;
                --nvidia-gray: #1a1f1d;
                --text-primary: #e8f3e8;
                --text-muted: #9fb09e;
                --card-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
                --border-color: rgba(118, 185, 0, 0.35);
            }}

            * {{ box-sizing: border-box; }}

            body {{
                margin: 0;
                min-height: 100vh;
                font-family: 'Inter', system-ui, -apple-system, sans-serif;
                background: radial-gradient(circle at 20% 20%, rgba(118, 185, 0, 0.1), transparent 25%),
                            radial-gradient(circle at 80% 0%, rgba(118, 185, 0, 0.12), transparent 20%),
                            linear-gradient(135deg, #0b1510, #040806 55%, #0a0f0d 100%);
                color: var(--text-primary);
            }}

            header {{
                padding: 32px 28px 8px;
            }}

            .eyebrow {{
                display: inline-flex;
                gap: 8px;
                align-items: center;
                padding: 8px 12px;
                background: rgba(118, 185, 0, 0.16);
                border: 1px solid var(--border-color);
                border-radius: 999px;
                color: var(--text-primary);
                letter-spacing: 0.03em;
                text-transform: uppercase;
                font-weight: 600;
                font-size: 12px;
            }}

            h1 {{
                margin: 12px 0 6px;
                font-size: 32px;
                letter-spacing: -0.02em;
            }}

            .subhead {{
                margin: 0 0 10px;
                color: var(--text-muted);
                max-width: 720px;
                line-height: 1.5;
            }}

            main {{
                padding: 0 28px 32px;
            }}

            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                gap: 20px;
                align-items: start;
            }}

            .card {{
                background: linear-gradient(145deg, rgba(26, 31, 29, 0.75), rgba(12, 16, 14, 0.8));
                border: 1px solid rgba(255, 255, 255, 0.04);
                border-radius: 16px;
                padding: 18px 18px 20px;
                box-shadow: var(--card-shadow);
                backdrop-filter: blur(10px);
            }}

            .card h3 {{
                margin: 4px 0 6px;
                font-size: 18px;
                letter-spacing: 0.01em;
            }}

            .card p {{
                margin: 0 0 12px;
                color: var(--text-muted);
                line-height: 1.4;
            }}

            .upload {{
                border: 1px dashed var(--border-color);
                border-radius: 14px;
                padding: 14px;
                background: rgba(118, 185, 0, 0.04);
            }}

            .upload label {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 10px 14px;
                background: var(--nvidia-green);
                color: #0b0f0c;
                border-radius: 12px;
                font-weight: 700;
                cursor: pointer;
                transition: transform 0.1s ease, box-shadow 0.1s ease;
                box-shadow: 0 10px 20px rgba(118, 185, 0, 0.35);
            }}

            .upload label:hover {{
                transform: translateY(-1px);
            }}

            .upload input {{
                display: none;
            }}

            .meta-row {{
                display: flex;
                gap: 12px;
                flex-wrap: wrap;
                align-items: center;
                margin: 12px 0;
            }}

            select {{
                background: var(--nvidia-gray);
                color: var(--text-primary);
                border: 1px solid var(--border-color);
                border-radius: 10px;
                padding: 10px 12px;
                font-weight: 600;
                min-width: 170px;
            }}

            button {{
                background: linear-gradient(135deg, var(--nvidia-green), #95e000);
                border: none;
                color: #0b0f0c;
                border-radius: 12px;
                padding: 11px 16px;
                font-weight: 800;
                letter-spacing: 0.01em;
                cursor: pointer;
                min-width: 160px;
                box-shadow: 0 12px 24px rgba(118, 185, 0, 0.35);
                transition: transform 0.12s ease, box-shadow 0.12s ease;
            }}

            button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 16px 30px rgba(118, 185, 0, 0.45);
            }}

            button:disabled {{
                background: #4a6142;
                color: #d7e5d6;
                cursor: not-allowed;
                box-shadow: none;
                transform: none;
            }}

            .preview {{
                width: 100%;
                max-width: 500px;
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.04);
                background: #070a07;
                box-shadow: inset 0 0 0 1px rgba(118, 185, 0, 0.08);
                min-height: 280px;
                object-fit: contain;
            }}

            .status {{
                margin-top: 8px;
                display: inline-flex;
                align-items: center;
                gap: 10px;
                padding: 10px 14px;
                border-radius: 999px;
                background: rgba(118, 185, 0, 0.12);
                border: 1px solid var(--border-color);
                color: var(--text-primary);
                font-weight: 600;
                box-shadow: 0 10px 18px rgba(0, 0, 0, 0.25);
            }}

            .status-dot {{
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: var(--nvidia-green);
                box-shadow: 0 0 12px rgba(118, 185, 0, 0.8);
            }}

            footer {{
                color: var(--text-muted);
                text-align: center;
                padding: 16px;
                font-size: 13px;
            }}
        </style>
    </head>
    <body>
        <header>
            <div class=\"eyebrow\">NVIDIA Inspired</div>
            <h1>AI Mood Mirror</h1>
            <p class=\"subhead\">Upload a portrait to detect the dominant emotion and generate an NVIDIA-inspired artwork. No webcam access is requested.</p>
        </header>

        <main>
            <div class=\"grid\">
                <section class=\"card\">
                    <h3>1 · Upload a face photo</h3>
                    <p>Select a recent portrait. The image stays on your machine until you click "Send to AI".</p>
                    <div class=\"upload\">
                        <label for=\"photo\">📤 Choose image</label>
                        <input type=\"file\" id=\"photo\" accept=\"image/*\" />
                        <div class=\"meta-row\">
                            <div>
                                <label for=\"style\">Style template</label><br />
                                <select id=\"style\">{options}</select>
                            </div>
                            <button id=\"send\" disabled>Send to AI</button>
                        </div>
                    </div>
                    <div class=\"status\" id=\"status\">
                        <span class=\"status-dot\"></span>
                        <span>Connecting...</span>
                    </div>
                </section>

                <section class=\"card\">
                    <h3>2 · Preview & Results</h3>
                    <p>Review your uploaded image and see the generated NVIDIA-flavored portrait.</p>
                    <img id=\"preview\" class=\"preview\" alt=\"Your selected image preview\" />
                    <img id=\"portrait\" class=\"preview\" style=\"margin-top:12px;\" alt=\"Generated portrait will appear here\" />
                </section>
            </div>
        </main>

        <footer>
            Built for private, on-device exploration — we never request webcam access.
        </footer>

        <canvas id=\"canvas\" width=\"360\" height=\"270\" style=\"display:none;\"></canvas>
        <script>
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            const statusEl = document.getElementById('status');
            const statusText = statusEl.querySelector('span:last-child');
            const portrait = document.getElementById('portrait');
            const preview = document.getElementById('preview');
            const styleSelect = document.getElementById('style');
            const photoInput = document.getElementById('photo');
            const sendBtn = document.getElementById('send');

            let socket;
            let ready = false;
            let pendingDataUrl = null;

            function setStatus(message, connected = ready) {{
                statusText.textContent = message;
                statusEl.querySelector('.status-dot').style.background = connected ? 'var(--nvidia-green)' : '#b94c00';
            }}

            async function downscaleImage(dataUrl) {{
                return new Promise((resolve, reject) => {{
                    const img = new Image();
                    img.onload = () => {{
                        const maxWidth = 360;
                        const maxHeight = 270;
                        const scale = Math.min(maxWidth / img.width, maxHeight / img.height, 1);
                        canvas.width = img.width * scale;
                        canvas.height = img.height * scale;
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                        resolve(canvas.toDataURL('image/jpeg', 0.82));
                    }};
                    img.onerror = reject;
                    img.src = dataUrl;
                }});
            }}

            function openSocket() {{
                socket = new WebSocket(`ws://${{location.host}}/ws`);

                socket.onopen = () => {{
                    ready = true;
                    sendBtn.disabled = !pendingDataUrl;
                    setStatus('Connected. Upload an image to begin.');
                }};

                socket.onclose = () => {{
                    ready = false;
                    sendBtn.disabled = true;
                    setStatus('Disconnected. Reconnecting...', false);
                    setTimeout(openSocket, 1000);
                }};

                socket.onerror = (err) => {{
                    console.error('WebSocket error', err);
                }};

                socket.onmessage = (event) => {{
                    const payload = JSON.parse(event.data);
                    if (payload.emotion) {{
                        setStatus(`Emotion: ${{payload.emotion}}`);
                    }} else {{
                        setStatus('No face detected');
                    }}
                    if (payload.generated_image) {{
                        portrait.src = `data:image/jpeg;base64,${{payload.generated_image}}`;
                    }}
                }};
            }}

            photoInput.addEventListener('change', async (event) => {{
                const file = event.target.files[0];
                if (!file) return;

                const reader = new FileReader();
                reader.onload = async () => {{
                    pendingDataUrl = await downscaleImage(reader.result);
                    preview.src = pendingDataUrl;
                    setStatus('Image ready. Click "Send to AI".');
                    sendBtn.disabled = !ready;
                }};
                reader.readAsDataURL(file);
            }});

            sendBtn.addEventListener('click', () => {{
                if (!ready) {{
                    setStatus('Connecting to server, please wait...', false);
                    return;
                }}
                if (!pendingDataUrl) {{
                    setStatus('Upload an image first.');
                    return;
                }}

                setStatus('Processing...');
                socket.send(JSON.stringify({{ frame: pendingDataUrl, style: styleSelect.value }}));
            }});

            openSocket();
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
