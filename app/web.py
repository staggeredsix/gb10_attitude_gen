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
    mode: str = "single"
    last_latency_ms: Optional[float] = None


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
        self, frame: cv2.typing.MatLike, style_key: Optional[str], mode: str, state: SessionState
    ) -> tuple[Optional[str], Optional[np.ndarray], bool, Optional[float]]:
        """Process a frame and update session state as needed."""

        boxes = self.detector.detect(frame)
        emotion: Optional[str] = None
        face: Optional[np.ndarray] = None
        gen_latency_ms: Optional[float] = None

        state.mode = mode if mode in {"single", "dual"} else state.mode

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
            gen_start = time.time()
            generated = self.generator.generate(prompt, face)
            if generated is not None:
                state.generated_img = generated
                state.last_emotion = emotion
                state.last_gen_time = now
                state.has_pending_image = True
                gen_latency_ms = (time.time() - gen_start) * 1000
                state.last_latency_ms = gen_latency_ms

        return emotion, state.generated_img, state.has_pending_image, gen_latency_ms or state.last_latency_ms


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


def _build_html(default_mode: str) -> str:
    options = "\n".join(
        f'<option value="{key}">{key.title()}</option>' for key in STYLE_MAP.keys()
    )
    default_mode_label = "Dual node" if default_mode == "dual" else "Single node"
    template = """
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8" />
        <title>AI Mood Mirror</title>
        <style>
            :root {
                --bg: #f6f7fb;
                --card: #ffffff;
                --border: #d7dce7;
                --text: #111827;
                --muted: #4b5563;
                --accent: #2563eb;
                --danger: #b91c1c;
                --shadow: 0 16px 40px rgba(17, 24, 39, 0.08);
                --radius: 14px;
            }

            * { box-sizing: border-box; }

            body {
                margin: 0;
                min-height: 100vh;
                font-family: 'Inter', system-ui, -apple-system, sans-serif;
                background: var(--bg);
                color: var(--text);
            }

            header {
                padding: 32px 28px 8px;
            }

            h1 {
                margin: 0 0 8px;
                font-size: 32px;
                letter-spacing: -0.01em;
            }

            .subhead {
                margin: 0;
                color: var(--muted);
                max-width: 840px;
                line-height: 1.5;
            }

            main {
                padding: 0 28px 40px;
            }

            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
                gap: 20px;
                align-items: start;
            }

            .card {
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                padding: 18px 18px 20px;
                box-shadow: var(--shadow);
            }

            .card h3 {
                margin: 4px 0 10px;
                font-size: 18px;
            }

            .card p {
                margin: 0 0 12px;
                color: var(--muted);
                line-height: 1.5;
            }

            .controls {
                display: flex;
                gap: 12px;
                flex-wrap: wrap;
                align-items: center;
                margin: 12px 0;
            }

            .field {
                display: flex;
                flex-direction: column;
                gap: 6px;
                min-width: 180px;
            }

            label {
                font-weight: 600;
                color: var(--text);
            }

            select, button, input[type="file"] {
                font: inherit;
            }

            select {
                padding: 10px 12px;
                border-radius: 10px;
                border: 1px solid var(--border);
                background: #f9fafb;
            }

            .upload-area {
                border: 1px dashed var(--border);
                border-radius: 12px;
                padding: 14px;
                background: #fafbff;
            }

            .upload-area input {
                display: block;
                width: 100%;
            }

            button {
                background: var(--accent);
                color: #fff;
                border: 1px solid var(--accent);
                border-radius: 10px;
                padding: 11px 16px;
                font-weight: 700;
                cursor: pointer;
                transition: filter 120ms ease;
            }

            button:hover {
                filter: brightness(0.95);
            }

            button:disabled {
                background: #e5e7eb;
                border-color: #e5e7eb;
                color: var(--muted);
                cursor: not-allowed;
            }

            .switch {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                cursor: pointer;
                user-select: none;
            }

            .switch input {
                width: 44px;
                height: 24px;
                appearance: none;
                background: #e5e7eb;
                border-radius: 999px;
                position: relative;
                outline: none;
                transition: background 140ms ease;
            }

            .switch input::after {
                content: '';
                position: absolute;
                top: 3px;
                left: 3px;
                width: 18px;
                height: 18px;
                border-radius: 50%;
                background: #fff;
                box-shadow: 0 1px 3px rgba(0,0,0,0.15);
                transition: transform 140ms ease;
            }

            .switch input:checked {
                background: var(--accent);
            }

            .switch input:checked::after {
                transform: translateX(20px);
            }

            .preview {
                width: 100%;
                max-width: 560px;
                border-radius: 12px;
                border: 1px solid var(--border);
                background: #f9fafb;
                min-height: 280px;
                object-fit: contain;
            }

            .status {
                margin-top: 10px;
                display: inline-flex;
                align-items: center;
                gap: 10px;
                padding: 10px 14px;
                border-radius: 999px;
                background: #eef2ff;
                border: 1px solid var(--border);
                color: var(--text);
                font-weight: 600;
            }

            .status-dot {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: var(--accent);
                box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.15);
            }

            .metrics {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 12px;
                margin-top: 14px;
            }

            .metric {
                padding: 12px;
                border: 1px solid var(--border);
                border-radius: 12px;
                background: #f9fafb;
            }

            .metric .label {
                font-size: 12px;
                letter-spacing: 0.04em;
                color: var(--muted);
                text-transform: uppercase;
                margin-bottom: 6px;
            }

            .metric .value {
                font-weight: 700;
                font-size: 18px;
            }
        </style>
    </head>
    <body>
        <header>
            <h1>AI Mood Mirror</h1>
            <p class="subhead">Stream your webcam or upload a portrait. We crop the face, infer the mood, and send it to the diffusion model to generate a live portrait.</p>
        </header>

        <main>
            <div class="grid">
                <section class="card">
                    <h3>Input</h3>
                    <p>Toggle the webcam for live streaming or upload a still image.</p>
                    <label class="switch"><input type="checkbox" id="use-webcam" /> Enable webcam</label>
                    <div class="upload-area" id="upload-area">
                        <label for="photo">Choose an image file</label>
                        <input type="file" id="photo" accept="image/*" />
                    </div>
                    <div class="controls">
                        <div class="field">
                            <label for="style">Style template</label>
                            <select id="style">{options}</select>
                        </div>
                        <div class="field">
                            <label for="mode">Inference mode</label>
                            <select id="mode">
                                <option value="single">Single node</option>
                                <option value="dual">Dual node</option>
                            </select>
                        </div>
                        <button id="send" disabled>Send to AI</button>
                    </div>
                    <div class="status" id="status">
                        <span class="status-dot"></span>
                        <span>Connecting...</span>
                    </div>
                </section>

                <section class="card">
                    <h3>Live preview</h3>
                    <p>Webcam frames or uploaded images are sent to the model. Generated results update below.</p>
                    <video id="webcam" class="preview" autoplay playsinline muted style="display:none;"></video>
                    <img id="preview" class="preview" alt="Preview frame" />
                    <img id="portrait" class="preview" style="margin-top:12px;" alt="Generated portrait" />
                    <div class="metrics">
                        <div class="metric">
                            <div class="label">Active mode</div>
                            <div class="value" id="mode-label">{default_mode_label}</div>
                        </div>
                        <div class="metric">
                            <div class="label">Generation latency</div>
                            <div class="value" id="latency-label">--</div>
                        </div>
                    </div>
                </section>
            </div>
        </main>

        <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
        <script>
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            const statusEl = document.getElementById('status');
            const statusText = statusEl.querySelector('span:last-child');
            const portrait = document.getElementById('portrait');
            const preview = document.getElementById('preview');
            const webcamEl = document.getElementById('webcam');
            const styleSelect = document.getElementById('style');
            const photoInput = document.getElementById('photo');
            const sendBtn = document.getElementById('send');
            const webcamToggle = document.getElementById('use-webcam');
            const uploadArea = document.getElementById('upload-area');
            const modeSelect = document.getElementById('mode');
            const modeLabel = document.getElementById('mode-label');
            const latencyLabel = document.getElementById('latency-label');

            const defaultMode = '{default_mode_value}';
            modeSelect.value = defaultMode;
            let inferenceMode = defaultMode;

            let socket;
            let ready = false;
            let pendingDataUrl = null;
            let webcamStream = null;
            let frameTimer = null;

            function setStatus(message, connected = ready) {
                statusText.textContent = message;
                statusEl.querySelector('.status-dot').style.background = connected ? 'var(--accent)' : 'var(--danger)';
            }

            function updateSendAvailability() {
                sendBtn.disabled = !(ready && pendingDataUrl && !webcamToggle.checked);
            }

            function captureFrame(sourceEl) {
                const maxWidth = 640;
                const maxHeight = 480;
                const width = sourceEl.videoWidth || sourceEl.naturalWidth;
                const height = sourceEl.videoHeight || sourceEl.naturalHeight;
                if (!width || !height) return null;
                const scale = Math.min(maxWidth / width, maxHeight / height, 1);
                canvas.width = width * scale;
                canvas.height = height * scale;
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(sourceEl, 0, 0, canvas.width, canvas.height);
                return canvas.toDataURL('image/jpeg', 0.82);
            }

            function sendFrame(dataUrl) {
                if (!ready) {
                    setStatus('Connecting to server, please wait...', false);
                    return;
                }
                if (!dataUrl) {
                    setStatus('No frame available to send', false);
                    return;
                }
                socket.send(JSON.stringify({ frame: dataUrl, style: styleSelect.value, mode: inferenceMode }));
            }

            async function startWebcam() {
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    console.error('Browser does not support getUserMedia or page is not served over a secure context');
                    setStatus('Webcam is not available in this browser or context', false);
                    webcamToggle.checked = false;
                    return;
                }
                try {
                    webcamStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                    webcamEl.srcObject = webcamStream;
                    webcamEl.style.display = 'block';
                    preview.style.display = 'none';
                    await webcamEl.play();
                    setStatus('Streaming from webcam');
                    frameTimer = setInterval(() => {
                        const dataUrl = captureFrame(webcamEl);
                        if (dataUrl) {
                            portrait.dataset.lastSource = 'webcam';
                            sendFrame(dataUrl);
                        }
                    }, 600);
                } catch (error) {
                    console.error(error);
                    setStatus('Unable to access webcam', false);
                    webcamToggle.checked = false;
                }
            }

            function stopWebcam() {
                if (frameTimer) {
                    clearInterval(frameTimer);
                    frameTimer = null;
                }
                if (webcamStream) {
                    webcamStream.getTracks().forEach((track) => track.stop());
                    webcamStream = null;
                }
                webcamEl.pause();
                webcamEl.srcObject = null;
                webcamEl.style.display = 'none';
                preview.style.display = 'block';
                setStatus('Webcam stopped');
            }

            function openSocket() {
                socket = new WebSocket(`ws://${location.host}/ws`);

                socket.onopen = () => {
                    ready = true;
                    setStatus('Connected. Choose a source to begin.');
                    updateSendAvailability();
                };

                socket.onclose = () => {
                    ready = false;
                    sendBtn.disabled = true;
                    setStatus('Disconnected. Reconnecting...', false);
                    setTimeout(openSocket, 1000);
                };

                socket.onerror = (err) => {
                    console.error('WebSocket error', err);
                };

                socket.onmessage = (event) => {
                    const payload = JSON.parse(event.data);
                    if (payload.emotion) {
                        setStatus(`Emotion: ${payload.emotion}`);
                    } else {
                        setStatus('No face detected');
                    }
                    if (payload.mode) {
                        inferenceMode = payload.mode;
                        modeSelect.value = payload.mode;
                        modeLabel.textContent = payload.mode === 'dual' ? 'Dual node' : 'Single node';
                    }
                    if (payload.latency_ms !== undefined) {
                        latencyLabel.textContent = `${payload.latency_ms.toFixed(1)} ms`;
                    }
                    if (payload.generated_image) {
                        portrait.src = `data:image/jpeg;base64,${payload.generated_image}`;
                    }
                };
            }

            photoInput.addEventListener('change', (event) => {
                const file = event.target.files[0];
                if (!file) return;
                webcamToggle.checked = false;
                stopWebcam();
                const reader = new FileReader();
                reader.onload = () => {
                    const img = new Image();
                    img.onload = () => {
                        pendingDataUrl = captureFrame(img);
                        preview.src = pendingDataUrl;
                        setStatus('Image ready. Click "Send to AI".');
                        updateSendAvailability();
                    };
                    img.src = reader.result;
                };
                reader.readAsDataURL(file);
            });

            sendBtn.addEventListener('click', () => {
                setStatus('Processing...');
                sendFrame(pendingDataUrl);
            });

            webcamToggle.addEventListener('change', (event) => {
                if (event.target.checked) {
                    pendingDataUrl = null;
                    updateSendAvailability();
                    startWebcam();
                    uploadArea.style.opacity = 0.6;
                } else {
                    stopWebcam();
                    uploadArea.style.opacity = 1;
                    setStatus('Webcam disabled. Upload an image or re-enable.');
                    updateSendAvailability();
                }
            });

            modeSelect.addEventListener('change', (event) => {
                inferenceMode = event.target.value;
                modeLabel.textContent = inferenceMode === 'dual' ? 'Dual node' : 'Single node';
            });

            window.addEventListener('beforeunload', stopWebcam);
            openSocket();
        </script>
    </body>
    </html>
    """
    return (
        template.replace("{options}", options)
        .replace("{default_mode_value}", default_mode)
        .replace("{default_mode_label}", default_mode_label)
    )




def create_app(config: AppConfig) -> FastAPI:
    """Create the FastAPI application with configured routes."""

    app = FastAPI()
    pipeline = InferencePipeline(config)

    @app.get("/")
    async def index() -> HTMLResponse:  # noqa: D401
        """Serve the basic HTML UI."""

        return HTMLResponse(content=_build_html(config.default_mode), media_type="text/html")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        state = SessionState(mode=config.default_mode)
        LOGGER.info("Client connected")
        try:
            while True:
                data = await websocket.receive_json()
                frame_data = data.get("frame")
                style_key = data.get("style")
                mode = data.get("mode") or state.mode
                if not frame_data:
                    continue

                frame = _decode_frame(frame_data)
                if frame is None:
                    await websocket.send_json({"error": "invalid_frame"})
                    continue

                emotion, generated, has_pending, latency_ms = pipeline.process(frame, style_key, mode, state)
                response: dict[str, Optional[str]] = {
                    "emotion": emotion,
                    "style": style_key,
                    "mode": state.mode,
                }
                if generated is not None and has_pending:
                    encoded = _encode_image_b64(generated)
                    if encoded:
                        response["generated_image"] = encoded
                        state.has_pending_image = False
                if latency_ms is not None:
                    response["latency_ms"] = latency_ms
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
