# AI Mood Mirror

A simple demo that captures webcam video, detects faces, classifies emotions, and generates a matching portrait using SDXL Turbo.

## Features

- Real-time face detection with MediaPipe
- Emotion classification from face crops (default: `trpakov/vit-face-expression`)
- Prompt construction mapped from dominant emotion
- SDXL Turbo image generation (GPU optional, CPU fallback)
- OpenCV windows for webcam and generated portrait
- Optional browser UI that streams webcam frames and returns generated portraits with style templates

## Installation (local)

1. Ensure Python 3.10+ is available.
2. Install system dependencies for OpenCV (e.g., `libgl1`, `libglib2.0-0` on Debian/Ubuntu).
3. Install the package:

```bash
pip install .
```

## CLI usage

Run the application:

```bash
ai-mood-mirror
```

Run the browser UI server (streams your webcam from the browser and pushes generated portraits back):

```bash
ai-mood-mirror-web --port 8000
# then open http://localhost:8000 in your browser
```

Useful flags:

- `--camera-index`: Webcam index (default `0`).
- `--emotion-model`: Hugging Face model id for emotion detection.
- `--diffusion-model`: Diffusion model id (default `stabilityai/sdxl-turbo`).
- `--detection-confidence`: Minimum confidence for face detection (default `0.5`).
- `--generation-interval`: Seconds between portrait generations (default `3.0`).
- `--use-cuda` / `--no-cuda`: Force enable/disable CUDA.
- `--no-ui`: Run headless without OpenCV windows.
- `--host` / `--port`: Host/port for the web UI server (defaults `0.0.0.0:8000`).

Environment variables (fallbacks for the flags above) use the `AI_MOOD_MIRROR_` prefix, e.g. `AI_MOOD_MIRROR_CAMERA_INDEX`, `AI_MOOD_MIRROR_EMOTION_MODEL`, and `AI_MOOD_MIRROR_USE_CUDA`.

Press `q` in the webcam window to exit.

## Docker

Build the image (requires NVIDIA GPU drivers):

```bash
docker build -t ai-mood-mirror .
```

Run with compose (shares X11 display and `/dev/video0`, exposes the web UI on `0.0.0.0:8000`):

```bash
docker compose up
```

Or run directly:

```bash
xhost +local:root  # allow X11 from container

docker run --rm -it \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --device /dev/video0 \
  -p 8000:8000 \
  ai-mood-mirror
```

To run the web UI from the container, replace the command with `ai-mood-mirror-web --port 8000` and open `http://localhost:8000` in your browser.

## Notes

- GPU acceleration is used when available and enabled; otherwise, models run on CPU.
- The app throttles image generation to avoid excessive GPU load and regenerates when the detected emotion changes.
- If no face is detected, a visible overlay is shown and portraits are not refreshed.
