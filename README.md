# AI Mood Mirror

A simple demo that captures webcam video, detects faces, classifies emotions, and generates a matching portrait using a Stable Diffusion + ControlNet pipeline.

## Features

- Real-time face detection with MediaPipe
- Emotion classification from face crops (default: `trpakov/vit-face-expression`)
- Prompt construction mapped from dominant emotion
- Diffusion image generation conditioned on your webcam frame (GPU optional, CPU fallback)
- OpenCV windows for webcam and generated portrait
- Optional browser UI that streams webcam frames and returns generated portraits with style templates

## Installation (local)

1. Ensure Python 3.10+ is available.
2. Install system dependencies for OpenCV (e.g., `libgl1`, `libglib2.0-0` on Debian/Ubuntu).
3. Install a CUDA 13 build of PyTorch (required for GB10 GPUs), including torchvision for preprocessing utilities:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

4. Install the package:

```bash
pip install .
```

## CLI usage

Run the browser UI server (streams your webcam **from the browser**, so the backend never needs direct webcam access). The server
binds HTTPS with an auto-generated self-signed certificate by default—click through the browser warning:

```bash
ai-mood-mirror-web --port 8000
# then open https://localhost:8000 in your browser
```

If you want the legacy local-camera mode with OpenCV windows, run:

```bash
ai-mood-mirror
```

Useful flags:

- `--camera-index`: Webcam index (default `0`, used only by the legacy OpenCV mode).
- `--emotion-model`: Hugging Face model id for emotion detection.
- `--diffusion-model`: Diffusion model id (default `runwayml/stable-diffusion-v1-5`).
- `--controlnet-model`: ControlNet id to condition on the webcam frame (default `lllyasviel/sd-controlnet-canny`).
- `--detection-confidence`: Minimum confidence for face detection (default `0.5`).
- `--generation-interval`: Seconds between portrait generations (default `3.0`).
- `--use-cuda` / `--no-cuda`: Force enable/disable CUDA.
- `--no-ui`: Run headless without OpenCV windows.
- `--host` / `--port`: Host/port for the web UI server (defaults `0.0.0.0:8000`).
- `--https` / `--no-https`: Enable or disable HTTPS for the web UI (HTTPS is on by default and will generate a self-signed cert
  if missing).
- `--ssl-certfile` / `--ssl-keyfile`: Provide your own certificate and key instead of the generated self-signed pair.

Environment variables (fallbacks for the flags above) use the `AI_MOOD_MIRROR_` prefix, e.g. `AI_MOOD_MIRROR_CAMERA_INDEX`, `AI_MOOD_MIRROR_EMOTION_MODEL`, and `AI_MOOD_MIRROR_USE_CUDA`.

Press `q` in the webcam window to exit.

## Docker

Build the image (requires NVIDIA GPU drivers). The Dockerfile uses the CUDA 13 runtime and installs PyTorch from the `cu130` channel for GB10 compatibility:

```bash
docker build -t ai-mood-mirror .
```

Run with compose (exposes the web UI on `0.0.0.0:8000` and lets the browser access your webcam):

```bash
docker compose up
```

Or run directly (backend does **not** need webcam access—your browser provides the video stream):

```bash
docker run --rm -it \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -p 8000:8000 \
  ai-mood-mirror-web --host 0.0.0.0 --port 8000
```

### Deployment scripts

- **Single node:** `./scripts/single_spark.sh` builds the image locally and launches the web UI on the current machine (defaults to port `8000`).
- **Dual node:** `VISION_HOST=<dgx1> DIFFUSION_HOST=<dgx2> ./scripts/dual_spark.sh` configures the dual 100G ports, syncs the repo to each host, builds the container, and runs a quick connectivity/GPU sanity check.

## Notes

- GPU acceleration is used when available and enabled; otherwise, models run on CPU.
- The app throttles image generation to avoid excessive GPU load and regenerates when the detected emotion changes.
- If no face is detected, a visible overlay is shown and portraits are not refreshed.
- Models are pulled automatically from Hugging Face the first time the server starts; watch the logs for the "Models ready" line to confirm downloads finished.

## Cluster-accelerated demo concept

Looking for a larger, more visual showcase? See [`docs/cluster_accelerated_mind_mirror.md`](docs/cluster_accelerated_mind_mirror.md) for a two-DGX design that streams webcam input to a vision stack on DGX #1 (VLM-based face+emotion) and ships tensors over dual 100 Gb links to DGX #2 for diffusion rendering at 10–20 FPS.
