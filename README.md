# AI Mood Mirror

A simple demo that captures webcam video, detects faces, classifies emotions, and generates a matching portrait using a FLUX + ControlNet pipeline.

## Features

- GPU face segmentation (no MediaPipe) using a lightweight transformer
- Emotion classification from face masks using a lightweight VLM (default: `Qwen/Qwen2-VL-2B-Instruct`)
- Prompt construction mapped from dominant emotion
- Diffusion image generation conditioned on your webcam frame (GPU-required)
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
- `--diffusion-model`: Diffusion model id (default `city96/FLUX.2-dev-gguf`).

- `--controlnet-model`: ControlNet id to condition on the webcam frame (default `InstantX/FLUX.1-dev-Controlnet-Union`).
- `--face-segmentation-model`: Hugging Face model id for face segmentation.
- `--segmentation-min-area`: Minimum area ratio for a valid face mask (default `0.01`).
- `--generation-interval`: Seconds between portrait generations (default `7.0`).
- `--use-cuda` / `--no-cuda`: Force enable/disable CUDA (GPU is mandatory; disabling will raise an error).
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

Models are downloaded once to a shared `./models` directory and mounted into the container at runtime. Download them locally (optionally with a Hugging Face token) before building:

```bash
./scripts/download_models.sh
# optionally: HUGGINGFACE_TOKEN=<token> ./scripts/download_models.sh
```

Build the image and start with compose (exposes the web UI on `0.0.0.0:8000` and lets the browser access your webcam):

```bash
docker compose up --build
```

The compose file explicitly requests NVIDIA GPUs (`deploy.reservations.devices`, `gpus: all`, and `NVIDIA_VISIBLE_DEVICES=all`).
If startup fails with "GPU execution is required", confirm GPU visibility from inside the container:

```bash
docker compose exec -T ai-mood-mirror python3 -c "import torch; print({'cuda_available': torch.cuda.is_available(), 'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})"
```

Troubleshooting GPU visibility:

- The container **must** start with GPU access (`docker run --gpus all ...` or the compose file above). If you are running in an environment without NVIDIA hardware (e.g., CI runners or cloud sandboxes without GPU passthrough), startup will fail with the `GPU execution is required` error seen in the logs.
- Verify the NVIDIA container runtime is installed locally and that `nvidia-smi` works on the host before launching compose.
- On some Docker Compose versions you may need to enable the NVIDIA runtime explicitly: add `DOCKER_DEFAULT_PLATFORM=linux/amd64` when building on non-native hosts and ensure `/etc/docker/daemon.json` contains `"default-runtime": "nvidia"` with the NVIDIA toolkit installed.
- If GPUs remain hidden, try `docker compose up --build --verbose` to confirm the GPU request is reaching the engine, or run the image directly with `docker run --gpus all ...` to bypass compose translation.
- For an interactive shell with a `/bin/bash` entrypoint (so you can run `nvidia-smi`, Python CUDA checks, etc.), use `./scripts/gpu_shell.sh`.

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
- **Dual node:** `./scripts/dual_spark.sh <SECOND_SPARK_IP> [ssh_user]` builds the image, deploys the primary vision/web stack locally with `CLUSTER_MODE=dual`, syncs the repo to the secondary Spark, and starts one diffusion worker per GPU on ports `9000+N`.

## Notes

- GPU acceleration is required for both the VLM-based emotion rater and the FLUX ControlNet pipeline; startup will fail if no CUDA or MPS device is detected.
- The default FLUX.2-dev GGUF diffusion model (`city96/FLUX.2-dev-gguf`, file `flux2-dev-Q4_K_S.gguf`) is optimized for the latest FLUX.2 pipeline while remaining compatible with existing CUDA setups.
- The app throttles image generation to avoid excessive GPU load and regenerates when the detected emotion changes.
- If no face is detected, a visible overlay is shown and portraits are not refreshed.
- Models must exist under `./models` (populated via `./scripts/download_models.sh`) and are mounted into the container at `/models`; the entrypoint will exit with a clear error if they are missing.

## Cluster-accelerated demo concept

Looking for a larger, more visual showcase? See [`docs/cluster_accelerated_mind_mirror.md`](docs/cluster_accelerated_mind_mirror.md) for a two-DGX design that streams webcam input to a vision stack on DGX #1 (VLM-based face+emotion) and ships tensors over dual 100 Gb links to DGX #2 for diffusion rendering at 10–20 FPS.
