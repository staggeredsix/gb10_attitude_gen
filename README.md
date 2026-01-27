# LTX-2 Fever Dream + Mood Mirror

Two runnable experiences powered **only** by LTX-2:

1. **AI Fever Dream** — continuously generates looping, surreal video streams from a prompt.
2. **AI Mood Mirror** — uses your browser webcam to estimate a lightweight “mood” signal and mirrors it in the generated video.

Both modes run from the same web UI and share the same backend.

## Requirements

- NVIDIA GPU (LTX-2 is heavy; this is a POC optimized for a local GPU).
- Python 3.10+
- System packages for OpenCV (Ubuntu/Debian: `libgl1`, `libglib2.0-0`).

## Install

```bash
pip install -r requirements.txt
```

**PyTorch note:** if you need a specific CUDA build, install it first from the official index, then install `requirements.txt`.

## Getting Models

```bash
export HF_TOKEN=...  # required if the model requires a license
./download_model.sh fp8   # fp8 checkpoint + upsampler + distilled lora + gemma
./download_model.sh fp4   # fp4 diffusers snapshot
./download_model.sh gemma # gemma only
./download_model.sh all   # fp8 + fp4 + gemma
```

Gemma downloads into `./models/gemma` by default. When running in Docker, mount that folder and set:

```bash
export LTX2_GEMMA_ROOT=/models/gemma
```

Expected Gemma model id: `google/gemma-3-12b`.

## Run

```bash
python -m app --host 0.0.0.0 --port 8000
```

Open from the same machine: `http://localhost:8000`  
Open from your LAN: `http://<your-machine-ip>:8000`

## Docker (profiled backends)

FP8 (pipelines):

```bash
docker compose --profile fp8 up --build
```

FP4 (diffusers, WIP):

```bash
docker compose --profile fp4 up --build
```

The compose file mounts `./models` to `/models` and expects:

- FP8 (pipelines):
  - `LTX2_BACKEND=pipelines`
  - `LTX2_CHECKPOINT_PATH=/models/.../ltx-2-19b-dev-fp8.safetensors`
  - `LTX2_GEMMA_ROOT=/models/gemma`
  - optional upscaler/lora paths for two-stage output
- FP4 (diffusers):
  - `LTX2_BACKEND=diffusers`
  - `LTX2_MODEL_ID=Lightricks/LTX-2`
  - `LTX2_SNAPSHOT_DIR=/models/huggingface/hub/models--Lightricks--LTX-2/snapshots/<hash>`
  - `LTX2_FP4_FILE=ltx-2-19b-dev-fp4.safetensors`

## UI guide

- **Mode selector**: choose Fever Dream or Mood Mirror.
- **Prompt**: used for Fever Dream, and as the base for Mood Mirror.
- **Resolution / FPS / Streams**: changing any value restarts all streams.
- **Output mode**:
  - `native`: one-stage output at the requested size.
  - `upscaled`: two-stage output (stage 1 low-res, stage 2 spatial upsample + distilled LoRA).
- **Dream strength / Motion**: coarse controls over guidance/steps and motion density.
- **Mood Mirror**: enable the webcam, watch the live mood readout, and adjust the “retain identity” slider.

## Output modes & required artifacts

Environment variables and request fields:

```bash
# Base (native) generation size & fps
export LTX2_NATIVE_WIDTH=1280
export LTX2_NATIVE_HEIGHT=736
export LTX2_NATIVE_FPS=24

# Output mode (native or upscaled)
export LTX2_OUTPUT_MODE=native

# Backend selection
export LTX2_BACKEND=pipelines  # or diffusers

# Pipelines (FP8) required artifacts
export LTX2_GEMMA_ROOT=/models/ltx2/gemma
export LTX2_CHECKPOINT_PATH=/models/ltx2/ltx-2-19b-dev-fp8.safetensors

# Required only for upscaled mode
export LTX2_SPATIAL_UPSAMPLER_PATH=/models/ltx2/ltx-2-spatial-upscaler-x2-1.0.safetensors
export LTX2_DISTILLED_LORA_PATH=/models/ltx2/ltx-2-distilled-lora-x2-1.0.safetensors
export LTX2_DISTILLED_LORA_STRENGTH=0.6

# Diffusers (FP4) artifacts
export LTX2_MODEL_ID=Lightricks/LTX-2
export LTX2_SNAPSHOT_DIR=/models/huggingface/hub/models--Lightricks--LTX-2/snapshots/<hash>
export LTX2_FP4_FILE=ltx-2-19b-dev-fp4.safetensors
```

When posting to `/api/config`, you can also set:

```json
{
  "output_mode": "upscaled"
}
```

Upscaled output requires width and height to be multiples of 64 (stage 1 uses half resolution).

## Troubleshooting

- **Missing model_index.json**: ensure the path you mounted is the snapshot root (the folder that contains `model_index.json`). The loader fails fast if this file is missing.
- **Container can’t see host cache**: the container only sees `/models` by default. Either mount `./models` with the download script output or uncomment the optional HF cache mount in
  `docker-compose.yml`.
- **LTX-2 load failed**: check your CUDA installation and make sure the model download completed.
- **Webcam not streaming**: ensure your browser has webcam permissions and the page is served over HTTP(s) you trust.
- **Slow FPS**: reduce resolution, reduce streams, or lower FPS.
