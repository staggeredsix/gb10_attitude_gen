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
./download_model.sh all
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

## Docker (recommended offline snapshot)

```bash
python scripts/prefetch_ltx2.py --fp4-only
docker compose up --build
```

The compose file mounts `./models` to `/models` and sets:

- `LTX2_MODEL_ID=/models/LTX-2`
- `LTX2_VARIANT=fp4`
- `LTX2_ALLOW_DOWNLOAD=false`

If you prefer using an existing Hugging Face cache, uncomment the cache mount in
`docker-compose.yml` and set `LTX2_MODEL_ID` to the repo id (`Lightricks/LTX-2`).

## Model notes (LTX-2 NVFP4)

This project targets the **NVFP4** weights from the LTX-2 repo (`ltx-2-19b-dev-fp4.safetensors`).
By default the backend loads:

- model id: `Lightricks/LTX-2`

If you need to change those:

```bash
export LTX2_MODEL_ID="Lightricks/LTX-2"
export LTX2_VARIANT="fp4"
```

To download a local snapshot explicitly:

```bash
python scripts/prefetch_ltx2.py --fp4-only
```

If you want to allow in-container downloads instead, set:

```bash
export LTX2_ALLOW_DOWNLOAD=true
export LTX2_SNAPSHOT_DIR=/models/LTX-2
```

The first run will download the model to your Hugging Face cache if downloads are enabled. If you have a token:

```bash
export HF_HOME=~/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=~/.cache/huggingface/hub
```

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

# Required LTX-2 artifacts
export LTX2_GEMMA_ROOT=/models/ltx2/gemma
export LTX2_CHECKPOINT_PATH=/models/ltx2/ltx2-fp4.safetensors

# Required only for upscaled mode
export LTX2_SPATIAL_UPSAMPLER_PATH=/models/ltx2/ltx2-spatial-upsampler.safetensors
export LTX2_DISTILLED_LORA_PATH=/models/ltx2/ltx2-distilled-lora.safetensors
export LTX2_DISTILLED_LORA_STRENGTH=0.6
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
- **Container can’t see host cache**: the container only sees `/models` by default. Either mount `./models` with the prefetch script output or uncomment the optional HF cache mount in `docker-compose.yml`.
- **LTX-2 load failed**: check your CUDA installation and make sure the model download completed.
- **Webcam not streaming**: ensure your browser has webcam permissions and the page is served over HTTP(s) you trust.
- **Slow FPS**: reduce resolution, reduce streams, or lower FPS.
