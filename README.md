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

## Run

```bash
python -m app --host 0.0.0.0 --port 8000
```

Open from the same machine: `http://localhost:8000`  
Open from your LAN: `http://<your-machine-ip>:8000`

## Model notes (LTX-2 NVFP4)

This project targets the **NVFP4** weights from the LTX-2 repo (`ltx-2-19b-dev-fp4.safetensors`).
By default the backend loads:

- model id: `Lightricks/LTX-2`
- variant: `fp4`

If you need to change those:

```bash
export LTX2_MODEL_ID="Lightricks/LTX-2"
export LTX2_VARIANT="fp4"
```

The first run will download the model to your Hugging Face cache. If you have a token:

```bash
export HF_HOME=~/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=~/.cache/huggingface/hub
```

## UI guide

- **Mode selector**: choose Fever Dream or Mood Mirror.
- **Prompt / negative prompt**: used for Fever Dream, and as the base for Mood Mirror.
- **Resolution / FPS / Streams**: changing any value restarts all streams.
- **Dream strength / Motion**: coarse controls over guidance/steps and motion density.
- **Mood Mirror**: enable the webcam, watch the live mood readout, and adjust the “retain identity” slider.

## Troubleshooting

- **LTX-2 load failed**: check your CUDA installation and make sure the model download completed.
- **Webcam not streaming**: ensure your browser has webcam permissions and the page is served over HTTP(s) you trust.
- **Slow FPS**: reduce resolution, reduce streams, or lower FPS.
