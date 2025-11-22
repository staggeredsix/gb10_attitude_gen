#!/usr/bin/env bash
set -euo pipefail

# Ensure required model snapshots are available from the mounted ./models volume
# before starting the application.

MODEL_ROOT="${MODEL_ROOT:-/models}"
export HF_HOME="${HF_HOME:-${MODEL_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HUGGINGFACE_HUB_CACHE}"

python3 - <<'PY'
import os
import sys
from pathlib import Path

REQUIRED = {
    "emotion": "Qwen/Qwen2-VL-2B-Instruct",
    "diffusion": "black-forest-labs/FLUX.1-schnell",
    "controlnet": "InstantX/FLUX.1-dev-Controlnet-Union",
    "face-segmentation": "briaai/RMBG-1.4",
}

hf_home = Path(os.environ["HF_HOME"])
cache_root = Path(os.environ["HUGGINGFACE_HUB_CACHE"])

missing = []
for label, repo in REQUIRED.items():
    cache_dir = cache_root / f"models--{repo.replace('/', '--')}"
    if not cache_dir.exists():
        missing.append((label, repo, cache_dir))

if missing:
    print("[error] Missing model snapshots in mounted models volume:")
    for label, repo, path in missing:
        print(f" - {label} ({repo}) not found at {path}")
    print("Run scripts/download_models.sh on the host to populate ./models before starting the container.")
    sys.exit(1)

print(f"[startup] Using Hugging Face cache at {hf_home}")
PY

exec "$@"
