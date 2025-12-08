#!/usr/bin/env bash
set -euo pipefail

# Download all required Hugging Face models into ./models so they can be mounted
# into the runtime container instead of being baked into the image.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${REPO_ROOT}/models"
export HF_HOME="${HF_HOME:-${MODELS_DIR}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HUGGINGFACE_HUB_CACHE}"

token="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}}"
export FLUX_MODEL_ID="${FLUX_MODEL_ID:-black-forest-labs/FLUX.1-dev-fp4}"

python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

MODELS = {
    "emotion": "Qwen/Qwen2-VL-2B-Instruct",
    "diffusion": os.environ.get("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-dev-fp4"),
    "controlnet": "InstantX/FLUX.1-dev-Controlnet-Union",
    "face-segmentation": "briaai/RMBG-1.4",
}

token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
cache_dir = os.environ["HUGGINGFACE_HUB_CACHE"]
print(f"[info] Writing snapshots to {cache_dir}")
os.makedirs(cache_dir, exist_ok=True)

for label, repo in MODELS.items():
    print(f"[download] {label}: {repo}")
    snapshot_download(
        repo_id=repo,
        cache_dir=cache_dir,
        token=token,
        local_files_only=False,
        resume_download=True,
        allow_patterns=None,
    )

print("[ok] All models cached under", cache_dir)
PY
