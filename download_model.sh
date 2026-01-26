#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
HF_HOME_DIR="${MODELS_DIR}/huggingface"
HUB_CACHE_DIR="${HF_HOME_DIR}/hub"

MODEL_ID="${LTX2_MODEL_ID:-Lightricks/LTX-2}"
MODEL_FILE="${LTX2_MODEL_FILE:-ltx-2-19b-dev-fp4.safetensors}"

mkdir -p "${HUB_CACHE_DIR}"

export HF_HOME="${HF_HOME_DIR}"
export HUGGINGFACE_HUB_CACHE="${HUB_CACHE_DIR}"
export MODEL_ID
export MODEL_FILE

echo "Downloading ${MODEL_ID}/${MODEL_FILE} into ${HUB_CACHE_DIR}..."

MODEL_PATH="$(python3 - <<'PY'
import os
import sys

try:
    from huggingface_hub import snapshot_download
except Exception as exc:  # noqa: BLE001
    print("huggingface_hub is not available. Install it with: pip install huggingface_hub", file=sys.stderr)
    raise SystemExit(1) from exc

model_id = os.environ.get("MODEL_ID") or os.environ.get("LTX2_MODEL_ID") or "Lightricks/LTX-2"
filename = os.environ.get("MODEL_FILE") or os.environ.get("LTX2_MODEL_FILE") or "ltx-2-19b-dev-fp4.safetensors"
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
cache_dir = os.environ.get("HUGGINGFACE_HUB_CACHE")
allow_patterns = os.environ.get("LTX2_ALLOW_PATTERNS")
if allow_patterns:
    patterns = [p.strip() for p in allow_patterns.split(",") if p.strip()]
else:
    patterns = [
        "model_index.json",
        "**/*.json",
        "**/*.safetensors",
        "**/*.txt",
        "**/*.py",
        "connectors/**",
    ]

snapshot_dir = snapshot_download(
    repo_id=model_id,
    token=token,
    cache_dir=cache_dir,
    allow_patterns=patterns,
)
model_path = os.path.join(snapshot_dir, filename)
print(model_path)
PY
)"

if [[ -z "${MODEL_PATH}" || ! -f "${MODEL_PATH}" ]]; then
    echo "Download failed: file not found in cache." >&2
    exit 1
fi

echo "Model cached at: ${MODEL_PATH}"
echo "Done. The model cache lives under ${HUGGINGFACE_HUB_CACHE}."
