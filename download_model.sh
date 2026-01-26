#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
HF_HOME_DIR="${MODELS_DIR}/huggingface"
HUB_CACHE_DIR="${HF_HOME_DIR}/hub"

MODEL_ID="${LTX2_MODEL_ID:-Lightricks/LTX-2}"

# What you actually want:
FP4_FILE="${LTX2_FP4_FILE:-ltx-2-19b-dev-fp4.safetensors}"
SPATIAL_UPSCALER_FILE="${LTX2_SPATIAL_UPSCALER_FILE:-ltx-2-spatial-upscaler-x2-1.0.safetensors}"
TEMPORAL_UPSCALER_FILE="${LTX2_TEMPORAL_UPSCALER_FILE:-ltx-2-temporal-upscaler-x2-1.0.safetensors}"

mkdir -p "${HUB_CACHE_DIR}"

export HF_HOME="${HF_HOME_DIR}"
export HUGGINGFACE_HUB_CACHE="${HUB_CACHE_DIR}"
export MODEL_ID FP4_FILE SPATIAL_UPSCALER_FILE TEMPORAL_UPSCALER_FILE

echo "Downloading LTX-2 (pipeline components) + FP4 + upscalers into ${HUB_CACHE_DIR}..."
python3 - <<'PY'
import os, sys

try:
    from huggingface_hub import snapshot_download
except Exception as exc:
    print("huggingface_hub is not available. Install it with: pip install huggingface_hub", file=sys.stderr)
    raise SystemExit(1) from exc

model_id = os.environ["MODEL_ID"]
cache_dir = os.environ["HUGGINGFACE_HUB_CACHE"]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

fp4 = os.environ["FP4_FILE"]
spatial = os.environ["SPATIAL_UPSCALER_FILE"]
temporal = os.environ["TEMPORAL_UPSCALER_FILE"]

# Complete enough for DiffusionPipeline.from_pretrained() to work:
# - model_index.json + configs
# - all component subfolders (audio_vae/vae/transformer/text_encoder/vocoder/etc.) incl weights
# - explicit top-level FP4 + upscalers
allow_patterns = [
    "model_index.json",
    "**/*.json",
    "**/*.txt",

    # pipeline component dirs (configs + weights)
    "audio_vae/**",
    "vae/**",
    "transformer/**",
    "text_encoder/**",
    "vocoder/**",
    "scheduler/**",
    "tokenizer/**",
    "connectors/**",

    # weight filenames diffusers may load
    "**/diffusion_pytorch_model.safetensors",
    "**/diffusion_pytorch_model.bin",
    "**/model.safetensors",
    "**/pytorch_model.bin",
    "**/*.safetensors",

    # explicit top-level artifacts (in case they're not under subfolders)
    fp4,
    spatial,
    temporal,
]

snapshot_dir = snapshot_download(
    repo_id=model_id,
    token=token,
    cache_dir=cache_dir,
    allow_patterns=allow_patterns,
)

# Quick validation: ensure key component weights exist somewhere.
required_dirs = ["audio_vae", "vae", "transformer", "text_encoder", "vocoder"]
missing_dirs = [d for d in required_dirs if not os.path.isdir(os.path.join(snapshot_dir, d))]
if missing_dirs:
    print("Snapshot missing expected component directories:", missing_dirs, file=sys.stderr)
    print("snapshot_dir:", snapshot_dir, file=sys.stderr)
    raise SystemExit(2)

def has_any_weight(d: str) -> bool:
    root = os.path.join(snapshot_dir, d)
    for base, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(".safetensors") or fn.endswith(".bin"):
                return True
    return False

missing_weights = [d for d in required_dirs if not has_any_weight(d)]
if missing_weights:
    print("Snapshot is missing weights in component dirs:", missing_weights, file=sys.stderr)
    print("snapshot_dir:", snapshot_dir, file=sys.stderr)
    raise SystemExit(3)

paths = {
    "snapshot_dir": snapshot_dir,
    "fp4": os.path.join(snapshot_dir, fp4),
    "spatial_upscaler": os.path.join(snapshot_dir, spatial),
    "temporal_upscaler": os.path.join(snapshot_dir, temporal),
}

# Note: fp4/upscalers are optional for base pipeline load, but you said you want them cached too.
missing_optional = [k for k in ("fp4","spatial_upscaler","temporal_upscaler") if not os.path.isfile(paths[k])]
if missing_optional:
    print("Warning: missing optional top-level artifacts:", missing_optional, file=sys.stderr)
    for k in missing_optional:
        print(f"  - {k}: {paths[k]}", file=sys.stderr)

print("OK:")
print(f"  snapshot_dir: {paths['snapshot_dir']}")
print(f"  fp4:          {paths['fp4']}")
print(f"  spatial:      {paths['spatial_upscaler']}")
print(f"  temporal:     {paths['temporal_upscaler']}")
PY

echo "Done."
echo "Cache: ${HUGGINGFACE_HUB_CACHE}"
