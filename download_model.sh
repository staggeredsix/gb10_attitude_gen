#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
HF_HOME_DIR="${MODELS_DIR}/huggingface"
HUB_CACHE_DIR="${HF_HOME_DIR}/hub"

LTX2_MODEL_ID="${LTX2_MODEL_ID:-Lightricks/LTX-2}"
LTX2_FP4_FILE="${LTX2_FP4_FILE:-ltx-2-19b-dev-fp4.safetensors}"
LTX2_FP8_FILE="${LTX2_FP8_FILE:-ltx-2-19b-dev-fp8.safetensors}"
LTX2_SPATIAL_UPSAMPLER_FILE="${LTX2_SPATIAL_UPSAMPLER_FILE:-ltx-2-spatial-upscaler-x2-1.0.safetensors}"
LTX2_DISTILLED_LORA_FILE="${LTX2_DISTILLED_LORA_FILE:-ltx-2-distilled-lora-x2-1.0.safetensors}"
GEMMA_MODEL_ID="${GEMMA_MODEL_ID:-google/gemma-3-12b}"
GEMMA_DIR="${GEMMA_DIR:-${MODELS_DIR}/gemma}"

# Optional: set your token here, or export HF_TOKEN=... in your shell.
HF_TOKEN_DEFAULT="${HF_TOKEN_DEFAULT:-}"
HF_TOKEN="${HF_TOKEN:-${HF_TOKEN_DEFAULT}}"

MODE="${1:-all}"

mkdir -p "${HUB_CACHE_DIR}"

export HF_HOME="${HF_HOME_DIR}"
export HUGGINGFACE_HUB_CACHE="${HUB_CACHE_DIR}"
export LTX2_MODEL_ID LTX2_FP4_FILE LTX2_FP8_FILE LTX2_SPATIAL_UPSAMPLER_FILE LTX2_DISTILLED_LORA_FILE
export GEMMA_MODEL_ID GEMMA_DIR
export HF_TOKEN

usage() {
  cat <<'USAGE'
Usage: ./download_model.sh [fp8|fp4|gemma|all]
Environment overrides:
  LTX2_MODEL_ID (default: Lightricks/LTX-2)
  LTX2_FP4_FILE (default: ltx-2-19b-dev-fp4.safetensors)
  LTX2_FP8_FILE (default: ltx-2-19b-dev-fp8.safetensors)
  LTX2_SPATIAL_UPSAMPLER_FILE (default: ltx-2-spatial-upscaler-x2-1.0.safetensors)
  LTX2_DISTILLED_LORA_FILE (default: ltx-2-distilled-lora-x2-1.0.safetensors)
  GEMMA_MODEL_ID (default: google/gemma-3-12b)
  GEMMA_DIR (default: ./models/gemma)
  HF_TOKEN (optional, required if the model requires a license)
USAGE
}

download_fp8() {
  echo "Downloading LTX-2 FP8 checkpoint + upsampler + distilled LoRA into ${HUB_CACHE_DIR}..."
  python3 - <<'PY'
import os, sys

try:
    from huggingface_hub import snapshot_download
except Exception as exc:
    print("huggingface_hub is not available. Install it with: pip install huggingface_hub", file=sys.stderr)
    raise SystemExit(1) from exc

model_id = os.environ["LTX2_MODEL_ID"]
cache_dir = os.environ["HUGGINGFACE_HUB_CACHE"]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

fp8 = os.environ["LTX2_FP8_FILE"]
spatial = os.environ["LTX2_SPATIAL_UPSAMPLER_FILE"]
distilled = os.environ["LTX2_DISTILLED_LORA_FILE"]

allow_patterns = [
    fp8,
    spatial,
    distilled,
]

snapshot_dir = snapshot_download(
    repo_id=model_id,
    token=token,
    cache_dir=cache_dir,
    allow_patterns=allow_patterns,
)

paths = {
    "snapshot_dir": snapshot_dir,
    "fp8": os.path.join(snapshot_dir, fp8),
    "spatial_upscaler": os.path.join(snapshot_dir, spatial),
    "distilled_lora": os.path.join(snapshot_dir, distilled),
}

missing = [k for k, v in paths.items() if k != "snapshot_dir" and not os.path.isfile(v)]
if missing:
    print("Warning: missing optional artifacts:", missing, file=sys.stderr)
    for k in missing:
        print(f"  - {k}: {paths[k]}", file=sys.stderr)

print("OK:")
print(f"  snapshot_dir: {paths['snapshot_dir']}")
print(f"  fp8:          {paths['fp8']}")
print(f"  spatial:      {paths['spatial_upscaler']}")
print(f"  distilled:    {paths['distilled_lora']}")
PY
  echo "Done."
  echo "Cache: ${HUGGINGFACE_HUB_CACHE}"
}

download_fp4() {
  echo "Downloading LTX-2 diffusers snapshot + FP4 into ${HUB_CACHE_DIR}..."
  python3 - <<'PY'
import os, sys

try:
    from huggingface_hub import snapshot_download
except Exception as exc:
    print("huggingface_hub is not available. Install it with: pip install huggingface_hub", file=sys.stderr)
    raise SystemExit(1) from exc

model_id = os.environ["LTX2_MODEL_ID"]
cache_dir = os.environ["HUGGINGFACE_HUB_CACHE"]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

fp4 = os.environ["LTX2_FP4_FILE"]

allow_patterns = [
    "model_index.json",
    "**/*.json",
    "**/*.txt",
    "audio_vae/**",
    "vae/**",
    "transformer/**",
    "text_encoder/**",
    "vocoder/**",
    "scheduler/**",
    "tokenizer/**",
    "connectors/**",
    "**/diffusion_pytorch_model.safetensors",
    "**/diffusion_pytorch_model.bin",
    "**/model.safetensors",
    "**/pytorch_model.bin",
    "**/*.safetensors",
    fp4,
]

snapshot_dir = snapshot_download(
    repo_id=model_id,
    token=token,
    cache_dir=cache_dir,
    allow_patterns=allow_patterns,
)

required_dirs = ["audio_vae", "vae", "transformer", "text_encoder", "vocoder"]
missing_dirs = [d for d in required_dirs if not os.path.isdir(os.path.join(snapshot_dir, d))]
if missing_dirs:
    print("Snapshot missing expected component directories:", missing_dirs, file=sys.stderr)
    print("snapshot_dir:", snapshot_dir, file=sys.stderr)
    raise SystemExit(2)

print("OK:")
print(f"  snapshot_dir: {snapshot_dir}")
print(f"  fp4:          {os.path.join(snapshot_dir, fp4)}")
PY
  echo "Done."
  echo "Cache: ${HUGGINGFACE_HUB_CACHE}"
}

download_gemma() {
  echo "Downloading Gemma (${GEMMA_MODEL_ID}) into ${GEMMA_DIR}..."
  if ! python3 - <<'PY'
import os, sys

try:
    from huggingface_hub import snapshot_download
except Exception as exc:
    print("huggingface_hub is not available. Install it with: pip install huggingface_hub", file=sys.stderr)
    raise SystemExit(1) from exc

model_id = os.environ["GEMMA_MODEL_ID"]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
local_dir = os.environ["GEMMA_DIR"]

allow_patterns = [
    "config.json",
    "generation_config.json",
    "tokenizer*",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "*.json",
    "*.safetensors",
    "*.bin",
    "*.model",
]

snapshot_download(
    repo_id=model_id,
    token=token,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    allow_patterns=allow_patterns,
)

required_files = ["config.json"]
missing = [f for f in required_files if not os.path.isfile(os.path.join(local_dir, f))]
if missing:
    print("Gemma download missing required files:", missing, file=sys.stderr)
    raise SystemExit(2)

preprocessor_found = False
for base, _, files in os.walk(local_dir):
    if "preprocessor_config.json" in files:
        preprocessor_found = True
        break
if not preprocessor_found:
    print(
        "Gemma download missing preprocessor_config.json. "
        "Re-run ./download_model.sh gemma (it must include preprocessor_config.json).",
        file=sys.stderr,
    )
    raise SystemExit(3)

print("OK:")
print(f"  gemma_dir:    {local_dir}")
print(f"  model_id:     {model_id}")
print("  preprocessor_config.json: found")
PY
  then
    if [[ -z "${HF_TOKEN:-}" ]]; then
      echo "Gemma requires accepting the HF license and using HF_TOKEN. Visit the model page and accept, then export HF_TOKEN=..." >&2
    fi
    return 1
  fi
  echo "Mount GEMMA_DIR into container at /models/gemma and set LTX2_GEMMA_ROOT=/models/gemma"
}

print_instructions() {
  cat <<'TEXT'

Next steps:
  - Mount ./models into the container as /models.
  - FP8 backend:
      LTX2_BACKEND=pipelines
      LTX2_CHECKPOINT_PATH=/models/huggingface/hub/models--Lightricks--LTX-2/.../ltx-2-19b-dev-fp8.safetensors
      LTX2_GEMMA_ROOT=/models/gemma
      LTX2_SPATIAL_UPSAMPLER_PATH=/models/huggingface/hub/models--Lightricks--LTX-2/.../ltx-2-spatial-upscaler-x2-1.0.safetensors
      LTX2_DISTILLED_LORA_PATH=/models/huggingface/hub/models--Lightricks--LTX-2/.../ltx-2-distilled-lora-x2-1.0.safetensors
  - FP4 backend:
      LTX2_BACKEND=diffusers
      LTX2_MODEL_ID=Lightricks/LTX-2
      LTX2_SNAPSHOT_DIR=/models/huggingface/hub/models--Lightricks--LTX-2/snapshots/<hash>
      LTX2_FP4_FILE=ltx-2-19b-dev-fp4.safetensors
TEXT
}

case "${MODE}" in
  fp8)
    download_fp8
    download_gemma
    print_instructions
    ;;
  fp4)
    download_fp4
    print_instructions
    ;;
  gemma)
    download_gemma
    print_instructions
    ;;
  all)
    download_fp8
    download_fp4
    download_gemma
    print_instructions
    ;;
  *)
    usage
    exit 1
    ;;
esac
