#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"
HF_HOME_DIR="${MODELS_DIR}/huggingface"
HUB_CACHE_DIR="${HF_HOME_DIR}/hub"

LTX2_MODEL_ID="${LTX2_MODEL_ID:-Lightricks/LTX-2}"
LTX2_FP4_FILE="${LTX2_FP4_FILE:-ltx-2-19b-dev-fp4.safetensors}"
LTX2_DEV_FP8_FILE="${LTX2_DEV_FP8_FILE:-ltx-2-19b-dev-fp8.safetensors}"
LTX2_SPATIAL_UPSCALER_FILE="${LTX2_SPATIAL_UPSCALER_FILE:-ltx-2-spatial-upscaler-x2-1.0.safetensors}"
LTX2_TEMPORAL_UPSCALER_FILE="${LTX2_TEMPORAL_UPSCALER_FILE:-ltx-2-temporal-upscaler-x2-1.0.safetensors}"
LTX2_USE_DISTILLED="${LTX2_USE_DISTILLED:-1}"
LTX2_DISTILLED_FP8_FILE="${LTX2_DISTILLED_FP8_FILE:-ltx-2-19b-distilled-fp8.safetensors}"
LTX2_DISTILLED_FP4_FILE="${LTX2_DISTILLED_FP4_FILE:-ltx-2-19b-distilled-fp4.safetensors}"
LTX2_DISTILLED_LORA_FILE="${LTX2_DISTILLED_LORA_FILE:-ltx-2-19b-distilled-lora-384.safetensors}"
GEMMA_MODEL_ID="${GEMMA_MODEL_ID:-google/gemma-3-12b-it}"
GEMMA_DIR="${GEMMA_DIR:-${MODELS_DIR}/gemma}"

MODE="${1:-all}"

mkdir -p "${HUB_CACHE_DIR}"

export HF_HOME="${HF_HOME_DIR}"
export HUGGINGFACE_HUB_CACHE="${HUB_CACHE_DIR}"
export LTX2_MODEL_ID LTX2_FP4_FILE LTX2_DEV_FP8_FILE LTX2_SPATIAL_UPSCALER_FILE LTX2_TEMPORAL_UPSCALER_FILE
export LTX2_USE_DISTILLED LTX2_DISTILLED_FP8_FILE LTX2_DISTILLED_FP4_FILE LTX2_DISTILLED_LORA_FILE
export GEMMA_MODEL_ID GEMMA_DIR

usage() {
  cat <<'USAGE'
Usage: ./download_model.sh [ltx2|gemma|all]
Environment overrides:
  LTX2_MODEL_ID (default: Lightricks/LTX-2)
  LTX2_FP4_FILE (default: ltx-2-19b-dev-fp4.safetensors)
  LTX2_DEV_FP8_FILE (default: ltx-2-19b-dev-fp8.safetensors)
  LTX2_USE_DISTILLED (default: 1)
  LTX2_DISTILLED_FP8_FILE (default: ltx-2-19b-distilled-fp8.safetensors)
  LTX2_DISTILLED_FP4_FILE (default: ltx-2-19b-distilled-fp4.safetensors)
  LTX2_DISTILLED_LORA_FILE (default: ltx-2-19b-distilled-lora-384.safetensors)
  LTX2_SPATIAL_UPSCALER_FILE (default: ltx-2-spatial-upscaler-x2-1.0.safetensors)
  LTX2_TEMPORAL_UPSCALER_FILE (default: ltx-2-temporal-upscaler-x2-1.0.safetensors)
  GEMMA_MODEL_ID (default: google/gemma-3-12b)
  GEMMA_DIR (default: ./models/gemma)
  HF_TOKEN (optional, required if the model requires a license)
USAGE
}

download_ltx2() {
  echo "Downloading LTX-2 (pipeline components) + FP4 + upscalers into ${HUB_CACHE_DIR}..."
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
dev_fp8 = os.environ["LTX2_DEV_FP8_FILE"]
use_distilled = os.environ.get("LTX2_USE_DISTILLED", "1").strip().lower() in {"1", "true", "yes", "on"}
distilled_fp8 = os.environ["LTX2_DISTILLED_FP8_FILE"]
distilled_fp4 = os.environ["LTX2_DISTILLED_FP4_FILE"]
distilled_lora = os.environ["LTX2_DISTILLED_LORA_FILE"]
spatial = os.environ["LTX2_SPATIAL_UPSCALER_FILE"]
temporal = os.environ["LTX2_TEMPORAL_UPSCALER_FILE"]

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
    dev_fp8,
    fp4,
    distilled_fp8,
    distilled_fp4,
    distilled_lora,
    spatial,
    temporal,
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
    "dev_fp8": os.path.join(snapshot_dir, dev_fp8),
    "fp4": os.path.join(snapshot_dir, fp4),
    "distilled_fp8": os.path.join(snapshot_dir, distilled_fp8),
    "distilled_fp4": os.path.join(snapshot_dir, distilled_fp4),
    "distilled_lora": os.path.join(snapshot_dir, distilled_lora),
    "spatial_upscaler": os.path.join(snapshot_dir, spatial),
    "temporal_upscaler": os.path.join(snapshot_dir, temporal),
}

mode = "distilled" if use_distilled else "dev"
missing_required = []
missing_optional = []
if not os.path.isfile(paths["dev_fp8"]):
    missing_required.append("dev_fp8")
if use_distilled:
    if not os.path.isfile(paths["distilled_fp8"]):
        missing_required.append("distilled_fp8")
    if not os.path.isfile(paths["distilled_lora"]):
        missing_required.append("distilled_lora")
    if not os.path.isfile(paths["distilled_fp4"]):
        missing_optional.append("distilled_fp4")
else:
    if not os.path.isfile(paths["fp4"]):
        missing_optional.append("fp4")

for key in ("spatial_upscaler", "temporal_upscaler"):
    if not os.path.isfile(paths[key]):
        missing_optional.append(key)

if missing_required:
    print("Missing required distilled artifacts:", missing_required, file=sys.stderr)
    for k in missing_required:
        print(f"  - {k}: {paths[k]}", file=sys.stderr)
    raise SystemExit(4)

if missing_optional:
    print("Warning: missing optional top-level artifacts:", missing_optional, file=sys.stderr)
    for k in missing_optional:
        print(f"  - {k}: {paths[k]}", file=sys.stderr)

print("OK:")
print(f"  snapshot_dir: {paths['snapshot_dir']}")
print(f"  mode:         {mode}")
print(f"  dev_fp8:      {paths['dev_fp8']}")
if use_distilled:
    print(f"  checkpoint:   {paths['distilled_fp8']}")
    print(f"  distilled_lora: {paths['distilled_lora']}")
    print(f"  distilled_fp4:  {paths['distilled_fp4']}")
else:
    print(f"  fp4:          {paths['fp4']}")
spatial_ok = os.path.isfile(paths["spatial_upscaler"])
temporal_ok = os.path.isfile(paths["temporal_upscaler"])
print(f"  spatial:      {paths['spatial_upscaler']}")
print(f"  temporal:     {paths['temporal_upscaler']}")
print(f"  upscalers:   spatial {'OK' if spatial_ok else 'MISSING'}  temporal {'OK' if temporal_ok else 'MISSING'}")

target_dir = os.path.join(os.path.dirname(snapshot_dir), "ltx2")
os.makedirs(target_dir, exist_ok=True)
for key in ("dev_fp8", "distilled_fp8", "distilled_lora"):
    src = paths.get(key)
    if src and os.path.isfile(src):
        dst = os.path.join(target_dir, os.path.basename(src))
        if not os.path.isfile(dst):
            try:
                import shutil
                shutil.copyfile(src, dst)
                print(f"  copied: {src} -> {dst}")
            except Exception as exc:
                print(f"  warning: failed to copy {src} -> {dst}: {exc}", file=sys.stderr)
PY
  echo "Done."
  echo "Cache: ${HUGGINGFACE_HUB_CACHE}"
}

download_gemma() {
  echo "Downloading Gemma (${GEMMA_MODEL_ID}) into ${GEMMA_DIR}..."
  if ! python3 - <<'PY'; then
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
    "preprocessor_config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.*",
    "special_tokens_map.json",
    "added_tokens.json",
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

required_files = ["config.json", "preprocessor_config.json", "tokenizer_config.json", "added_tokens.json"]
missing = [f for f in required_files if not os.path.isfile(os.path.join(local_dir, f))]
if missing:
    print("Gemma download missing required files:", missing, file=sys.stderr)
    raise SystemExit(2)

print("OK:")
print(f"  gemma_dir:    {local_dir}")
print(f"  model_id:     {model_id}")
PY
    if [[ -z "${HF_TOKEN:-}" ]]; then
      echo "Gemma requires accepting the HF license and using HF_TOKEN. Visit the model page and accept, then export HF_TOKEN=..." >&2
    fi
    return 1
  fi
  echo "Mount GEMMA_DIR into container at /models/gemma and set LTX2_GEMMA_ROOT=/models/gemma"
}

case "${MODE}" in
  ltx2)
    download_ltx2
    ;;
  gemma)
    download_gemma
    ;;
  all)
    download_ltx2
    download_gemma
    ;;
  *)
    usage
    exit 1
    ;;
esac
