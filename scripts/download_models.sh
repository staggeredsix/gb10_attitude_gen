#!/usr/bin/env bash
set -euo pipefail

# Download all required Hugging Face models into ./models so they can be mounted
# into the runtime container instead of being baked into the image. If an
# automatic download fails (for example due to missing auth), the script will
# continue and print clear instructions so models can be placed manually.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${REPO_ROOT}/models"

# Keep the user's HF_HOME untouched so existing CLI logins (including tokens for
# gated models) are respected. Only redirect the download cache to the repo so
# model files live under ./models. If you already have a cache elsewhere,
# export HUGGINGFACE_HUB_CACHE to point to it.
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${MODELS_DIR}/huggingface/hub}"
mkdir -p "${HUGGINGFACE_HUB_CACHE}"

token="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}}"
export FLUX_MODEL_ID="${FLUX_MODEL_ID:-black-forest-labs/FLUX.1-dev-onnx}"

python3 - <<'PY'
import os
import sys
from textwrap import indent

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError

MODELS = {
    "emotion": "Qwen/Qwen2-VL-2B-Instruct",
    "diffusion": os.environ.get("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-dev-onnx"),
    "controlnet": "InstantX/FLUX.1-dev-Controlnet-Union",
    "face-segmentation": "briaai/RMBG-1.4",
}

token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
cache_dir = os.environ["HUGGINGFACE_HUB_CACHE"]
print(f"[info] Writing snapshots to {cache_dir}")
os.makedirs(cache_dir, exist_ok=True)

if not token:
    print("[info] Using any locally cached Hugging Face credentials (if logged in via `huggingface-cli login`).")
else:
    print("[info] Using token provided via environment variables.")

failures = []
for label, repo in MODELS.items():
    print(f"[download] {label}: {repo}")
    try:
        snapshot_download(
            repo_id=repo,
            cache_dir=cache_dir,
            token=token,
            local_files_only=False,
            resume_download=True,
            allow_patterns=None,
        )
    except GatedRepoError as exc:
        print(f"[warn] Unable to download {repo}: {exc}")
        print(
            "  - This repository is gated. Authenticate with `huggingface-cli login` "
            "or set HUGGINGFACE_TOKEN/HF_TOKEN/HUGGINGFACE_HUB_TOKEN."
        )
        failures.append((label, repo, str(exc)))
    except Exception as exc:  # noqa: BLE001 - best-effort download
        print(f"[warn] Unable to download {repo}: {exc}")
        failures.append((label, repo, str(exc)))

if failures:
    print("\n[warn] Some models were not downloaded automatically.")
    manual_dir = os.path.join(cache_dir, "models")
    print(
        "You can place model snapshots manually under:\n"
        f"  - {cache_dir}\n"
        f"  - or nested under {manual_dir} (matching huggingface_hub layout)\n"
        "Missing entries:"
    )
    print(
        indent(
            "\n".join(f"- {label}: {repo}" for label, repo, _ in failures),
            prefix="  ",
        )
    )
else:
    print("[ok] All models cached under", cache_dir)

sys.exit(0)
PY
