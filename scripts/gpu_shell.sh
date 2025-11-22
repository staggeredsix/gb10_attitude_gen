#!/usr/bin/env bash
set -euo pipefail

# Launch an interactive shell inside the AI Mood Mirror container to troubleshoot GPU visibility.
# The container starts with a /bin/bash entrypoint so you can run nvidia-smi, torch checks, etc.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-ai-mood-mirror:latest}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models}"
NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"

usage() {
  cat <<USAGE
Usage: IMAGE_TAG=<tag> MODEL_ROOT=<path> NVIDIA_VISIBLE_DEVICES=<devices> $0 [-- <bash args>]

Environment variables:
  IMAGE_TAG                 Docker image tag to run (default: ai-mood-mirror:latest)
  MODEL_ROOT                Host path to the shared models directory (default: ./models)
  NVIDIA_VISIBLE_DEVICES    GPU devices to expose (default: all)
  NVIDIA_DRIVER_CAPABILITIES Driver capabilities to expose (default: all)

Examples:
  # Start a shell with all GPUs visible
  $0

  # Limit to GPU 0 and run nvidia-smi inside the container
  NVIDIA_VISIBLE_DEVICES=0 $0 -- -lc "nvidia-smi"
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "${MODEL_ROOT}" ]]; then
  echo "[error] Model root ${MODEL_ROOT} does not exist; run scripts/download_models.sh first." >&2
  exit 1
fi

# Pass through any additional arguments to bash after a literal '--'.
BASH_ARGS=()
if [[ "${1:-}" == "--" ]]; then
  shift
  BASH_ARGS=("$@")
fi

exec docker run --rm -it \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES}" \
  -e NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES}" \
  -e HF_HOME=/models/huggingface \
  -e HUGGINGFACE_HUB_CACHE=/models/huggingface/hub \
  -v "${MODEL_ROOT}:/models" \
  --entrypoint /bin/bash \
  "${IMAGE_TAG}" \
  "${BASH_ARGS[@]}"
