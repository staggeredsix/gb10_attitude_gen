#!/usr/bin/env bash
set -euo pipefail

# Single-node setup helper for AI Mood Mirror
# - Builds the container image
# - Launches the web UI on the local host
# - Prints connection info when ready

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-ai-mood-mirror:single}"
PORT="${PORT:-8000}"
ROLE="${ROLE:-single}"
DEFAULT_MODE="${DEFAULT_MODE:-single}"

usage() {
  cat <<USAGE
Usage: IMAGE_TAG=<tag> PORT=<port> ROLE=<role> $0

Environment variables:
  IMAGE_TAG   Docker image tag to build/use (default: ai-mood-mirror:single)
  PORT        Port to expose the web UI on (default: 8000)
  ROLE        Role label passed into the container (default: single)
  DEFAULT_MODE Default inference mode surfaced in the web UI (default: single)

Prereqs: docker, docker compose, NVIDIA Container Toolkit (for GPU acceleration).
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[error] Missing dependency: $1" >&2
    exit 1
  fi
}

require docker
require docker compose

echo "[info] Building ${IMAGE_TAG} from ${REPO_ROOT}"
DOCKER_BUILDKIT=1 docker build -t "${IMAGE_TAG}" "${REPO_ROOT}"

echo "[info] Starting ai-mood-mirror on port ${PORT}"
IMAGE_TAG="${IMAGE_TAG}" PORT="${PORT}" ROLE="${ROLE}" DEFAULT_MODE="${DEFAULT_MODE}" docker compose -f "${REPO_ROOT}/docker-compose.yml" up -d --force-recreate

echo "[ok] Single-node stack is live"
echo "     URL: http://localhost:${PORT}" 
echo "     To view logs: docker compose -f ${REPO_ROOT}/docker-compose.yml logs -f"
