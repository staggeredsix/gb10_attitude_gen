#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $0 <SECOND_SPARK_IP> [ssh_user]

Arguments:
  SECOND_SPARK_IP   IPv4 address of the secondary Spark node
  ssh_user          SSH username (defaults to $SECOND_SPARK_SSH_USER or "ubuntu")

Environment variables:
  IMAGE_TAG         Docker image tag to build/run (default: ai-mood-mirror:latest)
  PORT              Primary web UI port (default: 8000)
  REMOTE_DIR        Repository location on the secondary Spark (default: /opt/ai-mood-mirror)
  BASE_PORT         Base port for diffusion workers on the secondary Spark (default: 9000)
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

SECOND_SPARK_IP="$1"
SSH_USER="${2:-${SECOND_SPARK_SSH_USER:-ubuntu}}"
IMAGE_TAG="${IMAGE_TAG:-ai-mood-mirror:latest}"
PRIMARY_PORT="${PORT:-8000}"
REMOTE_DIR="${REMOTE_DIR:-/opt/ai-mood-mirror}"
BASE_PORT="${BASE_PORT:-9000}"

require() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[error] Missing dependency: $1" >&2
    exit 1
  fi
}

require git
require ssh
require docker
require rsync

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORIGIN_URL="$(git -C "${REPO_ROOT}" config --get remote.origin.url)"
CURRENT_BRANCH="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)"
CURRENT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"

check_ssh() {
  local host="$1"
  echo "[info] Checking SSH connectivity to ${host}"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "${SSH_USER}@${host}" "echo ok" >/dev/null
}

check_remote_tools() {
  echo "[info] Verifying git + docker on secondary Spark"
  ssh "${SSH_USER}@${SECOND_SPARK_IP}" "command -v git >/dev/null && command -v docker >/dev/null"
}

sync_remote_repo() {
  echo "[info] Syncing repository to ${SSH_USER}@${SECOND_SPARK_IP}:${REMOTE_DIR}"
  ssh "${SSH_USER}@${SECOND_SPARK_IP}" "set -euo pipefail; \
    if [[ ! -d '${REMOTE_DIR}' ]]; then git clone '${ORIGIN_URL}' '${REMOTE_DIR}'; fi; \
    cd '${REMOTE_DIR}'; \
    git fetch origin; \
    git checkout '${CURRENT_BRANCH}'; \
    git reset --hard '${CURRENT_COMMIT}'"
}

sync_models() {
  if [[ ! -d "${REPO_ROOT}/models" ]]; then
    echo "[warn] No local ./models directory found; secondary diffusion workers will download models as needed"
    ssh "${SSH_USER}@${SECOND_SPARK_IP}" "mkdir -p '${REMOTE_DIR}/models'"
    return
  fi

  echo "[info] Syncing models cache to secondary Spark"
  rsync -az --delete "${REPO_ROOT}/models/" "${SSH_USER}@${SECOND_SPARK_IP}:${REMOTE_DIR}/models/"
}

start_remote_workers() {
  echo "[info] Building image on secondary Spark (${SECOND_SPARK_IP})"
  ssh "${SSH_USER}@${SECOND_SPARK_IP}" "cd '${REMOTE_DIR}' && DOCKER_BUILDKIT=1 docker build -t '${IMAGE_TAG}' ."
  echo "[info] Starting diffusion workers on secondary Spark"
  ssh "${SSH_USER}@${SECOND_SPARK_IP}" "set -euo pipefail; \
    cd '${REMOTE_DIR}'; \
    NUM_GPUS=\$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l); \
    if [[ \${NUM_GPUS} -eq 0 ]]; then echo '[error] No GPUs detected on secondary Spark' >&2; exit 1; fi; \
    for i in \$(seq 0 $((NUM_GPUS - 1))); do \
      PORT=$(( ${BASE_PORT} + i )); \
      NAME=flux-worker-\${i}; \
      docker rm -f \"\${NAME}\" >/dev/null 2>&1 || true; \
      docker run -d --gpus \"device=\${i}\" --name \"\${NAME}\" \
        -e ROLE=diffusion \
        -e CLUSTER_MODE=single \
        -e AI_MOOD_MIRROR_DIFFUSION_DEVICE=\"cuda:\${i}\" \
        -e PORT=\"\${PORT}\" \
        -p \"\${PORT}:9000\" \
        -v '${REMOTE_DIR}/models:/models' \
        '${IMAGE_TAG}' \
        ai-mood-mirror-diffusion --host 0.0.0.0 --port 9000; \
    done"
}

launch_primary() {
  echo "[info] Building image on primary Spark"
  DOCKER_BUILDKIT=1 docker build -t "${IMAGE_TAG}" "${REPO_ROOT}"
  echo "[info] Launching primary application (vision/web) on port ${PRIMARY_PORT}"
  CLUSTER_MODE=dual SECOND_SPARK_IP="${SECOND_SPARK_IP}" SECOND_SPARK_SSH_USER="${SSH_USER}" ROLE=vision DEFAULT_MODE=dual \
    IMAGE_TAG="${IMAGE_TAG}" PORT="${PRIMARY_PORT}" docker compose -f "${REPO_ROOT}/docker-compose.yml" up -d ai-mood-mirror
}

echo "[step] Verifying SSH connectivity"
check_ssh "${SECOND_SPARK_IP}"

echo "[step] Verifying remote tooling"
check_remote_tools

echo "[step] Syncing repository to secondary Spark"
sync_remote_repo

echo "[step] Syncing model cache"
sync_models

echo "[step] Deploying diffusion workers"
start_remote_workers

echo "[step] Deploying primary application"
launch_primary

echo "[ok] Dual-Spark deployment ready"
echo "     Primary UI: http://localhost:${PRIMARY_PORT}"
echo "     Secondary diffusion base port: ${BASE_PORT} (one worker per GPU)"
