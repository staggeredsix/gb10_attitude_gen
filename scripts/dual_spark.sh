#!/usr/bin/env bash
set -euo pipefail

# Two-node orchestration helper for the cluster-accelerated Mind Mirror demo.
# - Sets MTU on the dual 100G ports on both nodes
# - Syncs the repo to both nodes
# - Builds and launches the containers on each node
# - Runs lightweight connectivity and GPU sanity checks

VISION_HOST="${VISION_HOST:-}"
DIFFUSION_HOST="${DIFFUSION_HOST:-}"
INTERFACES="${INTERFACES:-enp175s0f0,enp175s0f1}"
MTU="${MTU:-9000}"
REMOTE_DIR="${REMOTE_DIR:-/opt/ai-mood-mirror}"
IMAGE_TAG="${IMAGE_TAG:-ai-mood-mirror:dual}"
PORT="${PORT:-8000}"
DEFAULT_MODE="${DEFAULT_MODE:-dual}"

usage() {
  cat <<USAGE
Usage: VISION_HOST=<host> DIFFUSION_HOST=<host> $0

Required env vars:
  VISION_HOST       Hostname or IP for the vision node (DGX #1)
  DIFFUSION_HOST    Hostname or IP for the diffusion node (DGX #2)

Optional env vars:
  INTERFACES        Comma-separated NIC names to configure for 2x100G (default: enp175s0f0,enp175s0f1)
  MTU               MTU to apply to the fabric interfaces (default: 9000)
  REMOTE_DIR        Path on the remote hosts to deploy the repo (default: /opt/ai-mood-mirror)
  IMAGE_TAG         Docker image tag to use/build (default: ai-mood-mirror:dual)
  PORT              Port to expose the web UI on (default: 8000)
  DEFAULT_MODE      Default inference mode surfaced in the web UI (default: dual)

Prereqs: ssh, rsync, docker, docker compose, sudo access on both hosts, NVIDIA Container Toolkit.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${VISION_HOST}" || -z "${DIFFUSION_HOST}" ]]; then
  echo "[error] VISION_HOST and DIFFUSION_HOST must be set" >&2
  usage
  exit 1
fi

require() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[error] Missing dependency: $1" >&2
    exit 1
  fi
}

require ssh
require rsync
require docker
require docker compose

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IFS=',' read -ra FABRIC_IFS <<<"${INTERFACES}"

run_remote() {
  local host="$1"; shift
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$host" "$@"
}

configure_nics() {
  local host="$1"
  echo "[info] Configuring fabric on ${host} (${INTERFACES})"
  run_remote "$host" "sudo true"
  for nic in "${FABRIC_IFS[@]}"; do
    run_remote "$host" "sudo ip link set \"${nic}\" up && sudo ip link set \"${nic}\" mtu ${MTU} && ethtool \"${nic}\" | grep -E 'Speed|Link detected'"
  done
}

sync_repo() {
  local host="$1"
  echo "[info] Syncing repo to ${host}:${REMOTE_DIR}"
  run_remote "$host" "mkdir -p ${REMOTE_DIR}"
  rsync -az --delete "${REPO_ROOT}/" "${host}:${REMOTE_DIR}/"
}

build_and_launch() {
  local host="$1" role="$2"
  echo "[info] Building image on ${host} (${role})"
  run_remote "$host" "cd ${REMOTE_DIR} && DOCKER_BUILDKIT=1 IMAGE_TAG=${IMAGE_TAG} docker build -t ${IMAGE_TAG} ."
  echo "[info] Starting service on ${host} (${role})"
  run_remote "$host" "cd ${REMOTE_DIR} && IMAGE_TAG=${IMAGE_TAG} PORT=${PORT} ROLE=${role} DEFAULT_MODE=${DEFAULT_MODE} docker compose up -d --force-recreate"
}

sanity_checks() {
  echo "[info] Running GPU + connectivity checks"
  run_remote "$VISION_HOST" "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" || echo "[warn] Unable to query GPU on ${VISION_HOST}" >&2
  run_remote "$DIFFUSION_HOST" "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" || echo "[warn] Unable to query GPU on ${DIFFUSION_HOST}" >&2
  run_remote "$VISION_HOST" "nc -zv ${DIFFUSION_HOST} ${PORT}" || echo "[warn] TCP connectivity check failed vision->diffusion on port ${PORT}" >&2
  run_remote "$DIFFUSION_HOST" "nc -zv ${VISION_HOST} ${PORT}" || echo "[warn] TCP connectivity check failed diffusion->vision on port ${PORT}" >&2
}

echo "[step] Configuring dual 100G fabric"
configure_nics "${VISION_HOST}"
configure_nics "${DIFFUSION_HOST}"

echo "[step] Syncing repo to both nodes"
sync_repo "${VISION_HOST}"
sync_repo "${DIFFUSION_HOST}"

echo "[step] Building + launching containers"
build_and_launch "${VISION_HOST}" "vision"
build_and_launch "${DIFFUSION_HOST}" "diffusion"

echo "[step] Running sanity checks"
sanity_checks

echo "[ok] Dual-node stack deployed"
echo "     Vision host:    ${VISION_HOST}" 
echo "     Diffusion host: ${DIFFUSION_HOST}" 
echo "     Web UI:         http://${DIFFUSION_HOST}:${PORT}"
