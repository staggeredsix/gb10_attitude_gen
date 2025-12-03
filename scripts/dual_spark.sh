#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $0

This script now prompts for the required details of the secondary Spark node at runtime.

Prompts:
  SECOND_SPARK_IP   IPv4 address of the secondary Spark node
  ssh_user          SSH username (defaults to $SECOND_SPARK_SSH_USER or "ubuntu")
  ssh_password      SSH password (leave blank to rely on existing SSH keys/agent)
  sudo_password     Sudo password for remote commands (leave blank to reuse SSH password or rely on passwordless sudo)

Environment variables:
  IMAGE_TAG         Docker image tag to build/run (default: ai-mood-mirror:latest)
  PORT              Primary web UI port (default: 8000)
  REMOTE_DIR        Repository location on the secondary Spark (default: /opt/ai-mood-mirror)
  BASE_PORT         Base port for diffusion workers on the secondary Spark (default: 9000)
  SECOND_SPARK_USE_SUDO  Whether to run remote commands with sudo (default: true)
  SECOND_SPARK_SUDO_PASSWORD  Password to feed to sudo on the remote host (defaults to SSH password if provided)
  REPO_URL          Repository URL to clone on the secondary Spark (default: https://github.com/google/ai-mood-mirror.git)
  TARGET_REF        Branch or tag to deploy on the secondary Spark (default: main, or the current branch if running inside a git repo)
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

prompt_for_value() {
  local prompt default_value result
  prompt="$1"
  default_value="$2"
  if [[ -n "${default_value}" ]]; then
    read -r -p "${prompt} [${default_value}]: " result || exit 1
    result=${result:-${default_value}}
  else
    while [[ -z "${result:-}" ]]; do
      read -r -p "${prompt}: " result || exit 1
    done
  fi
  printf '%s' "${result}"
}

prompt_for_secret() {
  local prompt
  prompt="$1"
  read -rs -p "${prompt} (leave blank to use SSH keys): " result || exit 1
  echo
  printf '%s' "${result}"
}

DEFAULT_SECOND_SPARK_IP="${1:-}"
USE_REMOTE_SUDO="${SECOND_SPARK_USE_SUDO:-true}"

SECOND_SPARK_IP="$(prompt_for_value "Enter secondary Spark IP" "${DEFAULT_SECOND_SPARK_IP}")"
SSH_USER="$(prompt_for_value "Enter SSH username" "${SECOND_SPARK_SSH_USER:-ubuntu}")"
SECOND_SPARK_PASSWORD="$(prompt_for_secret "Enter SSH password")"
SECOND_SPARK_SUDO_PASSWORD="${SECOND_SPARK_SUDO_PASSWORD:-}"
if [[ "${USE_REMOTE_SUDO}" == "true" && -z "${SECOND_SPARK_SUDO_PASSWORD}" ]]; then
  SECOND_SPARK_SUDO_PASSWORD="$(prompt_for_secret "Enter sudo password (leave blank to reuse SSH password or passwordless sudo)")"
  if [[ -z "${SECOND_SPARK_SUDO_PASSWORD}" ]]; then
    SECOND_SPARK_SUDO_PASSWORD="${SECOND_SPARK_PASSWORD}"
  fi
fi
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
if [[ -n "${SECOND_SPARK_PASSWORD}" ]]; then
  require sshpass
fi

SSH_OPTS="-o StrictHostKeyChecking=accept-new"
if [[ -z "${SECOND_SPARK_PASSWORD}" ]]; then
  SSH_OPTS="${SSH_OPTS} -o BatchMode=yes"
fi

ssh_wrap() {
  if [[ -n "${SECOND_SPARK_PASSWORD}" ]]; then
    SSHPASS="${SECOND_SPARK_PASSWORD}" sshpass -e ssh ${SSH_OPTS} "$@"
  else
    ssh ${SSH_OPTS} "$@"
  fi
}

rsync_wrap() {
  local ssh_cmd
  ssh_cmd="ssh ${SSH_OPTS}"
  if [[ -n "${SECOND_SPARK_PASSWORD}" ]]; then
    SSHPASS="${SECOND_SPARK_PASSWORD}" sshpass -e rsync -e "${ssh_cmd}" "$@"
  else
    rsync -e "${ssh_cmd}" "$@"
  fi
}

remote_shell() {
  local host="$1"
  local command="$2"
  local use_sudo="${3:-${USE_REMOTE_SUDO}}"
  local runner="bash -lc"
  if [[ "${use_sudo}" == "true" ]]; then
    if [[ -n "${SECOND_SPARK_SUDO_PASSWORD}" ]]; then
      runner="printf '%s\\n' \"${SECOND_SPARK_SUDO_PASSWORD}\" | sudo -S -p '' -H bash -lc"
    else
      runner="sudo -H bash -lc"
    fi
  fi
  ssh_wrap "${SSH_USER}@${host}" "${runner} \"${command}\""
}

prepare_remote_dir() {
  echo "[info] Ensuring ${REMOTE_DIR} exists and is writable by ${SSH_USER}"
  if [[ "${USE_REMOTE_SUDO}" == "true" ]]; then
    remote_shell "${SECOND_SPARK_IP}" "mkdir -p '${REMOTE_DIR}' && chown ${SSH_USER}:${SSH_USER} '${REMOTE_DIR}'"
  else
    remote_shell "${SECOND_SPARK_IP}" "mkdir -p '${REMOTE_DIR}' && [[ -w '${REMOTE_DIR}' ]]"
  fi
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_URL="${REPO_URL:-}"
TARGET_REF="${TARGET_REF:-}"

if [[ -z "${REPO_URL}" ]] && git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  REPO_URL="$(git -C "${REPO_ROOT}" config --get remote.origin.url || true)"
  TARGET_REF="${TARGET_REF:-$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)}"
fi

REPO_URL="${REPO_URL:-https://github.com/google/ai-mood-mirror.git}"
TARGET_REF="${TARGET_REF:-main}"

if [[ -z "${REPO_URL}" ]]; then
  echo "[error] Unable to determine repository URL. Set REPO_URL explicitly." >&2
  exit 1
fi

check_ssh() {
  local host="$1"
  echo "[info] Checking SSH connectivity to ${host}"
  ssh_wrap "${SSH_USER}@${host}" "echo ok" >/dev/null
}

check_remote_tools() {
  echo "[info] Verifying git + docker on secondary Spark"
  remote_shell "${SECOND_SPARK_IP}" "command -v git >/dev/null && command -v docker >/dev/null"
}

sync_remote_repo() {
  echo "[info] Syncing repository to ${SSH_USER}@${SECOND_SPARK_IP}:${REMOTE_DIR}"
  remote_shell "${SECOND_SPARK_IP}" "set -euo pipefail; \
    if [[ ! -d '${REMOTE_DIR}/.git' ]]; then git clone --branch '${TARGET_REF}' '${REPO_URL}' '${REMOTE_DIR}'; fi; \
    cd '${REMOTE_DIR}'; \
    git fetch origin '${TARGET_REF}'; \
    git checkout '${TARGET_REF}'; \
    git reset --hard \"origin/${TARGET_REF}\"" false
}

sync_models() {
  if [[ ! -d "${REPO_ROOT}/models" ]]; then
    echo "[warn] No local ./models directory found; secondary diffusion workers will download models as needed"
    remote_shell "${SECOND_SPARK_IP}" "mkdir -p '${REMOTE_DIR}/models'"
    return
  fi

  echo "[info] Syncing models cache to secondary Spark"
  rsync_wrap -az --info=progress2 --human-readable --delete "${REPO_ROOT}/models/" "${SSH_USER}@${SECOND_SPARK_IP}:${REMOTE_DIR}/models/"
}

start_remote_workers() {
  echo "[info] Building image on secondary Spark (${SECOND_SPARK_IP})"
  remote_shell "${SECOND_SPARK_IP}" "cd '${REMOTE_DIR}' && DOCKER_BUILDKIT=1 docker build -t '${IMAGE_TAG}' ."
  echo "[info] Starting diffusion workers on secondary Spark"
  remote_shell "${SECOND_SPARK_IP}" "set -euo pipefail; \
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

echo "[step] Preparing remote directory"
prepare_remote_dir

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
