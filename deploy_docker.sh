#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-gpu}"
IMAGE_NAME="article-generator:${MODE}"
CONTAINER_NAME="article-generator-api-${MODE}"
PORT="${PORT:-8000}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required but not found in PATH"
  exit 1
fi

if [[ "${MODE}" == "gpu" ]]; then
  BUILD_ARGS=( -f Dockerfile.gpu -t "${IMAGE_NAME}" . )
  RUN_ARGS=( --rm --gpus all -p "${PORT}:8000" --name "${CONTAINER_NAME}" "${IMAGE_NAME}" )
elif [[ "${MODE}" == "cpu" ]]; then
  BUILD_ARGS=( -f Dockerfile -t "${IMAGE_NAME}" . )
  RUN_ARGS=( --rm -p "${PORT}:8000" --name "${CONTAINER_NAME}" "${IMAGE_NAME}" )
else
  echo "Invalid mode: ${MODE}"
  echo "Usage: ./deploy_docker.sh [gpu|cpu]"
  exit 1
fi

echo "Building image ${IMAGE_NAME}..."
docker build "${BUILD_ARGS[@]}"

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Removing existing container ${CONTAINER_NAME}..."
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting container ${CONTAINER_NAME} on port ${PORT}..."
docker run "${RUN_ARGS[@]}"
