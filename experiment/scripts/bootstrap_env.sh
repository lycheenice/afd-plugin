#!/usr/bin/env bash
# Bootstrap the AFD experiment environment on gpu-host.
#
# Requires: docker access (run as root, or as a user in the docker group).
# Idempotent: safe to re-run.
#
# Steps:
#   0. (optional) free GPUs from sglang containers
#   1. pull vLLM 0.19.1 image (via mirror)
#   2. download DeepSeek-V2-Lite (via hf-mirror) if missing
#   3. sync afd-plugin source to /data1/afd-plugin
#   4. (re)create the afd-exp container
#   5. install afd-plugin inside the container
set -euo pipefail

IMAGE_REPO="docker.1ms.run/vllm/vllm-openai:v0.19.1"
IMAGE_TAG="vllm/vllm-openai:v0.19.1"
MODEL_DIR="/data1/models/DeepSeek-V2-Lite"
CODE_SRC="/home/lychee/mycode/afd-plugin"
CODE_DST="/data1/afd-plugin"
CONTAINER="afd-exp"

echo "=== [0] GPU state ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true

if [[ "${CLEAN_SGLANG:-0}" == "1" ]]; then
  echo "=== [0a] Removing sglang containers (CLEAN_SGLANG=1) ==="
  # Stop and remove any containers whose processes hold the GPUs.
  mapfile -t containers < <(docker ps --format '{{.Names}}' || true)
  for c in "${containers[@]}"; do
    if docker inspect "$c" --format '{{.Config.Image}}' 2>/dev/null | grep -qi sglang; then
      echo "  stopping $c"
      docker kill "$c" 2>/dev/null || true
      docker rm -f "$c" 2>/dev/null || true
    fi
  done
  sleep 5
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true
fi

echo "=== [1] vLLM image ==="
if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -qx "${IMAGE_TAG}"; then
  docker pull "${IMAGE_REPO}"
  docker tag "${IMAGE_REPO}" "${IMAGE_TAG}"
else
  echo "  ${IMAGE_TAG} already present"
fi

echo "=== [2] Model ==="
if [[ ! -d "${MODEL_DIR}" ]] || [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "  downloading DeepSeek-V2-Lite via hf-mirror.com ..."
  HF_ENDPOINT=https://hf-mirror.com \
    huggingface-cli download deepseek-ai/DeepSeek-V2-Lite \
      --local-dir "${MODEL_DIR}"
else
  echo "  ${MODEL_DIR} already present"
fi

echo "=== [3] Sync source ==="
rsync -a --delete \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude '.venv' \
  "${CODE_SRC}/" "${CODE_DST}/"
echo "  synced $(git -C "${CODE_SRC}" rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo "=== [4] Container ==="
docker rm -f "${CONTAINER}" 2>/dev/null || true
docker run -d --name "${CONTAINER}" \
  --gpus all --network host \
  -v /data1/models:/models \
  -v "${CODE_DST}:/workspace/afd-plugin" \
  -e SETUPTOOLS_SCM_PRETEND_VERSION=0.0.1 \
  --entrypoint sleep \
  "${IMAGE_TAG}" infinity
echo "  ${CONTAINER} started"

echo "=== [5] Install afd-plugin ==="
docker exec "${CONTAINER}" pip install -e /workspace/afd-plugin --no-deps --no-build-isolation
docker exec "${CONTAINER}" python -c "import afd_plugin; print('afd_plugin OK')"

echo "=== [6] Cleanup stale vLLM procs ==="
docker exec "${CONTAINER}" bash -c 'pkill -9 -f vllm 2>/dev/null || true'
sleep 3
docker exec "${CONTAINER}" nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "=== Done. To run experiments: ==="
echo "  docker exec -e PYTHONPATH=/workspace/afd-plugin -e VLLM_LOGGING_LEVEL=INFO ${CONTAINER} \\"
echo "    python /workspace/afd-plugin/experiment/scripts/run_perf_tests.py"
