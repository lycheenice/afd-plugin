#!/usr/bin/env bash
# Run the DeepSeek-V2-Lite vLLM 0.25.0 GPU E2E matrix on gpu-host.
#
# This script intentionally does not stop services, kill processes, recreate
# containers, install packages, or modify source. Run it only after the user
# has explicitly confirmed that the production GPU workload is stopped.

set -euo pipefail

readonly EXPECTED_VLLM_VERSION="0.25.0"
readonly MIN_GPU_COUNT=4

CONTAINER_NAME="${AFD_V025_CONTAINER:-afd-v025-validate}"
CONTAINER_REPO_ROOT="${AFD_V025_REPO_ROOT:-/workspace/afd-plugin}"
MODEL_PATH="${AFD_GPU_E2E_MODEL:-/models/DeepSeek-V2-Lite}"
GPU_LIST="${AFD_GPU_E2E_GPUS:-0,1,2,3}"
VLLM_BIN="${AFD_GPU_E2E_VLLM_BIN:-/usr/local/bin/vllm}"
VIRTUAL_ENV_PATH="${AFD_V025_E2E_VENV:-/afd-v25/e2e-venv}"
CATEGORY="all"

usage() {
    echo "Usage: $0 [--category all|features|models|accuracy]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --category)
            if [[ $# -lt 2 ]]; then
                usage >&2
                exit 2
            fi
            CATEGORY="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$CATEGORY" in
    all)
        TEST_PATH="tests/e2e"
        EXPECTED_TEST_COUNT=20
        ;;
    features)
        TEST_PATH="tests/e2e/features"
        EXPECTED_TEST_COUNT=8
        ;;
    models)
        TEST_PATH="tests/e2e/models"
        EXPECTED_TEST_COUNT=10
        ;;
    accuracy)
        TEST_PATH="tests/e2e/accuracy"
        EXPECTED_TEST_COUNT=2
        ;;
    *)
        echo "Unsupported category: $CATEGORY" >&2
        usage >&2
        exit 2
        ;;
esac
readonly TEST_PATH EXPECTED_TEST_COUNT

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RUN_ID="${AFD_DSV2_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
GIT_DIRTY_FILES="$(git -C "$REPO_ROOT" status --porcelain | wc -l | tr -d ' ')"
LOG_DIR="$REPO_ROOT/experiment/logs/dsv2_v025/$RUN_ID"
RESULT_DIR="$REPO_ROOT/experiment/results/dsv2_v025/$RUN_ID"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

MANIFEST="$RESULT_DIR/manifest.txt"
COLLECTION_LOG="$LOG_DIR/pytest-collect.log"
PYTEST_LOG="$LOG_DIR/pytest-$CATEGORY.log"
POSTFLIGHT_LOG="$LOG_DIR/postflight.log"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required; run this script on gpu-host, not the development host" >&2
    exit 1
fi

if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" != "true" ]]; then
    echo "container $CONTAINER_NAME is not running" >&2
    exit 1
fi

container_exec() {
    docker exec \
        -e PYTHONPATH="$CONTAINER_REPO_ROOT" \
        -e VLLM_PLUGINS=afd \
        -e VLLM_USE_V2_MODEL_RUNNER=0 \
        -e AFD_GPU_E2E_MODEL="$MODEL_PATH" \
        -e AFD_GPU_E2E_GPUS="$GPU_LIST" \
        -e AFD_GPU_E2E_VLLM_BIN="$VLLM_BIN" \
        -e AFD_GSM8K_LIMIT="${AFD_GSM8K_LIMIT:-}" \
        -e AFD_GSM8K_THRESHOLD="${AFD_GSM8K_THRESHOLD:-0.20}" \
        -e AFD_GSM8K_TOLERANCE="${AFD_GSM8K_TOLERANCE:-0.05}" \
        -e UV_PROJECT_ENVIRONMENT="$VIRTUAL_ENV_PATH" \
        "$CONTAINER_NAME" "$@"
}

container_exec uv --version
container_exec test -x "$VIRTUAL_ENV_PATH/bin/python"

VLLM_VERSION="$(container_exec "$VIRTUAL_ENV_PATH/bin/python" -c 'import vllm; print(vllm.__version__)')"
if [[ "$VLLM_VERSION" != "$EXPECTED_VLLM_VERSION" ]]; then
    echo "expected vLLM $EXPECTED_VLLM_VERSION, got $VLLM_VERSION" >&2
    exit 1
fi

container_exec test -d "$MODEL_PATH"
container_exec test -x "$VLLM_BIN"
container_exec "$VIRTUAL_ENV_PATH/bin/python" -c 'import afd_plugin, pytest'

if [[ "$CATEGORY" == "all" || "$CATEGORY" == "accuracy" ]]; then
    if ! container_exec "$VIRTUAL_ENV_PATH/bin/python" -c 'import lm_eval, tenacity'; then
        echo "lm_eval with API extras is required for $CATEGORY; install it standalone in the container" >&2
        exit 1
    fi
fi

GPU_COUNT="$(container_exec bash -c 'nvidia-smi -L | wc -l')"
GPU_COUNT="${GPU_COUNT//[[:space:]]/}"
if (( GPU_COUNT < MIN_GPU_COUNT )); then
    echo "the full GPU matrix requires at least $MIN_GPU_COUNT GPUs; found $GPU_COUNT" >&2
    exit 1
fi
IFS=',' read -r -a SELECTED_GPUS <<<"$GPU_LIST"
if (( ${#SELECTED_GPUS[@]} < MIN_GPU_COUNT )); then
    echo "AFD_GPU_E2E_GPUS must select at least $MIN_GPU_COUNT GPUs; got $GPU_LIST" >&2
    exit 1
fi
SELECTED_GPU_COUNT="$(
    container_exec nvidia-smi \
        -i "$GPU_LIST" \
        --query-gpu=index \
        --format=csv,noheader,nounits | wc -l
)"
SELECTED_GPU_COUNT="${SELECTED_GPU_COUNT//[[:space:]]/}"
if (( SELECTED_GPU_COUNT != ${#SELECTED_GPUS[@]} )); then
    echo "failed to resolve every selected GPU: $GPU_LIST" >&2
    exit 1
fi

{
    echo "run_id=$RUN_ID"
    echo "scope=DeepSeek-V2-Lite-only"
    echo "category=$CATEGORY"
    echo "host=$(hostname)"
    echo "container=$CONTAINER_NAME"
    echo "container_image=$(docker inspect -f '{{.Image}}' "$CONTAINER_NAME")"
    echo "repo_root=$REPO_ROOT"
    echo "container_repo_root=$CONTAINER_REPO_ROOT"
    echo "git_commit=$GIT_COMMIT"
    echo "git_dirty_files=$GIT_DIRTY_FILES"
    echo "vllm_version=$VLLM_VERSION"
    echo "virtual_env=$VIRTUAL_ENV_PATH"
    echo "python_version=$(container_exec "$VIRTUAL_ENV_PATH/bin/python" --version 2>&1)"
    echo "torch_version=$(container_exec "$VIRTUAL_ENV_PATH/bin/python" -c 'import torch; print(torch.__version__)')"
    echo "cuda_version=$(container_exec "$VIRTUAL_ENV_PATH/bin/python" -c 'import torch; print(torch.version.cuda)')"
    echo "gpu_count=$GPU_COUNT"
    echo "gpu_list=$GPU_LIST"
    echo "model_path=$MODEL_PATH"
    echo "vllm_bin=$VLLM_BIN"
    echo "model_runner_v2=disabled"
    echo "gsm8k_limit=${AFD_GSM8K_LIMIT:-full}"
    echo "gsm8k_threshold=${AFD_GSM8K_THRESHOLD:-0.20}"
    echo "gsm8k_tolerance=${AFD_GSM8K_TOLERANCE:-0.05}"
} >"$MANIFEST"

container_exec nvidia-smi \
    --query-gpu=index,name,memory.total,memory.used \
    --format=csv,noheader >>"$MANIFEST"

echo "Collecting GPU tests into $COLLECTION_LOG"
container_exec bash -c '
    set -euo pipefail
    cd "$1"
    uv run --no-sync --project "$1" python -m pytest --collect-only -q -m gpu "$2"
' bash "$CONTAINER_REPO_ROOT" "$TEST_PATH" 2>&1 | tee "$COLLECTION_LOG"

COLLECTED_TEST_COUNT="$(grep -Ec '::test_' "$COLLECTION_LOG" || true)"
if (( COLLECTED_TEST_COUNT == 0 )); then
    COLLECTED_TEST_COUNT="$(
        awk -F': ' '/: [0-9]+$/ { count += $NF } END { print count + 0 }' \
            "$COLLECTION_LOG"
    )"
fi
if [[ "$COLLECTED_TEST_COUNT" != "$EXPECTED_TEST_COUNT" ]]; then
    echo "expected $EXPECTED_TEST_COUNT GPU tests for $CATEGORY, collected $COLLECTED_TEST_COUNT" >&2
    exit 1
fi
echo "collected_test_count=$COLLECTED_TEST_COUNT" >>"$MANIFEST"

echo "Running category=$CATEGORY; output=$PYTEST_LOG"
TEST_EXIT_CODE=0
container_exec bash -c '
    set -euo pipefail
    cd "$1"
    exec uv run --no-sync --project "$1" python -m pytest -m gpu "$2"
' bash "$CONTAINER_REPO_ROOT" "$TEST_PATH" 2>&1 | tee "$PYTEST_LOG" || TEST_EXIT_CODE=$?

{
    echo "test_exit_code=$TEST_EXIT_CODE"
    echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >>"$MANIFEST"

{
    echo "GPU compute applications after pytest:"
    container_exec nvidia-smi \
        --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
        --format=csv,noheader || true
    echo "AFD/vLLM processes after pytest:"
    container_exec bash -c \
        'pgrep -af "vllm serve|VLLM::EngineCore|VLLM::APIServer" || true'
    echo "AFD test ports after pytest:"
    container_exec bash -c \
        'ss -ltnp 2>/dev/null | grep -E ":(18[0-9]{3}|19[0-9]{3}|6[2-4][0-9]{2})\\b" || true'
} >"$POSTFLIGHT_LOG"

echo "manifest=$MANIFEST"
echo "postflight=$POSTFLIGHT_LOG"
exit "$TEST_EXIT_CODE"
