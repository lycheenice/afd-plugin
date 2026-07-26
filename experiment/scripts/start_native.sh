#!/bin/bash
# 原生 vLLM 启动脚本 (无 AFD, 作为基准)
# 用法: start_native.sh <mode> <gpu_list> [extra_vllm_args...]
set -e

MODEL_PATH="/models/DeepSeek-V2-Lite"
API_PORT=${API_PORT:-18000}
LOG_DIR="/workspace/afd-plugin/experiment/logs"
mkdir -p "$LOG_DIR"

MODE=$1
GPUS=$2
shift 2
EXTRA_ARGS="$@"

GRAPH_ARGS=""
DBO_ARGS=""
case "$MODE" in
  eager) GRAPH_ARGS="--enforce-eager" ;;
  graph)
    GRAPH_ARGS="--max-num-seqs 64 --max-num-batched-tokens 64 --max-cudagraph-capture-size 64 --cudagraph-capture-sizes 64 --compilation-config {\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}"
    ;;
  dbo)
    GRAPH_ARGS="--enforce-eager"
    DBO_ARGS="--enable-dbo --dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 12"
    ;;
  *) echo "Unknown mode: $MODE"; exit 1 ;;
esac

LOG_PREFIX="native_${MODE}"

echo "=== Starting native vLLM: $MODE on GPU $GPUS ==="

CUDA_VISIBLE_DEVICES=$GPUS \
  vllm serve "$MODEL_PATH" \
  --served-model-name deepseek-v2-lite-native \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  $GRAPH_ARGS $DBO_ARGS $EXTRA_ARGS \
  --trust-remote-code \
  --host 127.0.0.1 --port $API_PORT \
  > "$LOG_DIR/${LOG_PREFIX}.log" 2>&1 &
PID=$!
echo "$PID" > "$LOG_DIR/${LOG_PREFIX}.pid"

for i in $(seq 1 300); do
  if curl -s --max-time 2 "http://127.0.0.1:$API_PORT/v1/models" | grep -q "deepseek"; then
    echo "API ready after ${i}s, PID=$PID, PORT=$API_PORT"
    exit 0
  fi
  if ! kill -0 $PID 2>/dev/null; then
    echo "ERROR: process died"
    tail -30 "$LOG_DIR/${LOG_PREFIX}.log"
    exit 1
  fi
  sleep 2
done
echo "ERROR: timeout"
exit 1
