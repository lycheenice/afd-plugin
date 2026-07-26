#!/bin/bash
# 通用 AFD 启动脚本
# 用法: start_afd.sh <topology> <mode> <attention_gpus> <ffn_gpus> [extra_vllm_args...]
# topology: 1a1f | 2a2f
# mode: eager | graph | dbo | graph_dbo
set -e

MODEL_PATH="/models/DeepSeek-V2-Lite"
AFD_PORT=${AFD_PORT:-6239}
API_PORT=${API_PORT:-18000}
LOG_DIR="/workspace/afd-plugin/experiment/logs"
mkdir -p "$LOG_DIR"

TOPOLOGY=$1
MODE=$2
ATTN_GPUS=$3
FFN_GPUS=$4
shift 4
EXTRA_ARGS="$@"

# 根据拓扑设置 rank 数
case "$TOPOLOGY" in
  1a1f) NUM_ATTN_RANKS=1; NUM_FFN_RANKS=1 ;;
  2a2f) NUM_ATTN_RANKS=2; NUM_FFN_RANKS=2 ;;
  4a4f) NUM_ATTN_RANKS=4; NUM_FFN_RANKS=4 ;;
  *) echo "Unknown topology: $TOPOLOGY"; exit 1 ;;
esac

# 根据 mode 设置 vLLM 参数
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
  graph_dbo)
    GRAPH_ARGS="--max-num-seqs 64 --max-num-batched-tokens 64 --max-cudagraph-capture-size 64 --cudagraph-capture-sizes 64 --compilation-config {\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}"
    DBO_ARGS="--enable-dbo --dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 12"
    ;;
  *) echo "Unknown mode: $MODE"; exit 1 ;;
esac

# 计算端口
FFN_API_PORT=$((API_PORT + 1))

# 生成日志文件名
LOG_PREFIX="${TOPOLOGY}_${MODE}"

echo "=== Starting AFD: $TOPOLOGY $MODE ==="
echo "Attention GPUs: $ATTN_GPUS, FFN GPUs: $FFN_GPUS"
echo "AFD port: $AFD_PORT, API port: $API_PORT"

# 启动 FFN
echo "Starting FFN ($FFN_GPUS)..."
CUDA_VISIBLE_DEVICES=$FFN_GPUS VLLM_PLUGINS=afd PYTHONPATH=/workspace/afd-plugin \
  vllm serve "$MODEL_PATH" \
  --served-model-name deepseek-v2-lite-afd-ffn \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  $GRAPH_ARGS $DBO_ARGS $EXTRA_ARGS \
  --trust-remote-code \
  --host 127.0.0.1 --port $FFN_API_PORT \
  --additional-config '{"afd":{"role":"ffn","connector":"P2pNcclAFDConnector","host":"127.0.0.1","port":'$AFD_PORT',"num_attention_ranks":'$NUM_ATTN_RANKS',"num_ffn_ranks":'$NUM_FFN_RANKS'}}' \
  > "$LOG_DIR/${LOG_PREFIX}_ffn.log" 2>&1 &
FFN_PID=$!

# 启动 Attention
echo "Starting Attention ($ATTN_GPUS)..."
CUDA_VISIBLE_DEVICES=$ATTN_GPUS VLLM_PLUGINS=afd PYTHONPATH=/workspace/afd-plugin \
  vllm serve "$MODEL_PATH" \
  --served-model-name deepseek-v2-lite-afd-attention \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  $GRAPH_ARGS $DBO_ARGS $EXTRA_ARGS \
  --trust-remote-code \
  --host 127.0.0.1 --port $API_PORT \
  --additional-config '{"afd":{"role":"attention","connector":"P2pNcclAFDConnector","host":"127.0.0.1","port":'$AFD_PORT',"num_attention_ranks":'$NUM_ATTN_RANKS',"num_ffn_ranks":'$NUM_FFN_RANKS'}}' \
  > "$LOG_DIR/${LOG_PREFIX}_attn.log" 2>&1 &
ATTN_PID=$!

echo "$ATTN_PID" > "$LOG_DIR/${LOG_PREFIX}_attn.pid"
echo "$FFN_PID" > "$LOG_DIR/${LOG_PREFIX}_ffn.pid"

# 等待 API 就绪
echo "Waiting for API..."
for i in $(seq 1 300); do
  if curl -s --max-time 2 "http://127.0.0.1:$API_PORT/v1/models" | grep -q "deepseek"; then
    echo "API ready after ${i}s"
    echo "ATTN_PID=$ATTN_PID FFN_PID=$FFN_PID API_PORT=$API_PORT"
    exit 0
  fi
  if ! kill -0 $ATTN_PID 2>/dev/null; then
    echo "ERROR: Attention died"
    tail -30 "$LOG_DIR/${LOG_PREFIX}_attn.log"
    exit 1
  fi
  if ! kill -0 $FFN_PID 2>/dev/null; then
    echo "ERROR: FFN died"
    tail -30 "$LOG_DIR/${LOG_PREFIX}_ffn.log"
    exit 1
  fi
  sleep 2
done
echo "ERROR: API timeout"
exit 1
