#!/bin/bash
# 性能基准测试脚本
# 用法: bench_serve.sh <port> <model_name> [num_prompts] [request_rate] [max_concurrency] [input_len] [output_len]
set -e

PORT=$1
MODEL_NAME=$2
NUM_PROMPTS=${3:-256}
REQUEST_RATE=${4:-5}
MAX_CONCURRENCY=${5:-32}
INPUT_LEN=${6:-1024}
OUTPUT_LEN=${7:-128}
RESULT_DIR="/workspace/afd-plugin/experiment/results"
RESULT_FILE=${RESULT_FILE:-bench_${MODEL_NAME}_${NUM_PROMPTS}_${REQUEST_RATE}.json}
mkdir -p "$RESULT_DIR"

echo "=== Benchmark: port=$PORT model=$MODEL_NAME ==="
echo "  prompts=$NUM_PROMPTS rate=$REQUEST_RATE concurrency=$MAX_CONCURRENCY"
echo "  input_len=$INPUT_LEN output_len=$OUTPUT_LEN"

# 先发一个热身请求
echo "Warmup request..."
curl -s "http://127.0.0.1:$PORT/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_NAME\",\"prompt\":\"Hello\",\"max_tokens\":4,\"temperature\":0}" > /dev/null

# 运行 vllm bench serve
vllm bench serve \
  --host 127.0.0.1 \
  --port $PORT \
  --model "$MODEL_NAME" \
  --endpoint /v1/completions \
  --num-prompts $NUM_PROMPTS \
  --request-rate $REQUEST_RATE \
  --max-concurrency $MAX_CONCURRENCY \
  --input-len $INPUT_LEN \
  --output-len $OUTPUT_LEN \
  --output-file "$RESULT_DIR/$RESULT_FILE" 2>&1 | tee "$RESULT_DIR/${RESULT_FILE%.json}.log"

echo "Result saved to $RESULT_DIR/$RESULT_FILE"
