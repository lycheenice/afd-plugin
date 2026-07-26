#!/bin/bash
# F-01: 基础冒烟测试 - 1A1F eager 模式
# 验证 AFD completion 请求通路
set -e

MODEL_PATH="/models/DeepSeek-V2-Lite"
AFD_PORT=${AFD_PORT:-6239}
API_PORT=${API_PORT:-18000}
LOG_DIR="/workspace/afd-plugin/experiment/logs"
mkdir -p "$LOG_DIR"

echo "=== F-01: 基础冒烟 1A1F eager ==="

# 启动 FFN (GPU 1)
echo "Starting FFN worker on GPU 1..."
CUDA_VISIBLE_DEVICES=1 VLLM_PLUGINS=afd PYTHONPATH=/workspace/afd-plugin \
  vllm serve "$MODEL_PATH" \
  --served-model-name deepseek-v2-lite-afd-ffn \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --enforce-eager \
  --trust-remote-code \
  --host 127.0.0.1 --port $((API_PORT + 1)) \
  --additional-config '{"afd":{"role":"ffn","connector":"P2pNcclAFDConnector","host":"127.0.0.1","port":'$AFD_PORT',"num_attention_ranks":1,"num_ffn_ranks":1}}' \
  > "$LOG_DIR/f01_ffn.log" 2>&1 &
FFN_PID=$!
echo "FFN PID: $FFN_PID"

# 启动 Attention (GPU 0)
echo "Starting Attention worker on GPU 0..."
CUDA_VISIBLE_DEVICES=0 VLLM_PLUGINS=afd PYTHONPATH=/workspace/afd-plugin \
  vllm serve "$MODEL_PATH" \
  --served-model-name deepseek-v2-lite-afd-attention \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --enforce-eager \
  --trust-remote-code \
  --host 127.0.0.1 --port $API_PORT \
  --additional-config '{"afd":{"role":"attention","connector":"P2pNcclAFDConnector","host":"127.0.0.1","port":'$AFD_PORT',"num_attention_ranks":1,"num_ffn_ranks":1}}' \
  > "$LOG_DIR/f01_attn.log" 2>&1 &
ATTN_PID=$!
echo "Attention PID: $ATTN_PID"

cleanup() {
  echo "Cleaning up..."
  kill $ATTN_PID $FFN_PID 2>/dev/null || true
  wait $ATTN_PID $FFN_PID 2>/dev/null || true
}
trap cleanup EXIT

# 等待 API 就绪
echo "Waiting for API to be ready..."
for i in $(seq 1 300); do
  if curl -s --max-time 2 "http://127.0.0.1:$API_PORT/v1/models" | grep -q "deepseek"; then
    echo "API ready after ${i}s"
    break
  fi
  # 检查进程是否存活
  if ! kill -0 $ATTN_PID 2>/dev/null; then
    echo "ERROR: Attention process died"
    cat "$LOG_DIR/f01_attn.log" | tail -30
    exit 1
  fi
  if ! kill -0 $FFN_PID 2>/dev/null; then
    echo "ERROR: FFN process died"
    cat "$LOG_DIR/f01_ffn.log" | tail -30
    exit 1
  fi
  sleep 2
done

# 发送测试请求
echo "Sending test request..."
RESPONSE=$(curl -s http://127.0.0.1:$API_PORT/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v2-lite-afd-attention","prompt":"San Francisco is a","max_tokens":32,"temperature":0}')

echo "Response: $RESPONSE"

# 验证响应
echo "$RESPONSE" | python3 -c "
import sys, json
resp = json.load(sys.stdin)
assert 'error' not in resp, f'Error: {resp[\"error\"]}'
assert 'choices' in resp, 'No choices in response'
choice = resp['choices'][0]
assert 'text' in choice, 'No text in choice'
text = choice['text']
assert len(text) > 0, 'Empty text'
print(f'PASS: text=\"{text[:80]}...\"')
print(f'finish_reason: {choice[\"finish_reason\"]}')
usage = resp.get('usage', {})
print(f'prompt_tokens: {usage.get(\"prompt_tokens\", \"N/A\")}')
print(f'completion_tokens: {usage.get(\"completion_tokens\", \"N/A\")}')
print(f'total_tokens: {usage.get(\"total_tokens\", \"N/A\")}')
" && echo "F-01: PASS" || echo "F-01: FAIL"
