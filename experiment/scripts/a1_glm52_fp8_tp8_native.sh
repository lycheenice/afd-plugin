#!/bin/bash
# A1 决定性二分诊断:GLM-5.2-FP8 单实例 TP=8 native(无 AFD),greedy 冒烟。
# 目的:判定"GLM-5.2 乱码"是 W4AFP8 int4 反量化问题,还是 vLLM 0.19.1 对 GLM 架构
#      (head_dim=192 MLA/DSA)的支持问题。
#   - FP8 也乱码 → vLLM 架构支持问题(与 W4AFP8 无关)→ 升级 vLLM / 审计 MLA。
#   - FP8 正确   → 问题专属 W4AFP8 int4 专家路径 → 聚焦 w4afp8 remap/反量化。
# 在 gpu-host 的容器内跑(需 GPU 空闲)。用法:bash a1_glm52_fp8_tp8_native.sh
set -x
MODEL=/models/GLM-5.2-FP8
LOG=/workspace/afd-plugin/experiment/logs
mkdir -p "$LOG"
API_PORT=18000
LOGF="$LOG/a1_glm52_fp8_tp8_native.log"

# 清场
pkill -9 -f "[v]llm serve" 2>/dev/null; sleep 3

echo "=== A1: GLM-5.2-FP8 TP=8 native eager (no AFD) ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  vllm serve "$MODEL" --served-model-name glm52-fp8-native \
  --data-parallel-size 1 --tensor-parallel-size 8 --enable-expert-parallel \
  --enforce-eager --max-model-len 8192 --max-num-seqs 16 \
  --gpu-memory-utilization 0.92 --trust-remote-code --max-num-batched-tokens 2048 \
  --host 127.0.0.1 --port $API_PORT > "$LOGF" 2>&1 &
PID=$!
echo "$PID" > "$LOG/a1_glm52_fp8_tp8_native.pid"

# 等就绪(GLM-5.2 大模型载入慢,给 20min)
for i in $(seq 1 600); do
  if curl -s --max-time 2 "http://127.0.0.1:$API_PORT/v1/models" | grep -q "glm52-fp8-native"; then
    echo "API ready after $((i*2))s"
    break
  fi
  if ! kill -0 $PID 2>/dev/null; then
    echo "ERROR: process died"; tail -40 "$LOGF"; exit 1
  fi
  sleep 2
done

echo "=== greedy 冒烟(temp=0)==="
for P in "Hello, my name is" "The capital of France is" "1 + 1 ="; do
  echo "--- prompt: $P ---"
  curl -s --max-time 60 "http://127.0.0.1:$API_PORT/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"glm52-fp8-native\",\"prompt\":\"$P\",\"max_tokens\":32,\"temperature\":0}" \
    | python3 -c 'import sys,json; d=json.load(sys.stdin); print(repr(d["choices"][0]["text"]))' 2>&1
done
echo "=== 判读:输出连贯=FP8 OK(W4AFP8 专属问题);乱码=vLLM GLM 架构问题 ==="
