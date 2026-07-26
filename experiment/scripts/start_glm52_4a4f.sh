#!/bin/bash
# GLM-5.2-FP8 AFD 4A4F TP4 eager 冒烟启动脚本
# 用法: 在 afd-exp 容器内执行 (后台)
set -x
MODEL=/models/GLM-5.2-FP8
LOG=/workspace/afd-plugin/experiment/logs
mkdir -p "$LOG"
AFD_PORT=6250
API_PORT=18000
FFN_PORT=18001
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--enforce-eager --max-model-len 4096 --max-num-seqs 4 --gpu-memory-utilization 0.97 --trust-remote-code --max-num-batched-tokens 64"
ATT_CFG='{"afd":{"role":"attention","connector":"P2pNcclAFDConnector","host":"127.0.0.1","port":6250,"num_attention_ranks":4,"num_ffn_ranks":4}}'
FFN_CFG='{"afd":{"role":"ffn","connector":"P2pNcclAFDConnector","host":"127.0.0.1","port":6250,"num_attention_ranks":4,"num_ffn_ranks":4}}'

CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_PLUGINS=afd PYTHONPATH=/workspace/afd-plugin \
  vllm serve "$MODEL" --served-model-name glm52-afd-attn \
  --data-parallel-size 1 --tensor-parallel-size 4 --enable-expert-parallel \
  $COMMON --host 127.0.0.1 --port $API_PORT \
  --additional-config "$ATT_CFG" > "$LOG/glm52_4a4f_attn.log" 2>&1 &
ATTN_PID=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_PLUGINS=afd PYTHONPATH=/workspace/afd-plugin \
  vllm serve "$MODEL" --served-model-name glm52-afd-ffn \
  --data-parallel-size 1 --tensor-parallel-size 4 --enable-expert-parallel \
  $COMMON --host 127.0.0.1 --port $FFN_PORT \
  --additional-config "$FFN_CFG" > "$LOG/glm52_4a4f_ffn.log" 2>&1 &
FFN_PID=$!

echo "$ATTN_PID" > "$LOG/glm52_4a4f_attn.pid"
echo "$FFN_PID" > "$LOG/glm52_4a4f_ffn.pid"
wait
