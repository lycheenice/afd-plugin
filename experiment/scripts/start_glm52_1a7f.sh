#!/bin/bash
# GLM-5.2-FP8 AFD 1A7F fan-out TP1/TP7 eager smoke test
# Attention: 1 GPU (TP=1), FFN: 7 GPUs (TP=7) — fan-out ratio=7
# FFN 7×143GB=1001GB >> ~634GB MoE weights, should fit.
set -x
MODEL=/models/GLM-5.2-FP8
LOG=/workspace/afd-plugin/experiment/logs
mkdir -p "$LOG"
AFD_PORT=6251
API_PORT=18000
FFN_PORT=18001
COMMON="--enforce-eager --max-model-len 8192 --max-num-seqs 16 --gpu-memory-utilization 0.92 --trust-remote-code --max-num-batched-tokens 256"
ATT_CFG='{"afd":{"role":"attention","connector":"P2pNcclAFDConnector","host":"127.0.0.1","port":6251,"num_attention_ranks":1,"num_ffn_ranks":7}}'
FFN_CFG='{"afd":{"role":"ffn","connector":"P2pNcclAFDConnector","host":"127.0.0.1","port":6251,"num_attention_ranks":1,"num_ffn_ranks":7}}'

CUDA_VISIBLE_DEVICES=0 VLLM_PLUGINS=afd PYTHONPATH=/workspace/afd-plugin \
  vllm serve "$MODEL" --served-model-name glm52-afd-attn \
  --data-parallel-size 1 --tensor-parallel-size 1 \
  $COMMON --host 127.0.0.1 --port $API_PORT \
  --additional-config "$ATT_CFG" > "$LOG/glm52_1a7f_attn.log" 2>&1 &
ATTN_PID=$!

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 VLLM_PLUGINS=afd PYTHONPATH=/workspace/afd-plugin \
  vllm serve "$MODEL" --served-model-name glm52-afd-ffn \
  --data-parallel-size 1 --tensor-parallel-size 7 --enable-expert-parallel \
  $COMMON --host 127.0.0.1 --port $FFN_PORT \
  --additional-config "$FFN_CFG" > "$LOG/glm52_1a7f_ffn.log" 2>&1 &
FFN_PID=$!

echo "$ATTN_PID" > "$LOG/glm52_1a7f_attn.pid"
echo "$FFN_PID" > "$LOG/glm52_1a7f_ffn.pid"
wait
