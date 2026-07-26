#!/bin/bash
# GLM-5.2-W4AFP8 AFD 4A4F TP4 eager smoke test
# Attention: GPU 0-3 (TP=4), FFN: GPU 4-7 (TP=4), P2pNcclAFDConnector port 6252
set -x
MODEL=/models/GLM-5.2-W4AFP8
LOG=/workspace/afd-plugin/experiment/logs
mkdir -p "$LOG"
AFD_PORT=6252
API_PORT=18000
FFN_PORT=18001
COMMON="--enforce-eager --max-model-len 8192 --max-num-seqs 16 --gpu-memory-utilization 0.92 --trust-remote-code --max-num-batched-tokens 256"
ATT_CFG='{"afd":{"role":"attention","connector":"P2pNcclAFDConnector","host":"127.0.0.1","port":6252,"num_attention_ranks":4,"num_ffn_ranks":4}}'
FFN_CFG='{"afd":{"role":"ffn","connector":"P2pNcclAFDConnector","host":"127.0.0.1","port":6252,"num_attention_ranks":4,"num_ffn_ranks":4}}'

CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_PLUGINS=afd PYTHONPATH=/workspace/afd-plugin \
  vllm serve "$MODEL" --served-model-name glm52-afd-attn \
  --quantization w4afp8 \
  --data-parallel-size 1 --tensor-parallel-size 4 --enable-expert-parallel \
  $COMMON --host 127.0.0.1 --port $API_PORT \
  --additional-config "$ATT_CFG" > "$LOG/glm52_w4afp8_attn.log" 2>&1 &
ATTN_PID=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_PLUGINS=afd PYTHONPATH=/workspace/afd-plugin \
  vllm serve "$MODEL" --served-model-name glm52-afd-ffn \
  --quantization w4afp8 \
  --data-parallel-size 1 --tensor-parallel-size 4 --enable-expert-parallel \
  $COMMON --host 127.0.0.1 --port $FFN_PORT \
  --additional-config "$FFN_CFG" > "$LOG/glm52_w4afp8_ffn.log" 2>&1 &
FFN_PID=$!

echo "$ATTN_PID" > "$LOG/glm52_w4afp8_attn.pid"
echo "$FFN_PID" > "$LOG/glm52_w4afp8_ffn.pid"
wait
