#!/bin/bash
# 停止 AFD 服务
LOG_DIR="/workspace/afd-plugin/experiment/logs"
LOG_PREFIX=${1:-""}

if [ -n "$LOG_PREFIX" ] && [ -f "$LOG_DIR/${LOG_PREFIX}_attn.pid" ]; then
  ATTN_PID=$(cat "$LOG_DIR/${LOG_PREFIX}_attn.pid")
  FFN_PID=$(cat "$LOG_DIR/${LOG_PREFIX}_ffn.pid")
  kill $ATTN_PID $FFN_PID 2>/dev/null || true
  wait $ATTN_PID $FFN_PID 2>/dev/null || true
  echo "Stopped $LOG_PREFIX (attn=$ATTN_PID, ffn=$FFN_PID)"
else
  # 杀所有 vllm serve 进程
  pkill -f "vllm serve" 2>/dev/null || true
  echo "Stopped all vllm serve processes"
fi
sleep 2
