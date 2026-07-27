#!/bin/bash
# GLM-5.2-W4AFP8 AFD 4A4F 性能基准扫描（随时可跑）
#
# 前提：4A4F 服务已就绪（先跑 start_glm52_w4afp8_4a4f.sh 并等两侧
#       "Application startup complete"），attention 侧 API 在 127.0.0.1:18000。
#
# 重要说明：当前输出为乱码（精度问题未解决，见 SESSION_STATUS.md §5）。
# 但 forward pass 完整执行、token 正常生成到 max_tokens，因此
# 吞吐/延迟/TTFT/TPOT 等**性能指标有效**——它们与输出内容无关。
#
# 用法：docker exec afd-exp bash /workspace/afd-plugin/experiment/scripts/bench_glm52_w4afp8_4a4f.sh
set -u
PORT=${PORT:-18000}
MODEL_NAME=${MODEL_NAME:-glm52-afd-attn}
RESULT_DIR=/workspace/afd-plugin/experiment/results
LOG_DIR=/workspace/afd-plugin/experiment/logs
mkdir -p "$RESULT_DIR" "$LOG_DIR"
STAMP=${STAMP:-run1}

echo "=== warmup ==="
curl -s "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_NAME\",\"prompt\":\"Hello\",\"max_tokens\":8,\"temperature\":0}" >/dev/null || {
    echo "ERROR: server not reachable on :$PORT — start 4A4F first"; exit 1; }

# 扫描矩阵：(input_len, output_len, max_concurrency)
# 覆盖 prefill 重 / decode 重 / 混合，以及并发梯度
SWEEP=(
  "1024 128 1"
  "1024 128 8"
  "1024 128 16"
  "1024 128 32"
  "4096 256 8"
  "512  512 16"
  "256  1024 8"
)

SUMMARY="$RESULT_DIR/perf_glm52_w4afp8_4a4f_${STAMP}_summary.md"
echo "# GLM-5.2-W4AFP8 AFD 4A4F 性能扫描 ($STAMP)" > "$SUMMARY"
echo "" >> "$SUMMARY"
echo "| in_len | out_len | concurrency | throughput(tok/s) | req/s | TTFT_p50(ms) | TPOT_p50(ms) | e2e_p50(ms) |" >> "$SUMMARY"
echo "|--------|---------|-------------|-------------------|-------|--------------|--------------|-------------|" >> "$SUMMARY"

for row in "${SWEEP[@]}"; do
  read -r ILEN OLEN CONC <<< "$row"
  NPROMPTS=$(( CONC * 8 )); [ "$NPROMPTS" -lt 16 ] && NPROMPTS=16
  RF="perf_glm52_w4afp8_4a4f_${STAMP}_i${ILEN}_o${OLEN}_c${CONC}.json"
  echo "=== bench in=$ILEN out=$OLEN conc=$CONC nprompts=$NPROMPTS ==="
  vllm bench serve \
    --host 127.0.0.1 --port "$PORT" \
    --model "$MODEL_NAME" --endpoint /v1/completions \
    --dataset-name random \
    --num-prompts "$NPROMPTS" \
    --max-concurrency "$CONC" \
    --random-input-len "$ILEN" \
    --random-output-len "$OLEN" \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-dir "$RESULT_DIR" --result-filename "$RF" \
    2>&1 | tee "$LOG_DIR/${RF%.json}.log"

  # 从 JSON 提取关键指标进 summary（python3 容器内可用）
  python3 - "$RESULT_DIR/$RF" "$ILEN" "$OLEN" "$CONC" >> "$SUMMARY" <<'PY'
import json,sys
p,ilen,olen,conc=sys.argv[1:5]
try:
    d=json.load(open(p))
    def g(k): return d.get(k,"")
    tput=round(d.get("output_throughput",0),1)
    reqs=round(d.get("request_throughput",0),3)
    ttft=round(d.get("median_ttft_ms",0),1)
    tpot=round(d.get("median_tpot_ms",0),2)
    e2e=round(d.get("median_e2el_ms",0),1)
    print(f"| {ilen} | {olen} | {conc} | {tput} | {reqs} | {ttft} | {tpot} | {e2e} |")
except Exception as e:
    print(f"| {ilen} | {olen} | {conc} | ERR:{e} | | | | |")
PY
done

echo ""; echo "=== summary written to $SUMMARY ==="
cat "$SUMMARY"
