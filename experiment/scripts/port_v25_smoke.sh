#!/bin/bash
# GLM-5.2-W4AFP8 (0.25.0 + 移植插件) greedy 冒烟 + 判读。在 gpu-host root 跑。
# 需先 port_v25_setup_and_serve.sh 起服务且就绪。
API=http://127.0.0.1:18000/v1/completions
echo "=== 就绪检查 ==="
docker exec afd-v25 bash -c 'for i in $(seq 1 900); do curl -s --max-time 2 http://127.0.0.1:18000/v1/models 2>/dev/null | grep -q glm-w4-v25 && { echo "ready after $((i*2))s"; break; }; sleep 2; done'
echo "=== greedy 冒烟(temp=0)==="
for P in "Hello, my name is" "The capital of France is" "1 + 1 ="; do
  echo "--- $P ---"
  docker exec afd-v25 curl -s --max-time 60 "$API" -H 'Content-Type: application/json' \
    -d "{\"model\":\"glm-w4-v25\",\"prompt\":\"$P\",\"max_tokens\":32,\"temperature\":0}" \
    | python3 -c 'import sys,json;print(repr(json.load(sys.stdin)["choices"][0]["text"]))' 2>&1
done
echo "=== 判读:输出连贯(如 France->Paris)= W4AFP8 移植成功;乱码=数值仍有问题(查 remap/dequant)==="
echo "=== 若崩,取 worker 真实异常 ==="
docker exec afd-v25 bash -c 'L=/workspace/afd-plugin/experiment/logs/glm_w4afp8_v25_native.log; awk "/WorkerProc failed to start/{f=1} f" $L | grep -iE "Error|Exception|KeyError|\.py.*line [0-9]" | tail -12'
