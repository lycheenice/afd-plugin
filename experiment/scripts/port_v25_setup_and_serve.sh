#!/bin/bash
# 在 gpu-host 上一键验证:afd-plugin(移植版)+ vLLM 0.25.0 跑 native GLM-5.2-W4AFP8。
# 用途:验证 Priority-1(单实例 GLM-5.2-W4AFP8 经插件在 0.25.0 正确)。
# 前置:gpu-host GPU 空闲(先停 SGLang);/data1/afd-plugin 已同步移植后的代码。
# 用法(gpu-host root):bash port_v25_setup_and_serve.sh
#   起服务后用 port_v25_smoke.sh 或下方 curl 冒烟。
set -x
IMAGE=docker.1ms.run/vllm/vllm-openai:v0.25.0
MODEL=/models/GLM-5.2-W4AFP8
LOG=/workspace/afd-plugin/experiment/logs/glm_w4afp8_v25_native.log

echo "=== [0] GPU 状态(需空闲)==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head

echo "=== [1] (重)建 afd-v25 容器 ==="
docker rm -f afd-v25 2>/dev/null
docker run -d --name afd-v25 --gpus all --network host --ipc=host --shm-size=32g \
  -v /data1/models:/models -v /data1/afd-plugin:/workspace/afd-plugin \
  -e SETUPTOOLS_SCM_PRETEND_VERSION=0.0.1 --entrypoint sleep "$IMAGE" infinity
sleep 3

echo "=== [2] 安装插件(editable)==="
docker exec afd-v25 pip install -e /workspace/afd-plugin --no-deps --no-build-isolation 2>&1 | tail -2

echo "=== [3] 验证 w4afp8 量化注册 + native load_weights 补丁 ==="
docker exec -e VLLM_PLUGINS=afd -e PYTHONPATH=/workspace/afd-plugin afd-v25 python3 -c '
import afd_plugin; afd_plugin.register_afd()
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
print("w4afp8 registered:", "w4afp8" in QUANTIZATION_METHODS)
from vllm.model_executor.models import deepseek_v2
print("native load_weights patched:", getattr(deepseek_v2.DeepseekV2ForCausalLM.load_weights, "_afd_w4afp8_wrapped", False))
' 2>&1 | tail -4

echo "=== [4] 起 native GLM-5.2-W4AFP8 服务(detached)==="
docker exec -d -e VLLM_PLUGINS=afd -e PYTHONPATH=/workspace/afd-plugin afd-v25 bash -c \
  "vllm serve $MODEL --served-model-name glm-w4-v25 --tensor-parallel-size 8 --enable-expert-parallel --enforce-eager --max-model-len 8192 --max-num-seqs 16 --gpu-memory-utilization 0.90 --trust-remote-code --max-num-batched-tokens 2048 --host 127.0.0.1 --port 18000 > $LOG 2>&1"
echo "服务启动中。日志:$LOG"
echo "就绪判据:curl -s localhost:18000/v1/models | grep glm-w4-v25"
echo "冒烟:bash experiment/scripts/port_v25_smoke.sh  (或见下)"
