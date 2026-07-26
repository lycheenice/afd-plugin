#!/bin/bash
# AFD 实验环境初始化脚本
# 在 vLLM 0.19.1 容器内安装 afd-plugin
set -e

echo "=== 0. 检查环境 ==="
python3 -c "import vllm; print('vLLM version:', vllm.__version__)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo "=== 1. 安装 afd-plugin ==="
cd /workspace/afd-plugin
pip install -e . --no-deps 2>&1 | tail -3

echo "=== 2. 验证安装 ==="
python3 -c "
import vllm
import afd_plugin
print('vLLM version:', vllm.__version__)
print('AFD plugin loaded OK')
from afd_plugin.config import SUPPORTED_AFD_ROLES, SUPPORTED_AFD_CONNECTORS
print('Supported roles:', SUPPORTED_AFD_ROLES)
print('Supported connectors:', SUPPORTED_AFD_CONNECTORS)
"

echo "=== 3. 验证模型路径 ==="
ls -la /models/DeepSeek-V2-Lite/config.json && echo "Model found"

echo "=== 环境初始化完成 ==="
