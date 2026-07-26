# 实验环境备忘 (gpu-host)

> 本文件记录 AFD 插件性能实验的执行环境。任何新会话在跑实验前先读本文件,
> 不要把开发本机误当成实验机。

## 实验机器 (不是本机!)

| 项 | 值 |
|---|---|
| 角色 | 实验执行机 (跑 vLLM/AFD 性能测试) |
| 别名 | `gpu-host` (见开发机 `~/.ssh/config`) |
| 地址 | `REDACTED_IP` |
| 登录 | **`ssh root@gpu-host` 免密** (用 root,不要用 lychee) |
| 系统 | Ubuntu 22.04.5 |
| GPU | 8× NVIDIA H200 143GB |
| 磁盘 | `/data1` 3.5TB (代码/模型/容器卷) |

**关键**:开发本机是 `lychee@REDACTED_IP`(无 sudo、无 docker、GPU 被 sglang 占用),
**不能**直接在本机跑实验。所有 vLLM/AFD 实测都在 `gpu-host` 上以 `root` 通过 SSH 进行。

## 容器与挂载

| 项 | 值 |
|---|---|
| 容器名 | `afd-exp` (常驻, `docker restart` 即可恢复) |
| 镜像 | `vllm/vllm-openai:v0.19.1` |
| 启动方式 | `docker exec afd-exp ...` (容器以 `sleep infinity` 常驻) |
| 模型挂载 | host `/data1/models` → 容器 `/models` |
| 模型路径 | 容器内 `/models/DeepSeek-V2-Lite` |
| 代码挂载 | host `/data1/afd-plugin` → 容器 `/workspace/afd-plugin` (bind mount) |
| afd 安装 | 容器内 `pip install -e /workspace/afd-plugin --no-deps --no-build-isolation` |
| python | 容器内用 `python3` (`python` 不存在);vLLM CLI 入口 `vllm` 已就绪 |
| 环境变量 | 跑 AFD 时必带 `-e PYTHONPATH=/workspace/afd-plugin -e VLLM_PLUGINS=afd` |

## 代码同步

本机工作区 `/home/lychee/mycode/afd-plugin` 是 git 仓库(主开发地)。
改完代码后同步到 gpu-host 并重装:

```bash
cd /home/lychee/mycode/afd-plugin
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' \
  afd_plugin/ experiment/scripts/ \
  root@gpu-host:/data1/afd-plugin/   # 注意按子目录分别同步,保持 /data1 结构
# 重装(容器内,改动才生效):
ssh root@gpu-host 'docker exec afd-exp pip install -e /workspace/afd-plugin --no-deps --no-build-isolation'
```

## 跑实验

```bash
ssh root@gpu-host 'docker exec \
  -e PYTHONPATH=/workspace/afd-plugin -e VLLM_LOGGING_LEVEL=INFO \
  afd-exp python3 /workspace/afd-plugin/experiment/scripts/run_perf_tests.py'
```

清场:
```bash
ssh root@gpu-host 'docker exec afd-exp bash -c "pkill -9 -f vllm 2>/dev/null; sleep 3; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"'
```

## 结果与日志位置 (均在容器内,即 host /data1/afd-plugin/)

- 报告: `experiment/EXPERIMENT_REPORT.md`
- 性能 JSON: `experiment/results/performance_results.json`
- 单项: `experiment/results/p01_native.json` … `p06_*.json`
- 日志: `experiment/logs/` (p0x_*_attn/ffn/bench_*.log)

## 实验阶段规划

- 第一阶段(已完成): `experiment/EXPERIMENT_REPORT.md`
- 第二阶段(进行中): `experiment/PLAN_PHASE2.md`
- 环境引导脚本: `experiment/scripts/bootstrap_env.sh`
