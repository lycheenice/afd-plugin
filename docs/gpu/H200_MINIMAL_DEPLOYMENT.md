# NV H200 最小测试部署文档（AFD / P2pNcclAFDConnector）

> 目标：在单台 **NVIDIA H200** 机器上，用最少的资源（2 张卡）把 AFD 的
> Attention-FFN 分离跑起来并验证通路。模型用 **DeepSeek-V2-Lite**（最小可用 MoE），
> 连接器用 **`P2pNcclAFDConnector`**（NV+NV，同构 GPU 路径）。
>
> 本文所有命令均对应仓库现有代码/脚本（`recipe/gpu/P2pNcclAFDConnector/`、
> `docs/gpu/NCCL_P2P_CONNECTOR_USER_GUIDE.md`、`tests/e2e/runner.py`）。
> 对应提交：`ea7c56a`。

---

## ⚠️ 0. 开始前必读：vLLM 版本问题

afd-plugin **硬性要求 `vllm==0.19.1`**（见 [pyproject.toml](../../pyproject.toml)
第 37 行 `vllm = ["vllm==0.19.1"]`，以及 `compat/vllm.py` 的运行时版本门禁）。

但当前上层目录 `/Users/anne/mycode/vllm` 检出的是 **`v0.25.0` 分支**（且带有
mooncake 等本地定制改动）。**这两个版本不兼容**，直接拿上层目录的 vLLM 安装会导致：

- `register_afd()` 里的版本校验告警；
- 更可能的是 worker/model-runner 补丁基于 0.19.1 的上游函数签名，在 0.25.0 上
  `import` 或运行时报错（AGENTS.md 明确补丁是针对固定 tag 开发的）。

**处理方式（三选一，按推荐排序）：**

1. **（推荐）另装干净的 `vllm==0.19.1`**，不要用上层目录那份。见 §2。
2. 如果一定要用上层目录的 vLLM 做 0.19.1 测试，先把它切到 0.19.1：
   ```bash
   cd /Users/anne/mycode/vllm && git fetch --tags && git checkout v0.19.1
   ```
   （会丢弃当前 v0.25.0 分支的定制改动，请先确认这份 vLLM 不是你在改的目标。）
3. 若你的真实目标是把 AFD 适配到 vLLM 0.25.0，那是一次**上游升级工作**（要按
   AGENTS.md 重新对齐所有补丁的上游函数），不属于"最小测试部署"，需单独立项。

> 下文默认你走方式 1 或 2，运行环境里 `python -c "import vllm; print(vllm.__version__)"`
> 输出 `0.19.1`。

---

## 1. 硬件与前置条件

| 项 | 要求 |
| --- | --- |
| GPU | ≥ 2 张 NVIDIA H200（最小 `1A1F` 拓扑用 2 张：1 张 Attention + 1 张 FFN） |
| 驱动/CUDA | 支持 H200 的 NVIDIA 驱动 + CUDA 运行时；`nvidia-smi` 能看到卡 |
| NCCL | 随 PyTorch/vLLM 提供；连接器用 vLLM 的 `PyNcclCommunicator` |
| Python | 3.10–3.13（`requires-python = ">=3.10,<3.14"`） |
| 包管理 | [uv](https://docs.astral.sh/uv/) |
| 模型权重 | DeepSeek-V2-Lite（HF id `deepseek-ai/DeepSeek-V2-Lite` 或本地路径） |
| 网络端口 | AFD 交会端口 `6239` 及派生端口 `6239+subgroup+1`；HTTP 端口 `18000`/`18001` 需空闲 |

> H200 单机通常 8 卡，跑最小 `1A1F` 只用 2 卡即可，其余卡可留作扩展到 `2A2F`/`4A4F`。
> 卡间走 NVLink/NVSwitch，NCCL P2P 传 hidden states 是本机 GPU-to-GPU，无需额外网络配置。

---

## 2. 安装

从仓库根目录：

```bash
cd /Users/anne/mycode/afd-plugin
```

### 2.1 安装插件 + vLLM 0.19.1（推荐方式 1）

```bash
uv sync --group dev --extra vllm
```

`--extra vllm` 会拉入 pin 死的 `vllm==0.19.1`（[pyproject.toml](../../pyproject.toml)）。
分发包名是 `vllm-afd-plugin`，Python import 名是 `afd_plugin`。

### 2.2 使用上层目录的 vLLM（方式 2）

若已按 §0 方式 2 把 `/Users/anne/mycode/vllm` 切到 `v0.19.1`：

```bash
uv sync --group dev                       # 只装插件与 dev 依赖，不带 vllm extra
uv pip install -e /Users/anne/mycode/vllm  # 用本地 0.19.1 源码安装
```

### 2.3 验证安装

```bash
uv run python -c "import vllm, afd_plugin; print('vllm', vllm.__version__); print('afd', afd_plugin.__version__)"
```

期望看到 `vllm 0.19.1`。GPU 路径不需要构建 Ascend 算子（`AFD_BUILD_ASCEND_OPS`
默认在非昇腾环境自动跳过）。

---

## 3. 最小部署（`1A1F`，2 张 H200）

拓扑：1 个 Attention rank（GPU 0）＋ 1 个 FFN rank（GPU 1）。
`num_attention_ranks = num_ffn_ranks = 1`，满足 `A ≥ F` 且 `A % F == 0`。

有两条路子：**A. 一键 e2e runner（最省事，推荐先跑通）**；**B. 手动两进程**。

### 路子 A：一键冒烟（`tests/e2e/runner.py`）

runner 会自动拉起两个 serve 进程、等待就绪、发一条 completion 请求验证，然后清理。

```bash
uv run python tests/e2e/runner.py \
  --model deepseek-ai/DeepSeek-V2-Lite \
  --device-backend gpu \
  --num-attention-ranks 1 \
  --num-ffn-ranks 1 \
  --attention-gpus 0 \
  --ffn-gpus 1 \
  --api-port-base 18000 \
  --afd-port 6239 \
  --common-vllm-arg=--trust-remote-code
```

要点：
- `--model` 可换成本地权重路径。
- `--device-backend gpu` 会设 `VLLM_PLUGINS=afd` 并默认选 `P2pNcclAFDConnector`。
- 默认 `--enforce-eager`；加 `--enable-dbo`、`--cuda-graph-full-decode-only`
  可分别开启 DBO 和 `FULL_DECODE_ONLY` CUDA graph。
- 成功标志：runner 打印出 completion 结果且退出码为 0。

### 路子 B：手动两进程（贴近生产、便于观察日志）

**进程 1 — Attention（GPU 0，HTTP 18000）：**

```bash
CUDA_VISIBLE_DEVICES=0 uv run vllm serve deepseek-ai/DeepSeek-V2-Lite \
  --served-model-name deepseek-v2-lite-afd \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --enforce-eager \
  --trust-remote-code \
  --host 127.0.0.1 --port 18000 \
  --additional-config '{"afd":{"role":"attention","connector":"P2pNcclAFDConnector","host":"127.0.0.1","port":6239,"num_attention_ranks":1,"num_ffn_ranks":1}}' \
  > attn.log 2>&1 &
```

**进程 2 — FFN（GPU 1，HTTP 18001）：**

```bash
CUDA_VISIBLE_DEVICES=1 uv run vllm serve deepseek-ai/DeepSeek-V2-Lite \
  --served-model-name deepseek-v2-lite-afd \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --enforce-eager \
  --trust-remote-code \
  --host 127.0.0.1 --port 18001 \
  --additional-config '{"afd":{"role":"ffn","connector":"P2pNcclAFDConnector","host":"127.0.0.1","port":6239,"num_attention_ranks":1,"num_ffn_ranks":1}}' \
  > ffn.log 2>&1 &
```

关键约束（来自 NCCL P2P 用户指南）：
- 两进程 **`host`/`port`/rank 数必须完全一致**，只有 `role` 和 GPU 分配不同。
- 省略 `--worker-cls`，插件按平台自动选 GPU 的 Attention/FFN worker。
- 两个 HTTP 端口（18000/18001）是 vLLM 服务端口，与 AFD 交会端口 `6239` 相互独立。
- 初始化是**集合式**的：任一进程缺席、rank 数不一致会导致交会 hang 或超时。
- Attention 与 FFN 可任意顺序启动。

---

## 4. 就绪判断与冒烟验证（仅路子 B 需要手动做）

### 4.1 等待就绪

等到 **`attn.log` 打印出 `Application startup complete`** 再发请求：

```bash
tail -f attn.log        # 看到 Application startup complete 后 Ctrl-C
```

### 4.2 发请求（只发给 Attention 侧 18000）

> ⚠️ AFD 的不变量：**请求只发给 Attention 侧 API server**。FFN 是 connector
> 驱动的；直接给 FFN 侧发 `execute_model` 会 fail-fast。

```bash
curl -s http://127.0.0.1:18000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v2-lite-afd","prompt":"San Francisco is a","max_tokens":16,"temperature":0}'
```

期望返回一段补全文本。这条链路走通即证明：Attention 算注意力 → 经 NCCL P2P 把
hidden states 发到 FFN → FFN 算 MoE 回传 → Attention 出 token。

### 4.3 关停

```bash
kill %1 %2 2>/dev/null       # 或按 PID kill 两个 vllm serve 进程
```

---

## 5. 可选：进阶变体（验证通路后再试）

单机 H200 卡多，可直接升到 recipe 里的现成脚本（需 ≥ 4 卡）：

```bash
export MODEL_PATH=/path/to/DeepSeek-V2-Lite
# 2A2F colocation（TP=2）：
bash recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_colocation/2a2f_eager_dbo_dp1tp2.sh
```

开关速查（详见 [NCCL P2P 用户指南](NCCL_P2P_CONNECTOR_USER_GUIDE.md) 与
[recipe README](../../recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/README.md)）：

| 想要 | 加/改的参数 |
| --- | --- |
| 开 DBO（双微批重叠） | `--enable-dbo --dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 12` |
| eager → CUDA graph | 去掉 `--enforce-eager`，加 `--max-cudagraph-capture-size 64 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[64]}'` |
| `2A2F`（4 卡） | 两侧都设 `num_attention_ranks=2 num_ffn_ranks=2`，Attention 用 GPU 0,1，FFN 用 GPU 2,3 |

约束提醒：GPU CUDA graph 仅 `FULL_DECODE_ONLY`；DBO + CUDA graph 仅限恰好 2 个
ubatch；`P2pNcclAFDConnector` 不支持 `async`。

---

## 6. 故障排查

| 现象 | 可能原因 / 处理 |
| --- | --- |
| 启动卡住、交会超时 | 两进程 `host/port/rank 数`不一致，或只起了一个进程；核对两条命令的 `--additional-config` 完全对称 |
| `vLLM version ... not supported` 告警/报错 | vLLM 不是 0.19.1，见 §0 |
| 端口占用 / `Address already in use` | `6239`、`6239+1`、`18000/18001` 被占；换端口或清理残留进程 |
| `import afd_plugin` 失败 | 未在 `uv run` 环境；用 `uv run python ...` 或先 `uv sync` |
| CPU 张量被拒 | 该连接器要求 hidden states 在 CUDA 上；确认 `CUDA_VISIBLE_DEVICES` 正确、模型加载到 GPU |
| 给 18001（FFN）发请求报错 | 预期行为，请求只发 Attention 侧 18000 |

日志位置：路子 B 下 `attn.log` / `ffn.log`（当前目录）。

---

## 7. 与本仓库其他文档的关系

- 连接器契约与 rank 映射细节：[NCCL_P2P_CONNECTOR_USER_GUIDE.md](NCCL_P2P_CONNECTOR_USER_GUIDE.md)
- 更多 GPU 拓扑/DBO/graph 脚本：[recipe README](../../recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/README.md)
- 代码分层与执行流：[CODE_READING_MAP.md](../CODE_READING_MAP.md)
- 平台/连接器为什么是同构（NV+NV vs HW+HW）：见阅读地图第 4 节

---

*本文为最小验证用途，未覆盖跨节点、多副本、生产级 SLO 调优。跨节点 NCCL 配置在现有
recipe 中尚未验证，按未验证对待。*
