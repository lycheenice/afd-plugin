# AFD 插件实验 — 第二阶段排期

> 基于 2026-07-25 第一阶段实验结果制定
> 预计总耗时: ~6 小时
> 实验机器: gpu-host (8× NVIDIA H200 143GB)
> 容器: afd-exp (vllm/vllm-openai:v0.19.1)

---

## 第一阶段回顾

第一阶段完成了功能正确性验证 (F-01~F-05 全部 PASS) 和吞吐性能基准 (P-01~P-05)，
核心发现:

- AFD 1A1F eager 比原生 vLLM 吞吐提升 1.22x, decode 延迟降低 20%
- CUDA Graph 在 decode 阶段有 10% 加速, 但 benchmark 热身不足导致数据失真
- DBO 在 AFD P2P 架构下负优化 50% (TPOT 翻倍)
- 2A2F 拓扑存在端口冲突和 worker 崩溃, 性能仅为 1A1F 的 0.41x
- 并发扩展优秀: 1→128 并发, 吞吐 173→15335 tok/s

---

## 第二阶段任务

### 任务 1: 修复 2A2F 稳定性 (~2h)

**问题**: P-04 的 2A2F (DP2TP1) 测试中, FFN worker 崩溃
(`RuntimeError: AFD FFN worker loop failed`), Attention 侧两个 APIServer 竞争同一端口
(`port 18000 is used by process VLLM::APIServer_1`), 导致 TTFT 4.5s, 吞吐仅 1094 tok/s。

**排查步骤**:

1. **端口冲突分析** (~30min)
   - 阅读 vLLM 0.19.1 `DataParallelLauncher` 源码, 理解 DP>1 时端口分配逻辑
   - 确认 AFD attention 侧是否需要为每个 DP rank 指定不同 API port
   - 对比官方 recipe `2a2f_eager_dbo_dp2tp1.sh`: recipe 中两侧都用 `--port 18305`,
     说明 vLLM DP 模式内部应该自动管理端口分配, 可能是 AFD 插件干扰了该逻辑

2. **Worker 崩溃分析** (~45min)
   - 拉取 `experiment/logs/p04_2a2f_ffn.log` 中完整的 traceback
   - 检查 AFD FFN worker 在 DP>1 时的 connector 初始化: 是否所有 FFN rank 都正确
     注册到 NCCL group
   - 对比 `P2pNcclAFDConnector` 中 `num_attention_ranks=2, num_ffn_ranks=2` 的 rank 分配逻辑

3. **修复并重测** (~45min)
   - 尝试方案 A: 为 attention 侧两个 DP rank 分别指定不同端口
   - 尝试方案 B: 使用 `--data-parallel-api-port` 或相关 vLLM 参数
   - 修复后重跑 P-04, 目标: 2A2F 吞吐 ≥ 1A1F 的 1.5x

**关键文件**:
- `experiment/scripts/run_perf_tests.py` — `test_p04_topology()` 函数
- `experiment/logs/p04_2a2f_ffn.log` — FFN worker 崩溃日志
- `experiment/logs/p04_2a2f_attention.log` — Attention 端口冲突日志
- `recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_colocation/2a2f_eager_dbo_dp2tp1.sh`
- vLLM 源码: `vllm/data_parallel/` (DataParallelLauncher)

---

### 任务 2: 公平基线补测 (~30min)

**问题**: P-01 对比不公平 — AFD 1A1F 使用 2 张 GPU, 而原生 vLLM 仅用 1 张 GPU。
需要补充原生 vLLM 2 GPU (DP=2) 基线, 形成公平的 "2 GPU vs 2 GPU" 对比。

**步骤**:

1. 启动原生 vLLM with DP=2 (GPU 0,1), eager 模式
2. 运行相同 benchmark 参数 (128 prompts, 512 in / 128 out, conc=32, rate=inf)
3. 结果命名 `p01_native_dp2.json`
4. 更新报告中的 P-01 对比表, 增加 "Native DP2 (2 GPU)" 列

**预期**: AFD 1A1F 仍应有优势, 因为 AFD 是计算流水线并行 (Attn/FFN 重叠),
而 native DP2 只是数据并行 (两个独立副本), 无流水线重叠。

**关键文件**:
- `experiment/scripts/run_perf_tests.py` — `start_native()` 需支持 DP 模式

---

### 任务 3: Graph 模式重测 (~1h)

**问题**: P-02 的 graph 模式 TTFT 4858ms (eager 仅 247ms), 原因是 warmup 不足:
仅发了 1 个 max_tokens=4 的请求, 未触发所有 batch size 的 graph capture,
导致 128 个 benchmark 请求在运行时触发 capture 开销。

**步骤**:

1. **修改 benchmark warmup** (~20min)
   - 在 `run_bench()` 中增加 `--num-warmups` 参数 (vllm bench serve 已支持)
   - warmup 数量设为 max_concurrency (即 32), 确保所有 batch size 被预热
   - 或在 server 启动后手动发送不同 batch size 的预热请求

2. **重跑 P-02** (~20min)
   - eager vs graph, 参数同第一阶段
   - 记录 graph capture 阶段的耗时 (从 server 日志中提取)

3. **补充 graph + DBO 组合测试** (~20min)
   - 使用官方 recipe `2a2f_graph_dbo_dp2tp1.sh` 的配置
   - 1A1F graph_dbo 模式, 验证 graph 和 DBO 是否兼容

**预期**: 修正 warmup 后 graph TTFT 应回到 ~300ms 量级, throughput 应接近或超过 eager。

**关键文件**:
- `experiment/scripts/run_perf_tests.py` — `run_bench()` 和 `test_p02_graph_vs_eager()`

---

### 任务 4: DBO 根因分析 (~1.5h)

**问题**: P-03 中 DBO 导致 TPOT 翻倍 (59.3ms vs 29.7ms), 吞吐减半 (2596 vs 5038 tok/s)。
DBO 设计目标是重叠 Attention 和 FFN 的计算, 在 AFD 架构下反而劣化。

**假设**:
- H1: DBO 将 decode 拆成 2 个 micro-batch, 每个需独立 NCCL P2P 同步, 通信开销翻倍
- H2: `--dbo-decode-token-threshold 2` 配置过小, 导致所有 decode step 都进入 DBO 路径
- H3: DBO 在 AFD 下有额外的调度开销 (Python 层 if/else + 同步等待)

**排查步骤**:

1. **日志分析** (~30min)
   - 对比 DBO 和 non-DBO 的 server 日志, 查找 DBO 相关的 timing/warning
   - 检查 AFD connector 日志中 send/recv 调用频率是否翻倍
   - 搜索 AFD 源码中 DBO 相关的 patch 和调度逻辑

2. **参数调优实验** (~30min)
   - 提高 `--dbo-decode-token-threshold` 到 8/16/32, 减少 DBO 触发频率
   - 提高 `--dbo-prefill-token-threshold` 到 64/128, 观察 prefill DBO 行为
   - 在 concurrency=64/128 下重测 DBO, 验证是否大 batch 下有收益

3. **源码追踪** (~30min)
   - 查找 AFD 插件中 DBO 相关的 patch 函数
   - 理解 DBO 在 AFD 架构下的 micro-batch 调度路径
   - 确认是否存在不必要的同步或 Python 层开销

**关键文件**:
- `experiment/logs/p03_dbo_attn_attention.log`
- `experiment/logs/p03_dbo_ffn_ffn.log`
- AFD 源码中 `dbo` 相关的 patch: `afd_plugin/compat/patches/`

---

### 任务 5: 报告完善 (~1h)

**步骤**:

1. 更新 `EXPERIMENT_REPORT.md` 中 P-01~P-04 的结果表格
2. 补充第二阶段的新数据 (2A2F 修复后、公平基线、Graph 重测、DBO 分析)
3. 更新 "建议与后续方向" 部分
4. 如有重大发现, 更新 "核心发现" 部分

---

## 时间分配

| 任务 | 耗时 | 累计 |
|------|------|------|
| 1. 修复 2A2F 稳定性 | 2.0h | 2.0h |
| 2. 公平基线补测 | 0.5h | 2.5h |
| 3. Graph 模式重测 | 1.0h | 3.5h |
| 4. DBO 根因分析 | 1.5h | 5.0h |
| 5. 报告完善 | 1.0h | 6.0h |

---

## 环境准备清单

下个会话启动时需要:

```bash
# 1. 确认容器运行中
ssh root@gpu-host 'docker ps | grep afd-exp'

# 2. 如容器已停, 重启并安装
ssh root@gpu-host 'docker restart afd-exp && sleep 5'
ssh root@gpu-host 'docker exec afd-exp pip install -e /workspace/afd-plugin --no-deps --no-build-isolation'

# 3. 清理 GPU
ssh root@gpu-host 'docker exec afd-exp pkill -9 -f vllm 2>/dev/null; sleep 3'
ssh root@gpu-host 'docker exec afd-exp nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'

# 4. 同步最新脚本
rsync -avz /home/lychee/mycode/afd-plugin/experiment/scripts/ \
  root@gpu-host:/data1/afd-plugin/experiment/scripts/
```

## 当前结果文件位置

- 实验报告: `experiment/EXPERIMENT_REPORT.md`
- 性能数据: `experiment/results/performance_results.json`
- 功能数据: `experiment/results/functional_results.json`
- 服务器日志: `experiment/logs/` (p01~p05, f01~f05)
- gpu-host 容器: `afd-exp` (vllm/vllm-openai:v0.19.1)
- gpu-host 代码: `/data1/afd-plugin`
- gpu-host 模型: `/data1/models/DeepSeek-V2-Lite`
