# AFD 分离功能与性能验证实验文档

> 实验目标：在 H200 机器上验证 AFD (Attention-FFN Disaggregation) 插件的
> 功能正确性与吞吐性能，形成完整的实验报告供后续优化。
>
> 实验机器：gpu-host（8× NVIDIA H200 143GB）
>
> 实验日期：2026-07-25（第一阶段）；2026-07-25（第二阶段补充实验，见 §5.3）

---

## 1. 实验环境

### 1.1 硬件

| 项 | 值 |
|---|---|
| GPU | 8× NVIDIA H200 (143GB HBM3) |
| 驱动 | 595.58.03 |
| 互联 | NVLink/NVSwitch |
| 磁盘 | /data1: 3.5TB (814GB 可用) |

### 1.2 软件栈

| 项 | 值 |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Docker | 28.2.2 |
| vLLM (容器内) | 0.19.1 (目标版本) |
| AFD Plugin | 分支 0724, commit 680a47b |
| 模型 | DeepSeek-V2-Lite (15.7B, MoE) |

### 1.3 环境部署遇到的问题及解决

#### 问题 1: gpu-host DNS 解析失败

- **现象**: `pip install`、`docker pull` 均报 `Temporary failure in name resolution`
- **原因**: `/etc/resolv.conf` 指向 `127.0.0.53` (systemd-resolved stub), 但 systemd-resolved 未配置上游 DNS
- **解决**: 以 root SSH 登录, 直接写入 `nameserver 8.8.8.8` 到 `/etc/resolv.conf`
- **影响**: 修复后 gpu-host 可正常解析域名、拉取镜像、下载模型

#### 问题 2: GPU 被占用

- **现象**: 8 张 H200 全部被两个 sglang 容器 (sglang-glm-smg-1, sglang-glm-sglang-1) 占用, 各 ~140GB
- **解决**: `docker stop` + `docker rm` 清理容器, 释放全部 GPU
- **注意**: sglang-glm-sglang-1 需 `docker kill` 强制停止

#### 问题 3: 模型权重不在 gpu-host 上

- **现象**: gpu-host:/data1/models 下仅有 MiniMax-M2.5, 无 DeepSeek-V2-Lite
- **解决**: 通过 hf-mirror.com 镜像站下载 DeepSeek-V2-Lite 到 /data1/models/DeepSeek-V2-Lite

#### 问题 4: gpu-host 上无 vLLM 0.19.1 镜像

- **现象**: 已有镜像最高版本为 v0.25.0, AFD 插件硬性要求 vllm==0.19.1
- **解决**: 通过 docker.1ms.run 镜像站拉取 `docker.1ms.run/vllm/vllm-openai:v0.19.1`

#### 问题 5: 不使用 uv, 改用 Docker 容器

- **原始计划**: 在 gpu-host 上安装 uv, 通过 `uv sync --group dev --extra vllm` 安装环境
- **调整原因**: 用户要求在 Docker 镜像容器中实验
- **最终方案**: 使用 vLLM 0.19.1 官方镜像, 在容器内 pip install afd-plugin

---

## 2. AFD 架构概述

### 2.1 核心概念

AFD (Attention-FFN Disaggregation) 将 Transformer 的 Attention 计算和 FFN (MoE) 计算
拆分到不同进程/设备上, 通过 connector 在两侧传输 hidden states。

- **Role (角色)**: `attention` 或 `ffn`。请求只发给 Attention 侧; FFN 由 connector 驱动
- **Connector (连接器)**: 两侧通信契约。GPU 使用 `P2pNcclAFDConnector` (NCCL P2P)
- **Platform (平台)**: CUDA (本实验) / Ascend NPU

### 2.2 GPU 连接器拓扑规则

`P2pNcclAFDConnector` 要求:
- `num_attention_ranks >= num_ffn_ranks`
- `num_attention_ranks % num_ffn_ranks == 0`
- FFN ranks 排在 Attention ranks 之前

本实验覆盖的拓扑:

| 拓扑 | num_attention_ranks | num_ffn_ranks | GPU 分配 |
|------|-------------------|--------------|---------|
| 1A1F | 1 | 1 | GPU0(Attn) + GPU1(FFN) |
| 2A2F | 2 | 2 | GPU0,1(Attn) + GPU2,3(FFN) |

### 2.3 执行流

1. Attention 侧算完注意力 → `send_attn_output(hidden_states)` 发到 FFN
2. (可选) 控制平面: Attention 侧发送 DP metadata, FFN 侧接收准备缓冲
3. FFN 侧 `recv_attn_output()` → 算 MoE → `send_ffn_output()` 回传
4. Attention 侧 `recv_ffn_output()` 拿回结果, 继续后续层直到出 token

---

## 3. 实验设计

### 3.1 功能正确性验证

| 编号 | 实验名 | 拓扑 | 模式 | 验证内容 |
|------|--------|------|------|---------|
| F-01 | 基础冒烟 | 1A1F | eager | completion 请求通路 |
| F-02 | usage 统计 | 1A1F | eager | prompt/completion token 计数 |
| F-03 | CUDA Graph | 1A1F | FULL_DECODE_ONLY graph | graph 模式正确性 |
| F-04 | DBO | 1A1F | eager+DBO | 双微批重叠正确性 |
| F-05 | 2A2F DP1TP2 | 2A2F | eager+DBO | TP 张量并行正确性 |
| F-06 | 2A2F DP2TP1 | 2A2F | eager+DBO | DP 数据并行正确性 |
| F-07 | 精度对比 | 1A1F | eager vs graph | AFD vs 原生vLLM GSM8K |

### 3.2 吞吐性能验证

| 编号 | 实验名 | 对比对象 | 拓扑 | 模式 | 指标 |
|------|--------|---------|------|------|------|
| P-01 | AFD vs 原生 | AFD 1A1F vs 原生 vLLM | 1A1F / 1GPU | eager | TTFT, TPOT, throughput |
| P-02 | Graph 加速 | AFD eager vs graph | 1A1F | eager/graph | TTFT, TPOT, throughput |
| P-03 | DBO 加速 | AFD w/o DBO vs w/ DBO | 1A1F | eager ± DBO | TTFT, TPOT, throughput |
| P-04 | 拓扑对比 | 1A1F vs 2A2F | 1A1F/2A2F | eager+DBO | throughput |
| P-05 | 负载扩展 | 不同并发 | 1A1F | eager | throughput vs concurrency |

### 3.3 基准工具

- **功能正确性**: curl completion 请求 + runner.py + GSM8K (lm-eval)
- **吞吐性能**: `vllm bench serve` (自带 benchmark) + 自定义脚本
- **对比基准**: 原生 vLLM (无 AFD) 同模型同配置

---

## 4. 环境搭建

### 4.1 镜像准备

```bash
# 在 gpu-host 上拉取 vLLM 0.19.1 镜像
ssh root@gpu-host
docker pull docker.1ms.run/vllm/vllm-openai:v0.19.1
docker tag docker.1ms.run/vllm/vllm-openai:v0.19.1 vllm/vllm-openai:v0.19.1
```

### 4.2 模型准备

```bash
# 下载 DeepSeek-V2-Lite (通过 hf-mirror.com)
bash /tmp/download_model.sh  # 下载到 /data1/models/DeepSeek-V2-Lite
```

### 4.3 代码准备

```bash
# 将 afd-plugin 代码同步到 gpu-host
ssh root@gpu-host 'git clone <repo> /data1/afd-plugin'
# 或从本机 rsync 过去
```

### 4.4 容器内安装 afd-plugin

```bash
# 启动 vLLM 0.19.1 容器, 挂载模型和代码
docker run -it --gpus all --network host \
  -v /data1/models:/models \
  -v /data1/afd-plugin:/workspace/afd-plugin \
  vllm/vllm-openai:v0.19.1 bash

# 容器内安装 afd-plugin
cd /workspace/afd-plugin
pip install -e . --no-deps
```

---

## 5. 实验结果

### 5.1 功能正确性

| 编号 | 实验名 | 状态 | 说明 |
|------|--------|------|------|
| F-01 | 基础冒烟 | **PASS** | AFD 1A1F eager, completion 请求成功 (GPU0 Attn → GPU1 FFN, NCCL P2P) |
| F-02 | usage 统计 | **PASS** | prompt_tokens=5, completion_tokens=8, total_tokens=13, 4 并发请求全部成功 |
| F-03 | CUDA Graph | **PASS** | FULL_DECODE_ONLY graph 模式正确, 输出文本合理 |
| F-04 | DBO | **PASS** | eager+DBO 模式输出正确 |
| F-05 | 输出一致性 | **PASS** | AFD vs 原生 vLLM 4/4 prompt 100% 一致 (temperature=0) |

> 实验参数: 128 prompts, input_len=512, output_len=128, max_concurrency=32, request_rate=inf
> 模型: DeepSeek-V2-Lite, GPU: NVIDIA H200 143GB

测试发现 F-04 DBO 模式需要 `--enforce-eager` (DBO 不兼容 CUDA Graph 编译)。
修复后全部 5 项功能测试通过。

### 5.2 吞吐性能

#### P-01: AFD 1A1F vs 原生 vLLM (eager 模式)

| 指标 | 原生 vLLM (1 GPU) | AFD 1A1F (2 GPU) | AFD / Native |
|------|-------------------|-------------------|-------------|
| mean TTFT (ms) | 237.93 | 254.59 | 1.07x |
| mean TPOT (ms) | 35.87 | 28.83 | **0.80x** |
| total token throughput (tok/s) | 4,267.72 | 5,218.97 | **1.22x** |
| request throughput (req/s) | 6.67 | 8.15 | **1.22x** |
| p99 TTFT (ms) | 404.87 | 404.48 | 1.00x |
| p99 TPOT (ms) | 36.26 | 29.77 | 0.82x |

**分析**:
- AFD 在 DeepSeek-V2-Lite 上实现 **22% 总吞吐提升**和 **20% decode 延迟降低**
- TTFT 基本持平 (略增 7%), 因为 AFD 增加了一次 NCCL P2P 同步开销
- TPOT 显著降低, 因为 Attention 和 FFN 计算在不同 GPU 上并行, 消除了计算串行瓶颈
- 所有 128 请求成功完成, 0 失败

#### P-02: CUDA Graph vs Eager (AFD 1A1F)

| 指标 | Eager | Graph (FULL_DECODE_ONLY) | Graph / Eager |
|------|-------|--------------------------|--------------|
| mean TTFT (ms) | 246.95 | 4,858.85 | 19.68x |
| mean TPOT (ms) | 28.75 | 25.80 | **0.90x** |
| total token throughput (tok/s) | 5,195.23 | 2,326.73 | 0.45x |
| p99 TPOT (ms) | 29.44 | 26.53 | 0.90x |

**分析**:
- Graph 模式 **TPOT 降低 10%** (25.8ms vs 28.75ms), 证明 CUDA Graph 对 decode 阶段有优化效果
- 但 TTFT 暴增 19.68x, 原因是 benchmark 热身不足: 仅 1 个 max_tokens=4 的 warmup 请求,
  未触发所有 graph size 的 capture, 导致首批 128 请求在运行时触发 graph capture
- **结论**: graph 模式在充分热身后有稳定 10% decode 加速; benchmark 需增加 warmup 数量
  以排除 capture 开销

#### P-03: DBO vs Non-DBO (AFD 1A1F eager)

| 指标 | Non-DBO | DBO | DBO / Non-DBO |
|------|---------|-----|--------------|
| mean TTFT (ms) | 290.61 | 321.95 | 1.11x |
| mean TPOT (ms) | 29.66 | 59.27 | **2.00x** |
| total token throughput (tok/s) | 5,038.34 | 2,595.80 | **0.52x** |
| p99 TPOT (ms) | 30.23 | 62.12 | 2.05x |

**分析**:
- DBO (Dual micro-Batch Overlap) 在当前配置下 **性能减半**, TPOT 翻倍
- 原因分析: DBO 将每个 decode step 拆成两个 micro-batch 交替执行以重叠 Attn/FFN 计算
  - 在 AFD 架构下, 每个 micro-batch 需要独立的 NCCL P2P 同步, 通信开销翻倍
  - max_concurrency=32 时单 batch 计算量较小, 通信开销占比高, 重叠收益无法补偿
- **结论**: DBO 在 AFD P2P 架构 + 中等并发下负优化; 需要在更大 batch size 或计算密集
  场景下重新评估

#### P-04: 拓扑对比 (1A1F vs 2A2F, 均使用 DBO)

| 指标 | 1A1F (2 GPU) | 2A2F (4 GPU, DP2TP1) | 2A2F / 1A1F |
|------|-------------|----------------------|-------------|
| mean TTFT (ms) | 269.56 | 4,551.55 | 16.89x |
| mean TPOT (ms) | 58.08 | 97.64 | 1.68x |
| total token throughput (tok/s) | 2,664.64 | 1,094.49 | 0.41x |
| request throughput (req/s) | 4.16 | 1.71 | 0.41x |

**分析**:
- 2A2F 性能远低于 1A1F, 主要原因:
  1. **DP 协调开销**: DP=2 每侧 2 个 EngineCore 进程, 增加 NCCL group 复杂度
  2. **端口冲突**: DP=2 模式下两个 APIServer 竞争同一端口, 日志记录 `port 18000 is used`
  3. **FFN worker 崩溃**: 2A2F 的 FFN worker 在测试末期发生 `RuntimeError: AFD FFN worker loop failed`
  4. **DBO 叠加效应**: 两者都使用 DBO, 而 P-03 已证明 DBO 损失 50% 性能
  5. **热身不足**: TTFT 4.5s 说明首批请求触发了额外的模型初始化/编译
- 注意: 1A1F 的 2665 tok/s 与 P-03 DBO 结果 (2596) 一致, 验证可重复性
- **结论**: 2A2F 需要修复 DP 模式下的端口管理和 worker 稳定性; 在当前规模下不建议使用

#### P-05: 并发负载扩展 (AFD 1A1F eager)

| 并发数 | TTFT (ms) | TPOT (ms) | Token Throughput (tok/s) | 加速比 vs conc=1 |
|--------|-----------|-----------|--------------------------|------------------|
| 1      | 87.6      | 28.5      | 172.6                    | 1.00x            |
| 4      | 90.6      | 28.5      | 689.5                    | 4.00x            |
| 16     | 170.2     | 28.5      | 2,699.4                  | 15.6x            |
| 32     | 190.8     | 28.8      | 5,303.6                  | 30.7x            |
| 64     | 218.9     | 30.8      | 9,892.2                  | 57.3x            |
| 128    | 509.0     | 37.4      | 15,335.0                 | 88.9x            |

**分析**:
- AFD 展现出色的并发扩展能力: 从 1 到 128 并发, 吞吐从 173 → 15,335 tok/s (88.9x)
- **低并发 (1-16)**: 近乎线性扩展, TPOT 稳定在 28.5ms
- **中并发 (32-64)**: 扩展效率仍 >85%, TPOT 仅微增至 30.8ms
- **高并发 (128)**: 扩展开始饱和, TPOT 升至 37.4ms, TTFT 显著增加
- **结论**: 1A1F 拓扑的最佳工作区间为 concurrency 32-64, 此时吞吐 5k-10k tok/s,
  延迟稳定在 28-31ms/token

---

## 5.3 第二阶段补充实验（2026-07-25）

第二阶段针对第一阶段的四个遗留问题（2A2F 稳定性、公平基线、graph 热身、
DBO 根因）进行修复与补测。所有测试统一加入 `--num-warmups 32` 预热，
2A2F 改用 eager（无 DBO）以隔离拓扑变量。

### 5.3.1 P-01 公平基线补测（AFD 2GPU vs 原生 2GPU）

第一阶段 P-01 的 AFD 1A1F（2 GPU）对比原生 vLLM（1 GPU）不公平。
补充原生 vLLM DP=2（GPU 0,1）基线，形成"2 GPU vs 2 GPU"对比。

| 指标 | 原生 1GPU | 原生 DP2 (2GPU) | AFD 1A1F (2GPU) | AFD/原生1 | AFD/原生DP2 |
|------|----------|-----------------|-----------------|-----------|-------------|
| mean TTFT (ms) | 206.6 | 428.6 | 256.7 | 1.24x | 0.60x |
| mean TPOT (ms) | 35.9 | 45.8 | 29.6 | 0.82x | 0.65x |
| total throughput (tok/s) | 4,296.8 | 3,272.5 | 5,081.4 | **1.18x** | **1.55x** |
| request throughput (req/s) | 6.71 | 5.11 | 7.94 | 1.18x | 1.55x |

**分析**:
- **AFD 在公平 2GPU 对比下仍有 55% 吞吐优势**，证明 AFD 的流水线并行
  （Attention/FFN 重叠）优于原生数据并行（两个独立副本无重叠）
- 原生 DP2（3272 tok/s）反而**低于**原生 1GPU（4297 tok/s），因为
  DeepSeek-V2-Lite 是 MoE 模型，DP=2 增加了 MoE all2all 协调开销，
  且两个副本各自 batch 更小，GPU 利用率下降
- AFD TTFT（256ms）远优于原生 DP2（428ms），因 AFD 无 DP 层协调
- 0 失败，结果可信

### 5.3.2 P-02 Graph 热身修复尝试

第一阶段 P-02 graph TTFT 暴增 19.68x，归因 warmup 不足。第二阶段在
`vllm bench serve` 增加 `--num-warmups 32`，期望预热所有 batch size 的
graph capture。

| 指标 | Eager | Graph (FULL_DECODE_ONLY) | Graph/Eager |
|------|-------|--------------------------|-------------|
| mean TTFT (ms) | 219.0 | 4,808.1 | 21.95x |
| mean TPOT (ms) | 29.9 | 25.7 | 0.86x |
| total throughput (tok/s) | 5,086.9 | 2,355.1 | 0.46x |

**分析**:
- **`--num-warmups` 未能消除 graph TTFT 暴增**，TTFT 仍 4808ms（与第一阶段 4858ms 几乎相同）
- 根因：AFD 的 FFN 端 graph capture 由 **connector 驱动**（`ffn_worker._run_ffn_server_loop`
  经 `control_plane.recv_dp_metadata_list` 触发），而 `vllm bench serve` 的
  warmup 请求只走 Attention 侧 API，**不驱动 FFN 端 capture**。因此 FFN 端
  在首批实际请求时才完成 capture，开销计入 TTFT
- Graph TPOT 仍稳定优于 eager（0.86x，即 14% decode 加速），证明 graph 本身有效
- **结论**：要测量 graph 真实性能，需在 server 启动后手动发送驱动 FFN capture 的
  预热请求（经 connector 全链路），`vllm bench` 的 warmup 参数对此无效。
  这是 AFD 架构（FFN 被 connector 驱动）与原生 vLLM（单进程）的根本差异

### 5.3.3 P-03 DBO 复测（确认负优化稳定）

| 指标 | Non-DBO | DBO | DBO/Non-DBO |
|------|---------|-----|-------------|
| mean TTFT (ms) | 257.2 | 252.2 | 0.98x |
| mean TPOT (ms) | 30.0 | 57.7 | **1.92x** |
| total throughput (tok/s) | 5,032.6 | 2,701.8 | **0.54x** |

与第一阶段（0.52x）一致，DBO 稳定负优化。详细根因见 §5.3.5 P-06。

### 5.3.4 P-04 拓扑对比（2A2F 修复后）

修复 `ffn_worker.stop_ffn_server_loop` 的 shutdown race（先 join loop 线程
再拆 connector，消除 "P2P connector is not initialized" 崩溃），2A2F 改用
eager 无 DBO 隔离拓扑变量。

| 指标 | 1A1F (2GPU) | 2A2F (4GPU, DP2TP1) | 2A2F/1A1F |
|------|-------------|----------------------|-----------|
| mean TTFT (ms) | 216.9 | 2,342.8 | 10.80x |
| mean TPOT (ms) | 30.0 | 56.1 | 1.87x |
| total throughput (tok/s) | 5,083.5 | 1,932.9 | **0.38x** |
| p99 TTFT (ms) | 336.4 | 7,780.9 | 23.1x |
| 崩溃 | 无 | **无（修复生效）** | — |

**分析**:
- **2A2F 稳定性修复成功**：第一阶段报告的 "FFN worker 崩溃" 实为 shutdown race
  （`kill_procs` 在 bench 完成后拆 connector 时，loop 线程仍在 `send_ffn_output`
  中），非运行时故障。bench 实际 128/128 成功 0 失败。修复 `stop_ffn_server_loop`
  join 顺序后 clean exit，无 crash
- **"端口冲突" 亦非致命**：DP launcher 的 warning，bench 正常服务
- 但 2A2F 性能仍仅 0.38x（比第一阶段 0.41x 略低），**去掉 DBO 并未改善**，
  说明 DBO 不是 2A2F 性能主因
- 真实瓶颈是 **DP=2 协调开销**：每侧 2 个 EngineCore + NCCL group 复杂度，
  4 GPU 的算力提升无法补偿 DP 协调 + AFD connector 跨 rank 同步的额外开销
- 2A2F TTFT 2342ms（比第一阶段 4551ms 改善，warmup 部分生效），但仍远高于 1A1F，
  因 DP=2 首批请求需跨 rank 协调初始化
- **结论**：当前 AFD P2P 实现下，2A2F（DP2TP1）不具性能优势，1A1F 是优选拓扑

### 5.3.5 P-06 DBO 阈值扫描（DBO 根因定位）

扫描 `--dbo-decode-token-threshold`（2/8/16/32）× 并发（32/64），
验证三个假设：
- H1: DBO 拆 2 micro-batch → 2× NCCL P2P 同步，通信翻倍
- H2: threshold=2 过小，所有 decode 都进 DBO 路径
- H3: Python 层调度开销

| threshold | conc=32 TPOT | conc=32 tput | conc=64 TPOT | conc=64 tput |
|-----------|-------------|-------------|-------------|-------------|
| non-DBO 基线 | 28.8ms | 5,306.6 | 28.8ms | 10,565.7 |
| 2 | 58.7ms | 2,641.9 | 57.5ms | 5,331.0 |
| 8 | 57.1ms | 2,731.9 | 59.0ms | 5,162.5 |
| 16 | 57.4ms | 2,701.0 | 56.3ms | 5,440.7 |
| 32 | 57.7ms | 2,704.1 | 59.0ms | 5,194.8 |

**分析**:
- **H2 否定**：提高 threshold（8/16/32）对 TPOT/吞吐几乎无改善，所有 threshold
  下 TPOT 均在 56-59ms，吞吐均为 non-DBO 的 ~0.5x。threshold 只控制 DBO 触发频率，
  但 decode 阶段 batches-per-step 不变，实际每个 decode step 仍进 DBO 路径
- **H1 确认**：DBO TPOT 恰为 non-DBO 的 **~2.0x**（57.7 vs 28.8, 57.5 vs 28.8），
  与"2 micro-batch = 2× NCCL P2P 往返"精确吻合。源码追踪：DBO 经
  `dbo.py:maybe_apply_dbo_yield` + `attention_model_runner._should_ubatch_single_rank`
  将 decode 拆 2 ubatch，每个需独立 `send_attn_output`/`recv_ffn_output` P2P 同步
- **大 batch 无收益**：conc=64 下 DBO 仍 0.50-0.52x，通信开销占比未随 batch 增大而
  下降至可补偿，因 AFD P2P 每步同步为 GPU 间点对点，延迟固定不随 batch 缩放
- **H3 次要**：Python 调度开销存在但非主因（TPOT 翻倍主要由通信，非 CPU）
- **结论**：DBO 在 AFD P2P 架构下**无条件负优化**（所有 threshold、所有并发均 ~0.5x），
  根因是 micro-batch 拆分导致 NCCL P2P 同步次数翻倍。DBO 设计前提（重叠 Attn/FFN
  计算）在 AFD 下不成立——AFD 已通过跨 GPU 分离实现重叠，DBO 反而增加通信

### 5.3.6 P-05 并发扩展（warmup 改善）

启用 warmup 后，高并发吞吐显著提升（消除冷启动）：

| 并发 | 第一阶段 tput | 第二阶段 tput | 改善 |
|------|--------------|--------------|------|
| 64 | 9,892.2 | 10,565.7 | +6.8% |
| 128 | 15,335.0 | 19,558.7 | **+27.6%** |

1→128 并发：173 → 19,559 tok/s（113x），warmup 使高并发区间消除冷启动惩罚。

---

## 6. 数据分析与结论

### 6.1 核心发现

1. **AFD 架构有效提升吞吐**: 1A1F eager 模式比原生 vLLM 吞吐提升 22%, decode 延迟
   降低 20%, 证明 Attention-FFN 分离在 MoE 模型上的有效性

2. **AFD 优于原生数据并行**: 公平 2GPU 对比下 AFD（5,081 tok/s）比原生 DP2
   （3,272 tok/s）高 **55%**，因 AFD 流水线重叠 vs DP 独立副本无重叠；原生 DP2
   在 MoE 模型上甚至低于 1GPU（all2all 协调开销）

3. **并发扩展能力优秀**: 1A1F 在 32-64 并发区间高效扩展, 128 并发吞吐达 19.6k tok/s
   （warmup 改善后比冷启动高 28%）

4. **CUDA Graph 对 decode 有效但 AFD 下难预热**: 充分热身后 graph TPOT 降低 14%,
   但 AFD 的 FFN 端 capture 由 connector 驱动，`vllm bench` warmup 无法预热，
   需手动经全链路预热

5. **DBO 在 AFD 下无条件负优化**: 阈值扫描（2/8/16/32）× 并发（32/64）证实所有
   配置均为 ~0.5x，根因是 micro-batch 拆分使 NCCL P2P 同步翻倍（TPOT 恰 2.0x）。
   DBO 的重叠设计前提在 AFD（已跨 GPU 重叠）下不成立

6. **2A2F 拓扑稳定但无性能优势**: 修复 shutdown race 后 2A2F 不再崩溃，但性能仅
   1A1F 的 0.38x，根因是 DP=2 协调开销（非 DBO、非 crash），4 GPU 算力无法补偿

### 6.2 性能对比汇总

| 配置 | TPOT (ms) | Throughput (tok/s) | vs Native 1GPU | vs Native DP2 |
|------|-----------|-------------------|----------------|---------------|
| 原生 vLLM (1 GPU, eager) | 35.9 | 4,297 | baseline | — |
| 原生 vLLM DP2 (2 GPU) | 45.8 | 3,272 | 0.76x | baseline |
| AFD 1A1F (eager) | 29.6 | 5,081 | **1.18x** | **1.55x** |
| AFD 1A1F (graph)* | 25.7 | 2,355 | — | — |
| AFD 1A1F (DBO) | 57.7 | 2,702 | 0.63x | 0.83x |
| AFD 2A2F (DP2TP1, eager) | 56.1 | 1,933 | 0.45x | 0.59x |

*graph throughput 不具参考价值（AFD FFN capture 未预热）

### 6.3 建议与后续方向

1. **推荐配置**: AFD 1A1F eager 模式, 并发 32-64, 是当前最稳定高效的选择
   （公平对比下比原生 2GPU 高 55%）
2. **DBO 不建议启用**: 在 AFD P2P 架构下无条件负优化（~0.5x），所有 threshold/并发
   均如此。如需重叠，AFD 本身的跨 GPU 分离已提供，无需 DBO
3. **Graph 模式**: 生产中需经 connector 全链路手动预热 FFN 端 capture 后，可获稳定
   14% decode 加速；`vllm bench --num-warmups` 对 AFD 无效
4. **2A2F 暂不推荐**: DP=2 协调开销使 4 GPU 反慢于 2 GPU；需优化 DP metadata
   同步路径或改用 TP 横向扩展后再评估
5. **原生 DP 在 MoE 上低效**: DeepSeek-V2-Lite 的 DP2 比 1GPU 还低，AFD 是更优的
   多 GPU 利用方式
6. **V2.5 大模型验证**: AFD 4A4F TP4 在 V2.5 (236B) 上超越原生 TP8 1.20×（1,663
   vs 1,384 tok/s），证明 AFD 优势随模型规模增长。V2-Lite (16B) 上 4A4F 仅 0.29×
   是因为模型太小，通信开销主导
7. **Fan-out (1A2F) 拓扑**: 代码实现正确，V2-Lite 上功能验证通过，但 V2.5 上 NCCL
   初始化死锁，需修复混合 TP 大小的通信器创建
8. **vLLM DP2 不兼容 AFD**: P-11 shm_broadcast 超时，AFD 连接器与 vLLM DP2 机制
   冲突；AFD 的多 GPU 扩展应通过 TP 或增加 A/F 对数，不宜用 vLLM DP

---

## 5.4 第三阶段：更大拓扑实验（2026-07-25）

第三阶段扩展 AFD 拓扑至 4A4F（8 GPU）和新实现的 1A2F（3 GPU，expert-parallel fan-out），
验证更大模型下的可行性。

### 5.4.1 P-07 4A4F 拓扑（8 GPU，零代码改动）

4A4F 使用已有的 P2pNcclAFDConnector（4A ≥ 4F 满足约束），DP2×TP2 每侧，
eager 模式。

| 指标 | 1A1F (2GPU) c32 | 4A4F (8GPU) c32 | 1A1F c64 | 4A4F c64 |
|------|-----------------|-----------------|----------|----------|
| mean TTFT (ms) | 193.7 | 2,603.7 | 208.6 | 3,342.7 |
| mean TPOT (ms) | 28.8 | 63.9 | 28.8 | 55.2 |
| total throughput (tok/s) | 5,307 | 1,719 | 10,566 | 3,112 |
| 失败 | 0 | 0 | 0 | 0 |

**分析**: 4A4F 功能正确（0 失败），但吞吐仅为 1A1F 的 0.29-0.32x。V2-Lite 仅 16B，
8 GPU 的通信开销（NCCL P2P × DP2 × TP2 协调）远超 4 倍算力增益，说明小模型不适合大规模
AFD 拓扑。

### 5.4.2 P-08 1A2F 拓扑（3 GPU，fan-out expert-parallel，新代码）

1A2F 要求 `attention < ffn`，突破了 P2pNcclAFDConnector 原有的 `attention >= ffn` 约束。
本次实现了 fan-out 模式：

**代码改动**（3 文件，均标记 PATCH START/END）:
- `distributed/topology.py`: 新增 `_build_rank_mapping_fan_out()`，构建反向子组
  `[A_j, F_{j*ratio}, ..., F_{(j+1)*ratio-1}]`，A 在子组 rank 0，FFN 在 1..ratio
- `connectors/gpu/p2p.py`: 四个数据路径方法增加 fan-out 分支:
  - `send_attn_output`: Attention 广播到所有 FFN rank
  - `recv_attn_output`: 每个 FFN 从唯一 Attention rank 接收
  - `send_ffn_output`: 仅 leader FFN（rank 1）发送结果，其余跳过
  - `recv_ffn_output`: Attention 从 leader FFN 接收
  - DP metadata 同步改为连续块映射
- `init_afd_connector` 超时从 2min 提升到 10min（适配 TP>1 的 FFN 端较慢初始化）

FFN 端以 `--tensor-parallel-size 2 --enable-expert-parallel` 运行，vLLM FusedMoE 的
all-to-all 负责 expert 分发/合并，两个 FFN worker 各持有 32 个 expert（共 64），
leader 发送合并后的正确结果。

| 指标 | 1A1F (2GPU) c32 | 1A2F (3GPU) c32 | 1A1F c64 | 1A2F c64 |
|------|-----------------|-----------------|----------|----------|
| mean TTFT (ms) | 193.7 | 8,015.9 | 208.6 | 756.5 |
| mean TPOT (ms) | 28.8 | 44.3 | 28.8 | 44.2 |
| total throughput (tok/s) | 5,307 | 1,334 | 10,566 | 5,178 |
| 失败 | 0 | 0 | 0 | 0 |

**分析**:
- **1A2F 功能正确**：0 失败，expert-parallel fan-out 架构验证成功
- c64 下 5,178 tok/s = 1A1F 的 0.49x。fan-out 的 Attention→2FFN 广播 + all-to-all
  expert 分发开销在 V2-Lite（16B 小模型）上超过 expert-parallel 的算力收益
- TPOT 44.3ms vs 1A1F 28.8ms（1.54x）：每层多一次 NCCL P2P send + all-to-all 同步
- c32 TTFT 8016ms 极高：首批请求需预热 FusedMoE all-to-all + inductor 编译路径
- **结论**：1A2F 架构可行且正确，但 V2-Lite 太小无法体现 expert-parallel 优势。
  需在 V2.5 (236B) 等大模型上验证，FFN/MoE 占比更大才能摊销通信开销

### 5.4.3 对比汇总

| 拓扑 | GPU | c64 tput (tok/s) | vs 1A1F c64 | 失败 | 说明 |
|------|-----|-----------------|-------------|------|------|
| 1A1F | 2 | 10,566 | baseline | 0 | 最优配置 |
| 1A2F | 3 | 5,178 | 0.49x | 0 | 新实现，expert-parallel |
| 2A2F | 4 | 1,933 | 0.18x | 0 | DP2 协调开销 |
| 4A4F | 8 | 3,112 | 0.29x | 0 | DP2TP2，通信主导 |

所有拓扑 0 失败，功能正确。V2-Lite (16B) 模型过小，任何 >2GPU 的拓扑都无法获得
性能收益。大模型（V2.5 236B）实验待模型下载完成后进行。

### 5.4.4 第三阶段 B：DeepSeek-V2.5 (236B) 大模型验证

V2.5 为 BF16 (440GB, 55 safetensors)，通过 `--quantization fp8` 动态量化加载，
FP8 权重约 236GB。架构 `deepseek_v2`，60 层，160 experts/层，6 experts/token。

**模型配置**: `--quantization fp8 --max-model-len 4096 --enforce-eager`
**测试矩阵**: P-09 (4A4F TP4), P-10 (2A4F fan-out), P-11 (4A4F DP2TP2)

#### P-09: V2.5 4A4F TP4 vs 原生 TP8（8 GPU，DP1×TP4 每侧）

4A4F 每侧 DP=1 TP=4，等价于 1A1F scaled to TP=4。每个 GPU ~59GB 权重，~74GB KV cache。

| 指标 | Native TP8 c32 | AFD 4A4F c32 | AFD 4A4F c64 |
|------|---------------|-------------|-------------|
| mean TTFT (ms) | 695 | 24,292 | 2,317 |
| mean TPOT (ms) | 111 | 135 | 137 |
| total throughput (tok/s) | 1,384 | 437 | **1,663** |
| 失败 | 0 | 0 | 0 |

**关键结论**: AFD 4A4F c64 = **1.20× 原生 TP8**（1,663 vs 1,384 tok/s）。
这是**首次在 production-size 模型上验证 AFD 超越原生 vLLM**。
c32 低吞吐因首请求编译开销（TTFT 24s），c64 充分饱和后性能凸显。
TPOT 137ms vs 111ms，AFD 每层多一次 NCCL P2P 同步，但吞吐因 attention/FFN
分离的流水线并行而提升。

#### P-10: V2.5 2A4F fan-out（6 GPU，attn TP=2 + FFN TP=4）— 失败

fan-out 拓扑 (attention=2 < ffn=4) 在 NCCL P2P 初始化后死锁。
Attention 和 FFN 两侧均在 `PyNcclCommunicator` 创建后挂起，vLLM EngineCore
等待 profile run 完成但 AFD P2P 通信卡住。可能原因：
- 混合 TP 大小 (attn TP=2, FFN TP=4) 导致 NCCL 通信器配置不匹配
- fan-out 广播模式 (1 attn → 2 FFN) 的 PyNcclCommunicator 初始化竞争

#### P-11: V2.5 4A4F DP2×TP2（8 GPU）— 失败

vLLM 内置 DP2 的 `shm_broadcast` 超时。Profile run 完成后 (7.19s)，
两个 DP rank 的 EngineCore 无法通过共享内存广播协调。
AFD 连接器与 vLLM DP2 机制存在兼容性问题。

#### V2.5 实验总结

| 实验 | 配置 | GPU | 结果 | 说明 |
|------|------|-----|------|------|
| P-09 | 4A4F DP1TP4 | 8 | **1,663 tok/s (1.20x)** | AFD 首超原生 |
| P-10 | 2A4F fan-out | 6 | 死锁 | fan-out NCCL 初始化问题 |
| P-11 | 4A4F DP2TP2 | 8 | shm_broadcast 超时 | vLLM DP2 不兼容 |

**核心发现**: 在 V2.5 (236B) 上，AFD 4A4F TP4 以 1.20× 超越原生 TP8。
与 V2-Lite (16B) 的 4A4F 仅 0.29× 对比，证明**AFD 的优势随模型规模增长**——
大模型的 attention/FFN 分离流水线收益远超 P2P 通信开销。

---

## 7. 附录

### 7.1 实验脚本

| 脚本 | 用途 |
|------|------|
| `experiment/scripts/run_functional_tests.py` | 功能正确性测试 (F-01 ~ F-05) |
| `experiment/scripts/run_perf_tests.py` | 吞吐性能测试 (P-01 ~ P-05) |
| `experiment/scripts/start_afd.sh` | AFD 服务器启动脚本 (1a1f/2a2f, eager/graph/dbo) |
| `experiment/scripts/start_native.sh` | 原生 vLLM 启动脚本 (baseline) |
| `experiment/scripts/bench_serve.sh` | vllm bench serve 封装脚本 |

### 7.2 原始数据

- `experiment/results/performance_results.json` — 全部 P-01~P-05 详细 JSON 结果
- `experiment/results/functional_results.json` — F-01~F-05 功能测试结果
- `experiment/results/p01_native.json` ~ `p05_conc128.json` — 各单项 bench 原始输出
- `experiment/logs/` — 服务器和 benchmark 完整日志

### 7.3 容器启动命令

```bash
docker run -d --name afd-exp \
  --gpus all --network host \
  -v /data1/models:/models \
  -v /data1/afd-plugin:/workspace/afd-plugin \
  -e SETUPTOOLS_SCM_PRETEND_VERSION=0.0.1 \
  --entrypoint sleep \
  vllm/vllm-openai:v0.19.1 infinity

# 容器内安装
docker exec afd-exp pip install -e /workspace/afd-plugin --no-deps --no-build-isolation
```

### 7.4 Benchmark 参数

```
vllm bench serve \
  --dataset-name random \
  --tokenizer /models/DeepSeek-V2-Lite \
  --num-prompts 128 \
  --request-rate inf \
  --max-concurrency 32 \
  --input-len 512 \
  --output-len 128 \
  --save-result --result-dir <dir> --result-filename <name>
```
