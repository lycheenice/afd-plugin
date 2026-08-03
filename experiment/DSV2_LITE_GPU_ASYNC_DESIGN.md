# DeepSeek-V2-Lite GPU 异步 Attention–FFN 设计

> 状态：eager MVP 已在 gpu-host 通过语义、同步参考一致性、连续请求、1000 请求并发
> 长稳与 profiler timeline 门禁；性能门未通过，复杂拓扑扩展停止。
> 范围：gpu-host、DeepSeek-V2-Lite、vLLM 0.25.0、CUDA eager 优先。
> 隔离：不包含 GLM-5.2、W4AFP8、Ascend CAM/CAMP2P 的验收结论。

## 1. 当前同步路径

GPU 数据面位于 `afd_plugin/connectors/gpu/p2p.py`。当前 custom op 将
`PyNcclCommunicator.send/recv` 提交到调用方的 `torch.cuda.current_stream()`：

```text
Attention compute -> A2F send -> F2A recv -> next Attention layer
                            FFN recv -> FFN compute -> F2A send
```

Attention 侧的层间顺序由
`afd_plugin/model_executor/models/deepseek_v2.py::forward_with_afd` 驱动；FFN 侧由
`afd_plugin/v1/worker/ffn_model_runner.py::_ffn_forward` 按 layer/stage 串行执行。
DBO 可以把请求切成 stage，但通信仍在当前 compute stream 上提交。因此“存在两个
ubatch”不等于数据面已经异步，也不能据此宣称通信与计算发生重叠。

NPU 的 async CAM 流水代码可用于理解 stage 调度，但不能直接复用为 GPU 实现：其
top-k payload、传输状态、后端算子和 stream 语义均不同。GPU 方案应保持在 P2P connector
及 GPU runner/model wrapper 内，不给通用 connector 接口增加 NPU 专属状态。

## 2. 设计目标

第一阶段只支持 `1A1F + eager + exactly 2 ubatches`：

- A2F/F2A 通信使用 connector-owned CUDA stream；
- 用 CUDA event 表达 compute-ready、receive-ready 和 buffer-reusable 依赖；
- 允许 `A(stage 1)` 与 `F(stage 0)` 在不同 GPU 上流水；
- 保留同步路径，默认关闭异步，可在同一进程启动参数下回退；
- 输出、token usage、请求顺序和资源回收必须与同步路径一致。

graph、DP/TP、多 Attention/FFN rank 和 fan-out 不进入第一版实现。它们在 eager MVP
正确且 timeline 证明有收益后按独立阶段扩展。

## 3. Connector 所有权

异步资源由每个 `P2pNcclAFDConnector` 实例持有，避免引入新的可变全局状态：

- `a2f_comm_stream` / `f2a_comm_stream`；
- 每个方向、stage 和 peer 至多一个 receive buffer；shape 变化时替换而不累积，保证
  cache 大小由 slot/peer 数限定；
- compute-ready、receive-ready、send-complete event；
- 有界 slot 状态及 transaction/layer/stage 标识；
- shutdown 时的 drain、event/stream 引用释放和 communicator teardown。

现有 communicator id 注册表只承担 custom op 查找，不承载请求级异步状态。

配置应由 `connector_extra_config` 解析为不可变配置对象，至少包括：

- `async_transfer: bool = false`；
- `async_slots: int = 2`，MVP 固定校验为 2；
- 不支持的 graph/topology/ubatch 组合在初始化时明确拒绝，不静默降级。

## 4. 依赖与时序

### 4.1 Attention 发送

1. compute stream 完成本 stage Attention 输出后记录 `compute_ready`；
2. A2F stream 等待 `compute_ready`；
3. 在 A2F stream 上提交 NCCL send；
4. 记录 `send_complete`，在复用输入 slot 前等待它。

### 4.2 FFN 接收、计算与回传

1. A2F stream 预贴 receive 到稳定 slot buffer；
2. receive 完成后记录 `receive_ready`；
3. FFN compute stream 在消费该 buffer 前等待 `receive_ready`；
4. FFN 输出完成后在 compute stream 记录 `input_consumed`；
5. F2A stream 等待 compute event，提交 send，并记录 `send_complete`。

### 4.3 Attention 接收

1. F2A stream 预贴 receive；
2. receive 完成后记录 `receive_ready`；
3. Attention compute stream 在下一层消费输出前等待该 event；
4. `recv_ffn_output` 对调用方表现为依赖已建立的 tensor，而不是用全设备
   `torch.cuda.synchronize()` 强制串行。

所有 rank 必须使用相同的 layer/stage 顺序提交 NCCL 操作。MVP 沿用现有 runner 的确定
调用顺序，不引入乱序消息匹配；同一 stage 的下一次 A2F receive 等待
`input_consumed`，Attention 侧 F2A receive 在覆盖发送源之前等待 A2F `send_complete`。
显式 transaction id 和跨请求乱序支持不在 MVP 范围内。

## 5. 代码落点

| 位置 | MVP 状态 |
|---|---|
| `afd_plugin/connectors/gpu/p2p.py` | 已实现 typed config、双 stream、event 依赖、有界 buffer、drain/close |
| GPU model/runner | MVP 复用已有两个 DBO stage 的确定调用顺序；GPU timeline 后再判断是否需改调度 |
| `tests/unit/connectors/test_p2p_connector.py` | 已覆盖配置范围、event 顺序、同步回退与 close |
| `tests/e2e/` | 启动器已支持 DBO/connector extra config；独立 GPU 脚本已覆盖 sync/async greedy 文本一致与连续请求 |

实现时不修改 vLLM 安装目录。若必须 patch vLLM 0.25.0，使用
`afd_plugin/compat/patches/`，复制对应 pinned-tag 上游函数并用 AFD patch marker 标出差异。

## 6. 正确性门禁

按以下顺序验收，每一步失败都停止扩展：

1. 同步 `1A1F` 四条短 prompt：自动校验语义关键内容、上下文衔接、finish reason、
   token usage；
2. 同一 greedy prompt 集下 sync/async completion 全文、finish reason 和 usage 完全一致；
3. 两个 stage 的 transaction/layer/stage 序列无错配，slot 复用前 send 已完成；
4. 连续请求和并发请求中无串样、截断、重复段或无意义 token；
5. 现有 GPU E2E 在异步支持范围内通过，GSM8K 不低于同步基线阈值；
6. 正常与异常退出后无端口、进程、process group 和 GPU buffer 泄漏。

短 prompt 只用于快速阻断明显错误，不能替代完整 E2E 和精度评估。

这里的“上下文衔接”是自动断言，不是只做人工观感检查：completion 去除首尾空白后
必须以该 prompt 的预期续写前缀开头，并包含语义关键片段。例如 `The capital of
France is` 必须直接续写 `Paris`，Python 样例必须直接进入 `def` 且包含 `return` 与加法。

## 7. 性能验证与决策门

先用 timeline 确认 NCCL 处于 compute stream、两个 ubatch 均非空并量化 GPU idle。
随后在固定输入、输出、并发、seed 和预热条件下比较 native、AFD sync、AFD async。

MVP 继续扩展的条件：

- 目标通信受限场景吞吐提升或 TPOT 降低至少 10%；
- 并发 1 的 E2E latency 退化不超过 5%；
- 显存增量能由 slot 数和 tensor shape 解释；
- profiler 能看到真实通信/计算重叠，而非仅 CPU 调度重排。

若达不到门槛，异步保持默认关闭，记录瓶颈和停止原因，不扩展到 graph 或复杂拓扑。

## 8. 异步实现前的功能阻断：1A2F

`20260728T1308Z-topology-1a2f` 已确认不是启动抖动，而是 GPU FFN runner 的 DP/EP
metadata 错配：

- Attention DP world size 为 1，控制面发送的 `num_tokens_across_dp_cpu` 只有 1 项；
- FFN 侧以 DP=2、EP=2 启动，两个 FFN rank 都收到同一份 Attention hidden states；
- `GPUFFNModelRunner._ffn_forward` 把原始 Attention metadata 直接写回 vLLM forward
  context；
- vLLM `naive_dp_ep.py` 随后按 FFN DP/EP rank 索引这份单项列表，rank 1 在
  `sizes[rank_in_group]` 越界。

修复应放在 GPU FFN runner 的 forward-context 构造处，而不是修改 vLLM 的 MoE
dispatch：先把 Attention token counts 映射成 FFN role-rank counts，再投影为 vLLM
期望的 FFN DP counts。映射规则必须同时覆盖：

- fan-in（如 `2A1F`）：同一 FFN rank 收到的多个 Attention rank token counts 求和；
- one-to-one（如 `2A2F`、`4A4F`）：逐 rank 对应；
- fan-out（如 `1A2F`）：同一 Attention rank 的 token count 复制到其所有 FFN peers；
- TP：AFD role-rank counts 按 TP 分组投影回每个 DP rank 一项。

本地 repo 已实现 token-count 映射并新增纯 CPU 映射/forward-context 单测，在容器隔离
副本中与 P2P connector 回归合计 `47 passed`。补丁同步到 gpu-host 后，`1A2F` eager
四条自动语义门禁已 4/4 通过，证明 fan-out metadata 越界已修复；证据位于
`experiment/results/dsv2_v025/20260728T1608Z-topology-1a2f-fix-retry3/`。修复后的
`2A1F -> 2A2F -> 4A4F` 防回归仍需补齐，任何既有拓扑回归都停止扩大异步范围。

## 9. MVP 实现与审查状态

本地实现已完成 connector 调用点接线，并在 gpu-host 容器的隔离代码副本中与 FFN
metadata 回归合计 `59 passed`。证据位于
`experiment/results/dsv2_v025/20260728T1527Z-async-mvp-unit/`。

架构审查确认：同步默认路径保持不变；异步资源没有新增可变全局状态；两个方向 stream
分离；compute/通信依赖均由 event 表达；receive cache 有界；close 会先 drain 再释放
communicator。

gpu-host 的实际 CUDA/NCCL 门禁结果如下：

- 同步 `1A1F + eager + DBO` 四条语义 prompt 4/4 通过，保存为参考；
- 异步运行 4/4 通过，completion 全文、`finish_reason` 和 usage 与同步参考逐项完全一致；
- 连续三轮共 12 次请求均通过语义断言并与同步参考完全一致，无串样、截断或重复段；
- 32 并发下，async 与 paired sync control 各完成 1000 次请求，均为 1000/1000
  HTTP 200 且 1000/1000 通过 prefix + semantic fragment 门禁；两组输出、
  `finish_reason` 和 usage 逐项 1000/1000 完全一致；
- 两个并发组各有 248/1000 条未逐字匹配串行参考，全部集中于 Python prompt，差异只在
  completion 首尾换行位置。函数体、16 个 completion tokens 和 usage 不变。由于 sync
  control 呈现完全相同分布，该差异归因于并发 + DBO batching 的共同确定性边界，而非
  async connector 的错序或语义退化；
- torch profiler 中，同步 Attention/FFN 的主 NCCL stream 分别与 compute stream 共享
  `21`/`7`，通信—计算重叠均为 `0 us`；异步路径 compute stream 与 NCCL stream 完全
  分离，分别观测到 `1071.07 us` 和 `135616.44 us` 重叠。

原始 trace、输出与解析结果位于
`experiment/results/dsv2_v025/{20260728T1723Z-sync-profiler-timeline,20260728T1730Z-async-profiler-timeline}/`，
解析入口为 `experiment/scripts/analyze_dsv2_profiler_trace.py`。这些时间线证明了 stream
迁移和真实重叠，但不是吞吐/TPOT 收益结论。并发长稳的原始结果位于
`20260729T0200Z-async-semantic-soak1000` 和
`20260729T0205Z-sync-semantic-soak1000-control`；异步仍保持实验特性且默认关闭。

## 10. 固定负载性能结果与停止条件

两组 gpu-host paired benchmark 均在服务启动后执行 8 次 warmup、3 次重复，并在测量前后
各运行四条语义门禁。异步组前后均与同步语义参考完全一致，但性能门未通过：

| 负载 | sync 输出吞吐 | async 输出吞吐 | 差异 | sync mean TPOT | async mean TPOT | 差异 |
|---|---:|---:|---:|---:|---:|---:|
| 128→64 tokens，C=16，64 requests | 149.33 tok/s | 144.20 tok/s | -3.43% | 102.47 ms | 107.22 ms | +4.64% |
| 512→128 tokens，C=64，64 requests | 588.77 tok/s | 560.05 tok/s | -4.88% | 103.57 ms | 109.71 ms | +5.93% |

完整原始数据位于 `20260728T1800Z-sync-fixed-benchmark`、
`20260728T1830Z-async-fixed-benchmark`、`20260728T1900Z-sync-comm-heavy-benchmark`
和 `20260728T1930Z-async-comm-heavy-benchmark`；比较入口为
`experiment/scripts/compare_dsv2_async_benchmarks.py`。

profiler 显示数据面 NCCL P2P kernel 默认使用约 16 个 CTA；异步虽产生真实 overlap，
也让通信 kernel 与 FFN kernel 同时竞争 SM。尝试以 `NCCL_MIN_NCHANNELS=4` /
`NCCL_MAX_NCHANNELS=4` 限制 CTA，但 AFD/NCCL 初始化在 900 秒内未完成，故该方案记为
infra/兼容性失败，不进入性能比较。

据此，当前 MVP 已完成正确性和机制验证，但未达到“吞吐提升或 TPOT 降低至少 10%”的
扩展门槛。保持 `async_transfer=false` 默认值，不扩展到 graph 或复杂拓扑。下一轮只有在
能够安全降低 NCCL P2P 的 SM 占用、推迟长时间空等 receive，或用更细粒度调度减少
event/allocator 开销时才恢复性能优化；任何新方案仍须先过同一语义参考门。

长期处理原则：metadata 修复属于 AFD runner 的拓扑适配，继续保留并补齐回归；异步
stream 代码不修改 vLLM 安装目录，也不改变非 AFD 或默认同步路径。在达到性能门前不向
上游提交、不宣称生产支持；若后续无法通过 communicator 配置或调度方案消除 SM 竞争，
则删除实验异步分支及配置，只保留本报告和 profiler 证据作为停止依据。

DBO 与多 stream 的关系是“DBO 提供两个 ubatch 的流水机会，多 stream 让通信可以真正
离开 compute stream”，两者均不足以单独保证收益。当前正确性/性能实验把 decode 和
prefill threshold 设为 1，以稳定触发两个 stage；这是压力验证配置，不是推荐性能参数。
若继续优化，首先扫描 threshold，使小 batch 不拆分、足够大的 batch 才进入双 stage，
并继续以 sync/async paired workload 和语义前后门禁作为决策依据。
