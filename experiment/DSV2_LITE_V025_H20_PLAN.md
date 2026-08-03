# DeepSeek-V2-Lite vLLM 0.25.0 gpu-host 验证与 GPU 异步优化计划

> 状态：独立目标，进行中。
> 范围：gpu-host、CUDA、vLLM `0.25.0`、DeepSeek-V2-Lite。
> 隔离规则：不包含 GLM-5.2、W4AFP8、Ascend NPU 或 model runner v2；不得用这些
> 目标的结果替代本计划的验收证据。

## 1. 目标与完成定义

本目标需要同时完成以下三部分，缺一不可：

1. **功能完整性**：支持范围内的 serving、拓扑、并行模式、eager、CUDA graph、
   DBO、profiler、精度、稳定性和资源回收均有 gpu-host 实测证据。
2. **GPU 异步执行**：在同步 `P2pNcclAFDConnector` 基线上实现可回退的流级和流水线级
   Attention–FFN 异步执行，而不是只增加一个名为 `async` 的配置项。
3. **性能结论**：在相同模型、请求集、并发、并行度和预热条件下，对 native vLLM、
   AFD sync、AFD async 给出可复现的正确性、稳定性和性能对比。

当前已确认的起点是 DeepSeek-V2-Lite `1A1F`、DP=1、eager、同步 P2P 冒烟通过。
它只证明最小路径可用，不代表下方矩阵已经完成。

## 2. 与 GLM-5.2 目标的隔离

| 项 | 本目标 | GLM-5.2 目标 |
|---|---|---|
| 模型 | DeepSeek-V2-Lite | GLM-5.2 |
| 量化 | 模型现有格式，不做 W4AFP8 适配 | FP8/W4AFP8 与权重 remap |
| 主要风险 | AFD 功能矩阵、DBO、GPU 异步和性能 | 架构、量化和权重加载兼容 |
| 日志目录 | `experiment/logs/dsv2_v025/` | 现有 GLM 专项目录/文件 |
| 结果目录 | `experiment/results/dsv2_v025/` | 现有 GLM 专项目录/文件 |
| 报告 | 本文及后续 DS-V2-Lite 报告 | `PORT_TO_VLLM_0.25.0.md` 等 |

两项目标可以共享已经提交的通用 vLLM 0.25.0 兼容代码，但不能共享模型正确性、
量化正确性、E2E 或性能验收结论。

## 3. 执行约束与基线

- 所有 GPU 实测只在 gpu-host 上进行；开发机不运行 GPU 实验。
- 实验前必须获得用户明确的“业务服务已停”确认，不主动停止生产服务。
- 基线必须记录：git commit、dirty diff、镜像名称与 digest、vLLM/torch/CUDA/NCCL
  版本、模型路径、GPU 拓扑、环境变量、命令、端口和原始日志。
- v0.25.0 当前必须设置 `VLLM_USE_V2_MODEL_RUNNER=0`。
- 每个 case 结束后检查 vLLM 进程、GPU compute process 和监听端口；失败也必须清理。
- 先验证单一变量，再测试组合，避免把 graph、DBO、DP 和异步同时引入。

## 4. 功能验证矩阵

### 4.1 现有 GPU E2E 套件（20 项）

| 类别 | 数量 | 验证内容 | 状态 |
|---|---:|---|---|
| serving | 3 | completion、usage、非法模型 | 3/3 通过 |
| graph | 1 | `1A1F` eager 与 `FULL_DECODE_ONLY` 输出一致 | 1/1 通过 |
| TP | 2 | `2A2F/TP=2` eager、graph | 2/2 通过 |
| profiler | 2 | Attention/FFN eager、graph trace 产出 | 2/2 通过 |
| GSM8K | 2 | `1A1F` eager、graph 精度 | eager/graph 50题均以 0.24 通过有效阈值 0.15；全量仍待完成 |
| DS-V2-Lite 模型矩阵 | 10 | `2A2F` 的 base/TP/DBO/profiler 乘 eager/graph | 10/10 通过 |

运行完整套件前必须完成 GPU、模型、vLLM、插件、pytest 和 `lm_eval` preflight。
`lm_eval` 只允许单独安装在实验环境，不加入 `pyproject.toml` 或 `uv.lock`。

### 4.2 需要补充的目标级验证

| 维度 | Case | 主要判据 |
|---|---|---|
| 最小链路 | `1A1F` eager | HTTP、连续文本、显存拆分、NCCL 往返 |
| 拓扑 | `1A2F`、`2A1F`、`2A2F`、`4A4F` | `1A2F/2A1F/2A2F` 修复后通过；`4A4F` 修复前通过，待防回归 |
| 并行 | DP1TP2、DP2TP1、DP2TP2 | request 分配、DP metadata、输出一致 |
| 执行模式 | eager、graph、DBO、graph+DBO | 单变量正确后再验证组合 |
| 部署 | prefill/decode 共置与分离 | API 路由、角色生命周期、故障清理 |
| 稳定性 | 重复启停、长序列、高并发、异常请求 | 无 hang、泄漏、残留端口或进程 |
| 精度 | 固定 prompts、GSM8K | greedy 输出一致；精度不低于既定阈值 |

功能阶段的完成门是：现有 20 项 GPU E2E 全部通过，补充矩阵逐项有原始日志，所有
skip 都有明确且被接受的范围理由；不能用“未失败”替代“已执行并通过”。

## 5. 同步性能基线

在异步改动前建立三组可比基线：native vLLM、AFD sync 无 DBO、AFD sync + DBO。

建议固定工作负载：

- 输入长度：128、512、2048 tokens。
- 输出长度：32、128、512 tokens。
- 并发：1、8、32、64、128。
- 拓扑：先 `1A1F`，再选择验证通过的 `2A2F`/`4A4F`。
- 每组使用相同请求集、seed、预热、请求率和最大并发，至少重复三次。

记录 TTFT、TPOT、E2E latency、request/token throughput、P50/P95/P99、角色显存、
GPU 利用率、NCCL 时间和失败请求数。CUDA graph 的 capture 时间与稳态 replay 必须分开。

## 6. GPU 异步执行路线

### A0：时序观测

先用 Nsight Systems/torch profiler 回答：

1. DBO 是否真的生成两个非空 ubatch；
2. NCCL 是否和 Attention/FFN kernel 位于同一 CUDA stream；
3. 当前跨设备已有多少 `A/u1 || F/u0` 重叠；
4. 全 batch 与半 batch 的 Attention、FFN、通信耗时；
5. 控制面、NCCL、CPU yield、graph capture 和 GPU idle 各占多少。

若半 batch kernel 效率或 A/F 失衡已使理论收益不足，先调整 batch/ubatch 触发条件，
再进入多 stream 实现。

### A1：`1A1F` eager 异步 MVP

- 使用独立通信 stream；compute 完成后记录 event，通信 stream 等待后发送。
- 为每个 ubatch/layer 使用稳定 receive buffer 和完成 event。
- compute stream 在消费 FFN 输出前等待 receive-complete event。
- 保持确定的 NCCL send/recv 匹配顺序。
- 使用双缓冲，避免发送完成前复用 Attention buffer。
- 默认仍走同步路径，异步路径通过独立 connector 或 connector-owned 配置启用。

### A2：FFN 侧流水线

- 为两个 ubatch 预贴 receive。
- `recv(u1)` 与 `FFN(u0)`、`send(u0)` 与 `FFN(u1)` 在依赖允许时重叠。
- 引入有界 slot/backpressure；请求、layer、ubatch 标识必须可校验。
- 测量 NCCL 与 MoE kernel 的 SM 竞争，必要时恢复或调整 SM control。

### A3：扩展与组合

按以下顺序扩展：

1. `1A1F` 连续请求与高并发；
2. `2A2F` 和 DP/TP；
3. DBO 参数扫描；
4. `4A4F`；
5. 最后处理 `FULL_DECODE_ONLY` graph 的多 stream capture/replay。

graph 之前不得把 eager MVP 的动态分配或临时 event/buffer 直接带入 capture 路径。

## 7. 异步验收标准

### 正确性

- 同一 greedy 请求下，AFD async 与 AFD sync 输出一致。
- prompt 冒烟不能只检查 HTTP 200 或非空输出；每条样例必须同时检查上下文语义
  衔接、预期关键内容、`finish_reason` 和 token usage 一致性。
- 现有 GPU E2E 在 async 支持范围内通过；新增异步错序、buffer 复用、shutdown 单测。
- GSM8K 不低于同步基线允许的误差范围。

### 稳定性

- 每个主要拓扑至少完成 30 分钟或 1000 请求长稳。
- 零 deadlock、错序、CUDA illegal access、NCCL timeout 和请求丢失。
- 正常/异常退出后无 GPU 进程、端口、process group 或 buffer 泄漏。

### 性能

- 目标通信受限场景：吞吐提升或 TPOT 降低至少 10%。
- 低并发相对同步路径退化不超过 5%。
- 显存增量单独报告，且能由配置的 buffer/slot 数解释。
- 若收益不足以覆盖复杂度，异步保持实验特性且默认关闭，并记录停止条件。

## 8. 阶段交付

| 里程碑 | 交付物 | 完成门 |
|---|---|---|
| M0 基线冻结 | 环境清单、提交、命令、独立目录 | 可从干净环境复现 |
| M1 功能完整 | 20 项 E2E + 补充矩阵报告 | 全部通过或明确范围化 skip |
| M2 同步画像 | sync/native 数据与 GPU timeline | 瓶颈有证据，不靠推测 |
| M3 异步 MVP | `1A1F` eager 实现、单测、E2E | 输出一致，无错序/泄漏 |
| M4 扩展优化 | DP/TP/DBO/4A4F/graph | 每阶段独立回归通过 |
| M5 最终结论 | 数据、日志、报告、回滚说明 | 正确性、稳定性、性能同时达标 |

## 9. 下一执行批次

当前机器窗口内按以下顺序继续，GLM-5.2 不进入本批次：

1. GSM8K eager/graph 50题小样本均已通过；全量 1319 题另行增加预算；
2. metadata 修复后的 `2A1F`、`2A2F` 已通过；资源释放后继续 `4A4F` 四条语义门禁；
3. 32 并发下的 async/sync paired 1000 请求长稳已完成：两组均 1000/1000 语义通过，
   且彼此逐项完全一致；
4. 异步性能门已失败，保持默认关闭，不扩展 graph/复杂拓扑；仅在找到降低 NCCL P2P
   SM 竞争或空等 receive 的明确方案后再开新一轮 paired benchmark；
5. 汇总功能矩阵、精度、稳定性和停止条件，形成 DS-V2-Lite 独立最终报告。

容器内 GPU E2E 的标准入口为：

```bash
AFD_V025_CONTAINER=afd-v025-validate \
AFD_GPU_E2E_MODEL=/models/DeepSeek-V2-Lite \
bash experiment/scripts/run_dsv2_v025_gpu_e2e.sh --category all
```

脚本不会停止服务、杀进程或安装依赖。它会先校验 vLLM `0.25.0`、模型、GPU、pytest、
插件和 `lm_eval`，然后把 collection、pytest、manifest 和 postflight 分别写入
`experiment/{logs,results}/dsv2_v025/<run-id>/`。

## 10. 当前运行状态（2026-07-28）

- gpu-host 的 `afd-v025-validate` 容器、vLLM `0.25.0`、8 张 H20、
  `/models/DeepSeek-V2-Lite` 和当前工作区插件加载均已核验。
- `1A1F` eager 观测运行成功；四条短 prompt 的输出经人工检查均能承接上下文并表达
  正确含义。原始 JSON 保存在
  `experiment/results/dsv2_v025/20260728T1202Z-observation/prompt-smoke.json`。
- 上述运行发生在语义断言加入脚本之前，因此只登记为“观测通过”，不作为自动化语义
  门禁的最终通过证据。
- 自动语义门禁随后在 `1A1F` eager 上重跑并通过：4/4 prompt 均满足预设关键语义、
  上下文衔接、`finish_reason` 和 token usage 断言。结果与完整日志位于
  `experiment/results/dsv2_v025/20260728T1218Z-semantic-retry1/`。
- 在成功运行前有一次 infra failure：其他进程在 vLLM 显存 profiling 期间释放显存，
  触发 vLLM 0.25.0 的一致性断言；未进入 prompt 请求阶段。失败日志保留在
  `experiment/results/dsv2_v025/20260728T1215Z-semantic-infra-failure/`。
- P2P connector CPU 单测在 gpu-host 的 v0.25.0 容器内通过：`25 passed`。
- GPU feature E2E 完整通过：`8 passed, 10 deselected`，用时 659.94 秒；原始
  collection、pytest、postflight 与 manifest 位于
  `experiment/{logs,results}/dsv2_v025/20260728T1229Z-features-retry2/`。
- feature 执行前两次 runner preflight 分别暴露了 `nvidia-smi` GPU 列表参数和 pytest 9
  collection 输出格式兼容问题，均在未启动 GPU case 前修正并保留记录。
- `lm_eval==0.4.12` 已在隔离环境 `/afd-v25/e2e-venv` 中安装并通过导入校验；未写入
  项目的 `pyproject.toml` 或 `uv.lock`。首轮 `20260728T1340Z-accuracy` 因缺少可选依赖
  `tenacity` 未进入推理。依赖补齐后的 `20260728T1346Z-accuracy-retry1` 由独立 subagent
  跟踪：eager 在 7200 秒预算内完成 1139/1319 个 API 请求，均为 HTTP 200，但在汇总
  exact-match 前超时；随后 graph 因外部任务使可用显存 `120.79 GiB` 低于配置所需
  `128.62 GiB` 而未启动。两项均不能登记精度 pass/fail，需以较小代表性子集先取得信号，
  再为全量评测调整预算或并行策略。
- 首轮冒烟结束后发现其他容器启动了 8-GPU 外部任务。该任务已由独立 subagent
  只读跟踪至退出；未停止或清理未知任务。语义门禁成功后，实验进程、端口和 GPU
  compute process 的 postflight 均为空。
- DS-V2-Lite model matrix 完整通过：`10 passed, 11 deselected`，用时 1434.20 秒；
  原始 collection、pytest、postflight 与 manifest 位于
  `experiment/{logs,results}/dsv2_v025/20260728T1243Z-models/`。
- 现有 GPU E2E 非 accuracy 部分累计 18/18 通过。accuracy 首轮依赖失败和第二轮
  eager 超时/graph 显存阻断均已如实保留，不登记为模型精度 pass/fail；小样本重试由
  独立 subagent 跟踪。
- 补充拓扑语义冒烟：`2A1F` 重试后 4/4 prompt 通过；首次启动因随机内部 rendezvous
  端口冲突未进入请求。成功证据位于
  `experiment/results/dsv2_v025/20260728T1324Z-topology-2a1f-retry1/`。
- `1A2F` 暴露真实功能失败：FFN DP/EP rank 1 在 vLLM naive DP/EP MoE dispatch 中访问
  越界的 `sizes[rank_in_group]`，API 未就绪、未生成 prompt。证据位于
  `experiment/results/dsv2_v025/20260728T1308Z-topology-1a2f/`。修复前不得把 fan-out
  拓扑标记为支持。
- `2A1F` 首次 infra failure 的 launcher 未及时传播内部 EngineCore 死亡，标准退出后仍
  留下一个忽略 SIGTERM 的孤立 worker；已按精确 PID 清理 GPU/端口，但容器 PID 1 未
  reap zombie。该 shutdown 行为需纳入稳定性修复。
- `4A4F` eager 自动语义门禁通过：4/4 prompt 均满足关键语义、上下文衔接、
  `finish_reason` 和 token usage 断言；结果与完整日志位于
  `experiment/{results,logs}/dsv2_v025/20260728T1330Z-topology-4a4f/`。结束后 GPU compute
  process 与实验服务均为空。
- `1A2F` 的 FFN forward-context metadata 修复已同步到 gpu-host。新增 fan-in、
  one-to-one、fan-out、TP 投影、dummy-token 参数化单测；隔离回归证据位于
  `experiment/results/dsv2_v025/20260728T1402Z-1a2f-metadata-unit/`。修复后 `1A2F`
  eager 四条语义 prompt 4/4 通过，结果位于
  `experiment/results/dsv2_v025/20260728T1608Z-topology-1a2f-fix-retry3/`；仍需补齐
  `2A1F/2A2F/4A4F` 防回归后才可关闭该功能项。
- GPU async eager MVP 已落到当前 repo：默认关闭，仅允许 `1A1F + eager + DBO 两个
  ubatch`；A2F/F2A 分别使用 connector-owned CUDA stream，以 event 建立 compute-ready、
  send-complete、recv-complete 和 input-consumed 依赖，receive cache 按 stage/peer 有界，
  close 先 drain。它与 metadata 单测在 gpu-host 隔离代码副本中合计 `59 passed`，证据位于
  `experiment/results/dsv2_v025/20260728T1527Z-async-mvp-unit/`。
- gpu-host CUDA/NCCL 正确性门已通过：同步参考 4/4、异步 4/4、连续三轮 12/12，异步
  completion 全文、`finish_reason` 和 usage 均与同步参考完全一致；每条输出同时满足
  prefix + semantic fragment 断言。证据位于 `20260728T1645Z-async-sync-reference`、
  `20260728T1649Z-async-mvp-semantic` 和 `20260728T1652Z-async-mvp-continuous`。
- profiler 进一步证明同步主 NCCL kernel 与 compute 共 stream 且观测重叠为 0；异步
  NCCL 与 compute stream 分离，在 Attention/FFN trace 分别观测到 `1071.07 us` /
  `135616.44 us` 重叠。证据和可复现解析脚本位于
  `20260728T1723Z-sync-profiler-timeline`、`20260728T1730Z-async-profiler-timeline` 及
  `experiment/scripts/analyze_dsv2_profiler_trace.py`。
- 两组固定负载 paired benchmark 的语义前后门禁均通过，但异步输出吞吐分别回退
  `3.43%` 和 `4.88%`，mean TPOT 分别增加 `4.64%` 和 `5.93%`，未达到 10% 收益门。
  限制 NCCL channel 到 4 的试验在 API/NCCL 初始化阶段超时，未产生性能样本。因此
  M3 的正确性与机制验证完成，但 M4 性能扩展停止；异步继续默认关闭。原始数据位于
  `20260728T1800Z-sync-fixed-benchmark` 至 `20260728T1930Z-async-comm-heavy-benchmark`，
  比较入口为 `experiment/scripts/compare_dsv2_async_benchmarks.py`。
- GSM8K eager 50题小样本由独立 subagent 跟踪完成：50/50 HTTP 200，strict/flexible
  exact_match 为 `0.24/0.26`，测试采用 `0.24`，高于有效阈值 `0.15`；证据位于
  `20260728T235000Z-accuracy-limit50-eager-retry1`。这提供正向精度信号，但不能替代
  eager/graph 全量评测。
- GSM8K graph 50题小样本同样由独立 subagent 跟踪完成：50/50 HTTP 200，
  strict/flexible exact_match 均为 `0.24`，通过有效阈值 `0.15`；证据位于
  `20260729T0140Z-accuracy-limit50-graph`。eager/graph 小样本一致，但仍不能外推为
  全量精度结论。
- metadata 修复后的 `2A1F` 防回归 4/4 通过，输出继续满足 prefix + semantic fragment
  断言，结束后 GPU 5/6/7 均恢复为 4 MiB 且无实验端口/进程。证据位于
  `20260728T2358Z-topology-2a1f-fix-regression`。
- GPU4 释放后，metadata 修复后的 `2A2F` 防回归也已 4/4 通过，结束后 GPU4–7
  均恢复为 4 MiB 且目标端口为空；证据位于
  `20260729T0210Z-topology-2a2f-fix-regression`。当前仅 `4A4F` 受 GPU0–3 外部任务阻断。
- 32 并发 paired 长稳已完成：async 与 sync control 各执行 1000 次请求，均为
  1000/1000 HTTP 200 且 1000/1000 通过上下文前缀与语义片段断言；两组结果逐项
  1000/1000 完全一致。相对串行参考的 248 条逐字差异都来自 Python completion 首尾
  换行位置，函数内容、token 数和 usage 不变，且 sync/async 分布相同，因此不属于
  async connector 引入的正确性退化。证据位于
  `20260729T0200Z-async-semantic-soak1000` 与
  `20260729T0205Z-sync-semantic-soak1000-control`。
- GSM8K eager 全量已完成：1319/1319 HTTP 200，strict/flexible exact_match 为
  `0.3791/0.3829`，通过有效阈值 `0.15`；证据位于
  `20260729T0215Z-accuracy-full1319-eager`。graph 首轮全量在 968/1319、零 HTTP 错误时
  命中旧的 7200 秒内层超时，不登记精度结论；显式 14400 秒的 retry 已以
  1319/1319 HTTP 200、strict/flexible `0.3798/0.3836` 通过，证据位于
  `20260729T0453Z-accuracy-full1319-graph-retry1`。eager/graph 全量精度一致通过。
