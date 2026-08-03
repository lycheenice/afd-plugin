# DeepSeek-V2-Lite vLLM 0.25.0 实验结果索引

本目录只记录 gpu-host 上 DeepSeek-V2-Lite 专项目标的结果，与 GLM-5.2/W4AFP8 分开。

当前图文技术报告：`DSV2_LITE_V025_H20_REPORT.html`；其可审计源数据与布局定义为
`DSV2_LITE_V025_H20_REPORT.artifact.json`。全量 eager/graph 精度与修复后 4A4F
仍在后台运行，报告已按当前快照保留空白结果栏，完成后可原位补充。

| Run | 类型 | 结论 | 证据 |
|---|---|---|---|
| `20260728T1202Z-observation` | 1A1F eager prompt 观测 | 人工语义观测通过；不作为自动门禁 | `prompt-smoke.json`、`manifest.json` |
| `20260728T1210Z-p2p-unit` | P2P connector CPU 单测 | 25 passed | `result.txt` |
| `20260728T1215Z-semantic-infra-failure` | 自动语义冒烟首次启动 | infra failure，未发送 prompt | `prompt-smoke.log`、`manifest.json` |
| `20260728T1218Z-semantic-retry1` | 1A1F eager 自动语义门禁 | 4/4 passed | `prompt-smoke.json`、`prompt-smoke.log`、`manifest.json` |
| `20260728T1224Z-features` | feature runner preflight | collection 8/8，旧计数格式不兼容，未执行 case | `manifest.json`、对应 logs |
| `20260728T1226Z-features-retry1` | feature runner preflight | collection 8/8，`set -e` 提前退出，未执行 case | `manifest.json`、对应 logs |
| `20260728T1229Z-features-retry2` | GPU feature E2E | 8/8 passed | `manifest.txt`、对应 logs |
| `20260728T1243Z-models` | GPU model matrix E2E | 10/10 passed | `manifest.txt`、对应 logs |
| `20260728T1308Z-topology-1a2f` | 1A2F eager 拓扑 | 功能失败：FFN DP/EP metadata 越界 | `prompt-smoke.log`、`manifest.json` |
| `20260728T1314Z-topology-2a1f-infra-failure` | 2A1F eager 首次启动 | infra failure：内部端口冲突；未发送 prompt | `prompt-smoke.log`、`manifest.json` |
| `20260728T1324Z-topology-2a1f-retry1` | 2A1F eager 自动语义门禁 | 4/4 passed | `prompt-smoke.json`、`prompt-smoke.log`、`manifest.json` |
| `20260728T1330Z-topology-4a4f` | 4A4F eager 自动语义门禁 | 4/4 passed | `prompt-smoke.json`、`prompt-smoke.log`、`manifest.json` |
| `20260728T1340Z-accuracy` | GSM8K eager/graph 首轮 | infra failure：缺少 `tenacity`，未进入推理 | `manifest.txt`、对应 logs |
| `20260728T1346Z-accuracy-retry1` | GSM8K eager/graph 全量重试 | eager 1139/1319 请求后触发 7200 秒超时；graph 因外部显存占用未启动；无精度结论 | `manifest.txt`、对应 logs |
| `20260728T1402Z-1a2f-metadata-unit` | 1A2F metadata 本地补丁隔离 CPU 回归 | 47 passed | `pytest-unit.log`、`manifest.json` |
| `20260728T1527Z-async-mvp-unit` | 1A2F metadata + GPU async MVP 隔离 CPU 回归 | 59 passed | `pytest-unit.log`、`manifest.json` |
| `20260728T1600Z-topology-1a2f-fix-retry1` | 1A2F 修复后首次启动 | infra failure：`tests.e2e` 不可导入；未发送 prompt | `manifest.json`、对应 logs |
| `20260728T1602Z-topology-1a2f-fix-retry2` | 1A2F 修复后第二次启动 | infra failure：遗漏 model runner v2 禁用项；未发送 prompt | `manifest.json`、对应 logs |
| `20260728T1608Z-topology-1a2f-fix-retry3` | 1A2F 修复后 eager 自动语义门禁 | 4/4 passed，fan-out metadata 修复实机验证通过 | `prompt-smoke.json`、`manifest.json`、对应 logs |
| `20260728T1645Z-async-sync-reference` | 1A1F eager + DBO 同步参考 | 4/4 语义门禁通过 | `prompt-smoke.json`、`manifest.json`、对应 logs |
| `20260728T1649Z-async-mvp-semantic` | 1A1F eager + DBO 异步正确性 | 4/4，通过；全文、finish reason、usage 与同步参考完全一致 | `prompt-smoke.json`、`manifest.json`、对应 logs |
| `20260728T1652Z-async-mvp-continuous` | 异步连续请求 | 三轮 12/12，通过；逐项与同步参考完全一致 | `prompt-smoke.json`、`manifest.json`、对应 logs |
| `20260728T1723Z-sync-profiler-timeline` | 同步 DBO profiler | 4/4 语义通过；主 NCCL/compute 共 stream，观测重叠 0 us | trace、`prompt-smoke.json`、`manifest.json` |
| `20260728T1730Z-async-profiler-timeline` | 异步 DBO profiler | 4/4 且同步参考完全一致；NCCL/compute 分 stream，Attention/FFN 重叠 1071.07/135616.44 us | trace、`timeline-analysis.json`、`manifest.json` |
| `20260728T1800Z-sync-fixed-benchmark` | C16 固定负载同步参考 | 149.33 output tok/s，102.47 ms mean TPOT；前后语义 4/4 | `summary.json`、3 轮原始 JSON/log、`manifest.json` |
| `20260728T1830Z-async-fixed-benchmark` | C16 固定负载异步 | 语义参考前后完全一致；吞吐 -3.43%，TPOT +4.64%，性能门失败 | `summary.json`、3 轮原始 JSON/log、`manifest.json` |
| `20260728T1900Z-sync-comm-heavy-benchmark` | C64 长序列同步参考 | 588.77 output tok/s，103.57 ms mean TPOT；前后语义 4/4 | `summary.json`、3 轮原始 JSON/log、`manifest.json` |
| `20260728T1930Z-async-comm-heavy-benchmark` | C64 长序列异步 | 语义参考前后完全一致；吞吐 -4.88%，TPOT +5.93%，性能门失败 | `summary.json`、`comparison.json`、3 轮原始 JSON/log、`manifest.json` |
| `20260728T2000Z-sync-nccl4-benchmark` | NCCL 4-channel 调优对照 | infra failure：API/NCCL 初始化 900 秒超时；无性能样本 | `manifest.json`、对应 logs |
| `20260728T233615Z-accuracy-limit50-eager` | GSM8K eager 50题首次启动 | infra failure：虚拟环境无独立 pytest 入口；0 请求 | `manifest.txt`、对应 logs |
| `20260728T235000Z-accuracy-limit50-eager-retry1` | GSM8K eager 50题小样本 | 50/50 HTTP 200；strict/flexible exact_match 0.24/0.26；通过有效阈值 0.15 | `manifest.txt`、对应 logs |
| `20260728T2358Z-topology-2a1f-fix-regression` | metadata 修复后 2A1F eager 防回归 | 4/4 自动语义门禁通过 | `prompt-smoke.json`、`manifest.json`、对应 logs |
| `20260729T0140Z-accuracy-limit50-graph` | GSM8K graph 50题小样本 | 50/50 HTTP 200；strict/flexible exact_match 0.24/0.24；通过有效阈值 0.15 | `manifest.txt`、对应 logs |
| `20260729T0200Z-async-semantic-soak1000` | 1A1F eager + DBO + async，32 并发长稳 | 1000/1000 HTTP 200，1000/1000 语义通过；与 paired sync 逐项完全一致 | `prompt-smoke.json`、`manifest.txt`、对应 logs |
| `20260729T0205Z-sync-semantic-soak1000-control` | 1A1F eager + DBO sync，32 并发对照 | 1000/1000 HTTP 200，1000/1000 语义通过；与 async 逐项完全一致 | `prompt-smoke.json`、`manifest.txt`、对应 logs |
| `20260729T0210Z-topology-2a2f-fix-regression` | metadata 修复后 2A2F eager 防回归 | 4/4 自动语义门禁通过 | `prompt-smoke.json`、`manifest.json`、对应 logs |
| `20260729T0215Z-accuracy-full1319-eager` | GSM8K eager 全量 | 1319/1319 HTTP 200；strict/flexible exact_match 0.3791/0.3829；通过有效阈值 0.15 | `manifest.txt`、对应 logs |
| `20260729T0453Z-accuracy-full1319-graph-retry1` | GSM8K graph 全量 retry | 1319/1319 HTTP 200；strict/flexible exact_match 0.3798/0.3836；通过有效阈值 0.15 | `manifest.txt`、原始 results/samples、对应 logs |

当前已有 GPU E2E 非 accuracy 部分 18/18 通过。两项 GSM8K 尚无精度结论：eager 全量
评测在 86% 处超出 7200 秒预算，graph 被外部显存占用阻断。异步 eager MVP 已完成
gpu-host CUDA/NCCL 语义、同步参考一致性、连续请求和 timeline 机制验证。两组固定负载
性能门均失败，异步保持默认关闭且暂不扩展 graph/复杂拓扑。32 并发 paired 长稳中，
async/sync 各 1000/1000 语义通过且彼此逐项完全一致；两者相对串行参考均有 248 条仅
首尾换行位置不同，证明不是 async 特有退化。修复后的 2A1F、2A2F 防回归已通过；
4A4F 尚受八卡外部任务资源限制。GSM8K eager/graph 全量分别以 strict
`0.3791/0.3798` 通过，两种执行模式一致。graph 首轮全量在 968/1319、零 HTTP 错误时
命中旧的 7200 秒内层超时；显式 14400 秒的 retry 已完整通过并作为最终 graph 结论。
