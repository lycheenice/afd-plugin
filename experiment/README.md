# experiment/ 文档索引

> AFD 插件实验 + GLM-5.2 移植的所有记录入口。新会话先读本索引 + `EXPERIMENT_ENV.md`。

## 环境 / 工作流
- [`EXPERIMENT_ENV.md`](./EXPERIMENT_ENV.md) — 三机分工(dev-host 编辑 / archive-host 归档推 github / gpu-host 跑实验)、同步流、**实验前须等用户停服务**。
- 记忆:`afd-plugin-workflow.md`(工作流规则)、`w4afp8-glm52-afd-debug.md`(W4AFP8 调试轨迹)。

## GLM-5.2 移植到 vLLM 0.25.0(2026-07-27/28,进行中)
- [`MORNING_SUMMARY_20260727.md`](./MORNING_SUMMARY_20260727.md) — **先读**:决定性结论 + 当前状态 + 下一步。
- [`ANALYSIS_glm52_vllm0191_blocker_20260727.md`](./ANALYSIS_glm52_vllm0191_blocker_20260727.md) — 根因:0.19.1 缺 GLM-5.2 attention backend;0.25.0 native 正确。
- [`PORT_TO_VLLM_0.25.0.md`](./PORT_TO_VLLM_0.25.0.md) — **移植主文档**:分层、故障排查(#1-#5)、Priority-1 修复、Priority-2 AFD 栈依赖差异、roadmap。
- 脚本:`scripts/a1_glm52_fp8_tp8_native.sh`(0.19.1 诊断)、`scripts/port_v25_setup_and_serve.sh` + `scripts/port_v25_smoke.sh`(0.25.0 native W4AFP8 验证)。

## 早期 AFD 实验(DeepSeek-V2-Lite / GLM-5.2,vLLM 0.19.1)
- [`EXPERIMENT_REPORT.md`](./EXPERIMENT_REPORT.md)、[`ANALYSIS_REPORT.md`](./ANALYSIS_REPORT.md)、[`PLAN_PHASE2.md`](./PLAN_PHASE2.md)、[`W4AFP8_DESIGN.md`](./W4AFP8_DESIGN.md)、[`W4AFP8_STATUS_AND_PERF_REPORT.md`](./W4AFP8_STATUS_AND_PERF_REPORT.md)、[`SESSION_STATUS.md`](./SESSION_STATUS.md)。

## 文档约定
afd-plugin / 移植 / vLLM 改造 的所有工作(设计、脚本、代码改动、故障排查、分析)都记入 `experiment/` 下对应文档,并同步 archive-host。
