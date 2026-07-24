# afd-plugin 代码阅读地图

> 本文是一份"阅读地图"（reading map），目的是帮助读者在最短时间内建立对
> `afd-plugin` 仓库的整体心智模型：它是什么、代码怎么分层、执行流怎么走、历史怎么
> 演进过来的。**它是讨论实现细节的起点，不是最终的实现文档。** 标注 `⚠️待确认`
> 的地方是我从代码结构推断、但尚需与你确认的点。
>
> 最后更新对应提交：`ea7c56a`（main，共 404 次提交）。

---

## 0. 一句话定位

`afd-plugin` 是 [vLLM](https://github.com/vllm-project/vllm) 的**外部插件**，实现
**Attention-FFN Disaggregation（AFD，注意力/前馈解耦）**：把 Transformer 的
Attention 计算和 FFN（MoE）计算拆到不同的进程/设备角色上，通过"连接器"
（connector）在两侧之间传输 hidden states。它**不修改 vLLM 源码**，而是通过
`vllm.general_plugins` 入口点 + `--additional-config` + 角色化 worker + 窄范围
兼容补丁来注入 AFD 行为。目标运行时是 **vLLM `v0.19.1`**，同时支持
**CUDA GPU** 和 **Ascend NPU** 两条后端路径。

核心概念三元组：

- **Role（角色）**：`attention` 或 `ffn`。Attention 侧接收请求、算注意力、发出
  hidden states；FFN 侧接收 hidden states、算 MoE、回传结果。请求只发给 Attention
  侧 API server；FFN 是 connector 驱动的。
- **Connector（连接器）**：两个角色之间的通信契约与实现（P2P NCCL / CAM P2P /
  CAM async）。
- **Platform（平台）**：CUDA 与 Ascend NPU，两套 worker/runtime/算子实现。

---

## 1. 推荐阅读顺序（先读这些）

如果你只想快速上手，按这个顺序读：

1. **[README.md](../README.md)** — 定位、安装、启动命令、配置形状、当前支持矩阵。
2. **[docs/design/module/index.md](design/module/index.md)** — 官方模块设计索引，
   给出了"生产代码路径 → 归属文档"的权威映射表，是最好的导航起点。
3. 本文档（代码阅读地图）— 补充历史演进 + 执行流串讲。
4. 之后按"实现细节讨论"逐层深入 `afd_plugin/` 源码。

官方设计文档（`docs/design/module/`）已经比较完整，本地图与之互补：
设计文档偏"契约/边界/不变量"，本地图偏"从哪读起 + 历史脉络 + 执行流叙事"。

| 官方设计文档 | 覆盖内容 |
| --- | --- |
| [plugin_boundary.md](design/module/plugin_boundary.md) | 注册、配置、上游边界 |
| [attention_runtime.md](design/module/attention_runtime.md) | Attention 角色生命周期与执行流 |
| [ffn_runtime.md](design/module/ffn_runtime.md) | FFN 角色生命周期与执行流 |
| [connector_contracts.md](design/module/connector_contracts.md) | 角色间的握手契约 |
| [model_integration.md](design/module/model_integration.md) | 模型层接入（DeepSeek） |
| [execution_platforms.md](design/module/execution_platforms.md) | CUDA/NPU 机制（图、DBO、profiler、算子） |
| [compatibility_and_patches.md](design/module/compatibility_and_patches.md) | 上游兼容补丁 |

---

## 2. 代码架构分层

代码约 15k 行 Python（`afd_plugin/`）+ 一批 Ascend C++/AscendC 算子（`csrc/npu/`）。
分层如下（依赖方向：角色/模型层 → 共享边界层 → 平台/兼容层）：

```
┌─────────────────────────────────────────────────────────────────┐
│ 插件边界层  afd_plugin/__init__.py  register_afd() @ 入口点          │
│   配置: config.py / config_utils.py / validation.py / envs.py      │
└───────────────┬─────────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────┐   ┌──────────────────────────┐
│ 角色运行时层  afd_plugin/v1/worker │   │ 模型层  model_executor     │
│   GPU: attention_/ffn_ worker      │   │  models/deepseek_v2.py     │
│        + model_runner              │◄─►│  forward_context.py        │
│   NPU: v1/worker/npu/*             │   │  npu/ 变体                 │
│   图/DBO/ubatch: cuda_graph.py,    │   └──────────────────────────┘
│        dbo.py, ubatch_wrapper.py   │
└───────────────┬──────────────────┘
                │  send/recv hidden states
┌───────────────▼──────────────────────────────────────────────────┐
│ 连接器层  afd_plugin/connectors                                     │
│   契约: base.py (AFDConnectorBase / AFDControlPlane)               │
│   工厂: factory.py    元数据: metadata.py                          │
│   GPU: gpu/p2p.py (P2pNcclAFDConnector)                            │
│   NPU: npu/camp2p.py (同步) / npu/async_cam.py (异步)              │
│   拓扑/进程组: afd_plugin/distributed/*                            │
└───────────────┬──────────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────┐   ┌──────────────────────────┐
│ 平台/兼容层  compat/               │   │ 原生算子  csrc/npu         │
│   compat/vllm.py 版本门禁          │   │  a2e / e2a ACLNN 算子      │
│   compat/npu/* Ascend 运行时兼容   │   │  aclnn_torch_adapter       │
│   compat/patches/* 上游补丁        │   │  torch_extension           │
└──────────────────────────────────┘   └──────────────────────────┘
```

### 2.1 各层职责与关键文件

**A. 插件边界与注册** — [`afd_plugin/__init__.py`](../afd_plugin/__init__.py)
- `register_afd()` 是 `vllm.general_plugins` 入口点：无 vLLM 时安全 no-op（保证
  CPU/macOS 也能跑 import/config 测试）；有 vLLM 时依次做版本校验、应用兼容补丁、
  注册 DBO yield 自定义 op、按需应用 Ascend 补丁、注册 `AFD*` DeepSeek 模型架构。
- `_DEEPSEEK_MODEL_REGISTRATIONS`：把上游架构名映射到 `AFDDeepseek*ForCausalLM`
  包装类。
- 模块级 `__getattr__` 做惰性导出，避免 import 期强依赖 worker。

**B. 配置** — `config.py` / `config_utils.py` / `validation.py` / `envs.py`
- [`config.py`](../afd_plugin/config.py)：`AFDConfig` frozen dataclass，从
  `additional_config["afd"]` 解析。字段含 `role`/`connector`/`host`/`port`/
  `num_attention_ranks`/`num_ffn_ranks`/`afd_role_rank`/`compute_gate_on_attention`。
  保留 `afd_*` 兼容别名（`_ALIASES`）。
- `SUPPORTED_AFD_ROLES` / `SUPPORTED_AFD_CONNECTORS` 是白名单常量。
- [`validation.py`](../afd_plugin/validation.py)：`assert_compatible_afd_stack`
  等栈兼容性校验。

**C. 角色运行时层** — [`afd_plugin/v1/worker/`](../afd_plugin/v1/worker/)
- GPU 侧：
  - `attention_worker.py` → `AFDAttentionWorker(Worker)`
  - `attention_model_runner.py` → `AFDAttentionModelRunner(GPUModelRunner)`
  - `ffn_worker.py` → `AFDFFNWorker(Worker)`
  - `ffn_model_runner.py` → `GPUFFNModelRunner(LoRAModelRunnerMixin)`
  - `cuda_graph.py` → `AFDCUDAGraphPolicy` / `AFDGraphRunMode`（仅
    `FULL_DECODE_ONLY`）
  - `dbo.py` → Dual-Batch Overlap（双微批重叠）yield custom op
  - `ubatch_wrapper.py` → `AFDUBatchWrapper(UBatchWrapper)`
- NPU 侧：`v1/worker/npu/` 下有对应的 attention/ffn worker+runner、
  `forward_context.py`、`npu_ubatch_wrapper.py`、`ubatching.py`、`ubatch_utils.py`、
  `pcp_debug.py`。

**D. 连接器层** — [`afd_plugin/connectors/`](../afd_plugin/connectors/)
- [`base.py`](../afd_plugin/connectors/base.py)：核心契约。
  - `AFDConnectorBase`：生命周期（`init_afd_connector`/`close`/`is_initialized`）+
    数据路径（Attention 侧 `send_attn_output`/`recv_ffn_output`；FFN 侧
    `recv_attn_output`/`send_ffn_output`）。
  - `AFDControlPlane`：DP 元数据控制平面（`send/recv_dp_metadata_list`、
    `update_state_from_dp_metadata`）。可选，`None` 时 FFN 由 connector 接收循环
    直接驱动。
  - `ConnectorExtraInfo`：连接器自有配置基类，各连接器用 `parse_extra_config`
    严格校验 `connector_extra_config`。
- `factory.py`：`AFDConnectorFactory` 名称→实现的惰性注册表 + `create_connector`。
- `metadata.py`：传输数据结构族——`AFDDPMetadata`、`AFDControlPayload`、
  `AFDTransferState`、`AFDTransferMetadata`、`AFDTransferContext`、
  `AFDA2FTransferPayload`（Attn→FFN）、`AFDF2ATransferPayload`（FFN→Attn）、
  `AFDForwardContextMetadata`。
- 实现：
  - `gpu/p2p.py` → `P2pNcclAFDConnector` + `P2pNcclAFDControlPlane`（CUDA，同步，
    显式点对点传输）。
  - `npu/camp2p.py` → `CAMP2pAFDConnector` + 控制平面 + `_CAMP2PTopology`
    + `CAMP2PTransferState`（Ascend，同步，HCCL/CAMP2P 自定义 op）。
  - `npu/async_cam.py` → `CAMAsyncAFDConnector` + `AFDAsyncTopology`
    + `AFDAsyncFFNWorkItem`（Ascend，异步 DP，要求 `async=true`）。
- `distributed/`：`afd_process_group.py`（AFD 专用进程组）、`topology.py`（rank
  排布；README 指出 FFN ranks 排在 Attention ranks 之前）。

**E. 模型层** — [`afd_plugin/model_executor/`](../afd_plugin/model_executor/)
- `models/deepseek_v2.py`：`AFDDeepseek*ForCausalLM` 包装类。每个角色只构造并加载
  自己需要的模型组件（共享 embedding/norm/output 在生命周期需要处仍可用）。
  支持 DeepSeekV2 / V3 / V3.2（V3.2 复用 `AFDDeepseekV3ForCausalLM`）。
- `models/forward_context.py`、`model_utils.py`。
- `models/npu/`：`deepseek_v2_async_cam_forward.py`、
  `deepseek_v2_attention_gate.py`（NPU 专用 forward 变体，如把 gate 计算放在
  Attention 侧）。

**F. 平台/兼容层** — [`afd_plugin/compat/`](../afd_plugin/compat/)
- `compat/vllm.py`：`assert_vllm_version_supported` 版本门禁。
- `compat/npu/`：Ascend 运行时兼容（`runtime.py`、`runtime_config.py`、
  `forward_context.py`、`ops.py`、`profiler.py`、`feature_validation.py`），
  含 `ensure_afd_ascend_ops_loaded` / `apply_afd_ascend_patches_if_needed`。
- `compat/patches/`：上游补丁（`async_dp_engine.py`、
  `async_dp_forward_context.py`、`config_validation.py`、`engine_core.py`）+
  `patches/npu/`（`ascend_platform.py`、`force_load_balance.py`）。
  **补丁受 [AGENTS.md](../AGENTS.md) 严格约束**：必须用
  `# ### PATCH START/END` 标注、签名与上游一致、注释说明补丁原因。

**G. 原生算子** — [`csrc/npu/`](../csrc/npu/)
- `a2e/`（Attention→Expert/FFN）、`e2a/`（Expert/FFN→Attention）两组 AscendC
  算子（`op_host` + `op_kernel`）。
- `aclnn_torch_adapter/`（NPUBridge/NPUStorageImpl，把 ACLNN 接进 torch）、
  `torch_extension/`（torch binding）。
- 构建：`build.sh` / `build_aclnn.sh` / `CMakeLists.txt`，由 `setup.py` 在 Ascend
  环境自动触发（`AFD_BUILD_ASCEND_OPS` 可覆盖）。`csrc/gpu/` 目前是预留位。

---

## 3. 端到端执行流（叙事版）

以 GPU P2P 同步连接器为例（`⚠️待确认`：以下是按契约与命名推断的时序，细节需对代码
逐一核对）：

**启动期**
1. vLLM 加载插件 → `register_afd()` 应用补丁、注册 `AFD*` 模型架构。
2. 用户 `vllm serve ... --additional-config '{"afd":{...}}'` 分别拉起 Attention
   与 FFN 两个进程；省略 `--worker-cls` 时插件按平台+角色自动选 worker。
3. 各 worker 通过 `AFDConnectorFactory.create_connector` 构造连接器，
   `init_afd_connector()` 建进程组/通信器。

**推理期（每层/每 stage）**
4. Attention 侧算完注意力 → `send_attn_output(hidden_states, context)` 发到 FFN。
5.（可选）控制平面：Attention 侧 `send_dp_metadata_list`，FFN 侧
   `recv_dp_metadata_list` + `update_state_from_dp_metadata` 准备缓冲/形状。
6. FFN 侧 `recv_attn_output()` 拿到 `AFDA2FTransferPayload` → 算 MoE →
   `send_ffn_output(ffn_output, context)` 回传。
7. Attention 侧 `recv_ffn_output(ref_tensor)` 拿回结果，继续后续层直到出 token。

**关键不变量**：请求只进 Attention 侧；FFN 的 `execute_model()` 若被 scheduler
直接调用会 fail-fast（FFN 只应由 connector 驱动）。

**平台差异**：NPU 同步走 `CAMP2pAFDConnector`（HCCL + a2e/e2a 算子）；NPU 异步走
`CAMAsyncAFDConnector`（CAM async-DP，`async=true`，不支持图捕获）。

---

## 4. 连接器支持矩阵（速查）

| 连接器 | 平台 | 推荐阶段 | 同步/异步 | 图支持 | 关键约束 |
| --- | --- | --- | --- | --- | --- |
| `P2pNcclAFDConnector` | CUDA | Decode | 同步 | `FULL_DECODE_ONLY` CUDA graph | `num_attention_ranks ≥ num_ffn_ranks` 且可整除；FFN ranks 排在前 |
| `CAMP2pAFDConnector` | Ascend NPU | Decode | 同步 | `FULL_DECODE_ONLY` ACL graph | HCCL/CAMP2P 自定义 op，NPU 默认构建 |
| `CAMAsyncAFDConnector` | Ascend NPU | Prefill | 异步 | 不支持 | CAM async-DP，需 `async=true` |

已知短板（来自 README）：仅支持 vLLM `0.19.1`；不支持 model runner v2；GPU CUDA
graph 仅 `FULL_DECODE_ONLY`；GPU DBO+CUDA graph 仅限恰好两个 ubatch。

---

## 5. 提交历史演进脉络

404 次提交，主力贡献者 `jiangkuaixue123`（309）、`specture724`（51）、`zzh`（20）、
`yujuancao07`（12）等。大量工作以 Codex agent 分支 + PR 合并的方式推进。按主题可
归纳为几个阶段（非严格时间线，有交叠）：

**阶段 1 · 脚手架与起步**
- `Initial commit` → GitHub issue/PR 模板 → 迁移指南 → DeepSeekV2-Lite 示例。
- 建立插件骨架、配置解析、GPU P2P 连接器雏形。

**阶段 2 · NPU 适配**
- NPU 连接器契约、CAMP2P、DBO ubatch 重叠修复（#2 fix-npu-afd-dbo-ubatch-overlap）、
  NPU FFN graph key、profiler stack toggle。
- 引入 Ascend 算子（csrc/npu a2e/e2a）与 `csrc/` 按后端拆分（#44 split-csrc-backends）。

**阶段 3 · 大规模重构（契约收敛）**
- 控制平面拆分：#128 `refactor/SplitControlPlain`（把 connector trigger/控制平面
  从 model runner 中剥离）、#79 dpmetadataComm。
- 元数据体系整理：#108 `MetadataClassRenaming`（`AFDMetadata` 重命名）、
  #119 `RemoveMetadataHelpers`、#145 `StateSplit`（`AFDCustomTransferState`
  扁平化为 `AFDTransferState`，"HUGE REFACTOR"）。
- 连接器重命名与分组：#103 rename-connectors、按后端拆
  `connectors/gpu` 与 `connectors/npu`。

**阶段 4 · 配置与运行时清理**
- #96 RFC89 AFD config cleanup、#137 移除未用环境变量、#132 移除过时 multistream
  配置、#147 按平台自动选择 AFD worker、#144 为非 AFD worker 保留原生模型注册。
- 目录改名：#134 Ascend → NPU 目录重命名。

**阶段 5 · 文档与配方体系**
- 模块设计文档三阶段（#133 module-design-phase-3）、架构图（#116）、
  连接器用户指南（NCCL P2P / CAM P2P / CAM async）、
  recipe 按"连接器/模型"组织（#125）、DeepSeekV3.2 recipe（#67）、
  Ascend NPU 安装指南、DBO/capture 可调值文档（#148）。

> 阅读历史的建议入口：`git log --oneline --merges` 看 PR 标题即可把握主题演进；
> 每个 `refactor/*` 分支的合并提交对应一次契约层面的收敛。

---

## 6. 测试与配方（验证代码怎么跑）

- **单元测试** [`tests/unit/`](../tests/unit/)：CPU 安全，`uv run pytest` 默认跑。
  覆盖 config、connectors（各连接器 + base/factory）、compat/patches、
  v1/worker（cuda_graph、dbo、model runner、npu runtime、classpaths）、
  model_executor、package（打包与 Ascend 构建文件）。
- **E2E 测试** [`tests/e2e/`](../tests/e2e/)：需真实硬件 + 模型权重，opt-in。
  分 `accuracy`（gsm8k）、`features`（graph/tp/profiler/serving/ops）、
  `models/deepseek_v2_lite`。`tests/e2e/runner.py` 是本地冒烟入口，
  也有 [`run-e2e` skill](../.agents/skills/run-e2e/SKILL.md)。
- **配方** [`recipe/`](../recipe/README.md)：按 `平台/连接器/模型` 组织的可运行
  启动脚本（GPU P2P 的 colocation/disaggregation × eager/graph × dbo 组合；
  NPU CAMP2p / CAMAsync 的 DeepSeekV3.2）。

---

## 7. 贡献约束（读代码前必看）

[AGENTS.md](../AGENTS.md) 是硬性规范，尤其：
- **补丁要求**：所有 `compat/patches/` 补丁必须严格架构评审，用
  `# ### PATCH START/END` 标注 AFD 差异，签名与上游一致，注释说明补丁原因与新增参数。
  补丁基于固定的 vLLM / vLLM-Ascend tag 开发。
- **优先级**：AFD 功能优先通过 继承/组合 实现，其次才是补丁。
- Python 规范：imports 置顶（循环/惰性加载例外）、避免新可变全局、无魔法数字、
  描述性命名、直接访问上游属性（不用 `getattr`/`hasattr` 以便静态检查暴露上游变更）。

---

## 8. 待与你确认 / 深入讨论的清单

以下是我在读代码时标记的、需要你补充或一起深入的点，可作为后续完善本文档的议程：

1. **执行流细节**：第 3 节的时序是从契约推断的，需要对着
   `attention_model_runner.py` / `ffn_model_runner.py` 逐层核对每层调用点。
2. **控制平面 vs 接收循环**：`AFDControlPlane` 为 `None` 与非 `None` 两种驱动模式
   分别对应哪些连接器、FFN 侧循环具体长什么样。
3. **DBO / ubatch**：`dbo.py` yield custom op 的机制、与 CUDA graph "恰好两个
   ubatch" 约束的关系。
4. **NPU 异步路径**：`CAMAsyncAFDConnector` 的 `AFDAsyncFFNWorkItem` 工作队列模型、
   与 `async_dp_engine` 补丁的协作。
5. **`compute_gate_on_attention`**：gate 放在 Attention 侧计算的取舍与数据路径影响。
6. **拓扑与 rank 排布**：`distributed/topology.py` 中 FFN/Attention rank 排序规则
   与各连接器的 `world_rank` 计算差异。
7. **原生算子**：`csrc/npu` 的 a2e/e2a 算子接口与 Python 侧 `compat/npu/ops.py`
   的绑定关系。

---

*本地图基于当前 `main` 分支静态阅读整理，随讨论逐步细化为完整代码阅读文档。*
