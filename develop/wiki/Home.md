# AFD Plugin Wiki

> 本 Wiki 是对 `afd-plugin` 仓库代码架构与关键设计的系统性整理，**以源码为准**
> （每个论点尽量给出 `文件:行号` 引用），并与仓库已有的
> [设计文档](../../docs/design/module/index.md)、
> [代码阅读地图](../../docs/CODE_READING_MAP.md) 互补：
> 设计文档偏"契约 / 边界 / 不变量"，阅读地图偏"从哪读起 + 历史脉络"，
> 本 Wiki 偏"架构全景 + 按主题逐层展开的、可导航的知识库"。

## 一句话定位

`afd-plugin` 是 [vLLM](https://github.com/vllm-project/vllm) 的**外部插件**，
实现 **Attention-FFN Disaggregation（AFD，注意力 / 前馈解耦）**：把 Transformer
的 Attention 计算与 FFN（MoE）计算拆分到不同进程 / 设备角色上，通过"连接器"
（connector）在两侧之间传输 hidden states。它**不修改 vLLM 源码**，而是通过
`vllm.general_plugins` 入口点 + `--additional-config` + 角色化 worker + 窄范围
兼容补丁注入 AFD 行为。目标运行时 **vLLM `v0.19.1`**，同时支持 **CUDA GPU** 与
**Ascend NPU** 两条后端。

## 核心概念三元组

| 概念 | 含义 |
| --- | --- |
| **Role（角色）** | `attention` 或 `ffn`。Attention 侧接收请求、算注意力、发出 hidden states；FFN 侧接收 hidden states、算 MoE、回传结果。请求只发给 Attention 侧 API server；FFN 由 connector 驱动。 |
| **Connector（连接器）** | 两个角色之间的通信契约与实现：`P2pNcclAFDConnector`（GPU 同步）、`CAMP2pAFDConnector`（NPU 同步）、`CAMAsyncAFDConnector`（NPU 异步）。 |
| **Platform（平台）** | CUDA 与 Ascend NPU，分别对应两套 worker / model runner / 算子实现。 |

## 阅读导航

按主题组织的页面（建议按顺序阅读，但每页可独立查阅）：

| 页面 | 覆盖内容 |
| --- | --- |
| [01 - 总览与设计哲学](01-Overview.md) | AFD 是什么、为什么这样做、非侵入式插件边界、支持矩阵。 |
| [02 - 代码架构分层](02-Architecture.md) | 分层图、模块地图、依赖方向、关键文件索引。 |
| [03 - 插件边界与配置](03-Plugin-Boundary.md) | 入口点注册、`AFDConfig` 解析、校验、自动 worker 选择。 |
| [04 - Attention 运行时](04-Attention-Runtime.md) | Attention worker / model runner 生命周期与执行流（GPU + NPU）。 |
| [05 - FFN 运行时](05-FFN-Runtime.md) | FFN worker / model runner、connector 驱动、fail-fast 不变量。 |
| [06 - 连接器](06-Connectors.md) | 连接器基类契约、工厂、元数据体系、三种实现、拓扑与进程组。 |
| [07 - 模型集成](07-Model-Integration.md) | DeepSeek 包装类、按角色加载模型组件、NPU 变体。 |
| [08 - 执行平台机制](08-Execution-Platforms.md) | CUDA graph、DBO、ubatch、NPU 运行时、原生算子、profiler。 |
| [09 - 兼容补丁](09-Compatibility-Patches.md) | 补丁策略、版本门禁、逐个补丁说明。 |
| [10 - 端到端数据流](10-Data-Flow.md) | 启动 → 握手 → 推理 → 关闭 的完整时序叙事。 |
| [11 - 构建、打包与测试](11-Build-Packaging-Tests.md) | setup.py、Ascend 算子构建、CMake、单元 / E2E 测试、配方。 |
| [12 - 术语表](12-Glossary.md) | Role / Connector / Platform / ubatch / DBO 等术语速查。 |

## 约定

- **语言**：正文中文，代码标识符 / 文件路径 / CLI / 配置键保留英文原样。
- **引用**：源码引用使用 `路径:行号` 形式，便于在编辑器中跳转。
- **准确性**：所有结论以**当前 `main` 分支源码**为准；与设计文档冲突时，以源码为准
  并注明差异。推断或待确认处以 `⚠️` 标注。
- **图示**：必要时用 Mermaid 或 ASCII 框图。
- **不重复**：本 Wiki 不复述 `AGENTS.md` 的开发规范原文，只在相关处链接过去。

## 与既有文档的关系

- 想看"契约 / 边界 / 不变量"的权威定义 →
  [`docs/design/module/`](../../docs/design/module/index.md)
- 想看"从哪读起 + 历史演进脉络" →
  [`docs/CODE_READING_MAP.md`](../../docs/CODE_READING_MAP.md)
- 想看"架构全景 + 可导航知识库" → 本 Wiki
- 想看部署 / 基准命令 → [`recipe/`](../../recipe/README.md)、
  [`docs/gpu/`](../../docs/gpu/H200_MINIMAL_DEPLOYMENT.md)、
  [`docs/npu/`](../../docs/npu/CAM_P2P_CONNECTOR_USER_GUIDE.md)
- 开发规范（补丁要求、命名等）→ [`AGENTS.md`](../../AGENTS.md)
