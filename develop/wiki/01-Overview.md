# Overview

AFD（Attention-FFN Disaggregation，注意力/前馈解耦）是 [vLLM](https://github.com/vllm-project/vllm) 的一个外部插件，把 Transformer 的 Attention 计算与 FFN（含 MoE）计算拆分到不同的进程/设备角色上，二者通过"连接器"（connector）传输 hidden states。本页给出定位、动机、支持矩阵、已知短板与极速上手；分层与模块细节见 [架构](02-Architecture.md)，插件边界细节见 [插件边界](03-Plugin-Boundary.md)。

## AFD 解决什么问题

传统单进程推理把 Attention 与 FFN 捆绑在同一组设备上。对于 DeepSeek V2/V3 这类 MoE 模型，两部分对算力与显存的消耗特征差异很大：

- Attention 侧持有并持续增长 KV-cache，访存压力大。
- FFN/MoE 侧是计算密集的专家路由与专家计算，KV-cache 占用小。

AFD 的核心动机是把两者解耦到独立角色：

- **按 stage 独立扩缩容**：Attention 与 FFN 可部署到不同数量的设备/rank，各自按负载特征配比（见 `afd_plugin/config.py:58` 的 `num_attention_ranks` 与 `num_ffn_ranks`）。
- **KV-cache 与 MoE 计算资源分离**：Attention 进程专管 KV-cache 与注意力计算；FFN 进程专管 MoE，互不争用。
- **角色间用连接器传输 hidden states**：Attention 算完注意力后把 hidden states 发给 FFN，FFN 算完 MoE 回传结果，由连接器实现具体的通信契约（详见 [连接器](06-Connectors.md)）。

核心概念三元组：

- **Role（角色）**：`attention` 或 `ffn`（`afd_plugin/config.py:22`）。Attention 接收请求、算注意力、发出 hidden states；FFN 接收 hidden states、算 MoE、回传。请求只发给 Attention 侧 API server；FFN 由连接器驱动，若被 scheduler 直接调用 `execute_model()` 会 fail-fast。
- **Connector（连接器）**：两角色间的通信契约与实现，共三种（见下文支持矩阵）。
- **Platform（平台）**：CUDA GPU 与 Ascend NPU，各有一套 worker / model runner / 算子实现（见 [执行平台](08-Execution-Platforms.md)）。

## 为什么做成 vLLM 外部插件

AFD 不修改 vLLM 源码树，而是以非侵入式方式注入行为。设计文档 `docs/design/module/plugin_boundary.md:50` 把插件边界定位为"最低共享 AFD 层"，运行时/连接器/模型/平台各层均可消费它，但它不得反向 import 设备运行时依赖。

注入手段（均为外部插件机制）：

- **入口点注册**：`pyproject.toml:45` 声明 `vllm.general_plugins` 组入口点 `afd = "afd_plugin:register_afd"`。vLLM 启动时调用 `register_afd()`（`afd_plugin/__init__.py:66`）完成补丁应用、自定义 op 注册与模型架构注册。
- **`--additional-config` 配置通道**：AFD 通过 `additional_config["afd"]` 激活并配置（`afd_plugin/config.py:18` 的 `AFD_ADDITIONAL_CONFIG_KEY`）。存在即激活，省略即关闭，无需额外 CLI 开关。
- **自动选角色 worker**：省略 `--worker-cls` 时，插件按平台与角色自动选择 AFD worker（`afd_plugin/validation.py:70` 的 `afd_worker_qualname_for_platform_default`）。
- **插件自有模型包装类**：注册 `AFD*` 架构名到 `AFDDeepseek*ForCausalLM` 包装类（`afd_plugin/__init__.py:130`）。
- **窄范围版本兼容补丁**：仅针对固定 vLLM/vLLM-Ascend tag 的必要补丁（见 [兼容补丁](09-Compatibility-Patches.md)）。

这种边界使 AFD 与上游解耦：升级上游 tag 时，补丁按标记重新对齐即可，主路径通过继承/组合实现。

## 目标运行时与支持矩阵

### 运行时

- 目标运行时为 **vLLM `v0.19.1`**（`afd_plugin/compat/vllm.py:12` 的 `TARGET_VLLM_VERSION`）。`register_afd()` 在 `afd_plugin/__init__.py:88` 以 `strict=False` 做非严格版本检查；其它版本不被声明为支持。
- Python 范围 `>=3.10,<3.14`（`pyproject.toml:16`）。
- `vllm` 是可选 extra（`pyproject.toml:37`，`vllm==0.19.1`），便于无 CUDA 的 macOS/CPU 开发环境跑 import/config 测试。

### 模型支持

`_DEEPSEEK_MODEL_REGISTRATIONS`（`afd_plugin/__init__.py:47`）把上游架构名映射到 AFD 包装类：

| 上游架构名 | AFD 包装类 | 说明 |
| --- | --- | --- |
| `DeepseekForCausalLM` | `AFDDeepseekForCausalLM` | DeepSeek V2 系列 |
| `DeepseekV2ForCausalLM` | `AFDDeepseekV2ForCausalLM` | DeepSeek V2 |
| `DeepseekV3ForCausalLM` | `AFDDeepseekV3ForCausalLM` | DeepSeek V3 |
| `DeepseekV32ForCausalLM` | `AFDDeepseekV3ForCausalLM` | DeepSeek V3.2 复用 V3 包装类 |
| `GlmMoeDsaForCausalLM` | `AFDGlmMoeDsaForCausalLM` | GlM MoE DSA 架构 |

> ⚠️ 源码 `afd_plugin/__init__.py:60` 含 `GlmMoeDsaForCausalLM` 注册项，但 README "Model support" 表（`README.md:36`）未列出此项。以源码为准：插件确实注册了该架构。各角色只构造并加载自身所需的模型组件，共享 embedding/norm/output 在生命周期需要处仍可用。

### 连接器支持矩阵

白名单见 `afd_plugin/config.py:23` 的 `SUPPORTED_AFD_CONNECTORS`：

| 连接器 | 平台 | 推荐阶段 | 同步/异步 | 图支持 | 关键约束 |
| --- | --- | --- | --- | --- | --- |
| `P2pNcclAFDConnector` | CUDA | Decode | 同步 | `FULL_DECODE_ONLY` CUDA graph | `num_attention_ranks ≥ num_ffn_ranks` 且可整除；FFN ranks 排在 Attention ranks 之前 |
| `CAMP2pAFDConnector` | Ascend NPU | Decode | 同步 | `FULL_DECODE_ONLY` ACL graph | HCCL/CAMP2P 自定义算子；NPU 平台默认构建 |
| `CAMAsyncAFDConnector` | Ascend NPU | Prefill | 异步 | 不支持 | CAM async-DP 自定义算子；需 `async=true`（即 `async_dp`，`afd_plugin/config.py:34`）与 NPU worker |

异步约束由配置校验强制：`async_dp=true` 且 connector 非 `CAMAsyncAFDConnector` 会被拒（`afd_plugin/config.py:310`）；`CAMAsyncAFDConnector` 始终走 NPU worker 族（`afd_plugin/validation.py:121`）。详见 [连接器](06-Connectors.md)。

## 已知短板

来自 README "Known gaps"（`README.md:54`）：

- 仅声明支持 vLLM `0.19.1`，其它版本不作支持声明。
- 不支持 vLLM / vLLM-Ascend 的 model runner v2。
- GPU 与 NPU 的 E2E 测试为 opt-in，需真实硬件与模型权重。
- GPU CUDA graph 仅支持 `FULL_DECODE_ONLY`。
- GPU DBO（Dual-Batch Overlap）+ CUDA graph 仅限恰好两个 ubatch。

## 安装与使用极速入门

### 两条安装路径

**GPU（Linux/CUDA）**：需 uv。克隆仓库后安装 dev 组并加 `vllm` extra（pin `vllm==0.19.1`）：

```bash
uv sync --group dev --extra vllm
```

**Ascend NPU**：在 openEuler 22.03 (aarch64) + Ascend 910C / Atlas A3 上，先拉官方镜像 `quay.io/ascend/vllm-ascend:v0.19.1rc1-a3-openeuler`，在容器内于仓库根目录执行：

```bash
AFD_BUILD_ASCEND_OPS=1 \
SOC_VERSION=ascend910_9391 \
python -m pip install -v --no-build-isolation --no-deps -e .
```

`--no-deps` 保留匹配的 NPU 运行时，`--no-build-isolation` 使用其 CANN/torch-npu 工具链；`AFD_BUILD_ASCEND_OPS` 强制 Ascend 算子构建（不设则自动探测，详见 [构建打包测试](11-Build-Packaging-Tests.md)）。验证算子加载可用 `ensure_afd_ascend_ops_loaded`（`README.md:139`）。

### 配置形状

AFD 通过 vLLM `--additional-config` 配置，无独立 `--afd-config` 开关。规范形状（`README.md:235`）：

```json
{
  "afd": {
    "role": "attention",
    "connector": "P2pNcclAFDConnector",
    "host": "127.0.0.1",
    "port": 1239,
    "num_attention_ranks": 2,
    "num_ffn_ranks": 1,
    "afd_role_rank": 0,
    "compute_gate_on_attention": false,
    "connector_extra_config": {}
  }
}
```

- `role` 取 `attention` 或 `ffn`；`connector` 取三种连接器之一。
- 存在 `additional_config["afd"]` 且通过公共校验即激活；省略即关闭。
- 连器自有配置放 `connector_extra_config`，由所选连接器严格校验。
- 兼容别名 `afd_role`/`afd_connector`/`afd_host`/`afd_port`/`async` 仍接受（`afd_plugin/config.py:29` 的 `_ALIASES`）。

字段默认值与含义见 [插件边界](03-Plugin-Boundary.md)。

### 启动要点

- Attention 与 FFN 两侧各自 `vllm serve`，传入相同 connector/host/port 与各自 `role`。
- 省略 `--worker-cls` 时插件按平台+角色自动选 worker。
- **Attention 与 FFN 可任意顺序启动**（`README.md:209`）。
- **请求只发给 Attention 侧 API server**；FFN worker 由连接器驱动，scheduler 直接调用 FFN `execute_model()` 会 fail-fast。

GPU Attention 侧示例（`README.md:168`）：

```bash
vllm serve /path/to/DeepSeek-V2-Lite \
  --served-model-name deepseek-v2-lite-afd-attention \
  --enable-expert-parallel --enforce-eager \
  --host 127.0.0.1 --port 18000 \
  --additional-config '{"afd":{"role":"attention","connector":"P2pNcclAFDConnector","host":"127.0.0.1","port":6239,"num_attention_ranks":1,"num_ffn_ranks":1}}'
```

NPU 换用 `CAMP2pAFDConnector` 即可，插件自动选 NPU worker。本地冒烟可用 `tests/e2e/runner.py`（`README.md:216`）。数据如何在角色间流转见 [数据流](10-Data-Flow.md)，术语见 [术语表](12-Glossary.md)。
