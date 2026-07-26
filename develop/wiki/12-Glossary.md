# Glossary

本页是 AFD plugin 的术语速查表。每条给出中文名、英文原词、一句话定义与关键引用（`路径:行号` 或 Wiki 页链接）。按主题分组，最后一节放相关页面速查表。术语定义以当前 main 分支源码为准。

## 核心架构

**AFD（Attention-FFN Disaggregation，注意力/前馈解耦）**
将 Transformer 的 Attention 计算与 FFN（MoE）计算拆分到不同进程/设备角色、其间用连接器传输 hidden states 的推理范式。`afd_plugin/__init__.py:3` 的模块 docstring 即 "Attention-FFN Disaggregation support"。总览见 [01-Overview](01-Overview.md)。

**Plugin boundary（插件边界）**
"最低共享 AFD 层"——配置/契约/校验对上层运行时可见，但不得反向 import 设备运行时依赖；行为通过入口点 + `--additional-config` 注入，不修改 vLLM 源码。见 [03-Plugin-Boundary](03-Plugin-Boundary.md)、`afd_plugin/config.py:18` 的 `AFD_ADDITIONAL_CONFIG_KEY`。

**vllm.general_plugins entry point（入口点）**
声明于 `pyproject.toml:45-46` 的 `vllm.general_plugins` 组入口点 `afd = "afd_plugin:register_afd"`；vLLM 启动时调用 `register_afd()`（`afd_plugin/__init__.py:66`）应用补丁、注册自定义 op 与 `AFD*` 模型架构。无 vLLM 时该函数安全返回（`afd_plugin/__init__.py:80-83`），保证 CPU 安全。

**Patch（补丁）START/END 标记约定**
AFD 补丁复制上游函数后，用 `# ### PATCH START: <说明>` / `# ### PATCH END: <说明>` 圈出 AFD 改动，便于与上游比对；规则源自 `AGENTS.md`。实例见 `afd_plugin/compat/patches/config_validation.py:59`、`engine_core.py:45`、`async_dp_engine.py:112` 等共 17 处 `### PATCH START`。补丁策略见 [09-Compatibility-Patches](09-Compatibility-Patches.md)。

## 角色与配置

**Role（角色）**
一个进程承担的 AFD 职责，取值 `attention` 或 `ffn`。`AFDRole = Literal["attention","ffn"]`（`afd_plugin/config.py:20`），`SUPPORTED_AFD_ROLES`（`config.py:22`）。Attention 接请求、算注意力、发 hidden states；FFN 收 hidden states、算 MoE、回传；scheduler 直接调 FFN `execute_model()` 会 fail-fast。

**AFDConfig**
解析自 `additional_config["afd"]` 的 frozen dataclass，携带 connector/role/host/port/rank 数等拓扑信息。`afd_plugin/config.py:38-103`。激活规则：存在 `additional_config["afd"]` 且通过公共校验即激活，省略即关闭（`is_afd_active` `config.py:268-271`）。

**afd_role_rank**
本进程在其角色组内的 rank。`afd_plugin/config.py:62`，校验须 `0 <= afd_role_rank < 该角色 rank 数`（`config.py:340-344`）。

**num_attention_ranks / num_ffn_ranks**
拓扑中 Attention / FFN 的 rank 数。`afd_plugin/config.py:58` 与 `:60`，均须为正（`config.py:327-334`）。P2P 拓扑约束 `num_attention_ranks ≥ num_ffn_ranks` 且可整除（见 [06-Connectors](06-Connectors.md)）。

**compute_gate_on_attention**
Attention 侧是否在发出前先算 MoE gate 输出。`afd_plugin/config.py:64`。NPU 非 async 路径为 `True` 会被 `fail_if_unsupported_npu_afd_features` 拒绝（`afd_plugin/compat/npu/feature_validation.py:45-48`）；async MoE ubatching 路径则要求为 `True`（`feature_validation.py:107-134`）。

**async_dp（async）**
AFD 是否启用异步数据并行运行时补丁；别名 `"async" -> "async_dp"`（`config.py:34`），字段 `config.py:50`。`async_dp=true` 须配 `CAMAsyncAFDConnector`（`config.py:310-313`），轻量选择器 `is_afd_async_dp`（`config.py:274-287`）供 import-time async-DP 补丁使用。

**connector_extra_config / ConnectorExtraInfo**
`connector_extra_config` 是 `additional_config["afd"]` 中存放连接器私有配置的映射键（解析见 `config.py:113-117`，导出 `connector_extra_config_from_source` `config.py:261-265`）；`ConnectorExtraInfo` 是连接器自有配置的基类型（`afd_plugin/connectors/base.py:26-34`），由各连接器 `parse_extra_config` 用自有 schema 严格校验并产出子类（如 `CAMP2PExtraInfo` `camp2p.py:73`、`AFDAsyncExtraInfo` `async_cam.py:83`）。工厂入口 `AFDConnectorFactory.parse_connector_extra_info`（`factory.py:68-77`）。

**AFDGraphRunMode**
CUDA graph 运行时模式枚举 `{EAGER, WARMUP, CAPTURE, REPLAY}`，`afd_plugin/v1/worker/cuda_graph.py:24-28`；由 `graph_run_mode()`（`cuda_graph.py:142-149`）按 warmup/capturing/enabled/exists 决定，优先级 WARMUP>CAPTURE>REPLAY>EAGER。

**FULL_DECODE_ONLY**
AFD 当前唯一支持的 vLLM CUDA graph 模式字符串 `"FULL_DECODE_ONLY"`，`afd_plugin/v1/worker/cuda_graph.py:20`；`_SUPPORTED_GRAPH_MODES` 仅含此值（`cuda_graph.py:21`）。NPU 等价约束在 `feature_validation.py:59-64`。详见 [08-Execution-Platforms](08-Execution-Platforms.md#cuda-graph-策略)。

## 连接器与传输

**Connector（连接器）**
两个角色间的通信契约与实现，白名单 `SUPPORTED_AFD_CONNECTORS`（`config.py:23-27`）：`P2pNcclAFDConnector`（`connectors/gpu/p2p.py:112`，CUDA 同步）、`CAMP2pAFDConnector`（`connectors/npu/camp2p.py:214`，NPU 同步）、`CAMAsyncAFDConnector`（`connectors/npu/async_cam.py:206`，NPU 异步）。工厂注册 `connectors/factory.py:80-94`。基类 `AFDConnectorBase`（`base.py:37`）。详见 [06-Connectors](06-Connectors.md)。

**AFDConnectorFactory**
按 `AFDConfig.connector` 名延迟导入并实例化连接器的工厂，`afd_plugin/connectors/factory.py:22-77`。`_registry` 用名→loader 注册三种连接器（`factory.py:80-94`）。

**Control Plane（AFDControlPlane）**
连接器可选的 DP metadata 控制面，FFN 步可由其收到的 DP metadata 驱动。抽象基类 `AFDControlPlane`（`base.py:254-303`），三个抽象方法 `update_state_from_dp_metadata`/`send_dp_metadata_list`/`recv_dp_metadata_list`。实现 `P2pNcclAFDControlPlane`（`p2p.py:583`）、`CAMP2pAFDControlPlane`（`camp2p.py:619`）。挂载点 `AFDConnectorBase.control_plane`（`base.py:60`）。

**AFDControlPayload**
控制面的结构化 DP metadata 信封，含 `dp_metadata_list`（stage→`AFDDPMetadata`）、`is_graph_capturing`、`is_warmup`。`metadata.py:107-141`。`__post_init__` 把 vLLM `DPMetadata` 归一为 `AFDDPMetadata`（`metadata.py:132-141`）。JSON 序列化 `encode_control_payload`/`decode_control_payload`（`metadata.py:307-355`），点对点收发 `send_control_payload`/`recv_control_payload`（`metadata.py:358-402`）。

**AFDDPMetadata**
与 vLLM `DPMetadata` 兼容、可序列化的 plugin 自有 token 数元数据，`metadata.py:24-100`。核心字段 `num_tokens_across_dp_cpu`/`max_tokens_across_dp_cpu`，提供 SP 切分 (`sp_local_sizes`) 与 chunked sizes 工具。`AFDSingleDPMetadata` 是其别名（`metadata.py:103`）。

**AFDTransferMetadata**
一次 Attention/FFN 交换的后端中性通信元数据，`metadata.py:160-219`。字段 `layer_idx`/`stage_idx`/`seq_lens`，校验 `seq_lens` 非空且为正（`:180-184`），提供 `total_tokens` 与 `validate_tensor_shape`。工厂方法 `create_attention_metadata`/`create_ffn_metadata`（`:190-216`）。

**AFDTransferState**
连接器在一次交换的 receive/send 间读回的后端私有状态的基类，`metadata.py:145-156`。由后端子类化（如 `CAMP2PTransferState`、`AFDAsyncTransferState`）挂在 `AFDTransferContext.states`；不路由 payload 的 GPU P2P 连接器留 `None`。

**AFDTransferContext**
绑定 `AFDTransferMetadata` 与可选 `AFDTransferState` 的传输上下文，是连接器沿数据路径传递的对象，`metadata.py:223-250`。`states` 默认 `None`（非 `field(default_factory=...)`）以保持 `torch.compile` 可 trace（`metadata.py:237-241`）。

**AFDA2FTransferPayload**
FFN 侧 `recv_attn_output()` 的统一收载荷，含 `hidden_states` 与 `AFDTransferContext`，`metadata.py:254-269`。FFN runner 将 context 透传到 `send_ffn_output()`。

**AFDF2ATransferPayload**
FFN→Attention 的统一发载荷，分离 routed/shared 输出，`metadata.py:273-277`（`routed_output` + 可选 `shared_output`）。

**AFDForwardContextMetadata**
forward context 中对 plugin 模型包装类可见的 AFD 元数据，`metadata.py:281-299`。含 `tokens_start_loc`/`requests_start_loc`/`stage_idx`/`connector`/`num_stages`/`transaction_id` 等，提供 `clone()` 供 ubatch 切分。

**ubatch / UBatch**
vLLM 将一个调度 batch 沿 token 维切分为多个微批的机制；AFD 在两平台均仅支持恰好 2 个 ubatch（GPU `cuda_graph.py:71-77`，NPU `ubatch_utils.py:37-68` 与 `feature_validation.py:59-64`）。详见 [08-Execution-Platforms](08-Execution-Platforms.md#ubatch-包装)。

**AFDUBatchWrapper**
GPU 侧 vLLM `UBatchWrapper` 的薄子类，注入 AFD context provider、旁路 SM control、构建 ubatch metadata、处理 CUDA graph 捕获/重放。`afd_plugin/v1/worker/ubatch_wrapper.py:24`。NPU 对应 `AscendUBatchWrapper`（`afd_plugin/v1/worker/npu/npu_ubatch_wrapper.py:53`），独立实现不继承。

**DBO（Dual-Batch Overlap，双批重叠）**
ubatching 的执行时重叠策略，使两个微批在前向中交替执行以隐藏通信延迟。注册自定义 op `torch.ops.vllm.manual_dbo_yield`（`afd_plugin/v1/worker/dbo.py:26-49`），平台分发 `_yield_if_dbo_enabled`（`dbo.py:52-74`）优先 NPU 回退 vLLM 原生。详见 [08-Execution-Platforms](08-Execution-Platforms.md#dbodual-batch-overlap)。

## 平台与 NPU 算子

**Platform（平台）**
CUDA GPU 与 Ascend NPU 两条后端，各有一套 worker / model runner / 算子实现，不跨平台继承。连接器分组 `connectors.gpu` / `connectors.npu`（`README.md:50-52`）。详见 [08-Execution-Platforms](08-Execution-Platforms.md)。

**ACLNN（Ascend Computing Language Neural Network）**
Ascend 自定义算子的 host API 两段式接口（`GetWorkspaceSize` + 执行），AFD 用 `csrc/npu/a2e`/`e2a` 的 `aclnnA2e`/`aclnnE2a` 实现 CAMP2P 数据路径。桥接宏 `EXEC_NPU_CMD` 在 `csrc/npu/aclnn_torch_adapter/op_api_common.h:535-602`，通过 `AFD_CUST_OPAPI_LIB_PATH` 动态查找 `libcust_opapi.so`。vendor 包构建见 `csrc/npu/CMakeLists.txt:242`（aclnn 段）。

**HCCL（Huawei Collective Communication Library）**
华为集合通信库，CAMP2P 同步连接器的传输后端。README 连接器表注明 `CAMP2pAFDConnector` 使用 HCCL/CAMP2P 自定义算子（`README.md:47`）。

**CAMP2P**
Ascend 上的点对点通信库/机制，`CAMP2pAFDConnector`（`camp2p.py:214`）与 `CAMP2pAFDControlPlane`（`camp2p.py:619`）据此实现 NPU 同步 decode 数据路径。其 `CAMP2PExtraInfo`（`camp2p.py:73`）由 `fail_if_unsupported_npu_afd_features` 校验（`feature_validation.py:49-57`）。

**CAM（Communication Access Module）**
Ascend 通信访问模块；`CAMAsyncAFDConnector`（`async_cam.py:206`）用 CAM async-DP 自定义算子实现 NPU 异步 prefill 路径，需外部 `umdk_cam_op_lib`（`afd_plugin/compat/npu/ops.py:106-127`）。

**a2e / e2a 算子**
a2e（Attention-to-Expert）/ e2a（Expert-to-Attention）是 CAMP2P 的核心 CANN 自定义算子，注册于 `torch.ops.afd_ascend`（`csrc/npu/README.md:6-7`，namespace 常量 `AFD_ASCEND_OPS_NAMESPACE="afd_ascend"` `ops.py:11`）。binding 在 `csrc/npu/torch_extension/torch_binding.cpp`，延迟加载 `ensure_cam_p2p_ops_available()`（`ops.py:71-89`）。详见 [08-Execution-Platforms](08-Execution-Platforms.md#a2e--e2a-算子)。

**vLLM-Ascend 共存**
AFD 自定义算子与 vLLM-Ascend 同进程隔离规则：扩展归 plugin 所有（`afd_plugin._C_ascend`）；算子注册于 `torch.ops.afd_ascend` 而非 vLLM-Ascend 的 `torch.ops._C_ascend`；CANN vendor 包置于 `afd-plugin` vendor 路径；loader 用包内 `libcust_opapi.so`（`AFD_CUST_OPAPI_LIB_PATH`）而非 bare `dlopen`。`csrc/npu/README.md:62-74`、`tests/unit/package/test_ascend_build_files.py:154-173`。

## 相关页面速查

| 页面 | 内容 |
| --- | --- |
| [Home](Home.md) | Wiki 导航与约定 |
| [01-Overview](01-Overview.md) | AFD 定位、动机、支持矩阵 |
| [02-Architecture](02-Architecture.md) | 分层、模块地图、依赖方向 |
| [03-Plugin-Boundary](03-Plugin-Boundary.md) | 入口点、`AFDConfig`、自动 worker 选择 |
| [04-Attention-Runtime](04-Attention-Runtime.md) | Attention worker / model runner |
| [05-FFN-Runtime](05-FFN-Runtime.md) | FFN worker、connector 驱动 |
| [06-Connectors](06-Connectors.md) | 连接器基类、工厂、元数据、三种实现 |
| [07-Model-Integration](07-Model-Integration.md) | DeepSeek 包装类、按角色加载 |
| [08-Execution-Platforms](08-Execution-Platforms.md) | CUDA graph、DBO、ubatch、NPU 运行时、算子 |
| [09-Compatibility-Patches](09-Compatibility-Patches.md) | 补丁策略、版本门禁、逐补丁说明 |
| [10-Data-Flow](10-Data-Flow.md) | 启动→握手→推理→关闭 时序 |
| [11-Build-Packaging-Tests](11-Build-Packaging-Tests.md) | 打包、Ascend 构建、单元/E2E 测试、CI |
