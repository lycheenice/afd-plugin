# 兼容补丁

本页以当前 main 分支源码为准，逐行核对 `afd_plugin/compat/` 下的版本门禁与全部补丁模块，覆盖补丁总策略、应用时序、每个补丁的上游目标与 AFD 差异、升级操作清单。补丁规范权威来源是 [AGENTS.md](../../AGENTS.md) 的 Patching Requirement 段；既有设计文档 `docs/design/module/compatibility_and_patches.md` 作为参考，与源码冲突处以源码为准并注明。相关上下文见 [插件边界](03-Plugin-Boundary.md)、[执行平台](08-Execution-Platforms.md)、[连接器](06-Connectors.md)。

## 补丁总策略

AFD plugin 不修改 vLLM 源码，而是通过 `vllm.general_plugins` 入口点注入行为。当继承/组合无法达成目标时，才使用 monkey-patch。AGENTS.md（`AGENTS.md:78-124`）规定的硬性约束：

- **严格架构评审**：所有新补丁须经架构评审，验证补丁目标正确、最小聚焦、性能可理解、有长期上游贡献或移除计划（`AGENTS.md:80-88`）。
- **差异标注**：复制或包装上游代码时，每处 AFD 特有差异必须用 `# ### PATCH START: ...` 与 `# ### PATCH END: ...` 注释标注，标注文本短而具体，便于评审者快速对比上游（`AGENTS.md:90-93`）。
- **注释说明**：每个补丁函数上方必须有注释，解释为何补丁该上游函数、补丁改变或增加了什么行为（`AGENTS.md:95-99`）。
- **签名一致**：补丁函数签名（含返回类型）须与上游完全一致；若必须新增参数，须在函数上方注释中逐一说明（`AGENTS.md:99`）。
- **Pinned tag 开发**：补丁基于固定的 vLLM / vLLM-Ascend tag 开发。默认做法是抄写上游对应函数，仅用 `# ### PATCH START/END` 标注 AFD 差异（`AGENTS.md:101-104`）。
- **升级流程**：升级到新上游 tag 时，重新抄写新上游函数，再重新应用已标注的 AFD 差异（`AGENTS.md:104-105`）。
- **避免 `_original_*` 委托**：避免以 `_original_*` 委托作为非 AFD 主路径。例外仅当上游函数过大或不宜就地展开时允许，但必须在补丁函数注释中说明（`AGENTS.md:106-108`）。当前 `config_validation.py` 与 `npu/ascend_platform.py` 使用了该例外。
- **实现优先级**：AFD 功能优先通过继承/组合实现（`afd_plugin/v1/worker/` 角色运行时、`afd_plugin/connectors/` 连接器），其次才用补丁；外部上游贡献也是选项之一（`AGENTS.md:110-123`）。

补丁源码位于两个目录（`AGENTS.md:112-116`）：

- `afd_plugin/compat/patches/` — vLLM 兼容补丁。
- `afd_plugin/compat/patches/npu/` — NPU 专用兼容补丁。
- `afd_plugin/compat/npu/` — Ascend 运行时兼容（非补丁式适配器，见 [08-Execution-Platforms](08-Execution-Platforms.md)）。

`afd_plugin/compat/patches/__init__.py:3-7` 要求补丁包内的补丁必须幂等、版本感知、有文档，并尽可能由 CPU 安全测试覆盖。

## 版本门禁

### 目标版本与门禁函数

`afd_plugin/compat/vllm.py` 是版本门禁的唯一来源：

```python
TARGET_VLLM_VERSION: Final[str] = "0.19.1"   # vllm.py:12
```

| 函数 | 源码位置 | 行为 |
| --- | --- | --- |
| `get_installed_vllm_version` | `vllm.py:22-26` | 通过 `importlib.metadata.version("vllm")` 读取，未安装返回 `None` |
| `is_vllm_version_supported` | `vllm.py:29-35` | 解析主次修订号，要求与 `TARGET_VLLM_VERSION` 精确相等；`None` 视为不支持 |
| `assert_vllm_version_supported` | `vllm.py:38-50` | `strict=True`（默认）抛 `RuntimeError`；`strict=False` 发 `RuntimeWarning` 并继续 |

`_parse_release`（`vllm.py:15-19`）用正则 `^(\d+)\.(\d+)\.(\d+)` 提取前三位版本号，忽略后续后缀。`compat/__init__.py:5-10` 把这四个符号重新导出为包级 API。

### register_afd 用 strict=False 的容错原因

`register_afd` 在 `afd_plugin/__init__.py:88` 调用 `assert_vllm_version_supported(strict=False)`。`strict=False` 表示版本不匹配时只警告、不阻断注册流程。容错动机（见既有设计文档 `docs/design/module/compatibility_and_patches.md:60-63` 并经源码印证）：

1. 插件入口点不应因版本检查异常而完全无法加载——即使 vLLM 版本不符，也要让后续补丁导入与模型注册有机会执行。
2. 警告策略**不**使其它 vLLM 版本成为受支持版本；它只是避免在导入期硬失败。运行时路径若依赖某项适配而该适配缺失，应自行显式失败（见后文容错策略）。

该调用本身也被 `try/except` 包裹（`__init__.py:85-93`），仅记 debug 日志。

## 补丁应用总时序

`register_afd`（`afd_plugin/__init__.py:66-133`）是 `vllm.general_plugins` 入口点。无 vLLM 时安全 no-op（`__init__.py:80-83`），保证 CPU 机器能跑 import/config 测试。应用顺序如下：

```
register_afd()                                    afd_plugin/__init__.py:66
 ├─ 防重入: _registered 标志                      __init__.py:74-77
 ├─ vLLM 不存在则跳过                             __init__.py:80-83
 ├─ try: assert_vllm_version_supported(strict=False)   __init__.py:85-93
 │    └─ 失败仅 debug 日志
 ├─ try: 导入 4 个 vLLM 补丁 (一个 try 块)        __init__.py:95-104
 │    ├─ async_dp_engine                          __init__.py:96
 │    ├─ async_dp_forward_context                 __init__.py:97
 │    ├─ config_validation                        __init__.py:98
 │    └─ engine_core                              __init__.py:99
 │    └─ 失败仅 debug 日志 (无回滚)
 ├─ try: register_dbo_yield_custom_op()           __init__.py:106-114
 │    └─ 失败仅 debug 日志
 ├─ try: Ascend 补丁                             __init__.py:116-126
 │    ├─ apply_afd_ascend_patches_if_needed()     __init__.py:119
 │    └─ 若 vllm_ascend 可发现: import force_load_balance  __init__.py:120-121
 │    └─ 失败仅 debug 日志
 ├─ 模型注册 (无 try, 必须成功)                   __init__.py:128-131
 └─ _registered = True                            __init__.py:133
```

### 失败容错策略

每个阶段独立 `try/except`，失败时 `_logger.debug(..., exc_info=True)` 记录后继续（`__init__.py:89-93,100-104,110-114,122-126`）。关键特征（与既有设计文档 `docs/design/module/compatibility_and_patches.md:93-99` 一致）：

- **无事务/回滚**：4 个核心 vLLM 补丁共享同一个 `try`（`__init__.py:95-104`）。若中途某个导入失败，已安装的补丁保留、后续补丁被跳过，产生部分应用状态。
- **失败仅 debug 可见**：启动期补丁导入错误可能只在 debug 日志中体现。因此运行时路径若依赖某项适配，必须在缺失时显式失败，而非静默服务错误结果。
- **模型注册是硬门**：`__init__.py:128-131` 不在 `try` 内，模型注册失败会让 `register_afd` 抛出。

Python 模块导入提供进程级一次性执行；部分补丁额外保存 original 或设置 sentinel 以求幂等，但不一致（详见各补丁节）。

## 补丁模块逐一展开

### config_validation.py

`afd_plugin/compat/patches/config_validation.py`，补丁两个上游符号：

| 上游符号 | 补丁函数 | 源码位置 |
| --- | --- | --- |
| `vllm.engine.arg_utils.EngineArgs.create_engine_config` | `create_engine_config` | `config_validation.py:44-75` |
| `vllm.config.VllmConfig.__post_init__` | `__post_init__` | `config_validation.py:88-115` |

**为何补**：vLLM 0.19.1 校验原生 ubatching 时强制要求 DeepEP all2all backend（`config_validation.py:5-7,36-39`）。AFD 的 ubatching 由插件连接器实现，不依赖 DeepEP，因此该断言对 AFD 配置误报。此外上游在平台归一化阶段自动选择平台 worker，该 worker 不含 AFD 角色行为（`config_validation.py:78-83`）。

**改了什么行为**：

1. **all2all backend 校验放行**（两处 PATCH START/END）：
   - `create_engine_config`（`config_validation.py:59-74`）：对 AFD 配置，临时把 `self.all2all_backend` 设为 `_AFD_TEMP_BACKEND = "deepep_low_latency"`（`config_validation.py:31`），调用 original 构建并校验 `VllmConfig`，`finally` 恢复原 backend，再把 `config.parallel_config.all2all_backend` 设回原值。
   - `__post_init__`（`config_validation.py:98-109`）：重复校验场景同理，临时替换 `parallel_config.all2all_backend`，校验后恢复。
2. **auto worker 角色化重映射**（`config_validation.py:92-94, 111-114`）：`__post_init__` 入口记录 `worker_cls` 是否为 `"auto"`，上游归一化后若原为 auto，调用 `_select_afd_worker_for_auto`（`config_validation.py:123-141`）按平台 + 角色替换为 `afd_worker_qualname_for_platform_default(...)`。

**放行条件**：仅对已激活 AFD 配置且确实使用 ubatching 的场景放行，非 AFD 或显式 DeepEP backend 不受影响：

- `_should_relax_engine_args_backend`（`config_validation.py:144-165`）：需 AFD 配置 + (`enable_dbo` 或 `ubatch_size>1`) + backend 非 DeepEP。
- `_should_relax_vllm_config_backend`（`config_validation.py:168-185`）：需 AFD 配置 + `use_ubatching=True` + backend 非 DeepEP。

**`_original_*` 委托例外**：两个补丁函数都用 `_original_create_engine_config` / `_original_vllm_config_post_init` 委托非 AFD 路径。注释已说明例外原因——上游函数是大型配置构建器，不宜就地展开（`config_validation.py:40-42, 84-86`）。original 保存在上游模块的 AFD 专属属性 `_afd_plugin_original_create_engine_config` / `_afd_plugin_original_vllm_config_post_init` 下（`config_validation.py:29-33, 204-225`），重载时不覆盖已保存的 original。

**版本守卫与幂等**：`_is_target_vllm_compatible`（`config_validation.py:188-200`）接受目标版本前缀、dev 版本或缺失版本元数据。安装前用 `hasattr` 检查 AFD 专属属性是否已存在，避免重载时覆盖 original（`config_validation.py:204-217`）。

### engine_core.py

`afd_plugin/compat/patches/engine_core.py`，补丁五个上游符号（直接类赋值，`engine_core.py:557-561`）：

| 上游符号 | 补丁函数 | 源码位置 |
| --- | --- | --- |
| `EngineCore.__init__` | `__init__` | `engine_core.py:37-188` |
| `EngineCore._initialize_kv_caches` | `_initialize_kv_caches` | `engine_core.py:222-291` |
| `EngineCore.shutdown` | `shutdown` | `engine_core.py:196-215` |
| `EngineCoreProc.run_busy_loop` | `run_busy_loop` | `engine_core.py:299-371` |
| `DPEngineCoreProc.run_busy_loop` | 同上（共用） | `engine_core.py:561` |

**为何补**：AFD FFN 侧以连接器 daemon 方式运行，而非正常的请求调度 EngineCore（`engine_core.py:31-35, 8-9`）。FFN EngineCore 在构造完 model executor 后即返回，跳过 KV cache 与 scheduler 初始化，避免进入 `HybridKVCacheCoordinator`。

**改了什么行为**（对 `_is_afd_ffn_config(vllm_config)` 为真的配置生效，`engine_core.py:538-540`）：

1. `__init__`（`engine_core.py:45-58` PATCH）：FFN 配置走 `_initialize_ffn_engine_core`（`engine_core.py:403-461`），构造 executor、清零 cache blocks、设置若干占位属性后直接 return，跳过 KV/scheduler。非 AFD 走抄写的上游分支（`engine_core.py:60-187`）。
2. `_initialize_kv_caches`（`engine_core.py:223-229` PATCH）：FFN 配置走 `_prepare_late_loaded_ffn_engine_core`（`engine_core.py:464-485`）并返回 `_AFDFFNKVCacheConfig`（`engine_core.py:374-375`，空 groups）。`_AFDFFNNoopScheduler`（`engine_core.py:378-400`）作为占位 scheduler。
3. `shutdown`（`engine_core.py:197-208` PATCH）：FFN 引擎先 `_stop_ffn_worker_loop`（`engine_core.py:512-522`，`collective_rpc("stop_ffn_server_loop")`），再 `model_executor.shutdown()` 与 `gc.unfreeze`，跳过不存在的 scheduler/KV 状态。非 AFD 走上游 shutdown（`engine_core.py:210-214`）。
4. `run_busy_loop`（`engine_core.py:300-306` PATCH）：FFN 引擎走 `_run_ffn_busy_loop`（`engine_core.py:488-509`），调用 `collective_rpc("start_ffn_server_loop")` 启动 worker 端连接器循环，然后 0.5s 轮询 `raise_ffn_loop_error_if_any`。非 AFD 走抄写的上游 `DPEngineCoreProc` / 普通 busy loop（`engine_core.py:308-371`）。

**关键辅助**：`_get_afd_config`（`engine_core.py:543-554`）优先读 `vllm_config.afd_config`，否则 `parse_optional_afd_config(validate=False)` 兜底。FFN worker 端真正的循环由 `AFDFFNWorker.start_ffn_server_loop` 触发（见 [FFN 运行时](05-FFN-Runtime.md)、[10-Data-Flow](10-Data-Flow.md)）。

**无版本守卫、无 saved-original**：此补丁是直接类赋值，**没有** `_is_target_vllm_compatible` 守卫，也**没有**保存 original 的 sentinel（既有设计文档 `docs/design/module/compatibility_and_patches.md:110` 已指出，源码 `engine_core.py:557-561` 印证）。兼容性守卫完全依赖 package pin 与评审纪律。这是补丁集合中幂等性最弱的一项。

### async_dp_engine.py

`afd_plugin/compat/patches/async_dp_engine.py`，补丁三个上游符号（`async_dp_engine.py:390-393`）：

| 上游符号 | 补丁函数 | 源码位置 |
| --- | --- | --- |
| `EngineCoreProc.run_engine_core` | `run_engine_core` | `async_dp_engine.py:71-172` |
| `vllm.v1.engine.utils.launch_core_engines`（含 client 别名） | `launch_core_engines` | `async_dp_engine.py:180-318` |
| `DPAsyncMPClient.add_request_async` | `add_request_async` | `async_dp_engine.py:326-364` |

**为何补**：vLLM 0.19.1 原生 MoE DP 路径使用 `DPEngineCoreProc` 与 DP wave 通知。AFD async-DP 的 Attention ranks 是连接器驱动的，必须独立步进，同时保留原 DP/EP 拓扑用于 expert 放置与权重加载（`async_dp_engine.py:12-16, 66-69`）。

**改了什么行为**：

1. `run_engine_core`（`async_dp_engine.py:112-120` PATCH）：对 AFD async Attention 配置（`_is_afd_async_attention_config`，`async_dp_engine.py:367-373`：需 AFD 配置 + `is_afd_async_dp` + `role=="attention"`），实例化普通 `EngineCoreProc` 而非 `DPEngineCoreProc`，保留 DP rank 元数据。其余配置走抄写的上游分支（`async_dp_engine.py:119` 用 `DPEngineCoreProc`）。
2. `launch_core_engines`（`async_dp_engine.py:212-218` PATCH）：构造 `DPCoordinator` 时，对 AFD async-DP 配置置 `enable_wave_coordination=False`——保留 coordinator stats，禁用 wave 协调。其余上游启动逻辑保持不变。
3. `add_request_async`（`async_dp_engine.py:351-364` PATCH）：对 AFD async-DP 跳过 `FIRST_REQ` 协调器唤醒（保留正常路由与 stats 更新），因为 async 引擎独立步进。非 AFD 走上游分支（`async_dp_engine.py:332-349`）发 `FIRST_REQ`。

**版本守卫与幂等**：`_is_target_vllm_compatible`（`async_dp_engine.py:376-386`）接受目标版本前缀、dev 版本或缺失版本元数据。守卫通过后直接赋值三个符号（`async_dp_engine.py:390-393`）。`is_afd_async_dp`（`afd_plugin/config.py:274-287`）是轻量选择器：需 `async_dp=True` 且 `connector == AFD_ASYNC_CONNECTOR`，不做完整校验。

`launch_core_engines` 同时赋值 `engine_utils_module.launch_core_engines` 与 `core_client_module.launch_core_engines`（`async_dp_engine.py:391-392`），因为 client 模块通过导入别名持有该函数。

### async_dp_forward_context.py

`afd_plugin/compat/patches/async_dp_forward_context.py`，补丁一个上游符号并重绑已导入别名（`async_dp_forward_context.py:205-210`）：

| 上游符号 | 补丁函数 | 源码位置 |
| --- | --- | --- |
| `vllm.forward_context.set_forward_context` | `set_forward_context` | `async_dp_forward_context.py:62-190` |

**为何补**：vLLM 0.19.1 在 DP size > 1 时为 MoE DP ranks 构造 `DPMetadata` 并跨 rank 协调 token 计数。AFD async-DP 用连接器数据流代替 vLLM 的 DP metadata 控制平面，因此 all-reduce 与 metadata 路径须对 AFD async 配置跳过（`async_dp_forward_context.py:8-13, 57-61`）。

**改了什么行为**：补丁函数整体抄写上游 `set_forward_context`，仅改 DP metadata 构造段（`async_dp_forward_context.py:87-114` PATCH）：对非 AFD async 配置保留上游 `coordinate_batch_across_dp` + `DPMetadata.make`；对 AFD async 配置跳过，`dp_metadata` 保持 `None`。其余 forward-context 设置（batchsize 追踪、cudagraph descriptor、`set_additional_forward_context`、`create_forward_context`、override、统计）与上游一致。

**已导入别名重绑**：上游多个 worker 模块通过 `from vllm.forward_context import set_forward_context` 持有函数引用。补丁除替换 `forward_context_module.set_forward_context` 外，还遍历 `_FORWARD_CONTEXT_IMPORT_MODULES`（`async_dp_forward_context.py:49-54`）重绑已加载模块的引用，包括 `vllm.v1.worker.gpu_model_runner`、`gpu.model.runner`、`kv_connector_model_runner_mixin` 以及 `vllm_ascend.ascend_forward_context`。

**版本守卫**：`_is_target_vllm_compatible`（`async_dp_forward_context.py:192-202`），同上模式。

### npu/ascend_platform.py

`afd_plugin/compat/patches/npu/ascend_platform.py`，补丁一个上游符号：

| 上游符号 | 补丁函数 | 源码位置 |
| --- | --- | --- |
| `vllm_ascend.platform.NPUPlatform._fix_incompatible_config` | `_fix_incompatible_config` | `ascend_platform.py:48-57` |

**为何补**：vLLM-Ascend 的平台兼容性阶段会对普通 NPU 运行禁用 DBO/ubatching 字段。AFD 拥有自己的 NPU ubatching 路径，需要保留这些字段（`ascend_platform.py:22-27, 40-43`）。

**改了什么行为**：补丁函数先 `_snapshot_afd_dbo_config` 快照 `enable_dbo`/`use_ubatching`/`ubatch_size`（`ascend_platform.py:67-75`，仅对有效 AFD 配置生效，`ascend_platform.py:93-97`），调用 original 跑上游归一化，再 `_restore_afd_dbo_config` 恢复（`ascend_platform.py:78-91`，仅当快照值非全 falsy 时恢复）。非 AFD 配置快照为 `None`，行为与上游完全一致。包裹关系见 PATCH 标注 `ascend_platform.py:49-56`。

**`_original_*` 委托例外**：注释说明上游 `_fix_incompatible_config` 是平台拥有的归一化，保持窄范围 original 委托使补丁只拥有 AFD DBO 保留逻辑（`ascend_platform.py:44-46`）。

**幂等与守卫**：vLLM-Ascend 不可导入时直接 return（`ascend_platform.py:30-33`）。用类级 sentinel `_ASCEND_PLATFORM_PATCH_ATTR`（`ascend_platform.py:17`）防止重复安装：已安装则 return（`ascend_platform.py:35-36`），original 存在该属性上（`ascend_platform.py:60-64`）。无 patch-local 版本守卫——配合运行时 facade 的 `_PATCHES_APPLIED` sentinel 共同保证幂等（见下节）。

### npu/force_load_balance.py

`afd_plugin/compat/patches/npu/force_load_balance.py`，补丁两个上游符号（`force_load_balance.py:505-506`）：

| 上游符号 | 补丁函数 | 源码位置 |
| --- | --- | --- |
| `vllm_ascend.ops.fused_moe.fused_moe.AscendFusedMoE.__init__` | `__init__` | `force_load_balance.py:189-340` |
| `vllm_ascend.quantization.methods.w8a8_dynamic.AscendW8A8DynamicFusedMoEMethod.apply` | `apply` | `force_load_balance.py:349-502` |

**为何补**：vllm-ascend 的 W8A8 FusedMoE 用模型选出的 expert id 路由 token，AFD profiling 需要确定性均衡的 expert id（`force_load_balance.py:5-9, 183-188, 343-348`）。

**改了什么行为**：

1. `__init__`（两处 PATCH，`force_load_balance.py:234-245` 与 `319-327`）：从 `additional_config` 读取 `enable_force_load_balance` 与 `force_load_balance_topn_per_rank`，存为 layer 属性；W8A8 权重就绪后调用 `_init_force_lb_buffer`（`force_load_balance.py:133-153`）预建确定性 fake top-k buffer。其余 init 逻辑（quant method、eplb、expert map 等）抄写上游。
2. `apply`（`force_load_balance.py:431-445` PATCH）：当 layer 的 `enable_force_load_balance` 为真，用 `_get_force_lb_topk_ids`（`force_load_balance.py:156-180`）从预建 buffer 取确定性 topk_ids 替换模型路由结果（`mix_placement` 时拼接 shared topk）。上游 `enable_force_load_balance` 参数分支（随机 argsort，`force_load_balance.py:423-429`）保留不变。

**确定性 buffer**：`_build_expert_cycle`（`force_load_balance.py:95-118`）在 `topn_per_rank>0` 时按 EP rank 轮询构造确定性专家序列；为 0 时用固定种子 `_FORCE_LB_DETERMINISTIC_SEED = 1024`（`force_load_balance.py:38`）的 `randperm`。`_build_topk_buffer`（`force_load_balance.py:121-130`）按 `max_tokens * top_k` 展平重复切片。buffer 不足时 `_get_force_lb_topk_ids` 自动扩容（`force_load_balance.py:165-173`）。

**重要定性**：force load balance 改变模型输出，是 benchmark/profiling 开关，**不是**生产正确性特性（`force_load_balance.py:11-13`）。

**无版本守卫、无 sentinel**：此补丁**没有** patch-local 版本守卫，**没有**显式 reload sentinel（既有设计文档 `docs/design/module/compatibility_and_patches.md:112` 已指出，源码 `force_load_balance.py:505-506` 直接赋值印证）。函数整体抄写当前上游函数体并标注 AFD delta。仅在 `register_afd` 中当 `vllm_ascend` 可发现时才导入（`afd_plugin/__init__.py:120-121`）。

## Ascend 运行时补丁总入口

`apply_afd_ascend_patches_if_needed`（`afd_plugin/compat/npu/runtime.py:21-33`）是 Ascend 运行时补丁的幂等总入口，由 `compat/npu/__init__.py:21-27` 重新导出。

```
apply_afd_ascend_patches_if_needed()        compat/npu/runtime.py:21
 ├─ _PATCHES_APPLIED 守卫 (已应用则 return)  runtime.py:18,24-26
 ├─ 导入 apply_afd_ascend_dbo_config_patch   runtime.py:28-30
 ├─ apply_afd_ascend_dbo_config_patch()      runtime.py:32 -> ascend_platform.py:20
 └─ _PATCHES_APPLIED = True                   runtime.py:33
```

它当前只安装 `npu/ascend_platform.py` 的 DBO 配置补丁。该 facade 自身用模块级 `_PATCHES_APPLIED`（`runtime.py:18`）保证只执行一次；`ascend_platform.py` 内部另有类级 sentinel，双重防护。

`register_afd` 在 `afd_plugin/__init__.py:117-119` 调用它，置于第三个 `try` 块内。`vllm_ascend` 不可导入时 facade 内部 no-op（`ascend_platform.py:30-33`），外层 try 也吸收异常。

> 待确认：`runtime.py` 当前仅安装 `ascend_platform` 一个补丁。`compat/npu/__init__.py` 还导出了 `fix_all2all_backend_for_afd`、`fail_if_unsupported_npu_afd_features`、`ascend_forward_context`、`npu_afd_num_ubatches` 等非补丁式适配器（`compat/npu/__init__.py:21-27`），它们不替换全局符号，属于运行时适配，详见 [08-Execution-Platforms](08-Execution-Platforms.md)。

## 补丁总览矩阵

| 补丁文件 | 上游符号 | 应用/守卫/幂等 | `_original_*` 例外 | 版本守卫 |
| --- | --- | --- | --- | --- |
| `config_validation.py` | `EngineArgs.create_engine_config`, `VllmConfig.__post_init__` | `register_afd` 导入；AFD 属性存 original + `hasattr` 检查 | 是（注释说明） | 有 `_is_target_vllm_compatible` |
| `engine_core.py` | `EngineCore.__init__/_initialize_kv_caches/shutdown`, `EngineCoreProc/DPEngineCoreProc.run_busy_loop` | `register_afd` 导入；直接类赋值 | 否（抄写上游） | **无** |
| `async_dp_engine.py` | `EngineCoreProc.run_engine_core`, `launch_core_engines`+client 别名, `DPAsyncMPClient.add_request_async` | `register_afd` 导入；直接赋值 | 否（抄写上游） | 有 `_is_target_vllm_compatible` |
| `async_dp_forward_context.py` | `set_forward_context` + 已导入别名 | `register_afd` 导入；重绑多模块引用 | 否（抄写上游） | 有 `_is_target_vllm_compatible` |
| `npu/ascend_platform.py` | `NPUPlatform._fix_incompatible_config` | `apply_afd_ascend_patches_if_needed`；类 sentinel + facade sentinel | 是（注释说明） | 无（vLLM-Ascend 不导入即 no-op） |
| `npu/force_load_balance.py` | `AscendFusedMoE.__init__`, `AscendW8A8DynamicFusedMoEMethod.apply` | `register_afd` 当 vllm_ascend 可见时导入；直接赋值 | 否（抄写上游） | **无** |

## 升级上游 tag 操作清单

升级 pinned vLLM 或 vLLM-Ascend tag 时，对每个补丁执行以下可执行步骤（提炼自 AGENTS.md `AGENTS.md:101-108` 与既有设计文档 `docs/design/module/compatibility_and_patches.md:134-150`）：

1. **定位上游符号**：确认补丁针对的上游文件、符号、版本/tag、签名（逐个补丁核对上表）。
2. **确认归属层**：确认补丁目标是对应上游拥有层，而非在便利的全局 hook 里藏 AFD 行为。
3. **重新抄写上游函数**：从新 tag 复制对应上游函数体；对使用 `_original_*` 例外的补丁（`config_validation.py`、`ascend_platform.py`），只需确认 original 委托仍指向正确的上游函数。
4. **重新标注 AFD 差异**：用 `# ### PATCH START: ...` / `# ### PATCH END: ...` 重新标注 AFD 差异，保持标注短而具体。
5. **核查签名一致**：补丁函数签名（含返回类型）须与新上游完全一致；上游若改签名，补丁同步改并更新注释。
6. **更新版本守卫**：更新 `TARGET_VLLM_VERSION`（`vllm.py:12`）及各补丁的 `_is_target_vllm_compatible`；对无守卫的补丁（`engine_core.py`、`force_load_balance.py`）重新评估是否需要新增守卫。
7. **记录原始委托例外**：若有 `_original_*` 委托，在函数上方注释说明例外原因。
8. **测试双路径**：对 AFD 分支与非 AFD/上游分支均测试，覆盖初始化、失败、关闭、幂等场景。仅靠针对不同签名的单元测试通过**不**构成兼容性证据。
9. **评估影响**：hot-path、graph/compile、内存、分布式生命周期影响。
10. **记录移除计划**：记录上游贡献或具体移除条件。各补丁移除条件见既有设计文档 `docs/design/module/compatibility_and_patches.md:107-112`（如 `async_dp_engine` 等待 vLLM 暴露 role-selectable async-DP 调度 hook；`engine_core` 等待 headless connector-daemon 生命周期；`force_load_balance` 等待 vLLM-Ascend 上游确定性路由 profiling hook）。

> 待确认：既有设计文档指出仓库未在 `pyproject.toml` 声明精确 vLLM-Ascend 包依赖，Ascend 补丁刷新需维护者记录所用源 tag/commit（`docs/design/module/compatibility_and_patches.md:64-69, 192-201`）。此项属已知开放问题。
