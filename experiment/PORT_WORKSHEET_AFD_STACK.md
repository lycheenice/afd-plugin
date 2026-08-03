# AFD 拆分栈移植工作单(0.19.1 → 0.25.0)—— 2026-07-28 实施状态

> 目的:记录 AFD 栈在 v0.25.0 的逐项改动与验证结论。
> 方法:比对插件补丁替换函数(0.19.1 适配版)vs 0.25.0 实际目标。**所有目标结构在 0.25.0 都在**,以签名/行为再对齐为主。
> 附:vLLM 0.25.0 **原样使用、未改源码**;本工作单全是**插件侧**改动。

## 2026-07-28 结论

- ✅ H20 + vLLM 0.25.0 + DeepSeek-V2-Lite 的 GPU 同步 P2P `1A1F` eager 冒烟通过。
- ✅ 未传 `--worker-cls`;插件自动选择 `AFDAttentionWorker` / `AFDFFNWorker`。
- ✅ Attention 仅加载约 `1.54 GiB`,FFN 加载专家侧约 `29 GiB`;不是两份完整模型。
- ✅ FFN 日志出现 `AFD FFN EngineCore started; workers run connector loop`,请求返回 HTTP 200 和连贯文本。
- ⚠️ vLLM model runner v2 仍不支持;0.25.0 启动必须设置 `VLLM_USE_V2_MODEL_RUNNER=0`。
- ⚠️ 本次硬件冒烟覆盖同步 P2P / DP=1 / eager;DP>1、DBO、CUDA graph 和异步连接器由定向单测/静态对齐覆盖,仍需独立硬件回归。

## 结论:哪些要改、哪些不用
- ✅ **注入结构无需重写**:`config_validation.py` 的 EngineArgs.create_engine_config、VllmConfig.__post_init__ 签名与注入点未变（版本门仍需更新）;
  `run_engine_core`、`DPAsyncMPClient.add_request_async`(DROP-IN);`launch_core_engines`(签名同,override 逻辑仍需保留、无需改代码);
  AFDAttentionWorker、AFDAttentionModelRunner 主体(签名兼容)。
- 🔧 **需改**(下方按风险排序)。

## 移植改动清单(按风险)

### P0-0 [CRITICAL,已完成] 版本门导致原生 Worker 与 AFD EngineCore 混装

`config_validation.py` 原先只在 `0.19.1` 应用,但 `engine_core.py` 无条件应用。0.25.0 因而形成
“原生 Worker + AFD EngineCore”,在 `collective_rpc("start_ffn_server_loop")` 处报
`NotImplementedError`。现已把目标版本提升为 `0.25.0`,显式维护 `0.19.1/0.25.0`
支持集合,并让配置补丁在两个版本上统一完成自动 Worker 选择。

### P0-1 [CRITICAL] engine_core FFN 存根缺新属性 → AttributeError
`compat/patches/engine_core.py` `_initialize_ffn_engine_core()`(~line 421-462):0.25.0 `EngineCore.__init__` 新增了
`self.check_for_draft_tokens`(core.py:160)、`self.step_fn`(:221)、`self.async_scheduling`(:224),FFN 存根未设 →
`post_step()`/`_process_engine_step()` 访问时崩。
**状态:✅ 已完成。** FFN init 已补 `self.check_for_draft_tokens=False`、`self.step_fn=None`、`self.async_scheduling=False`。

### P0-2 [CRITICAL] set_forward_context 新增 `is_padding` 参数 + DP 判据扩展
`compat/patches/async_dp_forward_context.py`(replacement ~line 63):0.25.0 `set_forward_context`(forward_context.py:260)
新增 `is_padding: torch.Tensor|None=None`,且 DP-metadata 判据加了 `use_sequence_parallel_moe`,并有 num_tokens_across_dp fallback。
**状态:✅ 已完成。** (1) replacement 签名加 `is_padding=None`;(2) DP 判据 `data_parallel_size>1` → `(data_parallel_size>1 or parallel_config.use_sequence_parallel_moe)`;
(3) 补 `elif num_tokens_across_dp is None: num_tokens_across_dp=torch.tensor([num_tokens],dtype=int32)`;(4) `create_forward_context(..., is_padding=is_padding)` 透传。

### P0-3 [CRITICAL] DP run_busy_loop 缺 stats 发布 + dummy-batch 日志
`compat/patches/engine_core.py` run_busy_loop DP 分支(~line 308-362):0.25.0 `DPEngineCoreProc.run_busy_loop`(core.py:1925)
在 `_process_input_queue()`/`_process_engine_step()` 前后调 `_maybe_publish_request_counts()`,且 `execute_dummy_batch()` 包在
`with self.log_iteration_details(None):` 内。插件 DP 分支缺这些 → DP 负载均衡 stats 不发布。
**状态:✅ 已完成。** DP 分支对齐 0.25.0:step 前后加 `_maybe_publish_request_counts()`;dummy batch 包 `log_iteration_details(None)`,并跳过 sleeping executor。

### P1-1 [MEDIUM-HIGH] AFDFFNWorker.compile_or_warm_up_model 返回类型不符
`v1/worker/ffn_worker.py`(~line 79-84):返回 `0.0`(float),0.25.0 base 返回 `CompilationTimes`(dataclass)。
**状态:✅ 已完成。** 0.25.0 返回 `CompilationTimes(language_model=0.0, encoder=0.0)`;0.19.1 保留上游要求的 `float`。

### P1-2 [HIGH] GPUFFNModelRunner.load_model 完全自实现,可能漏 0.25.0 新初始化
`v1/worker/ffn_model_runner.py`(~line 105-120):不调 base,自己 `get_model_loader().load_model()`。0.25.0
`GPUModelRunner.load_model`(:5203)新增 LoRA init / drafter 加载 / EPLB。FFN 若不用 LoRA/drafter 大概率安全。
**状态:✅ 按 AFD 契约确认。** FFN daemon 不接受调度请求,当前不支持 speculative drafter/LoRA;保持只加载 FFN 模块的自实现,避免误建 KV/采样组件。后续若扩展 FFN LoRA/drafter,须单独设计而不是照搬原生 runner。

### P1-3 [MEDIUM] engine_core FFN shutdown 缺 cleanup_dist_env_and_memory + 其它签名扩展
- ✅ `engine_core.py` FFN shutdown:补 `cleanup_dist_env_and_memory()`。
- `initialize_from_config`(ffn_worker.py):0.25.0 base 会 `ensure_kv_transfer_initialized()`,AFD 全 override 跳过——确认 AFD 连接器自管即可。
- `AFDAttentionModelRunner._dummy_run`(attention_model_runner.py:408-418):0.25.0 `_dummy_run`(:5720)新增
  `is_profile/create_mixed_batch/profile_seq_lens` 等(默认安全);如需精确 profiling 补传 `profile_seq_lens`。
- ✅ `GPUFFNModelRunner.initialize_kv_cache`:补 `is_profiling=False` 参数。

### P1-4 [HIGH,已完成] MoE 专家映射 API 迁移

0.19.1 使用 `SharedFusedMoE.make_expert_params_mapping`;0.25.0 删除该模块并导出
`fused_moe_make_expert_params_mapping`。模型 wrapper 使用双版本导入,调用点统一到函数接口。

### P1-5 [HIGH,已完成] DPMetadata 控制面协议去耦

0.25.0 原生 `DPMetadata` 不再提供 `max_tokens_across_dp_cpu`。Attention 在发送控制面载荷前
统一转换为插件自有 `AFDDPMetadata`,由 token 向量重算最大值,避免连接器线格式依赖 vLLM 内部字段。

## 后续专项

P0/P1 和 1A1F eager 冒烟已完成。后续按独立任务验证 W4AFP8、4A4F、DP>1、DBO、
CUDA graph 与异步连接器；这些能力不应由本次最小冒烟结果外推。

> 数据来源:2026-07-28 三路静态 diff(engine_core / async_dp+forward_context / config+subclass);73 项依赖清单见 PORT_TO_VLLM_0.25.0.md。
