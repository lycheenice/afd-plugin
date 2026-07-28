# AFD 拆分栈移植工作单(0.19.1 → 0.25.0)—— 2026-07-28 静态预分析

> 目的:今晚(或后续)照此逐项改 AFD 栈,快速打通 GLM-5.2 AFD on 0.25.0。
> 方法:比对插件补丁替换函数(0.19.1 适配版)vs 0.25.0 实际目标。**所有目标结构在 0.25.0 都在**,是签名/行为再对齐。
> 附:vLLM 0.25.0 **原样使用、未改源码**;本工作单全是**插件侧**改动。

## 结论:哪些要改、哪些不用
- ✅ **完全不用改**:`config_validation.py`(EngineArgs.create_engine_config、VllmConfig.__post_init__ 签名+注入点未变);
  `run_engine_core`、`DPAsyncMPClient.add_request_async`(DROP-IN);`launch_core_engines`(签名同,override 逻辑仍需保留、无需改代码);
  AFDAttentionWorker、AFDAttentionModelRunner 主体(签名兼容)。
- 🔧 **需改**(下方 6 项,按风险排序)。

## 移植改动清单(按风险)

### P0-1 [CRITICAL] engine_core FFN 存根缺新属性 → AttributeError
`compat/patches/engine_core.py` `_initialize_ffn_engine_core()`(~line 421-462):0.25.0 `EngineCore.__init__` 新增了
`self.check_for_draft_tokens`(core.py:160)、`self.step_fn`(:221)、`self.async_scheduling`(:224),FFN 存根未设 →
`post_step()`/`_process_engine_step()` 访问时崩。
**改**:FFN init 补 `self.check_for_draft_tokens=False`;`self.step_fn=None`(或指向 no-op);`self.async_scheduling=False`。

### P0-2 [CRITICAL] set_forward_context 新增 `is_padding` 参数 + DP 判据扩展
`compat/patches/async_dp_forward_context.py`(replacement ~line 63):0.25.0 `set_forward_context`(forward_context.py:260)
新增 `is_padding: torch.Tensor|None=None`,且 DP-metadata 判据加了 `use_sequence_parallel_moe`,并有 num_tokens_across_dp fallback。
**改**:(1) replacement 签名加 `is_padding=None`;(2) DP 判据 `data_parallel_size>1` → `(data_parallel_size>1 or parallel_config.use_sequence_parallel_moe)`;
(3) 补 `elif num_tokens_across_dp is None: num_tokens_across_dp=torch.tensor([num_tokens],dtype=int32)`;(4) `create_forward_context(..., is_padding=is_padding)` 透传。

### P0-3 [CRITICAL] DP run_busy_loop 缺 stats 发布 + dummy-batch 日志
`compat/patches/engine_core.py` run_busy_loop DP 分支(~line 308-362):0.25.0 `DPEngineCoreProc.run_busy_loop`(core.py:1925)
在 `_process_input_queue()`/`_process_engine_step()` 前后调 `_maybe_publish_request_counts()`,且 `execute_dummy_batch()` 包在
`with self.log_iteration_details(None):` 内。插件 DP 分支缺这些 → DP 负载均衡 stats 不发布。
**改**:DP 分支对齐 0.25.0:step 前后加 `_maybe_publish_request_counts()`;dummy batch 包 `log_iteration_details(None)`。

### P1-1 [MEDIUM-HIGH] AFDFFNWorker.compile_or_warm_up_model 返回类型不符
`v1/worker/ffn_worker.py`(~line 79-84):返回 `0.0`(float),0.25.0 base 返回 `CompilationTimes`(dataclass)。
**改**:返回 `CompilationTimes(...)`(字段填 0/None);若无调用方读其字段可保留 float 但建议对齐类型。**先确认是否有调用方读返回值**。

### P1-2 [HIGH] GPUFFNModelRunner.load_model 完全自实现,可能漏 0.25.0 新初始化
`v1/worker/ffn_model_runner.py`(~line 105-120):不调 base,自己 `get_model_loader().load_model()`。0.25.0
`GPUModelRunner.load_model`(:5203)新增 LoRA init / drafter 加载 / EPLB。FFN 若不用 LoRA/drafter 大概率安全。
**改**:确认 FFN 场景不需要 drafter/LoRA;否则补齐。先按"安全跳过"跑,出问题再补。

### P1-3 [MEDIUM] engine_core FFN shutdown 缺 cleanup_dist_env_and_memory + 其它签名扩展
- `engine_core.py` FFN shutdown(~line 196-207):补 `cleanup_dist_env_and_memory()`(try/except 包)。
- `initialize_from_config`(ffn_worker.py):0.25.0 base 会 `ensure_kv_transfer_initialized()`,AFD 全 override 跳过——确认 AFD 连接器自管即可。
- `AFDAttentionModelRunner._dummy_run`(attention_model_runner.py:408-418):0.25.0 `_dummy_run`(:5720)新增
  `is_profile/create_mixed_batch/profile_seq_lens` 等(默认安全);如需精确 profiling 补传 `profile_seq_lens`。
- `GPUFFNModelRunner.initialize_kv_cache`:0.25.0 base 加 `is_profiling=False` 参数——存根签名加上以防调用方传入。

## 今晚顺序(Priority-1 通过后)
1. 先按 P0-1/P0-2/P0-3 改(阻断性:属性缺失/签名不符/DP loop);
2. 起 AFD 1A1F(GLM-5.2-FP8 先,避开 W4AFP8 变量)冒烟;按崩溃点对照 P1-x 逐个补;
3. 再上 W4AFP8 + 4A4F;correctness(vs 单实例)+ 简单 perf。

> 数据来源:2026-07-28 三路静态 diff(engine_core / async_dp+forward_context / config+subclass);73 项依赖清单见 PORT_TO_VLLM_0.25.0.md。
