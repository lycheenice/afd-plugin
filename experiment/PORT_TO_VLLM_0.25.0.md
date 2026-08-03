# 移植 afd-plugin → vLLM 0.25.0(设计 / 状态 / 故障排查)

> 2026-07-27 夜,无人值守。依据:vLLM 0.19.1 缺 GLM-5.2 attention backend(head_size=704),
> 0.25.0 已修且 native GLM-5.2-FP8 正确(见 `ANALYSIS_glm52_vllm0191_blocker_20260727.md`)。
> 用户指令:A 反复无果则移植 afd-plugin 到最新 vLLM,全程文档化。**目标 vLLM = 0.25.0**(已在 gpu-host)。

## 2026-07-28 AFD 拆分栈实施结论

GPU 同步 P2P 的最小可用路径已经迁移到 vLLM 0.25.0,并在 `redacted-host` 的 H20 上完成
DeepSeek-V2-Lite `1A1F` eager 冒烟。vLLM 源码和镜像均未修改,所有兼容行为位于插件侧。

### 支持边界

| 项 | v0.25.0 状态 | 证据/约束 |
|---|---|---|
| general plugin 注册 | 通过 | `register_afd()` 无版本警告 |
| 自动 Worker 选择 | 通过 | 启动命令未传 `--worker-cls`,FFN 成功进入 connector loop |
| DeepSeek-V2-Lite 1A1F eager | 通过 | HTTP 200,输出连贯;Attention 约 1.54 GiB,FFN 约 29 GiB |
| P2pNccl 控制/数据面 | 通过 | FFN/Attention NCCL 握手、逐层往返和生成完成 |
| model runner v2 | 不支持 | 必须设置 `VLLM_USE_V2_MODEL_RUNNER=0` |
| DP>1 / DBO / CUDA graph / async | 待专项硬件回归 | 已完成静态接口对齐与相关单测,不由本次 1A1F eager 冒烟外推 |
| v0.19.1 旧基线 | 保留 | 支持集合同时包含 `0.19.1` 与 `0.25.0` |
| Ascend NPU | 仍以 0.19.1rc1 为基线 | 本次不宣称 vLLM-Ascend 0.25 兼容 |

### 根因链与设计决策

1. **禁止半套运行栈。** 首次 0.25 验证中 `engine_core` patch 无条件生效,但
   `config_validation` 被 `TARGET_VLLM_VERSION=0.19.1` 门禁,形成“AFD EngineCore + 原生
   Worker”。FFN 因此在 `start_ffn_server_loop` RPC 上报 `NotImplementedError`。版本声明现改为
   `TARGET_VLLM_VERSION=0.25.0`,并用明确支持集合保留 0.19.1;所有版本门统一查询该集合。
2. **Worker 继续由配置规范化自动选择。** 不把 `--worker-cls` 固化到部署脚本。补丁在上游
   `VllmConfig.__post_init__` 完成平台默认选择后,仅对激活 AFD 的配置替换为角色 Worker。
3. **模型 wrapper 使用稳定能力接口。** 0.25 删除
   `fused_moe.shared_fused_moe.SharedFusedMoE`,改为公开函数
   `fused_moe_make_expert_params_mapping`;插件做双版本导入,不复制 0.25 MoE 内部实现。
4. **控制面协议不依赖 vLLM DPMetadata 字段。** 0.25 删除
   `max_tokens_across_dp_cpu`;发送前统一转换为 `AFDDPMetadata`,最大 token 数从
   `num_tokens_across_dp_cpu` 重算。连接器 JSON 线格式保持不变。
5. **EngineCore patch 以 0.25 为上游基准。** FFN daemon 只标注并保留 AFD 差异;
   非 AFD/Attention 路径补齐 0.25 的 request-count 发布、sleeping dummy guard、迭代日志和
   distributed cleanup。FFN 存根补齐 0.25 新增状态字段。
6. **FFN Runner 保持最小职责。** 它不继承完整 `GPUModelRunner` 的 KV、采样、LoRA 和 drafter
   生命周期;仅对齐被 vLLM Worker 调用的签名/返回类型。若未来支持 FFN LoRA/speculative,
   应单独设计,不能把本次“未用到”解释为已支持。

### 实现文件

| 文件 | 改动 |
|---|---|
| `afd_plugin/compat/vllm.py` | 目标版本 0.25.0 + 0.19.1/0.25.0 支持集合 |
| `compat/patches/config_validation.py` | 两版本自动 AFD Worker 选择 |
| `compat/patches/engine_core.py` | FFN 新状态、0.25 DP loop、distributed cleanup |
| `compat/patches/async_dp_forward_context.py` | `is_padding`、SP-MoE 判据及 DP fallback |
| `compat/patches/async_dp_engine.py` | 统一双版本门控 |
| `model_executor/models/deepseek_v2.py` | 双版本 MoE 专家映射 API |
| `v1/worker/ffn_worker.py` | 0.25 `CompilationTimes` 返回契约 |
| `v1/worker/ffn_model_runner.py` | `initialize_kv_cache(..., is_profiling=False)` |
| `v1/worker/attention_model_runner.py` | 控制面 DPMetadata 归一化 |

### 可复现环境与命令

- 主机:`redacted-host`,8× NVIDIA H20-3e 143 GiB。
- 代码:`/workspace/afd-plugin`,验证基线 `9d6aa37` 加本次工作区改动。
- 模型:`/models/DeepSeek-V2-Lite`。
- 镜像:`docker.m.daocloud.io/vllm/vllm-openai:v0.25.0`,ID
  `sha256:fc56161ee42a011aeee78b65d0a81b6683c7d04402fd40503d14d4d6c98f07cb`。
- 容器:`afd-v025-validate`;代码只读挂载为 `/workspace/afd-plugin`,模型只读挂载为 `/models`。

```bash
docker exec -e VLLM_USE_V2_MODEL_RUNNER=0 afd-v025-validate \
  python3 tests/e2e/runner.py \
  --model /models/DeepSeek-V2-Lite \
  --vllm-bin /usr/local/bin/vllm \
  --device-backend gpu \
  --num-attention-ranks 1 --num-ffn-ranks 1 \
  --attention-gpus 0 --ffn-gpus 1 \
  --api-port-base 18100 --afd-port 6339 \
  --startup-timeout 900 \
  --common-vllm-arg=--trust-remote-code
```

成功判据不是 runner 的退出码单项,而是同时满足:

1. FFN 出现 `AFD FFN EngineCore started; workers run connector loop` 且无 fatal traceback;
2. Attention 权重显存显著小于完整 29 GiB 模型(本次为 1.54 GiB);
3. FFN 侧加载专家参数并参与 NCCL 往返;
4. Attention API HTTP 200,生成文本连贯;
5. runner 退出后 GPU compute 进程与 18100/18101/6339 监听均清空。

### 验证结果

| 验证层 | 结果 | 说明 |
|---|---|---|
| 本地静态检查 | 通过 | `uv lock --check`、Ruff、`compileall`、`git diff --check` 均通过 |
| 本地轻量单测 | 通过 | package/version 与 config patch 测试通过（2 项按环境跳过） |
| v0.25.0 定向回归 | 通过 | 89 passed；覆盖版本门、config、EngineCore、forward context、FFN/Attention runner |
| v0.25.0 全量 unit | 迁移相关通过，待重跑收口 | `test_p2p_topology_validation_errors_are_clear` 的旧断言已改为 fan-out 正向映射测试，并保留真正非法拓扑断言；需在 v0.25.0 环境重跑全量 unit |
| v0.25.0 真实 1A1F | 通过 | GET/POST HTTP 200，生成 16 tokens；日志保存在 `<redacted-log-path>` |
| v0.19.1 兼容回归 | 静态/单测通过，实机重跑未完成 | 直接接口字段已核验；重跑时 8 张 H20 被外部 SGLang TP8 全部占用，启动在模型加载前因显存不足退出，不是代码异常 |
| 退出资源检查 | 通过 | runner/vLLM 进程、GPU compute process、18100/18101/6339 监听均为空 |

镜像按用户要求保留。其实际 containerd snapshot 占用约 27.4 GiB；系统盘仍有 124 GiB
可用空间（87% 使用率）。验证产生的服务进程和 GPU 资源已经全部释放。

## 前期调研记录（历史）

以下内容保留早期 native GLM-5.2/W4AFP8 调研过程；若状态与上方“实施结论”冲突，以上方
最终结论为准。

### 已确立的事实
1. **0.25.0 native GLM-5.2-FP8 正确**(无插件)——架构支持 OK,是移植的坚实地基。
2. **0.25.0 原生不认 w4afp8 量化**(`Unknown quantization method: w4afp8`)——w4afp8 是插件自带,GLM-5.2-W4AFP8 必须靠插件。
3. **afd-plugin 在 0.25.0 可安装 + register_afd 可运行**:`pip install -e` 成功;版本断言 `strict=False` 仅警告;
   **w4afp8 量化注册成功**(`"w4afp8" in QUANTIZATION_METHODS == True`)。register_afd 各步 try/except 静默,鲁棒。

### 移植分层与工作量（调研阶段估算）
| 层 | 文件 | 0.19.1→0.25.0 风险 | 本期 |
|---|---|---|---|
| 版本门 | compat/vllm.py | 低(strict=False 已不阻塞;可加 0.25.0 到支持集) | 已完成 |
| compat/patches | async_dp_engine/engine_core/forward_context/config_validation | 高(猴补丁贴 0.19.1 内部) | 1A1F 最小路径已完成；DP/async 待专项 |
| **量化桥接** | quantization/w4afp8.py | **中高**(见下故障) | 独立 W4AFP8 专项，不属于本次 BF16 最小冒烟 |
| AFD 模型 wrapper | model_executor/models/deepseek_v2.py | 中(subclass vLLM Deepseek,__init__/load_weights 随上游变) | DeepSeek-V2-Lite 最小路径已完成 |
| AFD worker/连接器 | v1/worker, connectors/gpu | 高(hook vLLM worker/executor 内部) | P2pNccl 1A1F 已完成；扩展拓扑待专项 |

## 故障排查日志(逐个)
### #1 FusedMoE 由类变工厂函数(已定位)
- 现象:`w4afp8.py:132 isinstance(layer, FusedMoE) → TypeError: arg 2 must be a type`。
- 根因:0.25.0 `vllm.model_executor.layers.fused_moe.FusedMoE` 是**工厂函数**(返回 `MoERunner`);
  MoE 层类改为 `MoERunner`(`fused_moe/runner/moe_runner.py`,MRO: MoERunner→MoERunnerInterface→PluggableLayer)。
- 修法:import MoE 层类时版本兼容(0.25.0 用 MoERunner,0.19.1 用 FusedMoE 类),isinstance 用之。
- 连带待查:`layer.moe_config`(line 133)在 MoERunner 上是否存在;`CompressedTensorsW4A8Fp8MoEMethod` 路径/签名;
  `_patch_convert_bf16_scales_to_fp8` 目标函数是否仍在;remap 目标参数名布局。

（后续故障 #2… 随迭代追加）

## 复现环境
gpu-host 容器 `afd-v25`(镜像 `docker.1ms.run/vllm/vllm-openai:v0.25.0`,挂 /models、/workspace/afd-plugin,
`pip install -e` 已装)。启动:`docker exec -e VLLM_PLUGINS=afd -e PYTHONPATH=/workspace/afd-plugin afd-v25
vllm serve /models/GLM-5.2-W4AFP8 --tensor-parallel-size 8 --enable-expert-parallel --enforce-eager ...`。
日志 `experiment/logs/glm_w4afp8_v25_native.log`。

### #2 CompressedTensorsW4A8Fp8MoEMethod 移位(已修)
- 0.25.0:`compressed_tensors_moe` 变为**包**;类在
  `compressed_tensors_moe/compressed_tensors_moe_w4a8_fp8.py`。`__init__(weight_quant, input_quant, moe, layer_name=None)`
  与插件 create() 传参兼容。
- 修法:w4afp8.py `_W4AFP8MoEMethod.create` 版本兼容 import(0.25.0 新路径 / 0.19.1 旧路径)。

### #3 get_quant_method 收到的 MoE 层类变了(已修)
- 0.25.0 MoE 栈:`FusedMoE(工厂)` → `MoERunner` → **`RoutedExperts`**;
  `RoutedExperts._get_quant_method` 用 `quant_config.get_quant_method(self=RoutedExperts, prefix)`,返回 None 就
  `UnquantizedFusedMoEMethod`(→ 专家未量化 → 巨大 → OOM)。
- 修法:w4afp8.py isinstance 匹配 `RoutedExperts`(0.25.0)/ `FusedMoE`(0.19.1)。`RoutedExperts.moe_config` 存在,`create(layer.moe_config)` OK。
- 效果:MoE 走 W4A8 量化(int4),不再 OOM,进入权重加载。

### #4 残留 GPU 进程 OOM(运维,已处理)
- 前次崩溃的 vllm worker 残留占满显存(pid 独占 139GB)。清场需宿主 `kill -9` 所有 `nvidia-smi` compute-apps,不能只 `docker exec pkill`(跨容器杀不到)。

### #5 插件模型 wrapper 未覆盖原生模型 + W4A8 专家权重名不匹配(待修,当前阻塞)
- 现象:权重加载 `KeyError: 'layers.3.mlp.experts.routed_experts.w2_weight'`(deepseek_v2.py:1685)。
- 分析:0.25.0 `CompressedTensorsW4A8Fp8MoEMethod.create_weights` 注册的是
  `w13/w2_weight_packed`(+`_scale`/`_shape`/`_chan_scale`),路径含**新的 `.routed_experts.` 中缀**。
  KeyError 是 `w2_weight`(无 `_packed`)→ 说明 **w4afp8 的 remap(`.weight`→`.weight_packed`)没被应用**。
- 根因:traceback 跑的是 **vLLM 原生 `deepseek_v2.load_weights`**,不是插件 `AFDGlmMoeDsa` wrapper 的
  load_weights(remap 钩子在 wrapper 里)。即**插件模型 wrapper 未激活**(ModelRegistry 注册未覆盖 0.25.0
  内置 GlmMoeDsa,或注册 API/优先级变化)。虽 `get_supported_archs()` 含 GlmMoeDsaForCausalLM,但实际实例化的是原生类。

## 剩余 roadmap(让 native GLM-5.2-W4AFP8 在 0.25.0 正确)
1. **模型 wrapper 激活**:核对 0.25.0 `ModelRegistry.register_model` 的注册/覆盖语义,确保
   `GlmMoeDsaForCausalLM` 解析到 `AFDGlmMoeDsaForCausalLM`(或改为 monkeypatch 原生 load_weights 注入 remap)。
2. **remap 适配 0.25.0**:确认 remap 后目标名产出 `...routed_experts.w2_weight_packed`(含 `.routed_experts.` 中缀
   与 `_packed`);核对 `make_expert_params_mapping` 在 0.25.0 的后缀保留逻辑。
3. **数值复核**:remap 的 `^0x88`(有符号 int4→uint4b8)与 `convert_bf16_scales_to_fp8` patch 在 0.25.0 的
   W4A8 dequant/CUTLASS 路径下是否仍正确(0.19.1 曾需 patch view bug;0.25.0 该函数仍在,但行为需再验)。
4. 通过后再谈 **AFD 拆分层移植**(compat/patches + v1/worker + connectors,hook vLLM executor 内部,工作量最大)。

## 已修改文件(本夜)
- `afd_plugin/quantization/w4afp8.py`:#1 MoE 层类→RoutedExperts;#2 CT-W4A8 import 路径。版本兼容(0.19.1/0.25.0 双向)。

## Priority-1 修复(已实现,待今晚验证)
- **新增 `afd_plugin/compat/patches/w4afp8_native_load_weights.py`**:monkeypatch
  `DeepseekV2ForCausalLM.load_weights`(GlmMoeDsa 继承之),quant==w4afp8 时先应用
  `remap_w4afp8_moe_checkpoint_weights` 再 delegate。解决故障 #5(native 路径不走 AFD wrapper → remap 未应用)。
  remap 幂等(第二遍 key 已 `.weight_packed`/dtype int32 → no-op),与 0.19.1 AFD wrapper 共存安全。已在 register_afd 独立 try/except 注册。
- 验证脚本:`experiment/scripts/port_v25_setup_and_serve.sh`(建 afd-v25 容器+装+起 native W4AFP8)+ `port_v25_smoke.sh`(greedy 冒烟+判读)。
- **今晚测试预案**:①停 SGLang → ②`bash port_v25_setup_and_serve.sh` → ③`bash port_v25_smoke.sh`
  → ④若输出连贯(France→Paris)= Priority-1 达成;若仍 KeyError/乱码,查下一故障(可能 `make_expert_params_mapping` 后缀逻辑 / 数值)。

## Priority-2 scoping:AFD 拆分栈 vs 0.25.0(静态依赖清单,73 项/12 文件)
**最高风险 = 11 处 monkeypatch(目标签名若变则静默失效/崩)**:
- `compat/patches/async_dp_engine.py`:`EngineCoreProc.run_engine_core`、`{engine.utils,core_client}.launch_core_engines`、`DPAsyncMPClient.add_request_async`(`vllm.v1.engine.*`)。
- `compat/patches/async_dp_forward_context.py`:`forward_context.set_forward_context`。
- `compat/patches/config_validation.py`:`EngineArgs.create_engine_config`、`VllmConfig.__post_init__`。
- `compat/patches/engine_core.py`:`EngineCore.__init__/_initialize_kv_caches/shutdown`、`EngineCoreProc.run_busy_loop`、`DPEngineCoreProc.run_busy_loop`。
**5 处 subclass**:`AFD{Attention,FFN}Worker`←`v1.worker.gpu_worker.Worker`;`AFDAttentionModelRunner`←`gpu_model_runner.GPUModelRunner`;`GPUFFNModelRunner`←`lora_model_runner_mixin.LoRAModelRunnerMixin`;`P2pNcclAFDConnector`←插件 base。
**引用最密集的 vLLM 模块(改动面)**:`vllm.v1.engine.core`(11)、`vllm.config`(9)、`vllm.v1.worker.*`(8)、`vllm.forward_context`(8)、`vllm.distributed.parallel_state`(5)。
**评估**:`compat/patches`(尤其 engine_core / async_dp)是最难、最脆的一块(hook 引擎/DP 内部,0.19.1→0.25.0 引擎重构面大);worker/connector subclass 次之。逐项 OK/MOVED/CHANGED 分类见下节(静态核对中)。

## Priority-2 差异分类(静态核对 0.25.0,2026-07-28)
**结构全部存活**——11 处猴补丁目标 + 5 处 subclass 基类在 0.25.0 均在原路径:
| vLLM 符号 | 0.25.0 位置 | 状态 |
|---|---|---|
| EngineCore / EngineCoreProc / DPEngineCoreProc | v1/engine/core.py:96/896/1745 | OK(路径同) |
| EngineCore._initialize_kv_caches / .shutdown / run_engine_core / run_busy_loop | core.py:240/644/1154/1259,DP:1925 | OK(路径同,**签名待逐个核对**) |
| launch_core_engines | v1/engine/utils.py:1072 | OK |
| forward_context.set_forward_context | forward_context.py:260 | OK(签名待核对) |
| EngineArgs.create_engine_config | engine/arg_utils.py:1829 | OK |
| VllmConfig.__post_init__ | config/vllm.py:916 | OK(0.19.1 路径 `config.vllm` 仍在) |
| gpu_worker.Worker / gpu_model_runner.GPUModelRunner / lora_model_runner_mixin.LoRAModelRunnerMixin | 原路径 | OK |

**结论**:AFD 拆分栈移植 = **对 ~11 猴补丁 + 5 subclass 逐个核对 0.25.0 的函数签名/行为并再对齐**,
非结构性重写。风险点从"目标不存在"降为"签名/内部行为变化"(需 diff 0.19.1↔0.25.0 各被 patch 函数体)。
完整依赖清单(73 项/12 文件)见本次会话静态分析(imports 57 / subclass 5 / monkeypatch 11)。

> 备注:本移植为**纯插件侧**改动;vLLM 0.25.0 **原样使用、未改源码**(0.25.0 已原生支持 GLM-5.2 架构)。
> 若后续发现需改 vLLM 源码,将单列 "vLLM 改造" 文档记录。
