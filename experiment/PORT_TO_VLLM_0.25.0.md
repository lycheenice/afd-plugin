# 移植 afd-plugin → vLLM 0.25.0(设计 / 状态 / 故障排查)

> 2026-07-27 夜,无人值守。依据:vLLM 0.19.1 缺 GLM-5.2 attention backend(head_size=704),
> 0.25.0 已修且 native GLM-5.2-FP8 正确(见 `ANALYSIS_glm52_vllm0191_blocker_20260727.md`)。
> 用户指令:A 反复无果则移植 afd-plugin 到最新 vLLM,全程文档化。**目标 vLLM = 0.25.0**(已在 gpu-host)。

## 已确立的事实
1. **0.25.0 native GLM-5.2-FP8 正确**(无插件)——架构支持 OK,是移植的坚实地基。
2. **0.25.0 原生不认 w4afp8 量化**(`Unknown quantization method: w4afp8`)——w4afp8 是插件自带,GLM-5.2-W4AFP8 必须靠插件。
3. **afd-plugin 在 0.25.0 可安装 + register_afd 可运行**:`pip install -e` 成功;版本断言 `strict=False` 仅警告;
   **w4afp8 量化注册成功**(`"w4afp8" in QUANTIZATION_METHODS == True`)。register_afd 各步 try/except 静默,鲁棒。

## 移植分层与工作量
| 层 | 文件 | 0.19.1→0.25.0 风险 | 本期 |
|---|---|---|---|
| 版本门 | compat/vllm.py | 低(strict=False 已不阻塞;可加 0.25.0 到支持集) | 待办 |
| compat/patches | async_dp_engine/engine_core/forward_context/config_validation | 高(猴补丁贴 0.19.1 内部,try/except 静默失败;AFD-async 才需要) | native 不需要,AFD 需 |
| **量化桥接** | quantization/w4afp8.py | **中高**(见下故障) | **本期主攻(native W4AFP8 正确性)** |
| AFD 模型 wrapper | model_executor/models/deepseek_v2.py | 中(subclass vLLM Deepseek,__init__/load_weights 随上游变) | 部分 |
| AFD worker/连接器 | v1/worker, connectors/gpu | 高(hook vLLM worker/executor 内部) | 下阶段(full AFD) |

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
gpu-host 容器 `afd-v25`(镜像 `docker.1ms.run/vllm/vllm-openai:v0.25.0`,挂 /data1/models、/data1/afd-plugin,
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
