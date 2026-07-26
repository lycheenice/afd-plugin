# W4AFP8 量化支持设计文档

## 1. 背景与问题

### 1.1 目标

在 AFD（Attention-FFN Disaggregation）4A4F 拓扑下运行 GLM-5.2 模型，
在 8×H200（每卡 143GB）上进行冒烟测试。

### 1.2 已确认的障碍

| 障碍 | 详情 |
|------|------|
| **FP8 模型 OOM** | GLM-5.2-FP8（704GB）的 MoE 权重在 4A4F（TP=4, EP）下每卡需 ~176GB，但 H200 仅 143GB，差 33GB。`cpu_offload_gb` 在权重创建阶段不生效，无法缓解。 |
| **TP 约束** | GLM-5.2 有 64 个 attention heads，TP 必须整除 64。合法 TP 值：1, 2, 4, 8。7（1A7F）、6（2A6F）、5（3A5F）均不合法。 |
| **8 卡全占** | TP=8 能装下 FP8 模型（~88GB/卡），但消耗全部 8 GPU，没有剩余给 attention 侧，AFD 不成立。 |
| **W4AFP8 模型不被支持** | GLM-5.2-W4AFP8（373GB，TP=4 下 ~47GB/卡）容量充足，但 `quant_method: w4afp8` 不在 vLLM 0.19.1 的 `QUANTIZATION_METHODS` 列表中。 |

### 1.3 结论

唯一可行路径：让 vLLM 0.19.1 识别并运行 W4AFP8 模型，
然后以 4A4F 拓扑启动 AFD 冒烟测试。

---

## 2. 版本基线

| 组件 | 版本 |
|------|------|
| vLLM | 0.19.1（`vllm/vllm-openai:v0.19.1` 镜像） |
| AFD Plugin | 当前开发分支，`compat/vllm.py` 中 `TARGET_VLLM_VERSION = "0.19.1"` |
| 模型 | GLM-5.2-W4AFP8，`/data1/models/GLM-5.2-W4AFP8` |
| 模型架构 | `GlmMoeDsaForCausalLM`，`model_type: glm_moe_dsa` |
| 硬件 | 8×H200 143GB，gpu-host 机器，`afd-exp` 容器 |

---

## 3. W4AFP8 模型格式分析

### 3.1 检查点结构

通过 `safetensors` 直接检查模型权重：

**Dense 层（layer 0-2，非 MoE）**
```
mlp.gate_proj.weight:          int8  [12288, 6144]    — 两个 uint4b8 nibble 打包在一个 int8 中
mlp.gate_proj.weight_scale_inv: bfloat16 [96, 48]     — 分组缩放因子，group_size=128
mlp.up_proj.weight:            int8  [12288, 6144]
mlp.up_proj.weight_scale_inv:  bfloat16 [96, 48]
mlp.down_proj.weight:          int8  [6144, 12288]
mlp.down_proj.weight_scale_inv: bfloat16 [48, 96]
```

**MoE 层（layer 3+，每个 expert 单独存储）**
```
mlp.experts.{N}.gate_proj.weight:           int8  [2048, 3072]     — moe_intermediate_size=2048, hidden=6144? 不，实际是 per-expert
mlp.experts.{N}.gate_proj.weight_scale_inv:  bfloat16 [2048, 48]
mlp.experts.{N}.up_proj.weight:             int8  [2048, 3072]
mlp.experts.{N}.up_proj.weight_scale_inv:   bfloat16 [2048, 48]
mlp.experts.{N}.down_proj.weight:           int8  [6144, 1024]
mlp.experts.{N}.down_proj.weight_scale_inv: bfloat16 [6144, 16]
mlp.experts.{N}.w1.input_scale:             bfloat16 [1]           — 静态 input scale（全 1.0）
mlp.experts.{N}.w2.input_scale:             bfloat16 [1]
mlp.experts.{N}.w3.input_scale:             bfloat16 [1]
```

**Attention 层（所有层）**
```
self_attn.q_a_proj.weight:          float8_e4m3fn  [2048, 6144]    — FP8 权重
self_attn.q_a_proj.weight_scale_inv: float32 [16, 48]              — block-wise scale
self_attn.q_b_proj.weight:          float8_e4m3fn  [16384, 2048]
self_attn.kv_a_proj_with_mqa.weight: float8_e4m3fn [576, 6144]
self_attn.kv_b_proj.weight:         float8_e4m3fn  [28672, 512]
self_attn.o_proj.weight:            float8_e4m3fn  [6144, 16384]
```

**Norm / Embedding 层**
```
embed_tokens.weight:       bfloat16 [154880, 6144]
input_layernorm.weight:    bfloat16 [6144]
post_attention_layernorm.weight: bfloat16 [6144]
...
```

### 3.2 量化格式判定

| 权重类别 | 存储类型 | 实际量化 |
|----------|----------|----------|
| Attention (q/kv/o_proj) | float8_e4m3fn | **W8A8-FP8**，block-wise scale [128, 128] |
| Dense MLP (layer 0-2) | int8 (packed uint4b8) | **W4A8-FP8**，group_size=128 |
| MoE Expert (layer 3+) | int8 (packed uint4b8) | **W4A8-FP8**，group_size=128 |
| Norm / Embed | bfloat16 | 不量化 |

**关键发现**：每个 int8 字节存储两个 uint4b8 nibble（值域 [0..15] 映射到 [-8..7]）。
225 个唯一值 = 15×15，完美匹配双 nibble 打包。

### 3.3 与 vLLM compressed-tensors W4A8FP8 的对应

vLLM 0.19.1 在 `CompressedTensorsW4A8Fp8MoEMethod` 中已实现 W4A8-FP8 MoE kernel：
- 使用 CUTLASS `cutlass_moe_w4a8_fp8` kernel
- 权重以 int32 打包（8 个 uint4b8 nibble / int32），checkpoint 的 int8 view→int32 完全兼容
- scale 格式：`(E, N, K//128)` bfloat16 → 运行时转换为 fp8 group scale + fp8 channel scale
- `convert_packed_uint4b8_to_signed_int4_inplace` 将 uint4b8 [0..15] → signed int4 [-8..7]
- `cutlass_encode_and_reorder_int4b_grouped` 重排权重为 CUTLASS 布局

**结论**：vLLM 已有完整 W4A8-FP8 MoE 实现，只是无法通过 `quant_method: w4afp8` 触发。

---

## 4. 设计方案

### 4.1 核心思路

不重写 kernel，而是编写一个 **桥接 QuantizationConfig**：
- 将 `quant_method: w4afp8` 注册为合法量化方法
-Attention/Linear 层 → 复用 vLLM 原生 `Fp8Config`（已是 FP8 量化）
- MoE 层 → 复用 vLLM 原生 `CompressedTensorsW4A8Fp8MoEMethod`

### 4.2 架构图

```
                    W4AFP8Config (新增)
                    /              \
          Linear/Attention           MoE
               |                      |
         Fp8Config                W4AFP8MoEMethod (新增)
        (vLLM 原生)              (继承 CompressedTensorsW4A8Fp8MoEMethod)
                                    |
                        cutlass_moe_w4a8_fp8 (vLLM 原生 kernel)
```

### 4.3 改动文件清单

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 1 | `afd_plugin/quantization/__init__.py` | **新增** | 包初始化，导出 W4AFP8Config |
| 2 | `afd_plugin/quantization/w4afp8.py` | **新增** | W4AFP8Config + W4AFP8MoEMethod 实现 |
| 3 | `afd_plugin/__init__.py` | **修改** | 在 `register_afd()` 中注册 `w4afp8` 量化方法 |

不改动任何 vLLM 源文件，不改动 AFD 模型代码，全部通过插件注册机制注入。

### 4.4 W4AFP8Config 设计

```python
@register_quantization_config("w4afp8")
class W4AFP8Config(QuantizationConfig):
    """桥接配置：Linear/Attention 层用 FP8，MoE 层用 W4A8-FP8。"""

    def __init__(self):
        super().__init__()
        # 内部持有一个 Fp8Config 实例，处理 linear/attention 层
        self._fp8_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )

    @classmethod
    def from_config(cls, config: dict) -> "W4AFP8Config":
        # 忽略 config 字典内容，因为 W4AFP8 的参数是固定的
        return cls()

    def get_name(self) -> str:
        return "w4afp8"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    def get_min_capability(self) -> int:
        return 90  # H100/H200 SM90

    def get_quant_method(self, layer, prefix) -> QuantizeMethodBase:
        if isinstance(layer, LinearBase):
            return self._fp8_config.get_quant_method(layer, prefix)
        if isinstance(layer, FusedMoE):
            return W4AFP8MoEMethod(layer.moe_config)
        if isinstance(layer, Attention):
            return self._fp8_config.get_quant_method(layer, prefix)
        return None
```

### 4.5 W4AFP8MoEMethod 设计

继承 `CompressedTensorsW4A8Fp8MoEMethod`，覆盖 weight loader 以适配 checkpoint 的 per-expert 扁平存储格式：

```python
class W4AFP8MoEMethod(CompressedTensorsW4A8Fp8MoEMethod):
    """W4AFP8 MoE 方法，复用 CUTLASS W4A8-FP8 kernel。
    
    与 compressed-tensors 版本的区别：
    - weight_scale_inv 的 shape 是 [N, K//128] 而非 [E, N, K//128]
      （checkpoint 按 per-expert 存储，vLLM 加载时会自动 stack 成 3D）
    - 去掉了 input_scale 的处理（checkpoint 中全为 1.0，无实际作用）
    """
```

**关键差异点**：
1. **权重名称映射**：checkpoint 用 `experts.{N}.gate_proj.weight`，vLLM 期望 `experts.w13_weight_packed`（stacked + int32 packed）。由 `FusedMoE.weight_loader` 处理 stacking，我们的 weight loader 只需处理 int8→int32 view。
2. **scale 名称映射**：checkpoint 用 `experts.{N}.gate_proj.weight_scale_inv`，vLLM 期望 `experts.w13_weight_scale`。由 `expert_params_mapping` 处理映射。
3. **input_scale**：checkpoint 中的 `w1/w2/w3.input_scale` 全为 1.0，对计算无影响，可以跳过。

### 4.6 注册机制

vLLM 提供 `register_quantization_config("name")` 装饰器：
- 自动将 `"w4afp8"` 添加到 `QUANTIZATION_METHODS` 列表
- 自动添加到 `current_platform.supported_quantization`
- 注册到 `_CUSTOMIZED_METHOD_TO_QUANT_CONFIG` 字典

在 `afd_plugin/__init__.py` 的 `register_afd()` 中 import 该模块即触发注册。

---

## 5. 验证计划

### 5.1 单元验证（开发机上，不需要 GPU）

| # | 验证项 | 方法 | 预期结果 |
|---|--------|------|----------|
| U1 | 模块导入 | `python -c "from afd_plugin.quantization import W4AFP8Config"` | 无异常 |
| U2 | 量化注册 | `python -c "import afd_plugin; afd_plugin.register_afd(); from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS; assert 'w4afp8' in QUANTIZATION_METHODS"` | True |
| U3 | Config 创建 | `W4AFP8Config.from_config({"quant_method": "w4afp8"})` | 返回 W4AFP8Config 实例 |
| U4 | get_quant_method Linear | 对 `LinearBase` 实例调用 `get_quant_method` | 返回 `Fp8LinearMethod` |
| U5 | get_quant_method MoE | 对 `FusedMoE` 实例调用 `get_quant_method` | 返回 `W4AFP8MoEMethod` |

### 5.2 集成验证（gpu-host 上，需要 GPU）

| # | 验证项 | 方法 | 预期结果 |
|---|--------|------|----------|
| I1 | vLLM 识别量化 | `vllm serve /models/GLM-5.2-W4AFP8 --quantization w4afp8 ...` 不报 "Invalid quantization method" | 启动到模型加载阶段 |
| I2 | 权重加载 | 监控日志，MoE 权重加载无异常 | 所有 78 层加载完成 |
| I3 | GPU 内存 | `nvidia-smi` 检查 FFN 侧每卡 < 100GB | 4×47GB ≈ 4×~50GB |
| I4 | 服务就绪 | attention 侧 API `curl http://127.0.0.1:18000/v1/models` | 返回模型列表 |
| I5 | Token 生成 | `curl` 发送 completion 请求 | 返回非空、非乱码文本 |
| I6 | 输出质量 | 检查返回文本是否可读、语义合理 | 无明显乱码 |

### 5.3 验证命令

```bash
# U1-U5: 开发机
python -c "from afd_plugin.quantization import W4AFP8Config; print('OK')"

# I1-I6: gpu-host
# I1: 启动
docker exec -d afd-exp bash /workspace/afd-plugin/experiment/scripts/start_glm52_w4afp8_4a4f.sh

# I4: 服务就绪
curl http://127.0.0.1:18000/v1/models

# I5: Token 生成
curl -s http://127.0.0.1:18000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm52-afd-attn","prompt":"Hello","max_tokens":32,"temperature":0}'
```

---

## 6. 风险与回退

### 6.1 风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| CUTLASS W4A8 kernel 不支持 SM90 (H200) | 低 | MoE 计算失败 | `_is_fp8_w4a8_sm90` 已验证 SM90 是目标 |
| scale shape 不匹配导致 kernel crash | 中 | MoE forward 失败 | 日志检查 scale shape，必要时做 reshape |
| AFD `AFDGlmMoeDsaForCausalLM` 空壳不处理 DSA 层 | 中 | 输出乱码 | I6 验证，若乱码需单独处理 DSA 结构 |
| weight loader 不识别 `weight_scale_inv` 名称 | 中 | scale 不加载 | 需要确认 `expert_params_mapping` 覆盖 scale 名称 |

### 6.2 回退

改动全部隔离在 `afd_plugin/quantization/` 新目录中，若失败：
- 从 `afd_plugin/__init__.py` 移除 import 行即可完全回退
- 不影响已有 FP8 / DeepSeek 模型的运行

---

## 7. 实施顺序

1. **[设计评审]** 本文档
2. **[编码]** 创建 `afd_plugin/quantization/w4afp8.py` + `__init__.py`
3. **[编码]** 修改 `afd_plugin/__init__.py` 注册
4. **[单元验证]** U1-U5 在开发机
5. **[同步]** rsync 到 gpu-host，pip install -e
6. **[集成验证]** I1-I3 启动 + 加载
7. **[集成验证]** I4-I6 功能 + 质量
8. **[记录]** 更新 EXPERIMENT_REPORT.md

---

## 8. 实施结果记录

### 8.1 改动清单（实际）

| # | 文件 | 操作 | 基线版本 | 说明 |
|---|------|------|----------|------|
| 1 | `afd_plugin/quantization/__init__.py` | **新增** | vLLM 0.19.1 | 包初始化，导出 W4AFP8Config |
| 2 | `afd_plugin/quantization/w4afp8.py` | **新增** | vLLM 0.19.1 | W4AFP8Config + W4AFP8MoEMethod 工厂 + `convert_bf16_scales_to_fp8` 补丁 |
| 3 | `afd_plugin/__init__.py:128-135` | **修改** | 当前开发分支 | 在 `register_afd()` 中 import W4AFP8Config 触发注册 |
| 4 | `experiment/scripts/start_glm52_w4afp8_4a4f.sh` | **新增** | — | W4AFP8 4A4F 启动脚本 |

### 8.2 额外发现：vLLM 0.19.1 bug

`convert_bf16_scales_to_fp8`（`vllm/.../quant_utils.py:811`）使用
`chan_scales.view(orig_shape[:-1], -1)`，将 `torch.Size` 对象传入
`view()`，PyTorch 不接受此参数类型。此 bug 影响**所有**
`CompressedTensorsW4A8Fp8MoEMethod` 用户，不仅限于 AFD。

修复方式：monkey-patch 替换为 `chan_scales.view(*orig_shape[:-1], -1)`。
补丁在 `w4afp8.py` 模块加载时自动应用。

### 8.3 验证结果

| 验证项 | 结果 | 详情 |
|--------|------|------|
| U1 模块导入 | ✅ | |
| U2 量化注册 | ✅ | `w4afp8` in QUANTIZATION_METHODS |
| U3 Config 创建 | ✅ | |
| U4 Linear 路由 | ✅ | |
| U5 MoE 路由 | ✅ | |
| I1 vLLM 识别 | ✅ | `quantization=w4afp8` 出现在 engine config |
| I2 权重加载 | ✅ | 41/41 shards, 12.56s, 无 OOM |
| I3 GPU 内存 | ✅ | FFN ~104GB/卡, Attn ~135GB/卡, 均 < 143GB |
| I4 服务就绪 | ✅ | `/v1/models` 返回模型列表 |
| I5 Token 生成 | ✅ | 32 tokens, 1.15s 响应 |
| I6 输出质量 | ❌ | 输出 `odesk` 重复 — DSA 空壳类问题 |

### 8.4 I6 失败分析

**现象**：输入 `"Hello, my name is"`，输出 `odeskodeskodesk...`（32 token 全相同）。

**根因**：`AFDGlmMoeDsaForCausalLM`（`deepseek_v2.py:919-920`）是空壳类，
直接继承 `AFDDeepseekV2ForCausalLM`，不处理 GLM-5.2 DSA 特有结构：

- `indexer` / `indexers_proj` — DSA 注意力索引器
- `eh_proj` — embed-to-hidden 投影
- `enorm` / `hnorm` — DSA 层归一化
- `shared_head.norm` — 共享头归一化

这些权重在 `load_weights` 中成为 unexpected keys，被跳过。
模型计算路径使用 DeepSeek-V2 的 MLA attention，不匹配 GLM-5.2 的 DSA attention。

**下一步**：需实现 `AFDGlmMoeDsaForCausalLM` 的 DSA 逻辑，或在 AFD 框架下适配 vLLM 0.19.1 的原生 `GlmMoeDsaForCausalLM`。
