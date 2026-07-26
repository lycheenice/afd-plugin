# W4AFP8 GLM-5.2 AFD 冒烟测试 — 会话状态

> **目的**：本文档记录上一会话的工作内容和当前阻塞点，供新会话直接接续。
> 最后更新：2026-07-26

---

## 1. 最终目标

在 AFD（Attention-FFN Disaggregation）4A4F 拓扑下运行 **GLM-5.2-W4AFP8** 模型，
在 8×H200（每卡 143GB）上产出**正确的输出文本**（而非乱码 token）。

---

## 2. 实验环境

| 项 | 值 |
|---|---|
| 实验机器 | `ssh root@gpu-host`（免密，IP REDACTED_IP） |
| 开发本机 | `lychee@REDACTED_IP`（无 sudo/docker，不可跑实验） |
| 容器 | `afd-exp`（镜像 `vllm/vllm-openai:v0.19.1`，bind mount `/data1`→容器） |
| 模型路径 | 容器内 `/models/GLM-5.2-W4AFP8`，host `/data1/models/GLM-5.2-W4AFP8` |
| 代码挂载 | host `/data1/afd-plugin` → 容器 `/workspace/afd-plugin` |
| GPU | 8×H200 143GB，4A4F = GPU 0-3 Attention / GPU 4-7 FFN |

详见 `experiment/EXPERIMENT_ENV.md`。

---

## 3. 当前进展

### 3.1 已完成 — W4AFP8 量化桥接

vLLM 0.19.1 已有 `CompressedTensorsW4A8Fp8MoEMethod` 内核，但不认识 `w4afp8` 这个
quant_method 名。我们通过 `W4AFP8Config` 桥接：

- Linear/Attention → 复用 vLLM `Fp8Config`（标准 FP8）
- FusedMoE → 复用 vLLM `CompressedTensorsW4A8Fp8MoEMethod`（W4A8-FP8 MoE）

**改动文件**（均已提交）：

| 文件 | 说明 |
|---|---|
| `afd_plugin/quantization/__init__.py` | 新增，导出 W4AFP8Config |
| `afd_plugin/quantization/w4afp8.py` | 新增，W4AFP8Config + MoE 工厂 + vLLM bug 修复 |
| `afd_plugin/__init__.py:128-135` | 修改，在 `register_afd()` 中 import 触发注册 |
| `experiment/scripts/start_glm52_w4afp8_4a4f.sh` | 新增，4A4F 启动脚本 |
| `experiment/W4AFP8_DESIGN.md` | 设计文档（含第 8 节实施结果） |

### 3.2 已完成 — vLLM 0.19.1 bug 修复

`convert_bf16_scales_to_fp8`（`vllm/.../quant_utils.py`）调用
`chan_scales.view(orig_shape[:-1], -1)` 传入 `torch.Size` 对象，
PyTorch 不接受 → `TypeError`。

修复：在 `w4afp8.py` 模块加载时 monkey-patch 为
`chan_scales.view(*orig_shape[:-1], -1)`。

### 3.3 已验证的里程碑

| 验证项 | 结果 |
|---|---|
| U1-U5 单元测试（模块导入、量化注册、Config 创建、Linear/MoE 路由） | ✅ |
| I1 vLLM 识别 `quantization=w4afp8` | ✅ |
| I2 权重加载 41/41 shards，无 OOM | ✅ |
| I3 GPU 内存：FFN ~104GB/卡，Attn ~135GB/卡，均 < 143GB | ✅ |
| I4 API 服务就绪 (`/v1/models` 返回) | ✅ |
| I5 Token 生成（32 tokens，正常响应） | ✅ |
| **I6 输出质量** | **❌ 乱码** |

---

## 4. 当前阻塞问题 — DSA 输出乱码

### 4.1 现象

```bash
# 输入
curl http://127.0.0.1:18000/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"glm52-afd-attn","prompt":"Hello, my name is","max_tokens":32,"temperature":0}'

# 输出
{"choices":[{"text":"odeskodeskodeskodeskodesk..."}]}
```

32 个 token 全部是重复的无意义 token `odesk`。

### 4.2 根因分析

vLLM 0.19.1 中 `GlmMoeDsaForCausalLM` 是空壳类：

```python
# vllm/model_executor/models/deepseek_v2.py:1638
class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    pass
```

`DeepseekV2ForCausalLM` 完全没有 DSA 相关代码（grep `indexer`, `eh_proj`,
`enorm`, `hnorm`, `shared_head` 均无命中）。

GLM-5.2 的 DSA attention 结构包含以下 checkpoint 权重，但模型类中没有
对应的模块，因此这些权重被静默跳过：

| DSA 权重 | 出现位置 | 用途 |
|---|---|---|
| `self_attn.indexer.wq_b` / `wk` | 所有 78 层 | indexer 查询/键投影 |
| `self_attn.indexer.k_norm` (weight+bias) | 所有 78 层 | indexer 键归一化 |
| `self_attn.indexer.weights_proj` | 所有 78 层 | indexer 权重投影 |
| `layers.78.eh_proj.weight` | 仅 layer 78 (MTP) | embed-to-hidden 投影 |
| `layers.78.enorm.weight` / `hnorm.weight` | 仅 layer 78 (MTP) | DSA 层归一化 |
| `layers.78.shared_head.norm.weight` | 仅 layer 78 (MTP) | 共享头归一化 |

没有这些权重参与计算，attention 部分产出垃圾 hidden states → FFN 输出乱码。

### 4.3 注意

vLLM 日志中确实选择了 `DEEPSEEK_V32_INDEXER` 和 `FLASHMLA_SPARSE` attention
backend，说明 vLLM 的 attention 层级有部分 DSA awareness。但
**模型类层级没有 indexer 模块定义**，所以即使 backend 存在，也没有地方
加载和使用 indexer 权重。问题在 model class，不在 attention backend。

### 4.4 关键架构参数（来自 config.json）

```json
{
  "architectures": ["GlmMoeDsaForCausalLM"],
  "model_type": "deepseek_v3",
  "head_dim": 192,
  "index_head_dim": 128,
  "index_n_heads": 32,
  "index_topk": 2048,
  "indexer_rope_interleave": true,
  "indexer_types": ["full","full","full","shared","shared",...],
  "num_hidden_layers": 78,
  "num_nextn_predict_layers": 1,
  "n_routed_experts": 256,
  "moe_intermediate_size": 2048,
  "kv_lora_rank": 512,
  "q_lora_rank": 2048,
  "qk_nope_head_dim": 192,
  "qk_rope_head_dim": 64,
  "quantization_config": {"quant_method": "w4afp8"}
}
```

---

## 5. 如何复现当前问题

### 5.1 同步代码到实验机

```bash
# 在开发本机 (/home/lychee/mycode/afd-plugin)
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' \
  afd_plugin/ experiment/scripts/ \
  root@gpu-host:/data1/afd-plugin/
ssh root@gpu-host 'docker exec afd-exp pip install -e /workspace/afd-plugin --no-deps --no-build-isolation'
```

### 5.2 启动 4A4F

```bash
ssh root@gpu-host 'docker exec -d afd-exp bash /workspace/afd-plugin/experiment/scripts/start_glm52_w4afp8_4a4f.sh'
```

脚本内容要点：
- Attention: `CUDA_VISIBLE_DEVICES=0,1,2,3`，TP=4，端口 18000
- FFN: `CUDA_VISIBLE_DEVICES=4,5,6,7`，TP=4，端口 18001
- 共同: `--quantization w4afp8 --enforce-eager --max-model-len 8192`
- AFD connector: `P2pNcclAFDConnector` port 6252
- 环境变量: `PYTHONPATH=/workspace/afd-plugin VLLM_PLUGINS=afd`

### 5.3 等待就绪（约 5 分钟）

```bash
# 查看进度
ssh root@gpu-host 'docker exec afd-exp bash -c "tail -n 5 /workspace/afd-plugin/experiment/logs/glm52_w4afp8_attn.log"'
# 看到 "Application startup complete" 即就绪
```

### 5.4 复现乱码

```bash
ssh root@gpu-host 'docker exec afd-exp curl -s http://127.0.0.1:18000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"glm52-afd-attn\",\"prompt\":\"Hello, my name is\",\"max_tokens\":32,\"temperature\":0}"'
# 输出: "odeskodeskodesk..."
```

### 5.5 清场

```bash
ssh root@gpu-host 'docker exec afd-exp bash -c "kill -9 \$(pgrep -f \"[V]LLM::\") 2>/dev/null; pkill -9 -f \"[v]llm serve\" 2>/dev/null; sleep 3; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"'
```

---

## 6. 下一步方向

需要实现 `AFDGlmMoeDsaForCausalLM` 的 DSA 逻辑，使 indexer / eh_proj / enorm /
hnorm / shared_head 等权重被正确加载和参与计算。

**可能的路线**：

1. **检查 vLLM 0.19.1 是否有更新的 DSA 实现**
   - 上游 `GlmMoeDsaForCausalLM` 也是空壳 `pass`，但 attention backend
     `DEEPSEEK_V32_INDEXER` 存在，说明 DSA 支持可能在 0.19.1 之后才完善。
   - 检查 vLLM main 分支或更新版本是否有完整的 `GlmMoeDsaForCausalLM` 实现。

2. **在 AFD plugin 中实现 DSA model class**
   - 基于上游 `DeepseekV2ForCausalLM` 结构，增加 indexer 模块定义
   - 实现 `load_weights` 中对 indexer / eh_proj 等权重的加载
   - DSA attention 的 forward 逻辑：indexer → top-k 稀疏选择 → MLA attention

3. **评估是否需要自定义 attention backend 集成**
   - vLLM 已有 `DEEPSEEK_V32_INDEXER` backend，但可能需要与 model class 配合

---

## 7. 相关文件索引

| 文件 | 说明 |
|---|---|
| `experiment/W4AFP8_DESIGN.md` | W4AFP8 设计文档（含第 8 节实施结果） |
| `experiment/SESSION_STATUS.md` | 本文件 |
| `experiment/EXPERIMENT_ENV.md` | 实验环境说明 |
| `experiment/scripts/start_glm52_w4afp8_4a4f.sh` | 4A4F 启动脚本 |
| `afd_plugin/quantization/w4afp8.py` | W4AFP8 配置 + vLLM bug 修复 |
| `afd_plugin/__init__.py` | 注册入口 |
| `afd_plugin/model_executor/models/deepseek_v2.py:919` | AFDGlmMoeDsaForCausalLM 空壳类 |
