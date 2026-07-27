# A 线结论:GLM-5.2 在 vLLM 0.19.1 上受阻于 MLA attention backend(2026-07-27)

> 无人值守夜跑记录。决定性证据 + 结论 + 转向"移植最新 vLLM"的依据。

## 结论(决定性)

**GLM-5.2(FP8 与 W4AFP8 同架构)在 vLLM 0.19.1 上无法正确运行,根因是缺少支持其 MLA 几何
(head_size=704)+ DSA sparse 的 attention backend——与量化(FP8/W4AFP8)、与 AFD 均无关。**

## 证据

### 1. A1:GLM-5.2-**FP8** 单实例 TP=8 native(无 AFD)→ 启动即崩
```
ValueError: No valid attention backend found for cuda with
  AttentionSelectorConfig(head_size=704, use_mla=True, use_sparse=True, ...)
  FLASH_ATTN_MLA / FLASHMLA / FLASHINFER_MLA / FLASHMLA_SPARSE: head_size not supported (+ sparse not supported)
  TRITON_MLA: sparse not supported
```
即 **head_size=704 + DSA sparse 没有任何可用 backend**。注意 TRITON_MLA 仅因 "sparse not supported"
被拒 → 关掉 DSA(dense MLA)时 TRITON_MLA 支持 704、可载入,但前会话实测 **dense 也乱码**。

### 2. 两模型 config 同架构(量化不是变量)
| 字段 | GLM-5.2-FP8 | GLM-5.2-W4AFP8 |
|---|---|---|
| architectures | GlmMoeDsaForCausalLM | GlmMoeDsaForCausalLM |
| model_type | glm_moe_dsa | deepseek_v3 |
| head_dim / qk_nope / qk_rope | 192 / 192 / 64 | 同 |
| kv_lora_rank / q_lora_rank / v_head_dim | 512 / 2048 / 256 | 同 |
| DSA: index_topk / index_head_dim / index_n_heads | 2048 / 128 / 32 | 同 |
| quant | fp8 (block 128) | w4afp8 |

head_size=704 来自 GLM 特有的 MLA 拼接(DeepSeek-V3 是 576),vLLM 0.19.1 的
`GlmMoeDsaForCausalLM` 是 `DeepseekV2ForCausalLM` 的空壳继承,其 backend 选择器不认 704。

### 3. 前会话(SESSION_STATUS §5)已排除的其它因素
- 单实例 TP=8(无 AFD)也乱码 → 非 AFD/连接器。
- 禁用 DSA 用 dense MLA 也乱码 → 非 DSA 稀疏路径特有。
- MoE 权重加载已修(W4AFP8 remap)、无 NaN、位置有区分 → 但输出系统性错误。

## 归因与判定
- **DSA sparse 路径**:head_size=704 无 backend → 根本载不进(A1 崩)。
- **dense MLA 路径**(TRITON_MLA,DSA off):能载入但**数值错误(乱码)** → vLLM 0.19.1 的
  GLM-5.2 dense MLA 实现对 704 几何 / q_lora=2048 / DSA indexer 有未适配假设。
- 两条路都不通 ⇒ **vLLM 0.19.1 对 GLM-5.2 支持不足**,非本插件/量化可修。

## 决定(按用户指令:A 反复无果 >3 则移植最新 vLLM)
累计证据(A1 崩 + 前会话多轮 dense/DSA 乱码 + config 佐证)已远超 3 次、且指向**上游架构支持缺口**。
→ 转向 **Track 2:移植 afd-plugin 到最新 vLLM**。**前置验证**:先确认最新 vLLM 是否
(a) 有 head_size=704 + DSA 的 MLA backend、(b) native 跑 GLM-5.2 能出正确输出。
- 若最新 vLLM native 正确 → 移植 afd-plugin 有意义,推进移植(设计文档 + 实施)。
- 若最新 vLLM 仍不支持 GLM-5.2 → 移植无益;记录并转 Track B(DeepSeek-V2-Lite AFD,已知可跑)产出结果。

## 复现
`experiment/scripts/a1_glm52_fp8_tp8_native.sh`(gpu-host 容器 afd-exp);日志
`experiment/logs/a1_glm52_fp8_tp8_native.log`。

---

## ★ 决定性验证(2026-07-27 夜):vLLM 0.25.0 修复了该缺口

gpu-host 上已有 `docker.1ms.run/vllm/vllm-openai:v0.25.0`(比 afd 目标 0.19.1 新)。用它跑
**native GLM-5.2-FP8 TP=8(无 AFD、无插件)**:

- **越过 attention backend 选择**(无 "No valid attention backend / head_size=704" 报错),权重正常加载 141 shards,`Application startup complete`。
- **greedy 输出正确连贯**:
  - `Hello, my name is` → ` [Your Name], and I am a [Your Profession/Role]. I have [Number] years of experience...`
  - `The capital of France is` → ` Paris. Distance from Paris to Lyon is 391 km...`(**Paris 正确**)

**结论(定案)**:多会话的"GLM-5.2 乱码"根因 = **vLLM 版本**。0.19.1 缺 GLM-5.2(head_size=704 + DSA)
的 MLA attention backend → 崩/乱码;**0.25.0 已实现,输出正确**。与 W4AFP8 量化、AFD 拆分、
afd-plugin 代码**均无关**(前会话的 W4AFP8 remap 修复本身是对的,只是被上游 backend 缺口掩盖)。

**行动**:按用户指令移植 afd-plugin 到最新 vLLM。**目标锁定 vLLM 0.25.0**(已验证 native 正确 +
已在 gpu-host)。W4AFP8 是插件自带量化,其在 0.25.0 上的正确性随移植一并验证(native FP8 已证架构 OK)。
复现:`docker run ... vllm-openai:v0.25.0 ... vllm serve /models/GLM-5.2-FP8 --tensor-parallel-size 8 ...`(容器 glm-v25)。
