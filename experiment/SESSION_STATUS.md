# W4AFP8 GLM-5.2 AFD 冒烟测试 — 会话状态

> 本文档记录调试进展，供新会话直接接续。
> 最后更新：2026-07-26（第二会话，纠正了上一会话的错误诊断）

---

## 1. 最终目标

在 AFD（Attention-FFN Disaggregation）4A4F 拓扑下运行 **GLM-5.2-W4AFP8**，
在 8×H200（每卡 143GB）上产出**正确的输出文本**（而非乱码 token）。

---

## 2. 实验环境

| 项 | 值 |
|---|---|
| 实验机器 | `ssh root@gpu-host`（免密，REDACTED_IP） |
| 开发/归档机 | 本机 archive-host（`/ceph/User/user/mycode/afd-plugin`，git 主仓库）。**无 rsync，用 `scp`**；editable 安装，重启即生效 |
| 容器 | `afd-exp`（镜像 `vllm/vllm-openai:v0.19.1`，`/data1`→容器 `/workspace`） |
| 模型 | 容器内 `/models/GLM-5.2-W4AFP8`（host `/data1/models/`），41 shards |
| GPU | 8×H200 143GB，4A4F = GPU0-3 Attn / GPU4-7 FFN |

工作流：本机改代码 → `scp` 到 gpu-host → 容器内重启服务 → 实验数据回拷本机。

---

## 3. 重大修正：上一会话根因诊断是错的

上一会话（§旧 4.2）称 vLLM 0.19.1 的 `GlmMoeDsaForCausalLM` 是空壳、DSA 未实现、
indexer 权重被跳过。**这是错的**，已逐行核实：

- vLLM 0.19.1 `deepseek_v2.py` **有完整 DSA**：`Indexer` 类、`SparseAttnIndexer`、
  `DeepseekV32IndexerBackend`、`is_v32` 处理，`GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)` 全部继承。
- AFD 的 `AFDDeepseekV2Model.__init__` **已正确创建 `topk_indices_buffer` 并穿线**到
  decoder layer → MLA attention → Indexer（与原生逐行一致）。
- indexer 权重（wq_b/wk FP8、weights_proj/k_norm BF16）本就会被通用 loader 正确加载。

**DSA 不是问题。**

---

## 4. 已定位并修复的真正主因：MoE routed-expert 权重被静默跳过

### 4.1 根因

GLM-5.2-W4AFP8 的 routed-expert 权重命名为
`experts.N.{gate,up,down}_proj.weight`（int8）/ `.weight_scale_inv`（bf16），
而 vLLM 的 `CompressedTensorsW4A8Fp8MoEMethod`（`compressed_tensors_moe.py:2234`）
在加载期注册的参数名是 `w13/w2_weight_packed`（int32）/ `w13/w2_weight_scale`。

`make_expert_params_mapping` 通过**字符串替换保留后缀**构造目标名：真正的
compressed-tensors checkpoint 后缀是 `.weight_packed`/`.weight_scale`（直接匹配），
而 GLM 用 `.weight`/`.weight_scale_inv` → 生成 `w13_weight`（无 `_packed`）→
`params_dict` 无此键 → AFD loader `if name_mapped not in params_dict: continue`
**静默跳过全部 MoE expert 权重** → MoE 输出垃圾。

### 4.2 附加根因：有符号 int4 vs uint4b8

nibble 直方图（expert0 gate_proj）：值 8 **零出现**，峰在 0/15 →
SGLang 存的是**有符号二补码 int4**（对称 [-7,7]，-8 不用），
本机 SGLang 源码 `.../quantization/w4afp8.py` 的 `process_weights_after_loading`
**不对权重减 8**（只 interleave scales），其 CUTLASS kernel 直接当有符号 int4。
而 vLLM 的 `convert_packed_uint4b8_to_signed_int4_inplace` 会**减 8**（假设 uint4b8）。

### 4.3 修复（已提交到本机，已同步 gpu-host）

| 文件 | 改动 |
|---|---|
| `afd_plugin/quantization/w4afp8.py` | 新增 `remap_w4afp8_moe_checkpoint_weights()`：对 routed-expert 权重流 `.weight`(int8)→`.weight_packed`、并 `^0x88`（有符号→uint4b8，使 vLLM 减 8 后还原）、`.view(int32)`；`.weight_scale_inv`→`.weight_scale` |
| `afd_plugin/model_executor/models/deepseek_v2.py` | `load_weights` 顶部按 `quant_config.get_name()=="w4afp8"` 包裹 weights 流 |

**MoE 修复已从 6 个角度严格验证正确**：CPU round-trip 精确还原有符号 int4 值和顺序；
`pack_rows` 确认 nibble i→bit 4i（LSB-first，与 int8→int32 小端 view 一致）；
CUTLASS/SGLang 均 low-first；scale 量级探针 sane；与 vLLM 测试参考路径一致。

---

## 5. 剩余阻塞：模型层的微妙数值问题（非 AFD、非 DSA、非 MoE）

修复 MoE 后输出**仍是乱码**（如 `odesk...`）。逐层调试（临时，已清理）结论：

- **单实例 TP=8（无 AFD 拆分）也乱码** → bug 在**模型层**，非 AFD 连接器/拆分。
- **禁用 DSA 用 dense MLA 也乱码** → 非 DSA 稀疏路径特有。
- 逐层 norm：**无 NaN**，残差正常增长（embed 7.66→layer2 13.0），真实 forward 中
  各位置 hidden state **有实质区分**（cross_pos_std≈0.84） → attention **未坍塌**。
- logprobs **非 NaN 但近乎均匀**（top token 仅 ~1%）→ 模型能跑但产出无意义分布。

**综合**：模型能跑、无 NaN、位置有区分，但输出系统性错误。这是一个需要**参考实现
对比**才能精确定位的微妙数值问题。

**最可能的嫌疑（GLM-5.2 特有、DeepSeek-V3 没有的）**：
`head_dim=192`（`qk_nope_head_dim=192`，DeepSeek-V3 是 128）。vLLM 的
`GlmMoeDsaForCausalLM` 只是 `DeepseekV2ForCausalLM` 的空壳子类，其 MLA/FlashMLA
kernel 可能对 GLM 的 192 head_dim 有未适配的假设（日志有
`Padding num_heads 16→64 for BF16 sparse prefill kernel`）。模型 README 明确
"4-bit 布局与 DSA 路径是 SGLang 专有，未在 vLLM 上验证过"。

---

## 6. 下一步选项（需你决策）

| 选项 | 说明 | 评估 |
|---|---|---|
| (a) 构建 SGLang 参考对比 | 本机有 SGLang 源码 `/ceph/.../sglang`。建 SGLang、跑同一 checkpoint、dump 逐层/逐权重中间值，与 vLLM diff，精确定位分歧点 | 最可靠，但工作量大（需编译 SGLang + GPU） |
| (b) 升级 vLLM | 查更新版 vLLM 是否有**真正实现** GLM-5.2 MLA（而非空壳继承 DeepseekV2），再把 AFD 的 patch 前移 | 若新版有正解则最省事；AFD patch 前移有成本 |
| (c) 深挖 vLLM MLA/FP8 attention | 审计 vLLM 0.19.1 对 GLM `head_dim=192`/`q_lora_rank=2048`/FP8 fused_qkv_a_proj scale 的处理 | 定向但可能耗时 |

**对"升级 vLLM vs 实现 DSA"的回答**：两者都不直接解决——阻塞点是 vLLM 对 GLM-5.2
的**数值正确性**（与 AFD 无关，单实例也复现）。核心问题是 vLLM 能否正确跑
GLM-5.2-W4AFP8。建议先做 (a) 或 (b) 确认 vLLM 层能否跑对，再谈 AFD 集成。

---

## 7. 复现

```bash
# 同步（本机→gpu-host）
scp afd_plugin/quantization/w4afp8.py afd_plugin/model_executor/models/deepseek_v2.py \
    root@gpu-host:/data1/afd-plugin/afd_plugin/...   # 按路径分别 scp

# 启动 4A4F（真实目标配置）
ssh root@gpu-host 'docker exec -d afd-exp bash /workspace/afd-plugin/experiment/scripts/start_glm52_w4afp8_4a4f.sh'
# 等 ~5min，两侧 "Application startup complete"

# 发请求（当前仍乱码，MoE 已修，剩 attention/数值问题）
ssh root@gpu-host 'docker exec afd-exp curl -s http://127.0.0.1:18000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"glm52-afd-attn\",\"prompt\":\"Hello, my name is\",\"max_tokens\":32,\"temperature\":0}"'

# 清场
ssh root@gpu-host 'docker exec afd-exp bash -c "pkill -9 -f \"[v]llm serve\"; sleep 3; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"'
```

---

## 8. 相关文件

| 文件 | 说明 |
|---|---|
| `afd_plugin/quantization/w4afp8.py` | W4AFP8Config + **remap 修复** + vLLM bug 补丁 |
| `afd_plugin/model_executor/models/deepseek_v2.py:670+` | `load_weights` 中的 w4afp8 remap 钩子 |
| `experiment/W4AFP8_DESIGN.md` | 设计文档（§3.1 dense-MLP 误标为 int4，实际是 FP8；只有 routed-expert 是 int4） |
| `experiment/scripts/start_glm52_w4afp8_4a4f.sh` | 4A4F 启动脚本 |
| 本机记忆 | `~/.claude/projects/-ceph-.../memory/w4afp8-glm52-afd-debug.md`（完整调试轨迹） |
