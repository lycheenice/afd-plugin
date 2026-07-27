# GLM-5.2-W4AFP8 AFD —— 状态与性能报告

> 日期：2026-07-27　机器：实验 gpu-host / 开发归档 archive-host
> 关联：详细调试轨迹见 `experiment/SESSION_STATUS.md`

---

## 0. 执行摘要

| 项 | 状态 |
|---|---|
| 模型能否 forward 跑通（起服务、生成 token） | **能**（无崩溃、无 NaN、生成到 max_tokens） |
| 输出精度（文本正确性） | **不正确**（乱码 token，根因见 §3） |
| 本次能否采集性能数据 | **暂不能** —— 实验容器 `afd-exp` 被删除，且 8×H200 已被他人 SGLang 工作负载占满（见 §1） |
| MoE 专家权重加载 bug | **已修复并严格验证**（见 §2） |

**结论**：性能数据采集在原理上可行（forward 完整执行，性能指标与输出内容无关），
但**当前实验环境不可用**，故本轮按"不能则详细记录"落实文档，并附**随时可跑的
基准脚本** `experiment/scripts/bench_glm52_w4afp8_4a4f.sh`，环境恢复后一条命令即可出数。

---

## 1. 实验环境现状（阻塞点）

2026-07-27 检查 gpu-host：

```
$ docker ps -a
sglang-glm-sglang-1   lmsysorg/sglang:v0.5.14-cu129   Up 36 minutes
sglang-glm-smg-1      lmsysorg/sglang:v0.5.14-cu129   Up 36 minutes
（afd-exp 容器已不存在）

$ nvidia-smi  (memory.used / 143GB)
GPU0 137702  GPU1 137744  GPU2 137000  GPU3 137000
GPU4 142742  GPU5 142742  GPU6 139232  GPU7 141512   ← 8 卡全满
```

- **`afd-exp` 容器已被删除**（过夜），`EXPERIMENT_ENV.md` 假设的"常驻容器 `docker restart` 即恢复"不再成立，需**重建容器**。
- **8×H200 已被他人的 `sglang-glm-*` 容器占满**（每卡 ~137–142GB）。4A4F 需要全部 8 卡，**当前无空间**，且不应干扰他人运行中的 SGLang。

**恢复实验的前置条件**（二选一，需协调）：
1. 等 `sglang-glm-*` 释放 GPU；或
2. 获得独占时段后重建 `afd-exp` 容器。

重建容器参考（镜像 `vllm/vllm-openai:v0.19.1`，挂载 `/data1`）：
```bash
docker run -d --name afd-exp --gpus all --ipc=host --shm-size=32g \
  -v /data1:/data1 -v /data1/models:/models -v /data1/afd-plugin:/workspace/afd-plugin \
  --entrypoint sleep vllm/vllm-openai:v0.19.1 infinity
docker exec afd-exp pip install -e /workspace/afd-plugin --no-deps --no-build-isolation
```

---

## 2. 本次代码改动（已修复：MoE 专家权重被静默跳过）

### 2.1 改动清单（本机 archive-host，未提交前）

| 文件 | 改动 | 行数 |
|---|---|---|
| `afd_plugin/quantization/w4afp8.py` | 新增 `remap_w4afp8_moe_checkpoint_weights()` 生成器 | +63 |
| `afd_plugin/model_executor/models/deepseek_v2.py` | `load_weights` 顶部按 quant 名 `w4afp8` 包裹 weights 流 | +18 |

### 2.2 修复内容

**问题**：GLM-5.2-W4AFP8 的 routed-expert 权重命名
`experts.N.{gate,up,down}_proj.weight`(int8) / `.weight_scale_inv`(bf16)，
与 vLLM `CompressedTensorsW4A8Fp8MoEMethod` 加载期注册的参数名
`w13/w2_weight_packed`(int32) / `w13/w2_weight_scale` 不匹配。
`make_expert_params_mapping` 靠字符串替换保留后缀，GLM 的 `.weight` 后缀
→ 生成 `w13_weight`（无 `_packed`）→ `params_dict` 无此键 →
AFD loader `if name_mapped not in params_dict: continue` **静默跳过全部 MoE 专家权重**。

**附加**：nibble 直方图（expert0）值 8 零出现 → SGLang 存**有符号二补码 int4**（对称 [-7,7]），
本机 SGLang 源码确认其 CUTLASS kernel 直接当有符号 int4；而 vLLM 的
`convert_packed_uint4b8_to_signed_int4_inplace` 会减 8（假设 uint4b8）。

**修复**：remap 生成器把 routed-expert 权重流：
- `.weight`(int8) → `.weight_packed`，先 `^0x88`（有符号→uint4b8，使 vLLM 减 8 后还原正确有符号值），再 `.view(torch.int32)`
- `.weight_scale_inv` → `.weight_scale`（仅重命名）
- Dense MLP / shared experts / attention / indexer 均为 FP8，路由已正确，不动

### 2.3 验证（6 角度，全部通过）

1. CPU round-trip：XOR 0x88 + int32 view + vLLM convert **精确还原**原始有符号 int4 值与顺序（无 XOR 则失败）。
2. `pack_rows` 源码：nibble i → bit 4i（LSB-first），与 int8→int32 小端 view **一致**。
3. CUTLASS/SGLang 均 low-first 约定，SGLang `process_weights` 不动权重（只 interleave scales），raw int8 直喂 kernel → checkpoint 即 low-first。
4. scale 量级探针：反量化 mean≈0、std≈0.008、absmax≈0.16，**合理**。
5. checkpoint dtype 权威核对：仅 routed-expert 是 int8/int4，其余 FP8/BF16。
6. 与 vLLM 自带测试 `test_cutlass_w4a8_moe.py` 的参考打包路径一致。

**效果**：修复后 MoE 专家权重正确加载（此前 100% 被跳过），输出从纯 `odesk` 重复
变化（说明专家参与计算），但仍乱码——因为存在**独立的第二个问题**（§3）。

---

## 3. 剩余问题（未解决）：模型层微妙数值错误

修复 MoE 后仍乱码。通过一系列分层实验（调试代码已清理）定位：

| 实验 | 结果 | 结论 |
|---|---|---|
| 单实例 TP=8（无 AFD 拆分） | 仍乱码 | bug 在**模型层**，非 AFD 连接器/拆分 |
| 禁用 DSA 用 dense MLA | 仍乱码 | 非 DSA 稀疏路径特有 |
| 逐层 hidden state norm | 无 NaN，残差正常增长，各位置有实质区分（cross_pos_std≈0.84） | attention **未坍塌** |
| logprobs | 非 NaN 但近乎均匀（top ~1%） | 能跑但产出无意义分布 |

**综合判断**：模型能跑、无 NaN、位置有区分，但输出**系统性错误**。这是需要
**参考实现对比**才能精确定位的微妙数值问题。

**最可能嫌疑**（GLM-5.2 特有、DeepSeek-V3 没有）：`head_dim=192`
（`qk_nope_head_dim=192`，DeepSeek-V3 为 128）。vLLM 0.19.1 的
`GlmMoeDsaForCausalLM` 只是 `DeepseekV2ForCausalLM` 的空壳子类，其 MLA / FlashMLA
kernel 可能对 GLM 的 192 head_dim 有未适配假设（日志：
`Padding num_heads 16→64 for BF16 sparse prefill kernel`）。模型 README 明确
"4-bit 与 DSA 路径为 SGLang 专有，未在 vLLM 验证"。

---

## 4. 性能数据采集方案（就绪，待环境恢复）

**脚本**：`experiment/scripts/bench_glm52_w4afp8_4a4f.sh`（本次新增）

- 基于 `vllm bench serve`（`--dataset-name random --ignore-eos`），忽略输出内容
- 扫描矩阵（in_len × out_len × concurrency）：
  `1024×128 @{1,8,16,32}`、`4096×256@8`、`512×512@16`、`256×1024@8`
- 采集：output throughput(tok/s)、request/s、TTFT p50、TPOT p50、e2e p50，写 summary.md + 每档 JSON

**执行（环境恢复后）**：
```bash
# 1) 起 4A4F
ssh root@gpu-host 'docker exec -d afd-exp bash /workspace/afd-plugin/experiment/scripts/start_glm52_w4afp8_4a4f.sh'
# 2) 等两侧 "Application startup complete"（~5min）
# 3) 跑基准
ssh root@gpu-host 'docker exec afd-exp bash /workspace/afd-plugin/experiment/scripts/bench_glm52_w4afp8_4a4f.sh'
# 4) 回拷结果
scp -r root@gpu-host:/data1/afd-plugin/experiment/results/perf_glm52_w4afp8_4a4f_* \
    /ceph/User/user/mycode/afd-plugin/experiment/results/
```

**重要口径**：所得性能数据是"**W4AFP8 4A4F 在乱码输出下**"的性能，仅反映
计算/调度/连接器开销，**不代表可用系统**（精度未过关）。适合用于：AFD 连接器开销、
W4A8 MoE kernel 吞吐、4A4F 拓扑显存/并发行为的工程评估；不适合作为对外性能结论。

---

## 5. 下一步建议（需决策）

| 优先级 | 选项 | 说明 |
|---|---|---|
| 高 | **(a) SGLang 参考对比** | 本机有 SGLang 源码；建 SGLang 跑同 checkpoint，dump 逐层/逐权重中间值与 vLLM diff，确定性定位数值分歧。工作量大但最可靠。 |
| 中 | **(b) 试更新版 vLLM** | 查是否有**真正实现** GLM-5.2 MLA（而非空壳继承 DeepseekV2）的版本；有则前移 AFD patch。 |
| 中 | **(c) 深挖 vLLM MLA/FP8** | 审计 `head_dim=192` / `q_lora_rank=2048` / FP8 `fused_qkv_a_proj` scale 处理。 |
| 并行 | **(d) 采集性能数据** | 环境恢复后按 §4 执行，出工程性能基线（明确标注精度未过关）。 |

**对"升级 vLLM vs 在当前版实现 DSA"的回答**：两者都不直接解决——阻塞点是
vLLM 对 GLM-5.2 的数值正确性（单实例即复现，与 AFD 无关），DSA 已在 vLLM 且已被
AFD 正确 wiring。核心是先确认 vLLM 层能否跑对 GLM-5.2-W4AFP8（选 a 或 b），再谈 AFD 集成与性能。

---

## 6. 附：文件索引

| 文件 | 说明 |
|---|---|
| `afd_plugin/quantization/w4afp8.py` | W4AFP8Config + MoE remap 修复 + vLLM bug 补丁 |
| `afd_plugin/model_executor/models/deepseek_v2.py:670+` | load_weights 中 w4afp8 remap 钩子 |
| `experiment/scripts/bench_glm52_w4afp8_4a4f.sh` | **本次新增**：4A4F 性能扫描脚本（就绪待跑） |
| `experiment/scripts/start_glm52_w4afp8_4a4f.sh` | 4A4F 启动脚本 |
| `experiment/SESSION_STATUS.md` | 完整调试状态与交接 |
| `experiment/logs/w4afp8_debug_evidence.txt` | 逐层 norm 调试证据存档 |
