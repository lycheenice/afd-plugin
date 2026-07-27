# 今晚实验计划(2026-07-27)—— Track A:W4AFP8 / GLM-5.2 正确性

> 机器 gpu-host(8×H200,容器 afd-exp)。**GPU 归生产,等用户明确"已停服务"再开始。**
> 同步/分工规则见 `EXPERIMENT_ENV.md`「三机分工与同步」。接续状态见 `SESSION_STATUS.md`。

## 目标
让 GLM-5.2 在 vLLM 0.19.1 上产出**正确输出**(当前:W4AFP8 4A4F 与单实例 TP=8 均乱码,
已排除 AFD/DSA/MoE-加载,疑在模型层数值,GLM head_dim=192)。今晚先做决定性二分,再定深入方向。

## A1(决定性二分,~15–25min):GLM-5.2-**FP8** 单实例 TP=8 native
脚本:`experiment/scripts/a1_glm52_fp8_tp8_native.sh`(无 AFD、无 `--quantization`,让 vLLM 自动识别 FP8)。
box 上有 `/models/GLM-5.2-FP8`(标准 FP8,非 int4 专家)。greedy 冒烟三个 prompt。

**判读**:
- **FP8 输出连贯正确** → 乱码**专属 W4AFP8 int4 专家路径** → 进 A2-W4AFP8:逐值核对
  `remap_w4afp8_moe_checkpoint_weights` 的 `^0x88`/int32 view/scale 重排,与 vLLM
  `CompressedTensorsW4A8Fp8MoEMethod` 期望布局逐字段对齐;必要时 dump 反量化后专家权重与参考比对。
- **FP8 也乱码** → 问题在 **vLLM 0.19.1 对 GLM-5.2 架构支持**(与 W4AFP8 无关)→ 进 A2-ARCH:
  ① 查更高版本 vLLM 是否真正实现 GLM-5.2 MLA(而非空壳继承 DeepseekV2),评估升级 + AFD patch 前移;
  ② 审计 `head_dim=192` / `q_lora_rank` / FP8 fused_qkv_a_proj scale / DSA indexer 的 kernel 假设
  (日志 `Padding num_heads 16→64 for BF16 sparse prefill kernel`)。

## A2:按 A1 分流深入(见上)。目标产出:定位到具体分歧点 + 修复或明确的升级/审计结论。

## 备选 Track B(若 A 卡住):DeepSeek-V2-Lite AFD 性能/稳定性(PLAN_PHASE2)
2A2F 稳定性修复 + 公平基线 native DP2。已知可跑,产出干净数据。

## 执行前置(用户停服务后)
1. 确认 GPU 空闲:`ssh root@gpu-host nvidia-smi`。
2. 起容器(afd-exp 当前不在):`docker start afd-exp` 或按 `scripts/bootstrap_env.sh` `docker run`
   (镜像 vllm/vllm-openai:v0.19.1,挂 /data1/models→/models、/data1/afd-plugin→/workspace/afd-plugin)。
   A1 是 native、**不需要** afd 插件;但用 afd-exp 容器方便(已挂模型)。
3. `docker exec afd-exp bash /workspace/afd-plugin/experiment/scripts/a1_glm52_fp8_tp8_native.sh`。
4. 清场:`docker exec afd-exp pkill -9 -f "[v]llm serve"`。

## 结果归档
A1 结论 + 输出样例记入本文件 + `SESSION_STATUS.md`;同步 archive-host(bundle)。
