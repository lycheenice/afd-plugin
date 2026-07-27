# 明早速览 — 2026-07-27 夜(无人值守)

> 先读本文件。详情见各链接文档。所有改动已提交并同步 archive-host + github(branch `0724`)。

## 一句话结论(重大突破)

**多会话的 "GLM-5.2 乱码" 谜题定案:根因是 vLLM 版本,不是 W4AFP8/AFD/插件代码。**
vLLM **0.19.1** 缺 GLM-5.2(head_size=704 + DSA)的 MLA attention backend → 崩/乱码;
vLLM **0.25.0**(已在 gpu-host)**native 跑 GLM-5.2-FP8 完全正确**("The capital of France is" → "Paris")。

## 今晚做了什么

1. **A1 诊断**:GLM-5.2-FP8 单实例 TP=8 在 0.19.1 **启动即崩** ——
   `ValueError: No valid attention backend for head_size=704, use_sparse=True`。两模型(FP8/W4AFP8)同架构
   (GlmMoeDsaForCausalLM,head_dim=192,DSA),故与量化无关。详见
   [`ANALYSIS_glm52_vllm0191_blocker_20260727.md`](./ANALYSIS_glm52_vllm0191_blocker_20260727.md)。
2. **决定性验证**:同机的 vLLM **0.25.0** native 跑 GLM-5.2-FP8 **正确**。⇒ 按你的指令,转向移植。
3. **移植 afd-plugin → vLLM 0.25.0**(按你"A 失败则移植"指令,已实质推进,详见
   [`PORT_TO_VLLM_0.25.0.md`](./PORT_TO_VLLM_0.25.0.md)):
   - ✅ 插件在 0.25.0 `pip install -e` 成功;`register_afd` 鲁棒运行(版本仅警告);**w4afp8 量化注册成功**。
   - ✅ 修复 3 个移植故障(w4afp8.py,双向版本兼容):#1 MoE 层类 `FusedMoE`→`RoutedExperts`;
     #2 `CompressedTensorsW4A8Fp8MoEMethod` import 路径;#3 MoE 走上 W4A8 量化(不再 OOM)。
   - ⏳ **当前阻塞(#5)**:GLM-5.2-W4AFP8 native 权重加载 `KeyError: ...routed_experts.w2_weight` ——
     插件模型 wrapper(带 remap 钩子)未覆盖 0.25.0 原生模型类,remap 未生效。roadmap 见 PORT 文档。

## 现在的可用状态(morning 可直接用)

- **GLM-5.2-FP8 在 0.25.0 native 可正确运行**(无需插件)。复现:
  ```
  ssh root@gpu-host  # 容器 glm-v25 已被清;重跑:
  docker run -d --name glm-v25 --gpus all --network host --ipc=host --shm-size=32g \
    -v /data1/models:/models --entrypoint bash docker.1ms.run/vllm/vllm-openai:v0.25.0 -c \
    'vllm serve /models/GLM-5.2-FP8 --served-model-name glm-v25 --tensor-parallel-size 8 \
     --enable-expert-parallel --enforce-eager --max-model-len 8192 --gpu-memory-utilization 0.92 \
     --trust-remote-code --host 127.0.0.1 --port 18000'
  ```
- **移植 dev 容器 `afd-v25`**(0.25.0 + 插件 editable 安装)常驻,可直接续移植。

## 建议的下一步(优先级)

1. **完成 W4AFP8 移植**(PORT 文档 roadmap #1-#3):激活插件模型 wrapper(或 monkeypatch 注入 remap)→
   remap 适配 0.25.0 的 `routed_experts.w*_weight_packed` 命名 → 数值复核。目标:native GLM-5.2-W4AFP8 在 0.25.0 正确。
2. **AFD 拆分层移植**(compat/patches + v1/worker + connectors)——工作量最大,待 native 正确后做。
3. **决策**:是否把 afd-plugin 的目标 vLLM 从 0.19.1 升到 0.25.0(README/pyproject/compat 版本门)。

## 环境与运维

- gpu-host GPU:今晚为实验**停了 SGLang 生产**(你已切流量)。GPU 现空闲供续跑。
  **恢复生产**:`ssh root@gpu-host 'docker start sglang-glm-sglang-1 sglang-glm-smg-1'`(容器保留;
  存档 `/data1/pd-exp/sglang_restore/`)。若需继续移植实验则先别恢复。
- 同步:本夜所有脚本/文档/代码已 commit 到 `/ceph` branch `0724` 并 `git bundle` 推 archive-host → github。
- 工作流规则(三机分工 + 等停服务)已固化进 `EXPERIMENT_ENV.md` 与 afd 项目记忆 `afd-plugin-workflow.md`。
