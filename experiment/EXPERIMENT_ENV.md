# 实验环境备忘 (gpu-host)

> 本文件记录 AFD 插件性能实验的执行环境。任何新会话在跑实验前先读本文件,
> 不要把开发本机误当成实验机。

## 实验机器 (不是本机!)

| 项 | 值 |
|---|---|
| 角色 | 实验执行机 (跑 vLLM/AFD 性能测试) |
| 别名 | `gpu-host` (见开发机 `~/.ssh/config`) |
| 地址 | `REDACTED_IP` |
| 登录 | **`ssh root@gpu-host` 免密** (用 root,不要用 lychee) |
| 系统 | Ubuntu 22.04.5 |
| GPU | 8× NVIDIA H200 143GB |
| 磁盘 | `/data1` 3.5TB (代码/模型/容器卷) |

**关键**:开发本机是 `lychee@REDACTED_IP`(无 sudo、无 docker、GPU 被 sglang 占用),
**不能**直接在本机跑实验。所有 vLLM/AFD 实测都在 `gpu-host` 上以 `root` 通过 SSH 进行。

## 容器与挂载

| 项 | 值 |
|---|---|
| 容器名 | `afd-exp` (常驻, `docker restart` 即可恢复) |
| 镜像 | `vllm/vllm-openai:v0.19.1` |
| 启动方式 | `docker exec afd-exp ...` (容器以 `sleep infinity` 常驻) |
| 模型挂载 | host `/data1/models` → 容器 `/models` |
| 模型路径 | 容器内 `/models/DeepSeek-V2-Lite` |
| 代码挂载 | host `/data1/afd-plugin` → 容器 `/workspace/afd-plugin` (bind mount) |
| afd 安装 | 容器内 `pip install -e /workspace/afd-plugin --no-deps --no-build-isolation` |
| python | 容器内用 `python3` (`python` 不存在);vLLM CLI 入口 `vllm` 已就绪 |
| 环境变量 | 跑 AFD 时必带 `-e PYTHONPATH=/workspace/afd-plugin -e VLLM_PLUGINS=afd` |

## 三机分工与同步(2026-07-27 定,长期规则)

> 铁律:**gpu-host 的 GPU 归生产,跑实验前必须等用户明确"已停服务"再开始**,不要擅自停生产。

| 机器 | 角色 | 说明 |
|---|---|---|
| **dev-host**(本机,`dev-host`,有 `/ceph`) | 权威编辑 | `/ceph/User/user/mycode/afd-plugin`,git branch `0724`。**不能 push github**(root key 未注册)。 |
| **archive-host** | git 归档 + 推 github | `/home/lychee/mycode/afd-plugin`,origin SSH,`sudo -u lychee git push`。**用户指定的同步目标**:脚本/设计文档/代码/报告都同步到这里。无 rsync。 |
| **gpu-host** | 实验执行机 | 见上表;容器 `afd-exp`,host `/data1/afd-plugin`→`/workspace/afd-plugin`。 |

### 同步流(已验证)

```bash
# 1) dev-host /ceph 编辑 + 提交
cd /ceph/User/user/mycode/afd-plugin && git add -A && git commit -m "..."   # branch 0724

# 2) → archive-host(dev-host 不能 push,用 git bundle 送过去由 archive-host 推 github)
H6HEAD=$(ssh root@archive-host 'cd /home/lychee/mycode/afd-plugin && git rev-parse 0724')
git bundle create /tmp/afd.bundle ${H6HEAD}..0724
scp /tmp/afd.bundle root@archive-host:/tmp/afd.bundle
ssh root@archive-host 'cd /home/lychee/mycode/afd-plugin && \
  git fetch /tmp/afd.bundle 0724:refs/remotes/bundle/0724 && \
  sudo -u lychee git merge --ff-only refs/remotes/bundle/0724 && \
  sudo -u lychee git push origin 0724'

# 3) → gpu-host 实验机(scp 改动的子目录;editable 安装,重启服务即生效,无需重装)
scp -r afd_plugin/<改动路径> experiment/scripts/<改动脚本> root@gpu-host:/data1/afd-plugin/<对应路径>
```

> 历史备注:早期文档误把 archive-host 当"本机"并用 rsync;实际编辑在 dev-host `/ceph`,archive-host 无 rsync,统一用 `scp`/`git bundle`。

## 跑实验

```bash
ssh root@gpu-host 'docker exec \
  -e PYTHONPATH=/workspace/afd-plugin -e VLLM_LOGGING_LEVEL=INFO \
  afd-exp python3 /workspace/afd-plugin/experiment/scripts/run_perf_tests.py'
```

清场:
```bash
ssh root@gpu-host 'docker exec afd-exp bash -c "pkill -9 -f vllm 2>/dev/null; sleep 3; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"'
```

## 结果与日志位置 (均在容器内,即 host /data1/afd-plugin/)

- 报告: `experiment/EXPERIMENT_REPORT.md`
- 性能 JSON: `experiment/results/performance_results.json`
- 单项: `experiment/results/p01_native.json` … `p06_*.json`
- 日志: `experiment/logs/` (p0x_*_attn/ffn/bench_*.log)

## 实验阶段规划

- 第一阶段(已完成): `experiment/EXPERIMENT_REPORT.md`
- 第二阶段(进行中): `experiment/PLAN_PHASE2.md`
- 环境引导脚本: `experiment/scripts/bootstrap_env.sh`
