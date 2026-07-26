#!/usr/bin/env python3
"""AFD 吞吐性能验证脚本

实验矩阵:
- P-01: AFD 1A1F eager vs 原生 vLLM (1 GPU) - 基准对比
- P-02: AFD 1A1F eager vs graph - CUDA Graph 加速
- P-03: AFD 1A1F w/o DBO vs w/ DBO - DBO 影响
- P-04: AFD 1A1F vs 2A2F - 拓扑对比
- P-05: 不同并发负载下的表现

使用 vllm bench serve 作为 benchmark 工具。
结果自动保存到 experiment/results/ 和 experiment/logs/
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MODEL_PATH = "/models/DeepSeek-V2-Lite"
MODEL_PATH_V25 = "/models/DeepSeek-V2.5"
LOG_DIR = Path("/workspace/afd-plugin/experiment/logs")
RESULT_DIR = Path("/workspace/afd-plugin/experiment/results")
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = {}

BENCH_PROMPTS = 128
BENCH_INPUT_LEN = 512
BENCH_OUTPUT_LEN = 128

def wait_for_api(port, timeout=300):
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False

def start_vllm(role, gpus, port, afd_port, mode="eager", num_attn=1, num_ffn=1,
               prefix="", dbo_decode_threshold=2, dbo_prefill_threshold=12,
               dp_size=1, tp_size=1, max_num_seqs=None, max_num_batched=None,
               model_path=None, model_name_prefix="deepseek-v2-lite",
               quantization=None, max_model_len=None, gpu_memory_utilization=None):
    if model_path is None:
        model_path = MODEL_PATH
    cmd = ["vllm", "serve", model_path,
        "--served-model-name", f"{model_name_prefix}-afd-{role}",
        "--data-parallel-size", str(dp_size), "--tensor-parallel-size", str(tp_size),
        "--enable-expert-parallel"]
    if max_num_seqs:
        cmd += ["--max-num-seqs", str(max_num_seqs)]
    if max_num_batched:
        cmd += ["--max-num-batched-tokens", str(max_num_batched)]
    if quantization:
        cmd += ["--quantization", quantization]
    if max_model_len:
        cmd += ["--max-model-len", str(max_model_len)]
    if gpu_memory_utilization:
        cmd += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
    if mode in ("eager", "dbo"):
        cmd += ["--enforce-eager"]
    elif mode in ("graph", "graph_dbo"):
        cmd += ["--max-num-seqs", "64", "--max-num-batched-tokens", "64",
                "--max-cudagraph-capture-size", "64", "--cudagraph-capture-sizes", "64",
                "--compilation-config", json.dumps({"cudagraph_mode": "FULL_DECODE_ONLY"})]
    if "dbo" in mode:
        cmd += ["--enable-dbo", "--dbo-decode-token-threshold", str(dbo_decode_threshold),
                "--dbo-prefill-token-threshold", str(dbo_prefill_threshold)]
    cmd += ["--trust-remote-code", "--host", "127.0.0.1", "--port", str(port)]
    afd_config = {"afd": {"role": role, "connector": "P2pNcclAFDConnector",
        "host": "127.0.0.1", "port": afd_port,
        "num_attention_ranks": num_attn, "num_ffn_ranks": num_ffn}}
    cmd += ["--additional-config", json.dumps(afd_config)]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["VLLM_PLUGINS"] = "afd"
    env["PYTHONPATH"] = "/workspace/afd-plugin"
    env["PYTHONUNBUFFERED"] = "1"

    log_file = str(LOG_DIR / f"{prefix}_{role}.log")
    with open(log_file, "w") as f:
        f.write(f"CMD: {' '.join(cmd)}\n\n")
        f.flush()
    log_f = open(log_file, "a")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)
    proc._log_file = log_f
    return proc

def start_native(gpus, port, mode="eager", prefix="native", dp_size=1,
                 model_path=None, model_name_prefix="deepseek-v2-lite",
                 tp_size=1, quantization=None, max_model_len=None,
                 gpu_memory_utilization=None):
    if model_path is None:
        model_path = MODEL_PATH
    cmd = ["vllm", "serve", model_path,
        "--served-model-name", f"{model_name_prefix}-native",
        "--data-parallel-size", str(dp_size), "--tensor-parallel-size", str(tp_size),
        "--enable-expert-parallel"]
    if quantization:
        cmd += ["--quantization", quantization]
    if max_model_len:
        cmd += ["--max-model-len", str(max_model_len)]
    if gpu_memory_utilization:
        cmd += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
    if mode == "eager":
        cmd += ["--enforce-eager"]
    elif mode == "graph":
        cmd += ["--max-num-seqs", "64", "--max-num-batched-tokens", "64",
                "--max-cudagraph-capture-size", "64", "--cudagraph-capture-sizes", "64",
                "--compilation-config", json.dumps({"cudagraph_mode": "FULL_DECODE_ONLY"})]
    cmd += ["--trust-remote-code", "--host", "127.0.0.1", "--port", str(port)]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("VLLM_PLUGINS", None)

    log_file = str(LOG_DIR / f"{prefix}.log")
    with open(log_file, "w") as f:
        f.write(f"CMD: {' '.join(cmd)}\n\n")
        f.flush()
    log_f = open(log_file, "a")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)
    proc._log_file = log_f
    return proc

def kill_procs(procs):
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                p.terminate()
    time.sleep(5)
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                p.kill()
    for p in procs:
        try:
            p.wait(timeout=5)
        except Exception:
            pass
    for p in procs:
        lf = getattr(p, "_log_file", None)
        if lf:
            lf.close()
    # Kill any surviving vLLM EngineCore subprocesses
    subprocess.run(["pkill", "-9", "-f", "vllm"], capture_output=True)
    time.sleep(5)

def run_bench(port, model_name, num_prompts, request_rate, max_concurrency,
              input_len, output_len, result_file, num_warmups=0,
              tokenizer_path=None):
    if tokenizer_path is None:
        tokenizer_path = MODEL_PATH
    cmd = ["vllm", "bench", "serve",
        "--host", "127.0.0.1", "--port", str(port),
        "--model", model_name,
        "--endpoint", "/v1/completions",
        "--dataset-name", "random",
        "--tokenizer", tokenizer_path,
        "--num-prompts", str(num_prompts),
        "--request-rate", str(request_rate),
        "--max-concurrency", str(max_concurrency),
        "--input-len", str(input_len),
        "--output-len", str(output_len),
        "--save-result",
        "--result-dir", str(RESULT_DIR),
        "--result-filename", result_file]
    if num_warmups > 0:
        cmd += ["--num-warmups", str(num_warmups)]

    bench_log = str(LOG_DIR / f"bench_{result_file}.log")
    with open(bench_log, "w") as f:
        f.write(f"Bench: {model_name} prompts={num_prompts} rate={request_rate} conc={max_concurrency}\n\n")
        f.flush()
    bench_f = open(bench_log, "a")
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    proc = subprocess.run(cmd, stdout=bench_f, stderr=subprocess.STDOUT, env=env, timeout=600)
    bench_f.close()

    result_path = RESULT_DIR / result_file
    if result_path.exists():
        with open(result_path) as f:
            return json.load(f)
    return None

def run_perf_test(test_name, fn):
    print(f"\n{'='*60}")
    print(f"  Running {test_name}")
    print(f"{'='*60}")
    try:
        result = fn()
        RESULTS[test_name] = result
        status = "PASS" if result.get("pass") else "FAIL"
        print(f"  {test_name}: {status}")
        if "summary" in result:
            for k, v in result["summary"].items():
                print(f"    {k}: {v}")
    except Exception as e:
        RESULTS[test_name] = {"pass": False, "error": str(e)}
        print(f"  {test_name}: FAIL - {e}")
        import traceback
        traceback.print_exc()

def bench_config(num_prompts=BENCH_PROMPTS, request_rate="inf", max_concurrency=32,
                 input_len=BENCH_INPUT_LEN, output_len=BENCH_OUTPUT_LEN,
                 num_warmups=0):
    return dict(num_prompts=num_prompts, request_rate=request_rate,
                max_concurrency=max_concurrency, input_len=input_len,
                output_len=output_len, num_warmups=num_warmups)

# ---- P-01: AFD 1A1F vs Native (1 GPU + DP2) ----
def test_p01_afd_vs_native():
    procs = []
    results = {}
    try:
        cfg = bench_config(num_warmups=32)
        # Native baseline (1 GPU)
        procs.append(start_native("0", 18000, "eager", "p01_native", dp_size=1))
        if not wait_for_api(18000, 300):
            return {"pass": False, "error": "Native API timeout"}
        native_result = run_bench(18000, "deepseek-v2-lite-native", **cfg,
                                   result_file="p01_native.json")
        kill_procs(procs)
        procs.clear()

        # Native DP2 baseline (2 GPU, data-parallel) — fair 2-GPU comparison
        procs.append(start_native("0,1", 18000, "eager", "p01_native_dp2",
                                   dp_size=2))
        if not wait_for_api(18000, 300):
            return {"pass": False, "error": "Native DP2 API timeout"}
        native_dp2_result = run_bench(18000, "deepseek-v2-lite-native", **cfg,
                                       result_file="p01_native_dp2.json")
        kill_procs(procs)
        procs.clear()

        # AFD 1A1F (2 GPUs)
        procs.append(start_vllm("ffn", "1", 18001, 6301, "eager", 1, 1, "p01_ffn"))
        procs.append(start_vllm("attention", "0", 18000, 6301, "eager", 1, 1, "p01_attn"))
        if not wait_for_api(18000, 300):
            return {"pass": False, "error": "AFD API timeout"}
        afd_result = run_bench(18000, "deepseek-v2-lite-afd-attention", **cfg,
                                result_file="p01_afd.json")

        summary = {}
        for label, r in [("native", native_result), ("native_dp2", native_dp2_result),
                         ("afd", afd_result)]:
            if r:
                for key in ["mean_ttft_ms", "mean_tpot_ms",
                            "total_token_throughput", "request_throughput"]:
                    val = r.get(key)
                    if val is not None:
                        summary[f"{label}_{key}"] = f"{val:.2f}"
        if native_result and afd_result:
            n = native_result.get("total_token_throughput") or 0
            a = afd_result.get("total_token_throughput") or 0
            if n:
                summary["afd_vs_native1_tput"] = f"{a/n:.2f}x"
        if native_dp2_result and afd_result:
            n2 = native_dp2_result.get("total_token_throughput") or 0
            a = afd_result.get("total_token_throughput") or 0
            if n2:
                summary["afd_vs_native_dp2_tput"] = f"{a/n2:.2f}x"

        return {"pass": True, "summary": summary,
                "native": native_result, "native_dp2": native_dp2_result,
                "afd": afd_result}
    finally:
        kill_procs(procs)

# ---- P-02: Graph vs Eager (AFD) ----
def test_p02_graph_vs_eager():
    procs = []
    results = {}
    try:
        # Eager
        procs.append(start_vllm("ffn", "1", 18001, 6302, "eager", 1, 1, "p02_eager_ffn"))
        procs.append(start_vllm("attention", "0", 18000, 6302, "eager", 1, 1, "p02_eager_attn"))
        if not wait_for_api(18000, 300):
            return {"pass": False, "error": "Eager API timeout"}
        cfg = bench_config(num_warmups=32)
        eager_result = run_bench(18000, "deepseek-v2-lite-afd-attention", **cfg,
                                  result_file="p02_eager.json")
        kill_procs(procs)
        procs.clear()

        # Graph — warmup triggers CUDA graph capture for all batch sizes before
        # the measured run, so TTFT is not dominated by on-demand capture.
        procs.append(start_vllm("ffn", "1", 18001, 6303, "graph", 1, 1, "p02_graph_ffn"))
        procs.append(start_vllm("attention", "0", 18000, 6303, "graph", 1, 1, "p02_graph_attn"))
        if not wait_for_api(18000, 600):
            return {"pass": False, "error": "Graph API timeout"}
        graph_result = run_bench(18000, "deepseek-v2-lite-afd-attention", **cfg,
                                  result_file="p02_graph.json")

        summary = {}
        if eager_result and graph_result:
            for key in ["mean_ttft_ms", "mean_tpot_ms", "total_token_throughput", "request_throughput"]:
                e_val = eager_result.get(key)
                g_val = graph_result.get(key)
                if e_val is not None and g_val is not None:
                    summary[f"eager_{key}"] = f"{e_val:.2f}"
                    summary[f"graph_{key}"] = f"{g_val:.2f}"
                    if "throughput" in key:
                        summary[f"graph_vs_eager_{key}"] = f"{g_val/e_val:.2f}x" if e_val else "N/A"
                    elif "ms" in key:
                        summary[f"graph_vs_eager_{key}"] = f"{g_val/e_val:.2f}x" if e_val else "N/A"

        return {"pass": True, "summary": summary,
                "eager": eager_result, "graph": graph_result}
    finally:
        kill_procs(procs)

# ---- P-03: DBO vs non-DBO (AFD) ----
def test_p03_dbo_vs_nodbo():
    procs = []
    try:
        # No DBO
        procs.append(start_vllm("ffn", "1", 18001, 6304, "eager", 1, 1, "p03_nodbo_ffn"))
        procs.append(start_vllm("attention", "0", 18000, 6304, "eager", 1, 1, "p03_nodbo_attn"))
        if not wait_for_api(18000, 300):
            return {"pass": False, "error": "No-DBO API timeout"}
        cfg = bench_config(num_warmups=32)
        nodbo_result = run_bench(18000, "deepseek-v2-lite-afd-attention", **cfg,
                                  result_file="p03_nodbo.json")
        kill_procs(procs)
        procs.clear()

        # With DBO
        procs.append(start_vllm("ffn", "1", 18001, 6305, "dbo", 1, 1, "p03_dbo_ffn"))
        procs.append(start_vllm("attention", "0", 18000, 6305, "dbo", 1, 1, "p03_dbo_attn"))
        if not wait_for_api(18000, 300):
            return {"pass": False, "error": "DBO API timeout"}
        dbo_result = run_bench(18000, "deepseek-v2-lite-afd-attention", **cfg,
                                result_file="p03_dbo.json")

        summary = {}
        if nodbo_result and dbo_result:
            for key in ["mean_ttft_ms", "mean_tpot_ms", "total_token_throughput", "request_throughput"]:
                n_val = nodbo_result.get(key)
                d_val = dbo_result.get(key)
                if n_val is not None and d_val is not None:
                    summary[f"nodbo_{key}"] = f"{n_val:.2f}"
                    summary[f"dbo_{key}"] = f"{d_val:.2f}"
                    if "throughput" in key:
                        summary[f"dbo_vs_nodbo_{key}"] = f"{d_val/n_val:.2f}x" if n_val else "N/A"
                    elif "ms" in key:
                        summary[f"dbo_vs_nodbo_{key}"] = f"{d_val/n_val:.2f}x" if n_val else "N/A"

        return {"pass": True, "summary": summary,
                "nodbo": nodbo_result, "dbo": dbo_result}
    finally:
        kill_procs(procs)

# ---- P-04: 1A1F vs 2A2F ----
def test_p04_topology():
    procs = []
    try:
        # 1A1F (eager, no DBO) — DBO is measured separately in P-03; keeping
        # both topologies DBO-free isolates the topology variable.
        procs.append(start_vllm("ffn", "1", 18001, 6306, "eager", 1, 1, "p04_1a1f_ffn"))
        procs.append(start_vllm("attention", "0", 18000, 6306, "eager", 1, 1, "p04_1a1f_attn"))
        if not wait_for_api(18000, 300):
            return {"pass": False, "error": "1A1F API timeout"}
        cfg = bench_config(num_warmups=32)
        r1a1f = run_bench(18000, "deepseek-v2-lite-afd-attention", **cfg,
                           result_file="p04_1a1f.json")
        kill_procs(procs)
        procs.clear()

        # 2A2F with DP=2 TP=1 (Attention: GPU 0,1 / FFN: GPU 2,3), eager, no DBO
        for role, gpus, port in [("ffn", "2,3", 18001), ("attention", "0,1", 18000)]:
            cmd = ["vllm", "serve", MODEL_PATH,
                "--served-model-name", f"deepseek-v2-lite-afd-{role}",
                "--data-parallel-size", "2", "--tensor-parallel-size", "1",
                "--enable-expert-parallel",
                "--max-num-seqs", "64", "--max-num-batched-tokens", "64",
                "--enforce-eager",
                "--trust-remote-code", "--host", "127.0.0.1", "--port", str(port)]
            afd_config = {"afd": {"role": role, "connector": "P2pNcclAFDConnector",
                "host": "127.0.0.1", "port": 6307,
                "num_attention_ranks": 2, "num_ffn_ranks": 2}}
            cmd += ["--additional-config", json.dumps(afd_config)]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpus
            env["VLLM_PLUGINS"] = "afd"
            env["PYTHONPATH"] = "/workspace/afd-plugin"
            env["PYTHONUNBUFFERED"] = "1"
            log_file = str(LOG_DIR / f"p04_2a2f_{role}.log")
            with open(log_file, "w") as f:
                f.write(f"CMD: {' '.join(cmd)}\n\n"); f.flush()
            log_f = open(log_file, "a")
            p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
            p._log_file = log_f
            procs.append(p)

        if not wait_for_api(18000, 300):
            return {"pass": False, "error": "2A2F API timeout"}
        r2a2f = run_bench(18000, "deepseek-v2-lite-afd-attention", **cfg,
                           result_file="p04_2a2f.json")

        summary = {}
        if r1a1f and r2a2f:
            for key in ["mean_ttft_ms", "mean_tpot_ms", "total_token_throughput", "request_throughput"]:
                v1 = r1a1f.get(key)
                v2 = r2a2f.get(key)
                if v1 is not None and v2 is not None:
                    summary[f"1a1f_{key}"] = f"{v1:.2f}"
                    summary[f"2a2f_{key}"] = f"{v2:.2f}"
            v1 = r1a1f.get("total_token_throughput") or 0
            v2 = r2a2f.get("total_token_throughput") or 0
            if v1:
                summary["2a2f_vs_1a1f_tput"] = f"{v2/v1:.2f}x"

        return {"pass": True, "summary": summary,
                "r1a1f": r1a1f, "r2a2f": r2a2f}
    finally:
        kill_procs(procs)

# ---- P-05: Different concurrency ----
def test_p05_concurrency():
    procs = []
    try:
        procs.append(start_vllm("ffn", "1", 18001, 6308, "eager", 1, 1, "p05_ffn"))
        procs.append(start_vllm("attention", "0", 18000, 6308, "eager", 1, 1, "p05_attn"))
        if not wait_for_api(18000, 300):
            return {"pass": False, "error": "API timeout"}

        concurrency_sweep = [1, 4, 16, 32, 64, 128]
        all_results = {}
        for conc in concurrency_sweep:
            cfg = bench_config(num_prompts=min(conc * 4, BENCH_PROMPTS),
                              request_rate="inf", max_concurrency=conc)
            r = run_bench(18000, "deepseek-v2-lite-afd-attention", **cfg,
                           result_file=f"p05_conc{conc}.json")
            if r:
                all_results[conc] = r
                print(f"  conc={conc}: ttft={r.get('mean_ttft_ms',0):.1f}ms "
                      f"tpot={r.get('mean_tpot_ms',0):.1f}ms "
                      f"tput={r.get('total_token_throughput',0):.1f}")

        summary = {}
        for conc, r in all_results.items():
            summary[f"conc{conc}_ttft"] = f"{r.get('mean_ttft_ms',0):.1f}ms"
            summary[f"conc{conc}_tput"] = f"{r.get('total_token_throughput',0):.1f}"

        return {"pass": len(all_results) > 0, "summary": summary,
                "all_results": all_results}
    finally:
        kill_procs(procs)

# ---- P-06: DBO threshold sweep (root-cause: does DBO help at larger batch?) ----
def test_p06_dbo_threshold():
    """DBO @ conc=32 halves throughput (P-03). Sweep the decode-token-threshold
    and concurrency to test H1/H2: whether raising the threshold (fewer DBO
    steps) or a larger batch (more compute to overlap) recovers DBO's overhead.
    """
    procs = []
    all_results = {}
    try:
        afd_port = 6309
        thresholds = [2, 8, 16, 32]
        concurrencies = [32, 64]
        for conc in concurrencies:
            for thr in thresholds:
                procs.append(start_vllm("ffn", "1", 18001, afd_port, "dbo", 1, 1,
                                        f"p06_t{thr}_c{conc}_ffn",
                                        dbo_decode_threshold=thr))
                procs.append(start_vllm("attention", "0", 18000, afd_port, "dbo", 1, 1,
                                        f"p06_t{thr}_c{conc}_attn",
                                        dbo_decode_threshold=thr))
                if not wait_for_api(18000, 300):
                    return {"pass": False, "error": f"DBO t{thr} c{conc} API timeout"}
                cfg = bench_config(num_warmups=32, max_concurrency=conc)
                r = run_bench(18000, "deepseek-v2-lite-afd-attention", **cfg,
                              result_file=f"p06_t{thr}_c{conc}.json")
                kill_procs(procs)
                procs.clear()
                if r:
                    key = f"t{thr}_c{conc}"
                    all_results[key] = r
                    print(f"  thr={thr} conc={conc}: "
                          f"tpot={r.get('mean_tpot_ms',0):.1f}ms "
                          f"tput={r.get('total_token_throughput',0):.1f}")

        summary = {}
        for key, r in all_results.items():
            summary[f"{key}_tpot"] = f"{r.get('mean_tpot_ms',0):.1f}ms"
            summary[f"{key}_tput"] = f"{r.get('total_token_throughput',0):.1f}"

        return {"pass": len(all_results) > 0, "summary": summary,
                "all_results": all_results}
    finally:
        kill_procs(procs)

# ---- P-08: 1A2F topology (1 attention GPU + 2 FFN GPUs, expert-parallel) ----
def test_p08_1a2f():
    """1A2F: 1 attention GPU (TP1) + 2 FFN GPUs (TP2, expert-parallel).
    Fan-out mode: attention broadcasts to both FFN ranks; FFN ranks shard
    experts via TP all-to-all; leader FFN sends combined result back.
    Tests whether adding FFN GPUs improves throughput for MoE models.
    Eager mode, no DBO.
    """
    procs = []
    try:
        afd_port = 6311
        # FFN: 2 GPUs, TP=2, expert-parallel
        ffn_cmd = ["vllm", "serve", MODEL_PATH,
            "--served-model-name", "deepseek-v2-lite-afd-ffn",
            "--data-parallel-size", "1", "--tensor-parallel-size", "2",
            "--enable-expert-parallel",
            "--max-num-seqs", "64", "--max-num-batched-tokens", "64",
            "--enforce-eager",
            "--trust-remote-code", "--host", "127.0.0.1", "--port", "18001"]
        ffn_cmd += ["--additional-config", json.dumps({
            "afd": {"role": "ffn", "connector": "P2pNcclAFDConnector",
                    "host": "127.0.0.1", "port": afd_port,
                    "num_attention_ranks": 1, "num_ffn_ranks": 2}})]
        ffn_env = os.environ.copy()
        ffn_env["CUDA_VISIBLE_DEVICES"] = "1,2"
        ffn_env["VLLM_PLUGINS"] = "afd"
        ffn_env["PYTHONPATH"] = "/workspace/afd-plugin"
        ffn_env["PYTHONUNBUFFERED"] = "1"
        ffn_log = open(str(LOG_DIR / "p08_1a2f_ffn.log"), "w")
        ffn_log.write(f"CMD: {' '.join(ffn_cmd)}\n\n"); ffn_log.flush()
        ffn_proc = subprocess.Popen(ffn_cmd, stdout=ffn_log,
            stderr=subprocess.STDOUT, env=ffn_env, start_new_session=True)
        ffn_proc._log_file = ffn_log
        procs.append(ffn_proc)

        # Attention: 1 GPU, TP=1
        attn_cmd = ["vllm", "serve", MODEL_PATH,
            "--served-model-name", "deepseek-v2-lite-afd-attention",
            "--data-parallel-size", "1", "--tensor-parallel-size", "1",
            "--enable-expert-parallel",
            "--max-num-seqs", "64", "--max-num-batched-tokens", "64",
            "--enforce-eager",
            "--trust-remote-code", "--host", "127.0.0.1", "--port", "18000"]
        attn_cmd += ["--additional-config", json.dumps({
            "afd": {"role": "attention", "connector": "P2pNcclAFDConnector",
                    "host": "127.0.0.1", "port": afd_port,
                    "num_attention_ranks": 1, "num_ffn_ranks": 2}})]
        attn_env = os.environ.copy()
        attn_env["CUDA_VISIBLE_DEVICES"] = "0"
        attn_env["VLLM_PLUGINS"] = "afd"
        attn_env["PYTHONPATH"] = "/workspace/afd-plugin"
        attn_env["PYTHONUNBUFFERED"] = "1"
        attn_log = open(str(LOG_DIR / "p08_1a2f_attn.log"), "w")
        attn_log.write(f"CMD: {' '.join(attn_cmd)}\n\n"); attn_log.flush()
        attn_proc = subprocess.Popen(attn_cmd, stdout=attn_log,
            stderr=subprocess.STDOUT, env=attn_env, start_new_session=True)
        attn_proc._log_file = attn_log
        procs.append(attn_proc)

        if not wait_for_api(18000, 360):
            return {"pass": False, "error": "1A2F API timeout"}

        summary = {}
        for conc in [32, 64]:
            cfg = bench_config(num_warmups=32, max_concurrency=conc)
            r = run_bench(18000, "deepseek-v2-lite-afd-attention", **cfg,
                          result_file=f"p08_1a2f_c{conc}.json")
            if r:
                summary[f"c{conc}_ttft"] = f"{r.get('mean_ttft_ms',0):.1f}ms"
                summary[f"c{conc}_tpot"] = f"{r.get('mean_tpot_ms',0):.1f}ms"
                summary[f"c{conc}_tput"] = f"{r.get('total_token_throughput',0):.1f}"
                print(f"  1A2F conc={conc}: ttft={r.get('mean_ttft_ms',0):.1f}ms "
                      f"tpot={r.get('mean_tpot_ms',0):.1f}ms "
                      f"tput={r.get('total_token_throughput',0):.1f}")

        return {"pass": len(summary) > 0, "summary": summary}
    finally:
        kill_procs(procs)

# ---- P-07: 4A4F topology (8 GPUs, DP2TP2 each side) ----
def test_p07_4a4f():
    """4A4F: 4 attention GPUs (DP2×TP2) + 4 FFN GPUs (DP2×TP2).
    Already supported by P2pNcclAFDConnector (attention >= ffn).
    Tests whether 8-GPU scale improves throughput vs 1A1F (2 GPUs).
    Eager mode, no DBO (Phase 2 confirmed DBO is always negative).
    """
    procs = []
    try:
        afd_port = 6310
        for role, gpus, port in [
            ("ffn", "4,5,6,7", 18001),
            ("attention", "0,1,2,3", 18000),
        ]:
            p = start_vllm(role, gpus, port, afd_port, "eager", 4, 4,
                           f"p07_4a4f_{role}", dp_size=2, tp_size=2,
                           max_num_seqs=64, max_num_batched=64)
            procs.append(p)

        if not wait_for_api(18000, 360):
            return {"pass": False, "error": "4A4F API timeout"}

        summary = {}
        for conc in [32, 64]:
            cfg = bench_config(num_warmups=32, max_concurrency=conc)
            r = run_bench(18000, "deepseek-v2-lite-afd-attention", **cfg,
                          result_file=f"p07_4a4f_c{conc}.json")
            if r:
                summary[f"c{conc}_ttft"] = f"{r.get('mean_ttft_ms',0):.1f}ms"
                summary[f"c{conc}_tpot"] = f"{r.get('mean_tpot_ms',0):.1f}ms"
                summary[f"c{conc}_tput"] = f"{r.get('total_token_throughput',0):.1f}"
                print(f"  4A4F conc={conc}: ttft={r.get('mean_ttft_ms',0):.1f}ms "
                      f"tpot={r.get('mean_tpot_ms',0):.1f}ms "
                      f"tput={r.get('total_token_throughput',0):.1f}")

        return {"pass": len(summary) > 0, "summary": summary}
    finally:
        kill_procs(procs)


# ---- P-09: V2.5 4A4F (8 GPUs, DP1×TP4 each side) — basic AFD ----
def test_p09_v25_4a4f_tp4():
    """V2.5 (236B FP8) basic AFD: 4A4F with DP=1, TP=4 per side.
    Equivalent to 1A1F scaled to TP=4 — no DP coordination overhead.
    Each GPU holds ~57.5GB weights, leaving ~74GB for KV cache.
    Native baseline: TP=8 on 8 GPUs (fair 8-GPU comparison).
    """
    procs = []
    results = {}
    try:
        v25 = dict(model_path=MODEL_PATH_V25, model_name_prefix="deepseek-v25",
                   quantization="fp8", max_model_len=4096,
                   gpu_memory_utilization=0.92)

        # Native TP=8 baseline (8 GPUs)
        procs.append(start_native("0,1,2,3,4,5,6,7", 18000, "eager", "p09_native_tp8",
                                   dp_size=1, tp_size=8, **v25))
        if not wait_for_api(18000, 600):
            return {"pass": False, "error": "Native TP8 API timeout"}
        cfg = bench_config(num_warmups=16, max_concurrency=32)
        native_result = run_bench(18000, "deepseek-v25-native", **cfg,
                                   result_file="p09_native_tp8.json",
                                   tokenizer_path=MODEL_PATH_V25)
        kill_procs(procs)
        procs.clear()

        # AFD 4A4F DP=1 TP=4 (8 GPUs)
        for role, gpus, port in [
            ("ffn", "4,5,6,7", 18001),
            ("attention", "0,1,2,3", 18000),
        ]:
            p = start_vllm(role, gpus, port, 6312, "eager", 4, 4,
                           f"p09_4a4f_{role}", dp_size=1, tp_size=4,
                           max_num_seqs=64, max_num_batched=64, **v25)
            procs.append(p)

        if not wait_for_api(18000, 600):
            return {"pass": False, "error": "V2.5 4A4F API timeout"}

        summary = {}
        for conc in [32, 64]:
            cfg = bench_config(num_warmups=16, max_concurrency=conc)
            r = run_bench(18000, "deepseek-v25-afd-attention", **cfg,
                          result_file=f"p09_v25_4a4f_c{conc}.json",
                          tokenizer_path=MODEL_PATH_V25)
            if r:
                summary[f"afd_c{conc}_ttft"] = f"{r.get('mean_ttft_ms',0):.1f}ms"
                summary[f"afd_c{conc}_tpot"] = f"{r.get('mean_tpot_ms',0):.1f}ms"
                summary[f"afd_c{conc}_tput"] = f"{r.get('total_token_throughput',0):.1f}"
                print(f"  V2.5 4A4F conc={conc}: ttft={r.get('mean_ttft_ms',0):.1f}ms "
                      f"tpot={r.get('mean_tpot_ms',0):.1f}ms "
                      f"tput={r.get('total_token_throughput',0):.1f}")

        if native_result:
            n = native_result.get("total_token_throughput") or 0
            a = r.get("total_token_throughput") if r else 0
            summary["native_c32_tput"] = f"{n:.1f}"
            summary["speedup_vs_native"] = f"{a/n:.2f}x" if n else "N/A"

        return {"pass": len(summary) > 0, "summary": summary}
    finally:
        kill_procs(procs)


# ---- P-10: V2.5 2A4F (6 GPUs, attn TP=2 + FFN TP=4) — fan-out ----
def test_p10_v25_2a4f():
    """V2.5 fan-out: 2A4F (attn TP=2, FFN TP=4).
    attention_size=2 < ffn_size=4 → fan-out mode, ratio=2.
    Each attention GPU: ~115GB weights (tight), FFN GPU: ~57.5GB.
    Tests whether expert-parallel scaling helps on large MoE (160 experts).
    """
    procs = []
    try:
        v25_attn = dict(model_path=MODEL_PATH_V25, model_name_prefix="deepseek-v25",
                        quantization="fp8", max_model_len=4096,
                        gpu_memory_utilization=0.95)
        v25_ffn = dict(model_path=MODEL_PATH_V25, model_name_prefix="deepseek-v25",
                       quantization="fp8", max_model_len=4096,
                       gpu_memory_utilization=0.92)

        # FFN: 4 GPUs, TP=4, expert-parallel
        ffn_cmd = ["vllm", "serve", MODEL_PATH_V25,
            "--served-model-name", "deepseek-v25-afd-ffn",
            "--data-parallel-size", "1", "--tensor-parallel-size", "4",
            "--enable-expert-parallel",
            "--max-num-seqs", "64", "--max-num-batched-tokens", "64",
            "--enforce-eager",
            "--quantization", "fp8", "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.92",
            "--trust-remote-code", "--host", "127.0.0.1", "--port", "18001"]
        ffn_cmd += ["--additional-config", json.dumps({
            "afd": {"role": "ffn", "connector": "P2pNcclAFDConnector",
                    "host": "127.0.0.1", "port": 6313,
                    "num_attention_ranks": 2, "num_ffn_ranks": 4}})]
        ffn_env = os.environ.copy()
        ffn_env["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
        ffn_env["VLLM_PLUGINS"] = "afd"
        ffn_env["PYTHONPATH"] = "/workspace/afd-plugin"
        ffn_env["PYTHONUNBUFFERED"] = "1"
        ffn_log = open(str(LOG_DIR / "p10_v25_2a4f_ffn.log"), "w")
        ffn_log.write(f"CMD: {' '.join(ffn_cmd)}\n\n"); ffn_log.flush()
        ffn_proc = subprocess.Popen(ffn_cmd, stdout=ffn_log,
            stderr=subprocess.STDOUT, env=ffn_env, start_new_session=True)
        ffn_proc._log_file = ffn_log
        procs.append(ffn_proc)

        # Attention: 2 GPUs, TP=2
        attn_cmd = ["vllm", "serve", MODEL_PATH_V25,
            "--served-model-name", "deepseek-v25-afd-attention",
            "--data-parallel-size", "1", "--tensor-parallel-size", "2",
            "--enable-expert-parallel",
            "--max-num-seqs", "64", "--max-num-batched-tokens", "64",
            "--enforce-eager",
            "--quantization", "fp8", "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.95",
            "--trust-remote-code", "--host", "127.0.0.1", "--port", "18000"]
        attn_cmd += ["--additional-config", json.dumps({
            "afd": {"role": "attention", "connector": "P2pNcclAFDConnector",
                    "host": "127.0.0.1", "port": 6313,
                    "num_attention_ranks": 2, "num_ffn_ranks": 4}})]
        attn_env = os.environ.copy()
        attn_env["CUDA_VISIBLE_DEVICES"] = "0,1"
        attn_env["VLLM_PLUGINS"] = "afd"
        attn_env["PYTHONPATH"] = "/workspace/afd-plugin"
        attn_env["PYTHONUNBUFFERED"] = "1"
        attn_log = open(str(LOG_DIR / "p10_v25_2a4f_attn.log"), "w")
        attn_log.write(f"CMD: {' '.join(attn_cmd)}\n\n"); attn_log.flush()
        attn_proc = subprocess.Popen(attn_cmd, stdout=attn_log,
            stderr=subprocess.STDOUT, env=attn_env, start_new_session=True)
        attn_proc._log_file = attn_log
        procs.append(attn_proc)

        if not wait_for_api(18000, 1200):
            return {"pass": False, "error": "V2.5 2A4F API timeout"}

        summary = {}
        for conc in [32, 64]:
            cfg = bench_config(num_warmups=16, max_concurrency=conc)
            r = run_bench(18000, "deepseek-v25-afd-attention", **cfg,
                          result_file=f"p10_v25_2a4f_c{conc}.json",
                          tokenizer_path=MODEL_PATH_V25)
            if r:
                summary[f"c{conc}_ttft"] = f"{r.get('mean_ttft_ms',0):.1f}ms"
                summary[f"c{conc}_tpot"] = f"{r.get('mean_tpot_ms',0):.1f}ms"
                summary[f"c{conc}_tput"] = f"{r.get('total_token_throughput',0):.1f}"
                print(f"  V2.5 2A4F conc={conc}: ttft={r.get('mean_ttft_ms',0):.1f}ms "
                      f"tpot={r.get('mean_tpot_ms',0):.1f}ms "
                      f"tput={r.get('total_token_throughput',0):.1f}")

        return {"pass": len(summary) > 0, "summary": summary}
    finally:
        kill_procs(procs)


# ---- P-11: V2.5 4A4F DP2×TP2 (8 GPUs) — DP scale ----
def test_p11_v25_4a4f_dp2tp2():
    """V2.5 DP scale: 4A4F with DP=2, TP=2 per side.
    Each GPU: ~115GB weights (tight, ~16GB for KV cache).
    Tests if DP2 helps throughput on V2.5 (unlike V2-Lite where DP coordination hurt).
    Compare against P-09 (4A4F TP=4, no DP) for same 8-GPU count.
    """
    procs = []
    try:
        v25 = dict(model_path=MODEL_PATH_V25, model_name_prefix="deepseek-v25",
                   quantization="fp8", max_model_len=4096,
                   gpu_memory_utilization=0.95)

        for role, gpus, port in [
            ("ffn", "4,5,6,7", 18001),
            ("attention", "0,1,2,3", 18000),
        ]:
            p = start_vllm(role, gpus, port, 6314, "eager", 4, 4,
                           f"p11_v25_4a4f_{role}", dp_size=2, tp_size=2,
                           max_num_seqs=64, max_num_batched=64, **v25)
            procs.append(p)

        if not wait_for_api(18000, 1200):
            return {"pass": False, "error": "V2.5 4A4F DP2TP2 API timeout"}

        summary = {}
        for conc in [32, 64]:
            cfg = bench_config(num_warmups=16, max_concurrency=conc)
            r = run_bench(18000, "deepseek-v25-afd-attention", **cfg,
                          result_file=f"p11_v25_4a4f_c{conc}.json",
                          tokenizer_path=MODEL_PATH_V25)
            if r:
                summary[f"c{conc}_ttft"] = f"{r.get('mean_ttft_ms',0):.1f}ms"
                summary[f"c{conc}_tpot"] = f"{r.get('mean_tpot_ms',0):.1f}ms"
                summary[f"c{conc}_tput"] = f"{r.get('total_token_throughput',0):.1f}"
                print(f"  V2.5 4A4F DP2TP2 conc={conc}: ttft={r.get('mean_ttft_ms',0):.1f}ms "
                      f"tpot={r.get('mean_tpot_ms',0):.1f}ms "
                      f"tput={r.get('total_token_throughput',0):.1f}")

        return {"pass": len(summary) > 0, "summary": summary}
    finally:
        kill_procs(procs)


if __name__ == "__main__":
    print("AFD 吞吐性能验证")
    print(f"Model: {MODEL_PATH}")
    print(f"Bench: {BENCH_PROMPTS} prompts, {BENCH_INPUT_LEN} in / {BENCH_OUTPUT_LEN} out")

    import sys
    only = sys.argv[1:] if len(sys.argv) > 1 else None

    if not only or "P-07" in only or "p07" in only:
        run_perf_test("P-07-Topology-4A4F", test_p07_4a4f)
    if not only or "P-08" in only or "p08" in only:
        run_perf_test("P-08-Topology-1A2F", test_p08_1a2f)
    if not only or "P-01" in only or "p01" in only:
        run_perf_test("P-01-AFD-vs-Native", test_p01_afd_vs_native)
    if not only or "P-02" in only or "p02" in only:
        run_perf_test("P-02-Graph-vs-Eager", test_p02_graph_vs_eager)
    if not only or "P-03" in only or "p03" in only:
        run_perf_test("P-03-DBO-vs-NoDBO", test_p03_dbo_vs_nodbo)
    if not only or "P-04" in only or "p04" in only:
        run_perf_test("P-04-Topology", test_p04_topology)
    if not only or "P-05" in only or "p05" in only:
        run_perf_test("P-05-Concurrency", test_p05_concurrency)
    if not only or "P-06" in only or "p06" in only:
        run_perf_test("P-06-DBO-Threshold", test_p06_dbo_threshold)
    if not only or "P-09" in only or "p09" in only:
        run_perf_test("P-09-V25-4A4F-TP4", test_p09_v25_4a4f_tp4)
    if not only or "P-10" in only or "p10" in only:
        run_perf_test("P-10-V25-2A4F-FanOut", test_p10_v25_2a4f)
    if not only or "P-11" in only or "p11" in only:
        run_perf_test("P-11-V25-4A4F-DP2TP2", test_p11_v25_4a4f_dp2tp2)

    print(f"\n{'='*60}")
    print("  PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    for name, result in RESULTS.items():
        status = "PASS" if result.get("pass") else "FAIL"
        print(f"\n  {name}: {status}")
        if "summary" in result:
            for k, v in result["summary"].items():
                print(f"    {k}: {v}")

    results_file = RESULT_DIR / "performance_results.json"
    with open(results_file, "w") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {results_file}")
