#!/usr/bin/env python3
"""Minimal 1A1F eager smoke test: verify the ffn_worker shutdown fix didn't
break the basic AFD path. Starts ffn+attn, sends one completions request,
then a tiny bench (8 prompts) to confirm end-to-end and clean shutdown.
"""
import sys
sys.path.insert(0, "/workspace/afd-plugin/experiment/scripts")
from run_perf_tests import (  # noqa: E402
    start_vllm, wait_for_api, run_bench, kill_procs, bench_config,
)
import json  # noqa: E402
import urllib.request  # noqa: E402

PORT_ATTN, PORT_FFN, AFD_PORT = 18000, 18001, 6320
procs = []
try:
    procs.append(start_vllm("ffn", "1", PORT_FFN, AFD_PORT, "eager", 1, 1, "smoke_ffn"))
    procs.append(start_vllm("attention", "0", PORT_ATTN, AFD_PORT, "eager", 1, 1, "smoke_attn"))
    print("waiting api...", flush=True)
    ok = wait_for_api(PORT_ATTN, 300)
    print(f"api_ready={ok}", flush=True)
    if not ok:
        raise SystemExit("API timeout")
    payload = json.dumps({
        "model": "deepseek-v2-lite-afd-attention",
        "prompt": "The capital of France is",
        "max_tokens": 8, "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT_ATTN}/v1/completions",
        data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read().decode()
        print("resp:", body[:300], flush=True)
    # tiny bench to exercise the loop + clean shutdown path
    r = run_bench(PORT_ATTN, "deepseek-v2-lite-afd-attention",
                  num_prompts=8, request_rate="inf", max_concurrency=4,
                  input_len=128, output_len=16, result_file="smoke.json",
                  num_warmups=4)
    print("bench tpot:", r.get("mean_tpot_ms") if r else None, flush=True)
    print("SMOKE PASS", flush=True)
finally:
    kill_procs(procs)
    print("clean exit", flush=True)
