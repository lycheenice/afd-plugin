#!/usr/bin/env python3
"""Quick DBO test - standalone"""
import json
import os
import signal
import subprocess
import time
import urllib.request

MODEL = "/models/DeepSeek-V2-Lite"
AFD_PORT = 6252

def start_proc(role, gpus, port):
    afd_config = {"afd": {"role": role, "connector": "P2pNcclAFDConnector",
        "host": "127.0.0.1", "port": AFD_PORT,
        "num_attention_ranks": 1, "num_ffn_ranks": 1}}
    cmd = ["vllm", "serve", MODEL,
        "--served-model-name", f"deepseek-v2-lite-afd-{role}",
        "--data-parallel-size", "1", "--tensor-parallel-size", "1",
        "--enable-expert-parallel", "--enforce-eager",
        "--enable-dbo", "--dbo-decode-token-threshold", "2",
        "--dbo-prefill-token-threshold", "12",
        "--trust-remote-code", "--host", "127.0.0.1", "--port", str(port),
        "--additional-config", json.dumps(afd_config)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["VLLM_PLUGINS"] = "afd"
    env["PYTHONPATH"] = "/workspace/afd-plugin"
    log_f = open(f"/tmp/dbo_{role}.log", "w")
    p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                         env=env, start_new_session=True)
    p._log_file = log_f
    return p

procs = []
try:
    p1 = start_proc("ffn", "1", 18001)
    procs.append(p1)
    time.sleep(3)
    p2 = start_proc("attention", "0", 18000)
    procs.append(p2)

    print("Waiting for API...")
    for i in range(60):
        if p1.poll() is not None:
            print(f"FFN died! rc={p1.returncode}")
            print(open("/tmp/dbo_ffn.log").read()[-500:])
            break
        if p2.poll() is not None:
            print(f"ATTN died! rc={p2.returncode}")
            print(open("/tmp/dbo_attn.log").read()[-500:])
            break
        try:
            with urllib.request.urlopen("http://127.0.0.1:18000/v1/models", timeout=5) as r:
                if r.status == 200:
                    print(f"API ready after {i*5}s")
                    break
        except Exception:
            pass
        time.sleep(5)

    payload = json.dumps({"model": "deepseek-v2-lite-afd-attention",
        "prompt": "Machine learning is", "max_tokens": 32, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:18000/v1/completions",
        data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read())
        text = resp["choices"][0]["text"]
        print(f"Response: {text[:80]}")
        print(f"DBO: PASS")
except Exception as e:
    print(f"DBO: FAIL - {e}")
    import traceback
    traceback.print_exc()
finally:
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                p.kill()
    subprocess.run(["pkill", "-9", "-f", "vllm serve"], capture_output=True)
