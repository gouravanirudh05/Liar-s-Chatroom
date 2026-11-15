"""Throughput/accuracy benchmark of several open-weight configs on Kaggle GPU T4 x2.

Each config runs in a fresh process (frees GPU memory), with a timeout. All models
are open-weight (Apache-2.0 Qwen3 family, <=70B). Train data only.
"""
import base64
import json
import shutil
import subprocess
import sys
import time

open("/kaggle/working/lib.py", "w").write(base64.b64decode(LIB_B64).decode())  # noqa: F821
open("/kaggle/working/worker.py", "w").write(base64.b64decode(WORKER_B64).decode())  # noqa: F821

t0 = time.time()
subprocess.run("nvidia-smi", shell=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)
print("vllm installed in", round(time.time() - t0), "s", flush=True)

for cfg in CONFIGS:  # noqa: F821
    print("=== CONFIG", json.dumps(cfg), flush=True)
    t1 = time.time()
    try:
        p = subprocess.run([sys.executable, "/kaggle/working/worker.py", json.dumps(cfg)],
                           capture_output=True, text=True, timeout=cfg.get("timeout", 1800))
        res = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if res:
            print(res[-1], flush=True)
        else:
            print("FAILED rc", p.returncode, "\n", (p.stdout + p.stderr)[-3000:], flush=True)
    except subprocess.TimeoutExpired:
        print("TIMEOUT after", round(time.time() - t1), "s", flush=True)
    shutil.rmtree("/root/.cache/huggingface", ignore_errors=True)
print("bench done in", round(time.time() - t0), "s")
