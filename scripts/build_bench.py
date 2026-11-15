"""Build the GPU benchmark kernel (kaggle/build/liars-bench). Push with --accelerator NvidiaTeslaT4."""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N = 96
CONFIGS = [
    {"model": "Qwen/Qwen3-8B", "tp": 2, "n": N},
    {"model": "Qwen/Qwen3-14B-AWQ", "tp": 2, "n": N, "extra": {"kv_cache_dtype": "fp8"}},
    {"model": "Qwen/Qwen3-30B-A3B-GPTQ-Int4", "tp": 2, "n": N},
    {"model": "Qwen/Qwen3-14B-AWQ", "tp": 2, "n": N},
    {"model": "Qwen/Qwen3-32B-AWQ", "tp": 2, "n": N, "timeout": 2700},
]
lib = (ROOT / "liars" / "rules.py").read_text() + "\n\n" + (ROOT / "liars" / "prompt.py").read_text()
worker = (ROOT / "kaggle" / "bench" / "worker.py").read_text()
main = (ROOT / "kaggle" / "bench" / "main.py").read_text()
src = (f"LIB_B64 = {base64.b64encode(lib.encode()).decode()!r}\n"
       f"WORKER_B64 = {base64.b64encode(worker.encode()).decode()!r}\n"
       f"CONFIGS = {json.dumps(CONFIGS)}\n\n" + main)
out = ROOT / "kaggle" / "build" / "liars-bench"
out.mkdir(parents=True, exist_ok=True)
(out / "kernel.py").write_text(src)
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": "gouravanirudh/liars-bench", "title": "liars-bench", "code_file": "kernel.py",
    "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
    "enable_tpu": "false", "enable_internet": "true", "dataset_sources": [],
    "competition_sources": ["the-liars-chatroom"], "kernel_sources": [], "model_sources": []}, indent=2))
print("built", out)
