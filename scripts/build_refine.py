"""Build the v2 refine kernel (kaggle/build/liars-refine). Usage: build_refine.py '<cfg json>'"""
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cfg = {"model": "Qwen/Qwen3-14B-AWQ", "tp": 2, "k": 3, "hard_tok": 2000, "max_tokens": 14000,
       "max_model_len": 16384}
if len(sys.argv) > 1:
    cfg.update(json.loads(sys.argv[1]))
lib = (ROOT / "liars" / "rules.py").read_text() + "\n\n" + (ROOT / "liars" / "prompt.py").read_text()
src = (f"LIB_B64 = {base64.b64encode(lib.encode()).decode()!r}\nCFG_JSON = {json.dumps(json.dumps(cfg))}\n\n"
       + (ROOT / "kaggle" / "refine" / "main.py").read_text())
out = ROOT / "kaggle" / "build" / "liars-refine"
out.mkdir(parents=True, exist_ok=True)
(out / "kernel.py").write_text(src)
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": "gouravanirudh/liars-refine", "title": "liars-refine", "code_file": "kernel.py", "language": "python",
    "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
    "enable_internet": "true", "dataset_sources": [], "competition_sources": ["the-liars-chatroom"],
    "kernel_sources": ["gouravanirudh/liars-chatroom-test", "gouravanirudh/liars-chatroom-eval"],
    "model_sources": []}, indent=2))
print("built", out, cfg)
