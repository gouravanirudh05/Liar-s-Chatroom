"""Bundle liars/rules.py + kaggle/llm_baseline/infer.py into one Kaggle script kernel.

Usage: python scripts/build_kernel.py [--mode eval|test] [--n-eval N] [--model ID]
           [--max-tokens N] [--n-samples N] [--slug SLUG]
Writes kaggle/build/<slug>/{kernel.py,kernel-metadata.json}; push with
    kaggle kernels push -p kaggle/build/<slug> --accelerator NvidiaTeslaT4
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="eval", choices=["eval", "test"])
ap.add_argument("--n-eval", type=int, default=300)
ap.add_argument("--model", default="Qwen/Qwen3-14B-AWQ")
ap.add_argument("--max-tokens", type=int, default=6000)
ap.add_argument("--n-samples", type=int, default=1)
ap.add_argument("--slug", default=None)
a = ap.parse_args()
slug = a.slug or f"liars-chatroom-{a.mode}"

rules = (ROOT / "liars" / "rules.py").read_text()
infer = (ROOT / "kaggle" / "llm_baseline" / "infer.py").read_text()
header = (
    "import os\n"
    f"os.environ.setdefault('LC_MODE', {a.mode!r})\n"
    f"os.environ.setdefault('LC_N_EVAL', {str(a.n_eval)!r})\n"
    f"os.environ.setdefault('LC_MODEL', {a.model!r})\n"
    f"os.environ.setdefault('LC_MAX_TOKENS', {str(a.max_tokens)!r})\n"
    f"os.environ.setdefault('LC_N_SAMPLES', {str(a.n_samples)!r})\n"
)
bundle = header + "\n# ===== bundled liars/rules.py =====\n" + rules + "\nRULES_SOLVE = solve\n" \
    + "\n# ===== kaggle/llm_baseline/infer.py =====\n" + infer

out = ROOT / "kaggle" / "build" / slug
out.mkdir(parents=True, exist_ok=True)
(out / "kernel.py").write_text(bundle)
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": f"gouravanirudh/{slug}",
    "title": slug,
    "code_file": "kernel.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "true",
    "dataset_sources": [],
    "competition_sources": ["the-liars-chatroom"],
    "kernel_sources": [],
    "model_sources": [],
}, indent=2))
print("built", out)
