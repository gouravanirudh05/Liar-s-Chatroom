"""Build the v3 vote-collection kernels: kaggle/build/liars-verify-{llm,rule}. Push with --accelerator NvidiaTeslaT4."""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
lib = (ROOT / "liars" / "rules.py").read_text() + "\n\n" + (ROOT / "liars" / "prompt.py").read_text()
main = (ROOT / "kaggle" / "verify" / "main.py").read_text()
base = {"model": "Qwen/Qwen3-14B-AWQ", "hard_tok": 2000, "max_model_len": 12288, "s1_tokens": 8000,
        "k2": 3, "s2_tokens": 10000, "n_train_rule": 300,
        # contested = v2 votes not unanimous (train analog from the liars-refine log); ids only, no test content
        "contested_train": ["LC00254", "LC04756", "LC00360", "LC01971"],
        "contested_test": json.load(open(ROOT / "scratch" / "test_nonunanimous.json"))}
for part in ("llm", "rule"):
    cfg = dict(base, part=part)
    slug = f"liars-verify-{part}"
    out = ROOT / "kaggle" / "build" / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel.py").write_text(f"LIB_B64 = {base64.b64encode(lib.encode()).decode()!r}\n"
                                   f"CFG_JSON = {json.dumps(json.dumps(cfg))}\n\n" + main)
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"gouravanirudh/{slug}", "title": slug, "code_file": "kernel.py", "language": "python",
        "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "true", "dataset_sources": [], "competition_sources": ["the-liars-chatroom"],
        "kernel_sources": ["gouravanirudh/liars-chatroom-test", "gouravanirudh/liars-chatroom-eval"],
        "model_sources": []}, indent=2))
    print("built", out)

# extra-vote shards on contested cases (ids only)
for part, ids in json.load(open(ROOT / "scratch" / "extra_split.json")).items():
    cfg = dict(base, part=part, k2=6, **ids)
    slug = f"liars-vote-{part}"
    out = ROOT / "kaggle" / "build" / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel.py").write_text(f"LIB_B64 = {base64.b64encode(lib.encode()).decode()!r}\n"
                                   f"CFG_JSON = {json.dumps(json.dumps(cfg))}\n\n" + main)
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"gouravanirudh/{slug}", "title": slug, "code_file": "kernel.py", "language": "python",
        "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "true", "dataset_sources": [], "competition_sources": ["the-liars-chatroom"],
        "kernel_sources": ["gouravanirudh/liars-chatroom-test", "gouravanirudh/liars-chatroom-eval"],
        "model_sources": []}, indent=2))
    print("built", out)

# Qwen3-32B-AWQ arbiter shards (ids only)
for part, ids in json.load(open(ROOT / "scratch" / "big_split.json")).items():
    cfg = dict(base, part="extra-" + part, model="Qwen/Qwen3-32B-AWQ", k2=3, k_train=3, gmu=0.90,
               max_model_len=10240, s2_tokens=8500, llm_extra={"enforce_eager": True, "max_num_seqs": 24}, **ids)
    if ids.get("bench96"):
        cfg["k_train"] = 1
    slug = f"liars-32b-{part}"
    out = ROOT / "kaggle" / "build" / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel.py").write_text(f"LIB_B64 = {base64.b64encode(lib.encode()).decode()!r}\n"
                                   f"CFG_JSON = {json.dumps(json.dumps(cfg))}\n\n" + main)
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"gouravanirudh/{slug}", "title": slug, "code_file": "kernel.py", "language": "python",
        "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "true", "dataset_sources": [], "competition_sources": ["the-liars-chatroom"],
        "kernel_sources": ["gouravanirudh/liars-chatroom-test", "gouravanirudh/liars-chatroom-eval"],
        "model_sources": []}, indent=2))
    print("built", out)
