"""TPU v5e-8 feasibility: vLLM (TPU build) + Qwen3-32B bf16 (open weights, Apache-2.0) on train only."""
import base64
import glob
import json
import subprocess
import sys
import time

open("/kaggle/working/lib.py", "w").write(base64.b64decode(LIB_B64).decode())  # noqa: F821
sys.path.insert(0, "/kaggle/working")

t0 = time.time()
print("python", sys.version, flush=True)
subprocess.run("pip list 2>/dev/null | grep -iE 'jax|torch|libtpu|vllm|tpu'", shell=True)
ok = False
for pkgs in (["vllm-tpu"], ["tpu-inference"]):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], capture_output=True, text=True)
    print("pip install", pkgs, "rc", r.returncode, (r.stdout + r.stderr)[-1500:], flush=True)
    if r.returncode == 0:
        try:
            import vllm  # noqa: F401
            print("vllm", vllm.__version__, flush=True)
            ok = True
            break
        except Exception as e:  # noqa: BLE001
            print("import failed:", repr(e)[:800], flush=True)
print("install phase", round(time.time() - t0), "s; ok =", ok, flush=True)
if not ok:
    sys.exit(0)

import pandas as pd  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

import lib  # noqa: E402

train = pd.read_csv(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
train["rule"] = [lib.solve(r)[0] for _, r in train.iterrows()]
df = train[train["rule"].isna()].sample(N_CASES, random_state=1).reset_index(drop=True)  # noqa: F821

for model in MODELS:  # noqa: F821
    try:
        t1 = time.time()
        llm = LLM(model=model, tensor_parallel_size=8, max_model_len=8192, seed=0)
        print(model, "loaded in", round(time.time() - t1), "s", flush=True)
        tok = llm.get_tokenizer()
        prompts = [tok.apply_chat_template([{"role": "user", "content": lib.build_prompt(r)}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True) for _, r in df.iterrows()]
        sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=6000, seed=0)
        t2 = time.time()
        outs = llm.generate(prompts, sp)
        gs = time.time() - t2
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        preds = [lib.extract_label(o.outputs[0].text, r) for o, (_, r) in zip(outs, df.iterrows())]
        acc = sum(p == lab for p, lab in zip(preds, df["label"])) / len(df)
        print("RESULT " + json.dumps(dict(model=model, gen_s=round(gs), tok_s=round(ntok / gs, 1),
                                          s_per_case=round(gs / len(df), 2), acc=round(acc, 4),
                                          answered=sum(p is not None for p in preds))), flush=True)
        del llm
    except Exception as e:  # noqa: BLE001
        print("MODEL FAILED", model, repr(e)[:2000], flush=True)
print("total", round(time.time() - t0), "s")
