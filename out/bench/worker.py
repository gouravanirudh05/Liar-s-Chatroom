"""Benchmark one vLLM config on train cases the rules do not cover.

Run as: python worker.py '<json config>'
Prints one line 'RESULT {json}'. Uses lib.py (rules + prompt) written next to it.
"""
import json
import sys
import time

import pandas as pd

sys.path.insert(0, "/kaggle/working")
import lib  # noqa: E402

cfg = json.loads(sys.argv[1])
from vllm import LLM, SamplingParams  # noqa: E402

import glob  # noqa: E402

train = pd.read_csv(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
train["rule"] = [lib.solve(r)[0] for _, r in train.iterrows()]
df = train[train["rule"].isna()].sample(cfg.get("n", 128), random_state=1).reset_index(drop=True)

t0 = time.time()
llm = LLM(model=cfg["model"], tensor_parallel_size=cfg.get("tp", 2), dtype="half",
          max_model_len=cfg.get("max_model_len", 8192), gpu_memory_utilization=cfg.get("gmu", 0.92),
          enable_prefix_caching=True, seed=0, **cfg.get("extra", {}))
load_s = time.time() - t0
tok = llm.get_tokenizer()
prompts = [tok.apply_chat_template([{"role": "user", "content": lib.build_prompt(r)}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True) for _, r in df.iterrows()]
sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=cfg.get("max_tokens", 6000), seed=0)
t1 = time.time()
outs = llm.generate(prompts, sp)
gen_s = time.time() - t1
ntok = sum(len(o.outputs[0].token_ids) for o in outs)
preds = [lib.extract_label(o.outputs[0].text, r) for o, (_, r) in zip(outs, df.iterrows())]
answered = [p is not None for p in preds]
correct = [p == lab for p, lab in zip(preds, df["label"])]
res = dict(cfg=cfg, load_s=round(load_s), gen_s=round(gen_s), tokens=ntok, tok_s=round(ntok / gen_s, 1),
           s_per_case=round(gen_s / len(df), 2), answered=sum(answered), acc=round(sum(correct) / len(df), 4),
           mean_tokens=round(ntok / len(df)))
print("RESULT " + json.dumps(res), flush=True)
