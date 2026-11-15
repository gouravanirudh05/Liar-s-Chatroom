"""v2: refine the hard cases of v1 with self-consistency + larger thinking budget + budget forcing.

Open-weight model only (Qwen3 family, Apache-2.0) via vLLM. No hosted APIs.
Inputs (attached as kernel sources): v1 test outputs (raw_test.jsonl, submission.csv) and the
300-case train eval outputs (raw_eval.jsonl, eval_preds.csv), plus competition data.
Selection criterion (identical for train and test): v1 trace >= HARD_TOK tokens or no thinking answer,
plus test cases whose rule prediction is withdrawn by the current (stricter) rules.
"""
import base64
import collections
import glob
import json
import subprocess
import sys
import time

open("/kaggle/working/lib.py", "w").write(base64.b64decode(LIB_B64).decode())  # noqa: F821
sys.path.insert(0, "/kaggle/working")
CFG = json.loads(CFG_JSON)  # noqa: F821

t0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)
print("vllm installed in", round(time.time() - t0), "s", flush=True)

import pandas as pd  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

import lib  # noqa: E402


def find(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    assert hits, pattern
    return hits[0]


train = pd.read_csv(find("train.csv")).set_index("id", drop=False)
test = pd.read_csv(find("test.csv")).set_index("id", drop=False)
v1_sub = pd.read_csv(find("submission.csv")).set_index("id")["label"]
raw_test = {json.loads(l)["id"]: json.loads(l) for l in open(find("raw_test.jsonl"))}
raw_eval = {json.loads(l)["id"]: json.loads(l) for l in open(find("raw_eval.jsonl"))}
eval_preds = pd.read_csv(find("eval_preds.csv")).set_index("id")


def is_hard(r):
    return (not r["votes"]) or r["n_tokens"][0] >= CFG["hard_tok"]


# ---- selection
train_hard = [i for i, r in raw_eval.items() if is_hard(r)]
test_hard = [i for i, r in raw_test.items() if is_hard(r)]
rule_now = {i: lib.solve(test.loc[i])[0] for i in test.index}
withdrawn = [i for i in test.index if i not in raw_test and rule_now[i] is None]  # v1 rule -> now abstains
changed_rule = [i for i in test.index if i not in raw_test and rule_now[i] is not None and rule_now[i] != v1_sub[i]]
print(f"train_hard {len(train_hard)} | test_hard {len(test_hard)} | rule-withdrawn {len(withdrawn)} | "
      f"rule-changed {len(changed_rule)}", flush=True)

EXTRA_RULE = ("- If more than one message seems impossible, re-check each of them strictly against the exact wording "
              "of the evidence: usually one of them is actually possible, because the evidence does not cover that "
              "time, person or place, or because an endpoint is included. Exactly one message is truly impossible.\n")


def prompt_for(row):
    p = lib.build_prompt(row)
    return p.replace("- Questions and opinions are never the answer.\n",
                     "- Questions and opinions are never the answer.\n" + EXTRA_RULE)


llm = LLM(model=CFG["model"], tensor_parallel_size=CFG.get("tp", 2), dtype="half",
          max_model_len=CFG["max_model_len"], gpu_memory_utilization=0.92, enable_prefix_caching=True,
          seed=0, **CFG.get("extra", {}))
tok = llm.get_tokenizer()
FORCE = ("\n\nConsidering the limited time, I have to give the answer based on the thinking directly now.\n"
         "</think>\n\nANSWER:")


def run(ids, df, k):
    """Return {id: [labels from k samples]} with budget forcing for truncated samples."""
    chats = {i: tok.apply_chat_template([{"role": "user", "content": prompt_for(df.loc[i])}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=True) for i in ids}
    sp = SamplingParams(n=k, temperature=0.6, top_p=0.95, top_k=20, max_tokens=CFG["max_tokens"], seed=1)
    t1 = time.time()
    outs = llm.generate([chats[i] for i in ids], sp)
    ntok = sum(len(c.token_ids) for o in outs for c in o.outputs)
    print(f"  generated {ntok} tokens in {time.time() - t1:.0f}s ({ntok / (time.time() - t1):.0f} tok/s)", flush=True)
    votes, force_jobs = {}, []
    for i, o in zip(ids, outs):
        votes[i] = []
        for c in o.outputs:
            lab = lib.extract_label(c.text, df.loc[i]) if "</think>" in c.text else None
            if lab:
                votes[i].append(lab)
            else:
                force_jobs.append((i, chats[i] + c.text + FORCE))
    if force_jobs:
        fo = llm.generate([p for _, p in force_jobs], SamplingParams(temperature=0.0, max_tokens=12))
        for (i, _), o in zip(force_jobs, fo):
            lab = lib.extract_label("ANSWER:" + o.outputs[0].text, df.loc[i])
            if lab:
                votes[i].append(lab)
        print(f"  budget-forced {len(force_jobs)} samples", flush=True)
    return votes


def combine(new, old):
    """Majority over new samples plus the v1 answer (if it came from a thinking trace); ties -> new samples."""
    pool = list(new) + ([old] if old else [])
    if not pool:
        return None
    c = collections.Counter(pool)
    best = max(c.values())
    for lab in list(new) + [old]:
        if lab and c[lab] == best:
            return lab


# ---- train check (same procedure)
tv = run(train_hard, train, CFG["k"])
before = after = 0
for i in train_hard:
    old = raw_eval[i]["votes"][0] if raw_eval[i]["votes"] else None
    lab = train.loc[i, "label"]
    before += eval_preds.loc[i, "llm_pred"] == lab
    after += combine(tv[i], old) == lab
    print(f"  TRAIN {i} label {lab} v1 {eval_preds.loc[i, 'llm_pred']} new {tv[i]} -> {combine(tv[i], old)}")
print(f"TRAIN-HARD accuracy: v1 {before}/{len(train_hard)} -> v2 {after}/{len(train_hard)}", flush=True)

# ---- test
todo = withdrawn + test_hard
final = v1_sub.copy()
for i in changed_rule:  # current rules disagree with v1 rule (should be none)
    final[i] = rule_now[i]
sv, n_changed, CH = {}, 0, 100
for s in range(0, len(todo), CH):  # chunked: a partial submission is always on disk
    chunk = todo[s:s + CH]
    sv.update(run(chunk, test, CFG["k"]))
    for i in chunk:
        old = (raw_test[i]["votes"][0] if raw_test[i]["votes"] else None) if i in raw_test else None
        lab = combine(sv[i], old) or v1_sub[i]
        n_changed += lab != v1_sub[i]
        final[i] = lab
    sub = final.rename("label").reset_index()
    assert len(sub) == 3500 and sub["label"].isin(list("ABCD")).all()
    sub.to_csv("/kaggle/working/submission.csv", index=False)
    json.dump(sv, open("/kaggle/working/votes_test.json", "w"))
    print(f"TEST chunk done: {s + len(chunk)}/{len(todo)} refined, {n_changed} labels changed vs v1, "
          f"elapsed {time.time() - t0:.0f}s", flush=True)
print("total runtime", round(time.time() - t0), "s")
