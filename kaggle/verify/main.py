"""v3 vote collection: second opinions on cases v2 did not re-check. Decisions are made offline.

PART == "llm":  one fresh sample on every v1 LLM case answered in < hard_tok tokens (train: eval300 analog),
                then k2 samples on disagreements and on contested v2 cases.
PART == "rule": one sample on every rule-covered case (train: random sample), then k2 samples on disagreements.
Open-weight model only (Qwen3-14B-AWQ, Apache-2.0) via vLLM; no hosted APIs. No test labels are used anywhere.
"""
import base64
import glob
import json
import subprocess
import sys
import time

open("/kaggle/working/lib.py", "w").write(base64.b64decode(LIB_B64).decode())  # noqa: F821
sys.path.insert(0, "/kaggle/working")
CFG = json.loads(CFG_JSON)  # noqa: F821
PART = CFG["part"]

t0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)
print("vllm installed in", round(time.time() - t0), "s", flush=True)

import pandas as pd  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

import lib  # noqa: E402


def find(pattern, inside=""):
    hits = [h for h in glob.glob(f"/kaggle/input/**/{pattern}", recursive=True) if inside in h]
    assert hits, (pattern, inside)
    return hits[0]


train = pd.read_csv(find("train.csv")).set_index("id", drop=False)
test = pd.read_csv(find("test.csv")).set_index("id", drop=False)
raw_test = {json.loads(l)["id"]: json.loads(l) for l in open(find("raw_test.jsonl"))}
raw_eval = {json.loads(l)["id"]: json.loads(l) for l in open(find("raw_eval.jsonl"))}


def easy(r):
    return bool(r["votes"]) and r["n_tokens"][0] < CFG["hard_tok"]


if PART == "llm":
    ref_train = {i: r["votes"][0] for i, r in raw_eval.items() if easy(r)}
    ref_test = {i: r["votes"][0] for i, r in raw_test.items() if easy(r)}
    extra_train, extra_test = CFG["contested_train"], CFG["contested_test"]
elif PART.startswith("extra"):
    tids = list(CFG["train_ids"])
    if CFG.get("bench96"):
        tr_rule = {i: lib.solve(train.loc[i])[0] for i in train.index}
        unc = train[[tr_rule[i] is None for i in train.index]]
        tids = unc.sample(96, random_state=1)["id"].tolist() + [i for i in tids if i not in set(unc.index)]
    ref_train = {i: None for i in tids}
    ref_test = {i: None for i in CFG["test_ids"]}
    extra_train, extra_test = [], []
else:
    rule = {i: lib.solve(test.loc[i])[0] for i in test.index}
    ref_test = {i: lab for i, lab in rule.items() if lab}
    tr_rule = {i: lib.solve(train.loc[i])[0] for i in train.index}
    covered = sorted(i for i, lab in tr_rule.items() if lab)
    ids = pd.Series(covered).sample(CFG["n_train_rule"], random_state=2).tolist()
    ref_train = {i: tr_rule[i] for i in ids}
    extra_train, extra_test = [], []
print(f"PART {PART}: train {len(ref_train)} (+{len(extra_train)}) | test {len(ref_test)} (+{len(extra_test)})",
      flush=True)

llm = LLM(model=CFG["model"], tensor_parallel_size=2, dtype="half", max_model_len=CFG["max_model_len"],
          gpu_memory_utilization=CFG.get("gmu", 0.92), enable_prefix_caching=True, seed=0,
          **CFG.get("llm_extra", {}))
tok = llm.get_tokenizer()
FORCE = ("\n\nConsidering the limited time, I have to give the answer based on the thinking directly now.\n"
         "</think>\n\nANSWER:")


def run(ids, df, k, max_tokens, seed):
    if not ids:
        return {}
    chats = {i: tok.apply_chat_template([{"role": "user", "content": lib.build_prompt(df.loc[i])}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=True) for i in ids}
    sp = SamplingParams(n=k, temperature=0.6, top_p=0.95, top_k=20, max_tokens=max_tokens, seed=seed)
    t1 = time.time()
    outs = llm.generate([chats[i] for i in ids], sp)
    ntok = sum(len(c.token_ids) for o in outs for c in o.outputs)
    print(f"  {len(ids)} cases x{k}: {ntok} tokens in {time.time() - t1:.0f}s", flush=True)
    votes, force = {}, []
    for i, o in zip(ids, outs):
        votes[i] = []
        for c in o.outputs:
            lab = lib.extract_label(c.text, df.loc[i]) if "</think>" in c.text else None
            if lab:
                votes[i].append(lab)
            else:
                force.append((i, chats[i] + c.text + FORCE))
    if force:
        fo = llm.generate([p for _, p in force], SamplingParams(temperature=0.0, max_tokens=12))
        for (i, _), o in zip(force, fo):
            lab = lib.extract_label("ANSWER:" + o.outputs[0].text, df.loc[i])
            if lab:
                votes[i].append(lab)
        print(f"  budget-forced {len(force)}", flush=True)
    return votes


out = {}
for name, df, ref, extra in (("train", train, ref_train, extra_train), ("test", test, ref_test, extra_test)):
    if PART.startswith("extra"):
        kk = CFG.get("k_train", CFG["k2"]) if name == "train" else CFG["k2"]
        out[name] = {"s3": run(sorted(ref), df, kk, CFG["s2_tokens"], seed=33)}
        if name == "train":
            lab = train["label"]
            got = out[name]["s3"]
            print("TRAIN acc (first vote):", sum(bool(got[i]) and got[i][0] == lab[i] for i in got), "/", len(got),
                  flush=True)
        json.dump(out, open(f"/kaggle/working/verify_{PART}.json", "w"))
        continue
    s1 = run(sorted(ref), df, 1, CFG["s1_tokens"], seed=11)
    dis = sorted({i for i in ref if s1[i] != [ref[i]]} | set(extra))
    print(f"{name}: stage-1 disagreements {sum(s1[i] != [ref[i]] for i in ref)} / {len(ref)}; stage-2 on {len(dis)}",
          flush=True)
    s2 = run(dis, df, CFG["k2"], CFG["s2_tokens"], seed=22)
    out[name] = {"ref": ref, "s1": s1, "s2": s2}
    json.dump(out, open(f"/kaggle/working/verify_{PART}.json", "w"))
    if name == "train":
        lab = train["label"]
        print("TRAIN ref acc", sum(ref[i] == lab[i] for i in ref), "/", len(ref), "| s1 acc",
              sum(s1[i] == [lab[i]] for i in ref), flush=True)
        for i in dis:
            print(f"  TRAIN {i} label {lab[i]} ref {ref.get(i)} s1 {s1.get(i)} s2 {s2.get(i)}", flush=True)
print("total runtime", round(time.time() - t0), "s")
