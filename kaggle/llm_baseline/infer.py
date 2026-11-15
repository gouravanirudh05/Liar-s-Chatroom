"""Liar's Chatroom — open-weight LLM baseline (Kaggle kernel, GPU T4 x2).

Rules compliance: runs an open-weight model (Qwen3, Apache-2.0, <=70B) locally
via vLLM. No hosted/closed APIs. test.csv is only fed through the model.

MODE = "eval": score on a random sample of train.csv (N_EVAL cases).
MODE = "test": predict all of test.csv and write submission.csv.
"""
import glob
import json
import os
import re
import subprocess
import sys
import time

# ---------------- config ----------------
MODE = os.environ.get("LC_MODE", "eval")
N_EVAL = int(os.environ.get("LC_N_EVAL", "60"))
MODEL = os.environ.get("LC_MODEL", "Qwen/Qwen3-14B-AWQ")
MAX_TOKENS = int(os.environ.get("LC_MAX_TOKENS", "6000"))
N_SAMPLES = int(os.environ.get("LC_N_SAMPLES", "1"))  # >1 => majority vote
SEED = 0
OUT = "/kaggle/working"

t0 = time.time()
subprocess.run("nvidia-smi", shell=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)

import pandas as pd  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

print("vllm installed in", round(time.time() - t0), "s", flush=True)


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, f"{name} not found"
    return hits[0]


# ---------------- parsing ----------------
MSG_RE = re.compile(r"^\s*(\d+)\.\s*([ABCD]):\s*(.*)$")
GENERIC_FILLER = re.compile(
    r"\b(confus|muddle|tangled|messy|hard to follow|clear answer|straight answer|"
    r"plain answer|clarity|thanks?\b|thank you|cheers|appreciate|keep (it|this)|"
    r"wrap this up|drag this out|go through it (once|one) more)",
    re.I,
)


def parse_speakers(s):
    out = {}
    for part in s.split(";"):
        lab, name = part.split("=", 1)
        out[lab.strip()] = name.strip()
    return out


def parse_chat(chat):
    msgs = []
    for line in str(chat).splitlines():
        m = MSG_RE.match(line.strip())
        if m:
            msgs.append((int(m[1]), m[2], m[3].strip()))
    return msgs


def is_filler(text):
    t = text.strip()
    if t.endswith("?"):
        return True
    # generic meta-chat with no digits ("This is confusing.", "Thanks, everyone.")
    return not re.search(r"\d", t) and bool(GENERIC_FILLER.search(t)) and len(t.split()) <= 9


INSTRUCTIONS = """You are checking a group chat against verified evidence.

Rules:
- Only the EVIDENCE and the SPEAKER TABLE are true. Chat messages are claims; one speaker's claim never proves another wrong.
- Exactly one message makes a claim that CANNOT be true at the same time as the evidence. Find it.
- A claim that is merely unverified, suspicious or unlikely is NOT a contradiction. Only a claim that is impossible given the evidence counts.
- Use the literal meaning of the words, basic logic, and careful clock/calendar arithmetic (12-hour clock with AM/PM; noon = 12:00 PM).
- "Between X and Y" and "from X to Y" include both endpoints. "After noon" is not "at noon".
- A person or object cannot be in two places at once when the evidence says the places are separate.
- Records, logs and lists are complete only if the evidence says they are.
- If the evidence says nothing about a particular time, place, person or thing, a claim about it cannot contradict the evidence (e.g. a rule that holds during certain hours says nothing about other hours).
- "I" / "my" in a message refers to the sender (see the speaker table). A message can be about someone else; the answer is still its sender.
- Questions and opinions are never the answer.

Check each claim message against the evidence one at a time, then give the number of the single contradicting message.
End your reply with a final line exactly of the form:
ANSWER: <message number>"""


def build_prompt(row):
    spk = parse_speakers(row["speakers"])
    lines = []
    for k, s, t in parse_chat(row["chat"]):
        if is_filler(t):
            continue
        lines.append(f"{k}. {s} ({spk.get(s, '?')}): {t}")
    table = "; ".join(f"{k} = {v}" for k, v in spk.items())
    return (
        f"{INSTRUCTIONS}\n\nSPEAKER TABLE: {table}\n\nEVIDENCE: {row['evidence']}\n\n"
        f"CHAT (questions and small talk removed):\n" + "\n".join(lines)
    )


ANS_RE = re.compile(r"ANSWER:\s*\**\s*(\d+)", re.I)


def extract_label(text, row):
    """Map the model's message number to its sender label; None if unusable."""
    body = text.split("</think>")[-1]
    hits = ANS_RE.findall(body) or ANS_RE.findall(text)
    if not hits:
        return None
    k = int(hits[-1])
    for kk, s, t in parse_chat(row["chat"]):
        if kk == k:
            return s
    return None


def candidate_fallback(row):
    """Last resort: first sender with a non-filler message."""
    for _, s, t in parse_chat(row["chat"]):
        if not is_filler(t):
            return s
    return "A"


# ---------------- data ----------------
if MODE == "eval":
    df = pd.read_csv(find("train.csv")).sample(N_EVAL, random_state=SEED).reset_index(drop=True)
else:
    df = pd.read_csv(find("test.csv"))
print(MODE, len(df), "cases", flush=True)

# exact rules (bundled by scripts/build_kernel.py); None => LLM decides
_solve = globals().get("RULES_SOLVE")
df["rule_pred"] = [(_solve(r)[0] if _solve else None) for _, r in df.iterrows()]
print("rules cover", df["rule_pred"].notna().sum(), "of", len(df), flush=True)
# In eval mode run the LLM on every sampled case (to measure it on all families);
# in test mode only on cases the rules do not settle.
full_df = df
if MODE != "eval":
    df = df[df["rule_pred"].isna()].reset_index(drop=True)
print("LLM cases:", len(df), flush=True)

# ---------------- model ----------------
llm = LLM(
    model=MODEL,
    tensor_parallel_size=2,
    dtype="half",
    max_model_len=8192,
    gpu_memory_utilization=0.92,
    enable_prefix_caching=True,
    seed=SEED,
)
tok = llm.get_tokenizer()


def chat_text(prompt, thinking=True):
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


prompts = [chat_text(build_prompt(r)) for _, r in df.iterrows()]
sp = SamplingParams(
    n=N_SAMPLES, temperature=0.6, top_p=0.95, top_k=20, max_tokens=MAX_TOKENS, seed=SEED
)
t1 = time.time()
outs = llm.generate(prompts, sp)
gen_s = time.time() - t1
n_tok = sum(len(c.token_ids) for o in outs for c in o.outputs)
print(f"generation {gen_s:.0f}s, {n_tok} tokens, {n_tok / gen_s:.0f} tok/s, "
      f"{gen_s / len(df):.2f}s/case", flush=True)

preds, raw = [], []
retry_idx = []
for i, (o, (_, r)) in enumerate(zip(outs, df.iterrows())):
    votes = [extract_label(c.text, r) for c in o.outputs]
    votes = [v for v in votes if v]
    if votes:
        preds.append(max(set(votes), key=votes.count))
    else:
        preds.append(None)
        retry_idx.append(i)
    raw.append({"id": r["id"], "votes": votes, "texts": [c.text[-1500:] for c in o.outputs],
                "n_tokens": [len(c.token_ids) for c in o.outputs]})

# retry unanswered cases without thinking (short direct answer)
if retry_idx:
    print("retrying", len(retry_idx), "unanswered cases without thinking", flush=True)
    rp = [chat_text(build_prompt(df.iloc[i]), thinking=False) for i in retry_idx]
    ro = llm.generate(rp, SamplingParams(temperature=0.0, max_tokens=1500))
    for i, o in zip(retry_idx, ro):
        lab = extract_label(o.outputs[0].text, df.iloc[i])
        preds[i] = lab or candidate_fallback(df.iloc[i])

df["llm_pred"] = preds
with open(f"{OUT}/raw_{MODE}.jsonl", "w") as f:
    for x in raw:
        f.write(json.dumps(x) + "\n")

if MODE == "eval":
    df["final"] = df["rule_pred"].fillna(df["llm_pred"])
    df[["id", "label", "rule_pred", "llm_pred", "final"]].to_csv(f"{OUT}/eval_preds.csv", index=False)
    cov = df["rule_pred"].notna()
    unc = ~cov
    print(f"EVAL LLM-only accuracy (all {len(df)}): {(df.llm_pred == df.label).mean():.4f}")
    print(f"EVAL LLM on rule-uncovered ({unc.sum()}): {(df.llm_pred == df.label)[unc].mean():.4f}")
    print(f"EVAL LLM on rule-covered ({cov.sum()}): {(df.llm_pred == df.label)[cov].mean():.4f}")
    if cov.any():
        print(f"EVAL rules on covered: {(df.rule_pred == df.label)[cov].mean():.4f}")
    print(f"EVAL ACCURACY (rules + LLM): {(df.final == df.label).mean():.4f} "
          f"(unanswered after thinking: {len(retry_idx)})", flush=True)
else:
    llm_map = dict(zip(df["id"], df["llm_pred"]))
    sub = full_df[["id"]].copy()
    sub["label"] = [rp if isinstance(rp, str) else llm_map[i]
                    for i, rp in zip(full_df["id"], full_df["rule_pred"])]
    assert sub["label"].isin(list("ABCD")).all()
    sub.to_csv(f"{OUT}/submission.csv", index=False)
    print("wrote submission.csv", sub.shape, sub["label"].value_counts().to_dict())
print("total runtime", round(time.time() - t0), "s")
