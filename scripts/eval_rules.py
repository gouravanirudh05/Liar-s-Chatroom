"""Score the rule modules on train.csv: coverage and precision, per module."""
import collections
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from liars.rules import parse_chat, solve  # noqa: E402

tr = pd.read_csv(Path(__file__).resolve().parents[1] / "Data" / "train.csv")
per = collections.defaultdict(lambda: [0, 0])  # module -> [cases flagged, cases where all flags are by liar]
covered = correct = 0
wrong = []
for _, r in tr.iterrows():
    lab, hits = solve(r)
    sender = {k: s for k, s, _ in parse_chat(r["chat"])}
    for mod, ks in hits.items():
        per[mod][0] += 1
        per[mod][1] += all(sender[k] == r["label"] for k in ks)
    if lab:
        covered += 1
        correct += lab == r["label"]
        if lab != r["label"]:
            wrong.append((r["id"], r["label"], lab, hits))

print(f"covered {covered}/{len(tr)} ({covered / len(tr):.1%}), precision {correct}/{covered}")
for mod, (n, ok) in sorted(per.items()):
    print(f"  {mod:22s} flagged {n:4d}  all-flags-correct {ok:4d}")
for w in wrong[:15]:
    print("WRONG", w)
if "-v" in sys.argv:
    for w in wrong:
        r = tr[tr.id == w[0]].iloc[0]
        print("\n", w, "\nEV:", r.evidence, "\n" + r.chat)
