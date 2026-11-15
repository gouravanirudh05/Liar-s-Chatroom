"""Combine v2b with v3 votes. Policies validated on the train analogs (see liars-verify-* logs)."""
import collections, json, pandas as pd
cur = pd.read_csv('out/refine_v2/submission_v2b.csv').set_index('id')['label']
raw = {json.loads(l)['id']: json.loads(l) for l in open('out/test_v1/raw_test.jsonl')}
v2 = json.load(open('out/refine_v2/votes_test.json'))
L = json.load(open('out/verify_llm/verify_llm.json'))['test']
R = json.load(open('out/verify_rule/verify_rule.json'))['test']
new = cur.copy(); log = collections.Counter()
# LLM cases: majority over every vote collected; tie -> keep current label
for i in set(L['s1']) | set(L['s2']):
    pool = []
    if i in raw and raw[i]['votes']: pool.append(raw[i]['votes'][0])
    pool += v2.get(i, []) + L['s1'].get(i, []) + L['s2'].get(i, [])
    c = collections.Counter(pool); best = max(c.values()); tops = [k for k in c if c[k] == best]
    lab = cur[i] if cur[i] in tops else (tops[0] if len(tops) == 1 else cur[i])
    if lab != cur[i]: log['llm-majority change'] += 1
    new[i] = lab
# Rule cases: override only if all 4 LLM votes agree on one other label (0 overrides on train)
for i, ref in R['ref'].items():
    votes = R['s1'].get(i, []) + R['s2'].get(i, [])
    if False and len(votes) == 4 and len(set(votes)) == 1 and votes[0] != ref:
        log['rule override'] += 1; new[i] = votes[0]
    elif votes and votes[0] != ref:
        log['rule disagreement kept'] += 1
print(dict(log), '| changed vs v2b:', (new != cur).sum())
new.rename('label').reset_index().to_csv('out/v3/submission_v3_norule.csv', index=False)
