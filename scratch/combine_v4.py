import collections, json, sys, pandas as pd
TH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8
S3 = {}
for p in ('extra1', 'extra2'):
    d = json.load(open(f'out/vote_{p}/verify_{p}.json'))
    for part in ('train', 'test'):
        S3.update(d[part]['s3'])
VL = json.load(open('out/verify_llm/verify_llm.json')); VR = json.load(open('out/verify_rule/verify_rule.json'))
# ---- train check of the rule-override policy
tr = pd.read_csv('Data/train.csv').set_index('id')['label']
for i, ref in VR['train']['ref'].items():
    v = VR['train']['s1'].get(i, []) + VR['train']['s2'].get(i, []) + S3.get(i, [])
    against = [x for x in v if x != ref]
    if against:
        c = collections.Counter(against).most_common(1)[0]
        ov = c[1] / len(v) >= TH
        print(f'TRAIN-RULE {i} label {tr[i]} rule {ref} votes {"".join(v)} -> {"OVERRIDE " + c[0] if ov else "keep"}')
for i in ('LC00254', 'LC02531'):
    print('TRAIN-LLM', i, tr[i], S3.get(i))
# ---- test
cur = pd.read_csv('out/v3/submission_v3.csv').set_index('id')['label']
raw = {json.loads(l)['id']: json.loads(l) for l in open('out/test_v1/raw_test.jsonl')}
v2 = json.load(open('out/refine_v2/votes_test.json'))
L, R = VL['test'], VR['test']
new = cur.copy(); log = collections.Counter()
for i in set(L['s1']) | set(L['s2']) | (set(S3) & set(raw)):
    if i not in raw: continue
    pool = ([raw[i]['votes'][0]] if raw[i]['votes'] else []) + v2.get(i, []) + L['s1'].get(i, []) + L['s2'].get(i, []) + S3.get(i, [])
    c = collections.Counter(pool); best = max(c.values()); tops = [k for k in c if c[k] == best]
    lab = cur[i] if cur[i] in tops else (tops[0] if len(tops) == 1 else cur[i])
    log['llm change'] += lab != cur[i]; new[i] = lab
for i in v2:  # rule-withdrawn cases (not in raw): v2 votes + extra votes
    if i in raw or i not in S3: continue
    c = collections.Counter(v2[i] + S3[i]); best = max(c.values()); tops = [k for k in c if c[k] == best]
    lab = cur[i] if cur[i] in tops else (tops[0] if len(tops) == 1 else cur[i])
    log['withdrawn change'] += lab != cur[i]; new[i] = lab
for i, ref in R['ref'].items():
    v = R['s1'].get(i, []) + R['s2'].get(i, []) + S3.get(i, [])
    against = [x for x in v if x != ref]
    lab = ref
    if against:
        c = collections.Counter(against).most_common(1)[0]
        if c[1] / len(v) >= TH: lab = c[0]
    log['rule-case change'] += lab != cur[i]; new[i] = lab
print(dict(log), '| changed vs v3b:', (new != cur).sum())
new.rename('label').reset_index().to_csv(f'out/v3/submission_v4_th{TH}.csv', index=False)
