"""v5 = v4, but where Qwen3-32B is unanimous (3/3) on a contested case, take its label.
Rule cases additionally need >=50% of the 10 Qwen3-14B votes for the same label (train-checked below)."""
import collections, json, pandas as pd
B = {'train': {}, 'test': {}}
for p in ('big1', 'big2'):
    try:
        d = json.load(open(f'out/32b_{p}/verify_extra-{p}.json'))
    except FileNotFoundError:
        print('missing', p); continue
    for part in d:
        B[part].update(d[part]['s3'])
S3 = {}
for p in ('extra1', 'extra2'):
    d = json.load(open(f'out/vote_{p}/verify_{p}.json'))
    for part in ('train', 'test'):
        S3.update(d[part]['s3'])
VR = json.load(open('out/verify_rule/verify_rule.json'))
tr = pd.read_csv('Data/train.csv').set_index('id')['label']
bench = [i for i in B['train'] if len(B['train'][i]) == 1]
print('32B bench acc:', sum(B['train'][i] == [tr[i]] for i in bench), '/', len(bench), '(14B: 94/96 = 97.9%)')

def decide(i, cur, ref, votes14, b):
    if len(b) == 3 and len(set(b)) == 1:
        x = b[0]
        if ref is None:
            return x
        if x != ref and votes14 and sum(v == x for v in votes14) / len(votes14) >= 0.5:
            return x
        if x == ref:
            return ref
    return cur

print('--- train analogs')
for i, b in B['train'].items():
    if len(b) != 3: continue
    ref = VR['train']['ref'].get(i)
    v14 = VR['train']['s1'].get(i, []) + VR['train']['s2'].get(i, []) + S3.get(i, []) if ref else []
    print(i, 'label', tr[i], 'rule', ref, '32B', ''.join(b), '-> decide', decide(i, ref or '?', ref, v14, b))
cur = pd.read_csv('out/v3/submission_v4_th0.8.csv').set_index('id')['label']
new = cur.copy()
R = VR['test']
for i, b in B['test'].items():
    ref = R['ref'].get(i)
    v14 = (R['s1'].get(i, []) + R['s2'].get(i, []) + S3.get(i, [])) if ref else []
    new[i] = decide(i, cur[i], ref, v14, b)
print('v5 changes vs v4:', (new != cur).sum(), '| 32B unanimous on', sum(len(set(b)) == 1 and len(b) == 3 for b in B['test'].values()), 'of', len(B['test']))
new.rename('label').reset_index().to_csv('out/v3/submission_v5.csv', index=False)
