import csv, collections
import numpy as np
import pandas as pd
csv.field_size_limit(10**9)
BASE = "/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try"
GEM = BASE + "/gemma_local_similarity/out/gemma_weapon_schema_FINAL.csv"
SUP = BASE + "/simcse_cuda_bundle/data/supervised_data.csv"
GPT = BASE + "/experiments/data_per_domain/similarity_scores_combined.csv"

sup = pd.read_csv(SUP); w = sup[sup.domain == "weapon"]
lo = {r.verdict: float(r.sentencing_range_low) for r in w.itertuples(index=False)}
hi = {r.verdict: float(r.sentencing_range_high) for r in w.itertuples(index=False)}
weapon = set(lo)

def load(path, restrict_to=None):
    best = collections.defaultdict(dict)
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        s = r.get("similarity_score")
        if not r.get("verdict_1") or s in (None, ""): continue
        try: s = float(s)
        except ValueError: continue
        a, b = r["verdict_1"], r["verdict_2"]
        if a in weapon and b in weapon:
            if restrict_to is not None and (b not in restrict_to.get(a, ())):
                pass
            if s > best[a].get(b, -1): best[a][b] = s
            if s > best[b].get(a, -1): best[b][a] = s
    return best

gem = load(GEM)
gpt = load(GPT)
# gemma restricted to gpt's candidate pairs
gem_pool = collections.defaultdict(dict)
for q in weapon:
    allowed = gpt.get(q, {})
    for c, s in gem.get(q, {}).items():
        if c in allowed: gem_pool[q][c] = s

pre = {name: {q: sorted(d.items(), key=lambda x: -x[1]) for q, d in b.items()}
       for name, b in [("GPT", gpt), ("Gemma_all", gem), ("Gemma_pool", gem_pool)]}

K_VALUES = [1,3,5,7,10,15,20,30,50]
def covered(name, K):
    return {q for q in weapon if pre[name].get(q, [])[:K]}

# common query set = queries all three can predict at K=10
common = covered("GPT",10) & covered("Gemma_all",10) & covered("Gemma_pool",10)
print(f"weapon cases total: {len(weapon)}")
print(f"common query set (all 3 predict, K=10): {len(common)}\n")

def mae(name, K, qset):
    pr = pre[name]; elo, ehi = [], []
    for q in qset:
        cands = [c for c, _ in pr.get(q, [])[:K]]
        if not cands: continue
        elo.append(abs(np.median([lo[c] for c in cands]) - lo[q]))
        ehi.append(abs(np.median([hi[c] for c in cands]) - hi[q]))
    return np.mean(elo), np.mean(ehi)

print("=== SAME query set (n=%d), best-K per method ===" % len(common))
print(f"{'method':22s} {'bestK':>5} {'MAE_lo':>7} {'MAE_hi':>7}")
for name, label in [("GPT","GPT + filter"),("Gemma_all","Gemma all-pairs"),("Gemma_pool","Gemma on GPT pool")]:
    bestK = min(K_VALUES, key=lambda K: mae(name, K, common)[0])
    mlo, mhi = mae(name, bestK, common)
    print(f"{label:22s} {bestK:5d} {mlo:7.2f} {mhi:7.2f}")

print("\n=== same query set, fixed K=10 ===")
for name, label in [("GPT","GPT + filter"),("Gemma_all","Gemma all-pairs"),("Gemma_pool","Gemma on GPT pool")]:
    mlo, mhi = mae(name, 10, common)
    print(f"{label:22s} K=10   {mlo:7.2f} {mhi:7.2f}")
