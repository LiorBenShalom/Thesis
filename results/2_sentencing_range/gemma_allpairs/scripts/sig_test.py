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

def load(path):
    best = collections.defaultdict(dict)
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        s = r.get("similarity_score")
        if not r.get("verdict_1") or s in (None, ""): continue
        try: s = float(s)
        except ValueError: continue
        a, b = r["verdict_1"], r["verdict_2"]
        if a in weapon and b in weapon:
            if s > best[a].get(b, -1): best[a][b] = s
            if s > best[b].get(a, -1): best[b][a] = s
    return best
gem = load(GEM); gpt = load(GPT)
gem_pool = collections.defaultdict(dict)
for q in weapon:
    al = gpt.get(q, {})
    for c, s in gem.get(q, {}).items():
        if c in al: gem_pool[q][c] = s

def errs(b, K=10):
    pre = {q: sorted(d.items(), key=lambda x: -x[1]) for q, d in b.items()}
    e = {}
    for q in weapon:
        cs = [c for c, _ in pre.get(q, [])[:K]]
        if not cs: continue
        e[q] = (abs(np.median([lo[c] for c in cs]) - lo[q]),
                abs(np.median([hi[c] for c in cs]) - hi[q]))
    return e

ea = errs(gem); ep = errs(gem_pool)
common = sorted(set(ea) & set(ep))
print(f"paired on n={len(common)} weapon cases (K=10, same model=Gemma)\n")

alo = np.array([ea[q][0] for q in common]); ahi = np.array([ea[q][1] for q in common])
plo = np.array([ep[q][0] for q in common]); phi = np.array([ep[q][1] for q in common])
print(f"Gemma all-pairs : MAE_lo={alo.mean():.2f}  MAE_hi={ahi.mean():.2f}")
print(f"Gemma on filter : MAE_lo={plo.mean():.2f}  MAE_hi={phi.mean():.2f}")
print(f"improvement (all - filter): lo={alo.mean()-plo.mean():+.2f}  hi={ahi.mean()-phi.mean():+.2f}\n")

# paired bootstrap of the mean difference (all-pairs minus filter); >0 => filter better
rng = np.random.default_rng(0)
dlo = alo - plo; dhi = ahi - phi
def boot(d, n=10000):
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])
clo = boot(dlo); chi = boot(dhi)
print(f"mean diff lo = {dlo.mean():+.2f}  95% CI [{clo[0]:+.2f}, {clo[1]:+.2f}]  -> {'SIGNIFICANT' if clo[0]>0 else 'not significant'}")
print(f"mean diff hi = {dhi.mean():+.2f}  95% CI [{chi[0]:+.2f}, {chi[1]:+.2f}]  -> {'SIGNIFICANT' if chi[0]>0 else 'not significant'}")
# wilcoxon
from scipy.stats import wilcoxon
print(f"\nWilcoxon lo p={wilcoxon(alo, plo).pvalue:.2e}   hi p={wilcoxon(ahi, phi).pvalue:.2e}")
