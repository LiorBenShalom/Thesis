import csv, collections
import numpy as np
import pandas as pd
csv.field_size_limit(10**9)
BASE = "/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try"
GEM = BASE + "/gemma_local_similarity/out/gemma_weapon_schema_FINAL.csv"
SUP = BASE + "/simcse_cuda_bundle/data/supervised_data.csv"

sup = pd.read_csv(SUP); w = sup[sup.domain == "weapon"]
lo = {r.verdict: float(r.sentencing_range_low) for r in w.itertuples(index=False)}
hi = {r.verdict: float(r.sentencing_range_high) for r in w.itertuples(index=False)}
weapon = set(lo)

best = collections.defaultdict(dict)
for r in csv.DictReader(open(GEM, encoding="utf-8-sig")):
    s = r.get("similarity_score")
    if not r.get("verdict_1") or s in (None, ""): continue
    try: s = float(s)
    except ValueError: continue
    a, b = r["verdict_1"], r["verdict_2"]
    if a in weapon and b in weapon:
        if s > best[a].get(b, -1): best[a][b] = s
        if s > best[b].get(a, -1): best[b][a] = s
pre = {q: sorted(d.items(), key=lambda x: -x[1]) for q, d in best.items()}

def wmedian(vals, wts):
    vals = np.asarray(vals, float); wts = np.asarray(wts, float)
    o = np.argsort(vals); v = vals[o]; cw = np.cumsum(wts[o])
    return float(v[np.searchsorted(cw, cw[-1] / 2.0)])

K_LIST = [10, 20, 30, 50, 100, 10**9]
def kname(K): return "all" if K > 10**8 else str(K)

print("baseline plain median K=10:  MAE_lo=13.05  MAE_hi=18.71\n")
for mode in ("median", "mean"):
    for p in (1, 2, 4, 8):
        row = []
        for K in K_LIST:
            elo, ehi = [], []
            for q in weapon:
                items = pre.get(q, [])[:K]
                if not items: continue
                cs = [c for c, _ in items]
                wt = np.array([s for _, s in items]) ** p
                if mode == "median":
                    plo = wmedian([lo[c] for c in cs], wt)
                    phi = wmedian([hi[c] for c in cs], wt)
                else:
                    plo = np.average([lo[c] for c in cs], weights=wt)
                    phi = np.average([hi[c] for c in cs], weights=wt)
                elo.append(abs(plo - lo[q])); ehi.append(abs(phi - hi[q]))
            row.append(f"{kname(K)}:{np.mean(elo):.2f}/{np.mean(ehi):.2f}")
        print(f"w-{mode} score^{p}:  " + "  ".join(row))
    print()
