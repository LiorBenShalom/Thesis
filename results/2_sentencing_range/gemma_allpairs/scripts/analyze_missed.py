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

# gemma all-pairs neighbors (sorted desc), and filter pool
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

filt = collections.defaultdict(set)
for r in csv.DictReader(open(GPT, encoding="utf-8-sig")):
    if r.get("domain") != "weapon": continue
    a, b = r["verdict_1"], r["verdict_2"]
    if a in weapon and b in weapon:
        filt[a].add(b); filt[b].add(a)

N = len(weapon)
print(f"weapon queries: {N}")
print(f"avg filter pool/query: {np.mean([len(filt[q]) for q in weapon]):.0f}")
print(f"avg all-pairs candidates/query: {np.mean([len(pre.get(q,[])) for q in weapon]):.0f}\n")

for eps in (6, 12):
    has_good_anywhere = good_only_out = found_out_top10 = found_out_top50 = 0
    out_good_counts = []
    salvageable = 0   # filter has NO good twin, but Gemma top-10 contains an out-of-filter good twin
    for q in weapon:
        nbrs = pre.get(q, [])
        rank = {c: i for i, (c, _) in enumerate(nbrs)}
        fset = filt.get(q, set())
        gi = go = 0
        out_in_top10 = False
        for c, _ in nbrs:
            if abs(lo[c]-lo[q]) <= eps and abs(hi[c]-hi[q]) <= eps:   # "good twin": both bounds within eps
                if c in fset:
                    gi += 1
                else:
                    go += 1
                    if rank[c] < 10: out_in_top10 = True
        out_good_counts.append(go)
        if gi + go > 0: has_good_anywhere += 1
        if go > 0 and gi == 0: good_only_out += 1
        if out_in_top10: found_out_top10 += 1
        if go > 0 and gi == 0 and out_in_top10: salvageable += 1
    pct = lambda x: f"{x} ({100*x/N:.0f}%)"
    print(f"=== good twin = both bounds within +/-{eps} months ===")
    print(f"  queries with >=1 good twin (anywhere):        {pct(has_good_anywhere)}")
    print(f"  median good twins OUTSIDE filter per query:    {int(np.median(out_good_counts))}  (mean {np.mean(out_good_counts):.1f})")
    print(f"  queries whose ONLY good twins are out-of-filter: {pct(good_only_out)}  <-- filter misses them")
    print(f"  queries where Gemma top-10 has an out-filter good twin: {pct(found_out_top10)}")
    print(f"  SALVAGEABLE (no good twin in filter, but Gemma top-10 finds one outside): {pct(salvageable)}\n")
