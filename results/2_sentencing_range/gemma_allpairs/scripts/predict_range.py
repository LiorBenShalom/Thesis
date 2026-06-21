import csv, collections
import numpy as np
import pandas as pd
csv.field_size_limit(10**9)
BASE = "/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try"
GEM = BASE + "/gemma_local_similarity/out/gemma_weapon_schema_FINAL.csv"
SUP = BASE + "/simcse_cuda_bundle/data/supervised_data.csv"
GPT = BASE + "/experiments/data_per_domain/similarity_scores_combined.csv"

# --- ground-truth weapon ranges (from supervised_data, local) ---
sup = pd.read_csv(SUP)
w = sup[sup.domain == "weapon"]
lo = {r.verdict: float(r.sentencing_range_low) for r in w.itertuples(index=False)}
hi = {r.verdict: float(r.sentencing_range_high) for r in w.itertuples(index=False)}
weapon = set(lo)
print(f"weapon cases with range: {len(weapon)}")

def load(path):
    """q -> dict(cand->score), weapon pairs only, deduped (max score)."""
    best = collections.defaultdict(dict)
    n = 0
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        s = r.get("similarity_score")
        if not r.get("verdict_1") or s in (None, ""):
            continue
        try:
            s = float(s)
        except ValueError:
            continue
        a, b = r["verdict_1"], r["verdict_2"]
        if a in weapon and b in weapon:
            if s > best[a].get(b, -1):
                best[a][b] = s
            if s > best[b].get(a, -1):
                best[b][a] = s
            n += 1
    return best, n

K_VALUES = [1, 3, 5, 7, 10, 15, 20, 30, 50]

def evaluate(best):
    pre = {q: sorted(d.items(), key=lambda x: -x[1]) for q, d in best.items()}
    out = {}
    for K in K_VALUES:
        elo, ehi = [], []
        for q in weapon:
            cands = [c for c, _ in pre.get(q, [])[:K]]
            if not cands:
                continue
            plo = float(np.median([lo[c] for c in cands]))
            phi = float(np.median([hi[c] for c in cands]))
            elo.append(abs(plo - lo[q]))
            ehi.append(abs(phi - hi[q]))
        out[K] = (np.mean(elo), np.mean(ehi), len(elo))
    return out, pre

print("loading scores...")
gem_b, gem_n = load(GEM)
gpt_b, gpt_n = load(GPT)
gem_cov = np.mean([len(d) for d in gem_b.values()]) if gem_b else 0
gpt_cov = np.mean([len(gpt_b.get(q, {})) for q in weapon])
print(f"gemma pairs={gem_n:,}  avg candidates/query={gem_cov:.0f}")
print(f"gpt   pairs={gpt_n:,}  avg candidates/query={gpt_cov:.1f}")

for name, b in [("GPT-filtered", gpt_b), ("Gemma all-pairs", gem_b)]:
    res, _ = evaluate(b)
    print(f"\n=== {name} (leave-one-out kNN, weapon) ===")
    print("  K   MAE_lo  MAE_hi   queries_covered")
    for K in K_VALUES:
        mlo, mhi, ncov = res[K]
        print(f"{K:4d}   {mlo:5.2f}   {mhi:5.2f}    {ncov}/{len(weapon)}")

# --- diagnostic: Gemma scores but ONLY on GPT's filtered candidate pairs ---
# (same candidate pool as GPT → isolates model quality from all-pairs coverage)
gem_filt = collections.defaultdict(dict)
for q in weapon:
    allowed = gpt_b.get(q, {})
    for c, s in gem_b.get(q, {}).items():
        if c in allowed:
            gem_filt[q][c] = s
res, _ = evaluate(gem_filt)
print("\n=== Gemma on GPT-filtered candidates (same pool as GPT; isolates model quality) ===")
print("  K   MAE_lo  MAE_hi   queries_covered")
for K in K_VALUES:
    mlo, mhi, ncov = res[K]
    print(f"{K:4d}   {mlo:5.2f}   {mhi:5.2f}    {ncov}/{len(weapon)}")

# --- DIRECT TEST of the all-pairs hypothesis: predict from Gemma's twins the filter EXCLUDED ---
gem_out = collections.defaultdict(dict)
for q in weapon:
    infilter = gpt_b.get(q, {})
    for c, s in gem_b.get(q, {}).items():
        if c not in infilter:        # "missed twins": Gemma-similar but NOT in the filter
            gem_out[q][c] = s
res, _ = evaluate(gem_out)
print("\n=== Gemma OUT-OF-FILTER twins only (the twins the filter 'missed') ===")
print("  K   MAE_lo  MAE_hi   queries_covered")
for K in K_VALUES:
    mlo, mhi, ncov = res[K]
    print(f"{K:4d}   {mlo:5.2f}   {mhi:5.2f}    {ncov}/{len(weapon)}")

# headline best-K (by MAE_lo) for each
print("\n=== BEST-K summary vs GPT table (table LLM-best weapon: lo=12.12 hi=17.42) ===")
for name, b in [("GPT-filtered", gpt_b), ("Gemma all-pairs", gem_b), ("Gemma on GPT pool", gem_filt)]:
    res, _ = evaluate(b)
    bestK = min(K_VALUES, key=lambda K: res[K][0])
    mlo, mhi, ncov = res[bestK]
    print(f"{name:20s} best K={bestK:3d}: MAE_lo={mlo:.2f}  MAE_hi={mhi:.2f}  (cov {ncov}/{len(weapon)})")
