"""
THESIS STORY — Part 1: What IS the variance in sentencing range in reality?

We use two proxies for "legally similar cases":
  (A) LLM-rated similar pairs   — what's |Δ_sentencing| as a function of LLM score?
  (B) Citation-connected pairs  — what's |Δ_sentencing| as a function of citation type?

No K filter, no sigma — just raw distributions.
The hypothesis: legally-similar cases have smaller sentencing gaps.
If true, this validates that there IS a signal in case similarity → sentencing.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
EXP  = ROOT / "experiments"

# Load sentencing ranges (the verdicts we care about: drugs+weapon, valid range, high confidence)
m = pd.read_csv(EXP / "data_per_domain/master_inventory.csv",
                usecols=["canonical_id","domain","sentencing_range_low",
                         "sentencing_range_high","sentencing_confidence"])
m = m[m.domain.isin(["drugs","weapon"])
      & m.sentencing_range_low.notna()
      & (m.sentencing_confidence == "גבוהה")].drop_duplicates("canonical_id")
rng_lo = dict(zip(m.canonical_id, m.sentencing_range_low))
rng_hi = dict(zip(m.canonical_id, m.sentencing_range_high))
dom_of = dict(zip(m.canonical_id, m.domain))
print(f"Valid verdicts (drugs+weapon, high-conf range): {len(rng_lo):,}")

# Reference: variance of sentencing across all verdicts (no similarity assumption)
all_lo = [v for v in rng_lo.values()]
all_hi = [v for v in rng_hi.values()]
print(f"  ALL verdicts: low mean={np.mean(all_lo):.1f} std={np.std(all_lo):.1f} | "
      f"high mean={np.mean(all_hi):.1f} std={np.std(all_hi):.1f}")
# By domain
for dom in ("drugs", "weapon"):
    sub = m[m.domain == dom]
    print(f"  {dom}: low mean={sub.sentencing_range_low.mean():.1f} std={sub.sentencing_range_low.std():.1f}  "
          f"| high mean={sub.sentencing_range_high.mean():.1f} std={sub.sentencing_range_high.std():.1f}")


# ===== Load all LLM pairs (375K) =====
print(f"\nLoading LLM scored pairs...")
llm_pairs = []
for path in [
    EXP / "data_per_domain/similarity_scores_combined.csv",
    EXP / "data_per_domain/similarity_batch_5fold/results/similarity_scores_5fold.csv",
    EXP / "data_per_domain/similarity_batch_simcse/results/similarity_scores_simcse.csv",
    EXP / "data_per_domain/similarity_batch_supervised/results/similarity_scores_supervised.csv",
    EXP / "data_per_domain/similarity_batch_5fold_v2/results/similarity_scores_5fold_v2.csv",
    EXP / "data_per_domain/similarity_batch_filtered/results/similarity_scores_filtered.csv",
]:
    if not path.exists(): continue
    df = pd.read_csv(path)
    for r in df.itertuples(index=False):
        if pd.notna(r.similarity_score):
            llm_pairs.append((r.verdict_1, r.verdict_2, float(r.similarity_score), r.domain))
print(f"  total LLM-scored rows: {len(llm_pairs):,}")

# Compute |Δ_low|, |Δ_high| per pair, keep only pairs where BOTH verdicts have range
rows = []
seen = set()
for v1, v2, score, dom in llm_pairs:
    if v1 not in rng_lo or v2 not in rng_lo: continue
    a, b = sorted([v1, v2])
    if (a, b) in seen: continue
    seen.add((a, b))
    rows.append({
        "v1": a, "v2": b, "domain": dom, "llm_score": score,
        "d_lo": abs(rng_lo[a] - rng_lo[b]),
        "d_hi": abs(rng_hi[a] - rng_hi[b]),
    })
df_llm = pd.DataFrame(rows)
print(f"  unique pairs with sentencing data: {len(df_llm):,}")


# ===== Bucket by LLM score & report variance =====
print(f"\n{'='*70}\n A. SENTENCING GAP vs LLM SIMILARITY SCORE (raw, all pairs)\n{'='*70}")
buckets = [(0, 25), (25, 50), (50, 75), (75, 90), (90, 101)]
for dom in ("drugs", "weapon"):
    sub_dom = df_llm[df_llm.domain == dom]
    print(f"\n  {dom.upper()}  ({len(sub_dom):,} pairs)")
    print(f"    {'LLM bucket':>12s}  {'n_pairs':>9s}  {'|Δlow| mean':>13s}  "
          f"{'|Δhi| mean':>12s}  {'|Δlow| median':>14s}  {'|Δhi| median':>13s}")
    for lo_b, hi_b in buckets:
        sub = sub_dom[(sub_dom.llm_score >= lo_b) & (sub_dom.llm_score < hi_b)]
        if len(sub) == 0: continue
        print(f"    {f'{lo_b}-{hi_b-1}':>12s}  {len(sub):>9,}  "
              f"{sub.d_lo.mean():>11.2f}    {sub.d_hi.mean():>10.2f}    "
              f"{sub.d_lo.median():>12.2f}    {sub.d_hi.median():>11.2f}")


# ===== Citation pairs (no LLM) =====
print(f"\n{'='*70}\n B. SENTENCING GAP vs CITATION TYPE (raw)\n{'='*70}")
cit_df = pd.read_csv(EXP / "data_per_domain/network_analysis/citation_pair_types.csv")
rows = []
seen = set()
for r in cit_df.itertuples(index=False):
    v1, v2 = r.verdict_1, r.verdict_2
    if v1 not in rng_lo or v2 not in rng_lo: continue
    a, b = sorted([v1, v2])
    if (a, b) in seen: continue
    seen.add((a, b))
    rows.append({
        "v1": a, "v2": b, "domain": r.domain, "cit_type": r.citation_type,
        "d_lo": abs(rng_lo[a] - rng_lo[b]),
        "d_hi": abs(rng_hi[a] - rng_hi[b]),
    })
df_cit = pd.DataFrame(rows)
print(f"  citation pairs with sentencing: {len(df_cit):,}")

for dom in ("drugs", "weapon"):
    sub_dom = df_cit[df_cit.domain == dom]
    print(f"\n  {dom.upper()}  ({len(sub_dom):,} pairs)")
    print(f"    {'cit_type':>12s}  {'n_pairs':>9s}  {'|Δlow| mean':>13s}  "
          f"{'|Δhi| mean':>12s}  {'|Δlow| median':>14s}  {'|Δhi| median':>13s}")
    for ct in ("1hop", "2hop", "cocite", "none"):
        sub = sub_dom[sub_dom.cit_type == ct]
        if len(sub) == 0: continue
        print(f"    {ct:>12s}  {len(sub):>9,}  "
              f"{sub.d_lo.mean():>11.2f}    {sub.d_hi.mean():>10.2f}    "
              f"{sub.d_lo.median():>12.2f}    {sub.d_hi.median():>11.2f}")


# ===== Random baseline = EXACT mean |Δ| over ALL C(n,2) pairs =====
# (Gini mean difference). Replaces the earlier 50K Monte-Carlo sample —
# the exact value is deterministic, fast (vectorized broadcasting), and
# removes the "why only 50K?" question. Verified: 50K sample == exact ±0.1mo.
print(f"\n{'='*70}\n C. RANDOM BASELINE — EXACT mean over ALL C(n,2) pairs\n{'='*70}")
for dom in ("drugs", "weapon"):
    vs = [v for v in rng_lo if dom_of.get(v) == dom]
    lo = np.array([rng_lo[v] for v in vs], dtype=float)
    hi = np.array([rng_hi[v] for v in vs], dtype=float)
    n = len(lo)
    total_pairs = n * (n - 1) // 2
    # sum_{i<j} |x_i - x_j| via broadcasting; matrix counts both i<j and i>j → /2
    exact_lo = np.abs(lo[:, None] - lo[None, :]).sum() / 2.0 / total_pairs
    exact_hi = np.abs(hi[:, None] - hi[None, :]).sum() / 2.0 / total_pairs
    print(f"\n  {dom.upper()} ALL pairs (n={n} verdicts, C(n,2)={total_pairs:,}):")
    print(f"    |Δlow|  mean={exact_lo:.4f}")
    print(f"    |Δhigh| mean={exact_hi:.4f}")

df_llm.to_csv("/tmp/story_llm_gaps.csv", index=False)
df_cit.to_csv("/tmp/story_citation_gaps.csv", index=False)
print(f"\n✅ Saved /tmp/story_llm_gaps.csv and /tmp/story_citation_gaps.csv")
