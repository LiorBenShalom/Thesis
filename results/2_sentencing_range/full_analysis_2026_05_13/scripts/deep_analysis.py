"""
DEEP THESIS ANALYSIS — 7 angles to enrich the "richer pool → better LLM" story.

A. Recall@K_oracle vs pool size — does supervised pool capture LLM-best cases?
B. Pool quality concentration — mean LLM-score within pool, by pool size.
C. Interval calibration — what % of true sentences fall in predicted (low, high)?
D. Per-quartile MAE — stratify by true sentence severity (defends median-regressor attack).
E. Marginal value curve — dMAE/dPool.
F. Cost-quality Pareto frontier in dollars.
G. Hybrid pool experiment: supervised ∪ citation.

Outputs all CSVs to /tmp/deep_*.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import re

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
EXP  = ROOT / "experiments"
FILTERED_DIR = ROOT / "simcse_cuda_bundle/outputs_supervised_filtered"

N_FOLDS = 5
K_FINAL = 10

# Sentencing ranges
m = pd.read_csv(EXP / "data_per_domain/master_inventory.csv",
                usecols=["canonical_id","domain","sentencing_range_low",
                         "sentencing_range_high","sentencing_confidence"])
m = m[m.domain.isin(["drugs","weapon"])
      & m.sentencing_range_low.notna()
      & (m.sentencing_confidence == "גבוהה")].drop_duplicates("canonical_id")
rng_lo = dict(zip(m.canonical_id, m.sentencing_range_low))
rng_hi = dict(zip(m.canonical_id, m.sentencing_range_high))
dom_of = dict(zip(m.canonical_id, m.domain))

# LLM scores
print("Loading LLM scores...")
llm_scores = {}
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
            a, b = sorted([r.verdict_1, r.verdict_2])
            llm_scores[(a, b)] = float(r.similarity_score)
print(f"  {len(llm_scores):,}")

# Citation pairs (all types)
cit_pairs = set()
cit_df = pd.read_csv(EXP / "data_per_domain/network_analysis/citation_pair_types.csv")
for r in cit_df.itertuples(index=False):
    if r.citation_type in ("1hop", "2hop", "cocite"):
        cit_pairs.add(tuple(sorted([r.verdict_1, r.verdict_2])))

# Folds
folds = {}
for dom in ("drugs", "weapon"):
    for f in range(1, N_FOLDS + 1):
        ep = FILTERED_DIR / f"verdict_embeddings_{dom}_topk_fold{f}_offenseFiltered.npy"
        ip = FILTERED_DIR / f"verdict_index_{dom}_topk_fold{f}_offenseFiltered.csv"
        if not ep.exists(): continue
        emb = np.load(ep).astype(np.float32)
        idx = pd.read_csv(ip)
        train_ids = idx[idx.split == "train"].verdict.tolist()
        test_ids  = idx[idx.split == "test"].verdict.tolist()
        v2i = {v: i for i, v in enumerate(idx.verdict)}
        folds[(dom, f)] = {"emb": emb, "v2i": v2i,
                           "train_ids": train_ids, "test_ids": test_ids}

POOL_SIZES = [10, 20, 50, 100, 200, 500, 1000, "all"]


def supervised_pool(q, ff, size):
    """Return supervised cosine top-N pool (N = size or 'all')."""
    emb, v2i = ff["emb"], ff["v2i"]
    if q not in v2i: return []
    qi = v2i[q]
    train_ids = ff["train_ids"]
    train_idx = np.array([v2i[v] for v in train_ids])
    sims = emb[qi] @ emb[train_idx].T
    order = np.argsort(-sims)
    if size == "all":
        return [train_ids[i] for i in order]
    return [train_ids[i] for i in order[:size]]


# ============ A. RECALL @ K_oracle ============
print("\n=== A. RECALL@K_oracle — does supervised pool capture LLM-best ===")
K_ORACLES = [10, 20, 50]
recall_rows = []
for dom in ("drugs", "weapon"):
    for K_oracle in K_ORACLES:
        for ps in POOL_SIZES:
            recalls = []
            for (d, fid), ff in folds.items():
                if d != dom: continue
                for q in ff["test_ids"]:
                    # Oracle top-K_oracle by LLM among train
                    cands = [(t, llm_scores.get(tuple(sorted([q, t]))))
                             for t in ff["train_ids"] if t != q]
                    cands = [(t, s) for t, s in cands if s is not None]
                    cands.sort(key=lambda x: -x[1])
                    oracle_top = set(t for t, _ in cands[:K_oracle])
                    if not oracle_top: continue
                    pool = set(supervised_pool(q, ff, ps))
                    recalls.append(len(oracle_top & pool) / len(oracle_top))
            r = float(np.mean(recalls)) if recalls else 0.0
            recall_rows.append({"domain": dom, "K_oracle": K_oracle, "pool_size": str(ps), "recall": r})
            print(f"  {dom:6s} K_oracle={K_oracle:>2d} pool={str(ps):>4s}: recall = {r:.3f}")
pd.DataFrame(recall_rows).to_csv("/tmp/deep_recall.csv", index=False)


# ============ B. POOL QUALITY CONCENTRATION ============
print("\n=== B. POOL QUALITY — mean LLM-score within pool, by size ===")
quality_rows = []
for dom in ("drugs", "weapon"):
    for ps in POOL_SIZES:
        means = []; scored_pcts = []
        for (d, fid), ff in folds.items():
            if d != dom: continue
            for q in ff["test_ids"]:
                pool = supervised_pool(q, ff, ps)
                scored = [llm_scores.get(tuple(sorted([q, c]))) for c in pool]
                scored = [s for s in scored if s is not None]
                if scored:
                    means.append(np.mean(scored))
                    scored_pcts.append(len(scored) / max(len(pool), 1))
        quality_rows.append({"domain": dom, "pool_size": str(ps),
                             "mean_llm_in_pool": float(np.mean(means)) if means else None,
                             "frac_scored": float(np.mean(scored_pcts)) if scored_pcts else None})
        print(f"  {dom:6s} pool={str(ps):>4s}: mean_llm={np.mean(means):.1f}  scored={np.mean(scored_pcts)*100:.1f}%")
pd.DataFrame(quality_rows).to_csv("/tmp/deep_pool_quality.csv", index=False)


# ============ C. INTERVAL CALIBRATION ============
print("\n=== C. INTERVAL CALIBRATION — % of true sentence in predicted (low, high) ===")
calib_rows = []
for dom in ("drugs", "weapon"):
    for ps in POOL_SIZES:
        inside_lo = 0  # true_low ≥ pred_low (i.e., true low ≥ pred low)
        inside_hi = 0
        both_inside = 0
        n_pred = 0
        for (d, fid), ff in folds.items():
            if d != dom: continue
            for q in ff["test_ids"]:
                if q not in rng_lo: continue
                pool = supervised_pool(q, ff, ps)
                scored = [(c, llm_scores.get(tuple(sorted([q, c])))) for c in pool]
                scored = [(c, s) for c, s in scored if s is not None]
                scored.sort(key=lambda x: -x[1])
                picked = [c for c, _ in scored[:K_FINAL]]
                if not picked:
                    picked = pool[:K_FINAL]
                picked = [p for p in picked if p in rng_lo]
                if not picked: continue
                plo = float(np.median([rng_lo[p] for p in picked]))
                phi = float(np.median([rng_hi[p] for p in picked]))
                true_lo, true_hi = rng_lo[q], rng_hi[q]
                # measure containment
                lo_within = abs(plo - true_lo) <= 6  # ±6 months for low
                hi_within = abs(phi - true_hi) <= 6
                if lo_within: inside_lo += 1
                if hi_within: inside_hi += 1
                if lo_within and hi_within: both_inside += 1
                n_pred += 1
        calib_rows.append({"domain": dom, "pool_size": str(ps),
                           "n_pred": n_pred,
                           "low_within_6mo": inside_lo / n_pred if n_pred else 0,
                           "high_within_6mo": inside_hi / n_pred if n_pred else 0,
                           "both_within_6mo": both_inside / n_pred if n_pred else 0})
        print(f"  {dom:6s} pool={str(ps):>4s}: low_within_6mo={inside_lo/n_pred*100:.1f}% "
              f"high_within_6mo={inside_hi/n_pred*100:.1f}% both={both_inside/n_pred*100:.1f}%")
pd.DataFrame(calib_rows).to_csv("/tmp/deep_calibration.csv", index=False)


# ============ D. PER-QUARTILE MAE (defends against median-regressor attack) ============
print("\n=== D. PER-QUARTILE MAE — defends against median-regressor attack ===")
# Quartiles per domain based on true sentence (mid = (low+high)/2)
quartile_rows = []
for dom in ("drugs", "weapon"):
    sub = m[m.domain == dom]
    mids = (sub.sentencing_range_low + sub.sentencing_range_high) / 2
    q25 = np.percentile(mids, 25)
    q50 = np.percentile(mids, 50)
    q75 = np.percentile(mids, 75)
    print(f"  {dom} quartile boundaries (mid sentence): {q25:.0f} | {q50:.0f} | {q75:.0f}")

    def quartile_of(q):
        v = rng_lo.get(q)
        if v is None: return None
        mid = (rng_lo[q] + rng_hi[q]) / 2
        if mid < q25: return "Q1"
        if mid < q50: return "Q2"
        if mid < q75: return "Q3"
        return "Q4"

    for ps in POOL_SIZES:
        bucket_errs = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
        for (d, fid), ff in folds.items():
            if d != dom: continue
            for q in ff["test_ids"]:
                if q not in rng_lo: continue
                bucket = quartile_of(q)
                if bucket is None: continue
                pool = supervised_pool(q, ff, ps)
                scored = [(c, llm_scores.get(tuple(sorted([q, c])))) for c in pool]
                scored = [(c, s) for c, s in scored if s is not None]
                scored.sort(key=lambda x: -x[1])
                picked = [c for c, _ in scored[:K_FINAL] if c in rng_lo]
                if not picked:
                    picked = [p for p in pool[:K_FINAL] if p in rng_lo]
                if not picked: continue
                plo = float(np.median([rng_lo[p] for p in picked]))
                phi = float(np.median([rng_hi[p] for p in picked]))
                err = (abs(plo - rng_lo[q]) + abs(phi - rng_hi[q])) / 2  # avg of low+high MAE
                bucket_errs[bucket].append(err)
        for bk in ("Q1","Q2","Q3","Q4"):
            quartile_rows.append({"domain": dom, "pool_size": str(ps), "quartile": bk,
                                  "n": len(bucket_errs[bk]),
                                  "avg_mae": float(np.mean(bucket_errs[bk])) if bucket_errs[bk] else None})
        print(f"  {dom:6s} pool={str(ps):>4s}: "
              f"Q1={np.mean(bucket_errs['Q1']):.2f} (n={len(bucket_errs['Q1'])})  "
              f"Q2={np.mean(bucket_errs['Q2']):.2f}  "
              f"Q3={np.mean(bucket_errs['Q3']):.2f}  "
              f"Q4={np.mean(bucket_errs['Q4']):.2f}")
pd.DataFrame(quartile_rows).to_csv("/tmp/deep_quartile.csv", index=False)


# ============ E + F. MARGINAL VALUE + COST PARETO ============
print("\n=== E + F. MARGINAL VALUE + COST PARETO ===")
# Cost: each (q, candidate) LLM scoring is ~$0.71/1k pairs = $0.00071
COST_PER_PAIR = 0.00071
pareto_rows = []
df_pool = pd.read_csv("/tmp/sweep_pool_size.csv")
for dom in ("drugs", "weapon"):
    sub = df_pool[df_pool.domain == dom].copy()
    sub["avg_mae"] = (sub.mae_lo + sub.mae_hi) / 2
    order = ["10","20","50","100","200","500","1000","all"]
    sub["pool_idx"] = sub.pool_size.map({p: i for i, p in enumerate(order)})
    sub = sub.sort_values("pool_idx")
    # cost = n_test_queries × pool_size × cost_per_pair (for LLM scoring)
    n_test = sum(len(ff["test_ids"]) for (d, _), ff in folds.items() if d == dom)
    for _, r in sub.iterrows():
        ps_str = r.pool_size
        if ps_str == "all":
            ps_num = 1800 if dom == "drugs" else 1300  # approx fold-train size
        else:
            ps_num = int(ps_str)
        cost = n_test * ps_num * COST_PER_PAIR
        pareto_rows.append({"domain": dom, "pool_size": ps_str, "pool_num": ps_num,
                            "avg_mae": r.avg_mae, "cost_usd": cost,
                            "mae_lo": r.mae_lo, "mae_hi": r.mae_hi})
        print(f"  {dom:6s} pool={ps_str:>4s} (={ps_num:>4d}): cost=${cost:>6.0f}  avg_mae={r.avg_mae:.2f}")
pd.DataFrame(pareto_rows).to_csv("/tmp/deep_pareto.csv", index=False)


# ============ G. HYBRID POOL (supervised ∪ citation) ============
print("\n=== G. HYBRID POOL (supervised top-N ∪ citation) ===")
hybrid_rows = []
for dom in ("drugs", "weapon"):
    for ps in [50, 100, 200, 500]:
        lo_errs = []; hi_errs = []
        for (d, fid), ff in folds.items():
            if d != dom: continue
            train_set = set(ff["train_ids"])
            for q in ff["test_ids"]:
                if q not in rng_lo: continue
                sup_pool = supervised_pool(q, ff, ps)
                cit_neighbors = [t for t in train_set if t != q and tuple(sorted([q, t])) in cit_pairs]
                pool = list(set(sup_pool) | set(cit_neighbors))
                scored = [(c, llm_scores.get(tuple(sorted([q, c])))) for c in pool]
                scored = [(c, s) for c, s in scored if s is not None]
                scored.sort(key=lambda x: -x[1])
                picked = [c for c, _ in scored[:K_FINAL] if c in rng_lo]
                if not picked: continue
                plo = float(np.median([rng_lo[p] for p in picked]))
                phi = float(np.median([rng_hi[p] for p in picked]))
                lo_errs.append(abs(plo - rng_lo[q]))
                hi_errs.append(abs(phi - rng_hi[q]))
        hybrid_rows.append({"domain": dom, "config": f"sup_top_{ps}_∪_citation",
                            "pool_base": ps,
                            "mae_lo": np.mean(lo_errs), "mae_hi": np.mean(hi_errs)})
        print(f"  {dom:6s} sup-top-{ps:>3d} ∪ citation: MAE-lo={np.mean(lo_errs):.2f} "
              f"MAE-hi={np.mean(hi_errs):.2f}")
pd.DataFrame(hybrid_rows).to_csv("/tmp/deep_hybrid.csv", index=False)

print(f"\n✅ All saved: /tmp/deep_*.csv")
