"""
Compare offense-filtered model (NEW) vs unfiltered supervised model (BASELINE):
  1. 5-fold MAE (k-NN sentencing range prediction)
  2. Spearman vs human-similarity GT
  3. Overlap with citation pool

Outputs:
  /tmp/comparison_5fold_mae.csv
  /tmp/comparison_spearman.csv
  /tmp/comparison_citation_overlap.csv
"""
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
EXP  = ROOT / "experiments"
BASELINE_DIR = EXP / "simcse_outputs/supervised"
FILTERED_DIR = ROOT / "simcse_cuda_bundle/outputs_supervised_filtered"

N_FOLDS = 5
K_VALUES = [3, 5, 10, 20]


def median_pred(picked, range_low, range_high):
    if not picked: return None, None
    return (float(np.median([range_low[p]  for p in picked])),
            float(np.median([range_high[p] for p in picked])))


def load_data():
    m = pd.read_csv(EXP / "data_per_domain/master_inventory.csv",
                    usecols=["canonical_id","domain","sentencing_range_low",
                             "sentencing_range_high","sentencing_confidence"])
    m = m[m.domain.isin(["drugs","weapon"])
          & m.sentencing_range_low.notna()
          & (m.sentencing_confidence == "גבוהה")].drop_duplicates("canonical_id")
    rng_lo = dict(zip(m.canonical_id, m.sentencing_range_low))
    rng_hi = dict(zip(m.canonical_id, m.sentencing_range_high))
    return rng_lo, rng_hi


def load_folds(emb_dir, suffix=""):
    """Load embeddings + index for all 5 folds × 2 domains.
       suffix: '' for baseline, '_offenseFiltered' for new model."""
    folds = {}
    for dom in ("drugs", "weapon"):
        for f in range(1, N_FOLDS + 1):
            ep = emb_dir / f"verdict_embeddings_{dom}_topk_fold{f}{suffix}.npy"
            ip = emb_dir / f"verdict_index_{dom}_topk_fold{f}{suffix}.csv"
            if not ep.exists() or not ip.exists():
                print(f"  ⚠ missing: {ep.name}")
                continue
            emb = np.load(ep).astype(np.float32)
            idx = pd.read_csv(ip)
            train_ids = idx[idx.split == "train"].verdict.tolist()
            test_ids  = idx[idx.split == "test"].verdict.tolist()
            v2i = {v: i for i, v in enumerate(idx.verdict)}
            folds[(dom, f)] = {"emb": emb, "v2i": v2i,
                               "train_ids": train_ids, "test_ids": test_ids}
    return folds


def predict_mae(folds, dom, K, range_low, range_high):
    """5-fold k-NN MAE: median over top-K cosine neighbors from fold-train."""
    rows = []
    for f in range(1, N_FOLDS + 1):
        if (dom, f) not in folds: continue
        ff = folds[(dom, f)]
        emb, v2i = ff["emb"], ff["v2i"]
        train_ids, test_ids = ff["train_ids"], ff["test_ids"]
        train_idx = np.array([v2i[v] for v in train_ids])
        for q in test_ids:
            if q not in v2i or q not in range_low: continue
            qi = v2i[q]
            sims = emb[qi] @ emb[train_idx].T
            top_idx = np.argsort(-sims)[:K]
            picked = [train_ids[i] for i in top_idx]
            picked = [p for p in picked if p in range_low]
            pred_lo, pred_hi = median_pred(picked, range_low, range_high)
            if pred_lo is None: continue
            rows.append({
                "qid": q, "fold": f, "domain": dom, "K": K,
                "true_lo": range_low[q], "true_hi": range_high[q],
                "lo_err": abs(pred_lo - range_low[q]),
                "hi_err": abs(pred_hi - range_high[q]),
            })
    return pd.DataFrame(rows)


def compute_mae_summary(name, folds, range_low, range_high):
    out = []
    for dom in ("drugs", "weapon"):
        for K in K_VALUES:
            df = predict_mae(folds, dom, K, range_low, range_high)
            if len(df) == 0: continue
            # per-fold MAE then mean ± std
            per_fold = df.groupby("fold").agg(
                lo_mae=("lo_err","mean"), hi_mae=("hi_err","mean")
            ).reset_index()
            out.append({
                "model": name, "domain": dom, "K": K,
                "lo_mae_mean": per_fold.lo_mae.mean(),
                "lo_mae_std":  per_fold.lo_mae.std(),
                "hi_mae_mean": per_fold.hi_mae.mean(),
                "hi_mae_std":  per_fold.hi_mae.std(),
                "avg_mae":     (per_fold.lo_mae.mean() + per_fold.hi_mae.mean()) / 2,
                "n_test":      len(df),
            })
    return pd.DataFrame(out)


# ===== 1. MAE Comparison =====
print("=" * 70)
print(" 1. 5-FOLD MAE COMPARISON")
print("=" * 70)
rng_lo, rng_hi = load_data()
print(f"  master inventory: {len(rng_lo):,} verdicts with valid range")

print(f"\n  Loading BASELINE folds (no filter)...")
folds_base = load_folds(BASELINE_DIR, suffix="")
print(f"    loaded {len(folds_base)} (domain, fold) tuples")

print(f"\n  Loading FILTERED folds (offense overlap + backfill cap=12mo)...")
folds_filt = load_folds(FILTERED_DIR, suffix="_offenseFiltered")
print(f"    loaded {len(folds_filt)} (domain, fold) tuples")

mae_base = compute_mae_summary("baseline (no filter)", folds_base, rng_lo, rng_hi)
mae_filt = compute_mae_summary("filtered (offense + backfill cap=12)", folds_filt, rng_lo, rng_hi)
mae_cmp  = pd.concat([mae_base, mae_filt], ignore_index=True)
mae_cmp.to_csv("/tmp/comparison_5fold_mae.csv", index=False)

print(f"\n  --- MAE per K (avg of low+high) ---")
print(f"  {'domain':8s} {'K':>3s}  {'baseline avg-MAE':>20s}  {'filtered avg-MAE':>20s}  {'delta':>8s}")
print("  " + "-" * 70)
for dom in ("drugs", "weapon"):
    for K in K_VALUES:
        b = mae_base[(mae_base.domain == dom) & (mae_base.K == K)]
        f = mae_filt[(mae_filt.domain == dom) & (mae_filt.K == K)]
        if len(b) == 0 or len(f) == 0: continue
        bvm, bvs = b.iloc[0].avg_mae, (b.iloc[0].lo_mae_std + b.iloc[0].hi_mae_std)/2
        fvm, fvs = f.iloc[0].avg_mae, (f.iloc[0].lo_mae_std + f.iloc[0].hi_mae_std)/2
        delta = fvm - bvm
        arrow = "↓" if delta < 0 else "↑"
        print(f"  {dom:8s} {K:>3d}  {bvm:>10.2f} ± {bvs:>5.2f}     {fvm:>10.2f} ± {fvs:>5.2f}    {delta:>+5.2f}{arrow}")


# ===== 2. Spearman vs human-similarity GT =====
print("\n" + "=" * 70)
print(" 2. SPEARMAN vs human-similarity GT")
print("=" * 70)

gt_files = {
    "drugs":  EXP / "data/final/drugs/facts.csv",
    "weapon": EXP / "data/final/wep/facts.csv",
}

def cosine_sim(emb1, emb2):
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-12))


def load_alias():
    p = ROOT / "innovation_submission/data_master_final/verdict_alias.csv"
    if not p.exists(): return {}
    al = pd.read_csv(p)
    m = {}
    for r in al.itertuples(index=False):
        m[r.original_id] = r.canonical_id
        m[r.hebrew_normalized] = r.canonical_id
        m[r.canonical_id] = r.canonical_id
    return m


alias = load_alias()
print(f"  loaded {len(alias):,} alias→canonical mappings")


def evaluate_spearman(name, folds, dom, gt_path):
    """For each GT pair, average cosine across all folds where both verdicts have embeddings."""
    if not gt_path.exists():
        return None
    gt = pd.read_csv(gt_path)
    gt = gt.dropna(subset=["similarity_scale"])

    sims = []
    gts = []
    for _, r in gt.iterrows():
        v1 = alias.get(r.verdict_1, r.verdict_1)
        v2 = alias.get(r.verdict_2, r.verdict_2)
        # average cosine across folds where both verdicts are present
        fold_sims = []
        for f in range(1, N_FOLDS + 1):
            if (dom, f) not in folds: continue
            ff = folds[(dom, f)]
            if v1 in ff["v2i"] and v2 in ff["v2i"]:
                emb = ff["emb"]
                fold_sims.append(cosine_sim(emb[ff["v2i"][v1]], emb[ff["v2i"][v2]]))
        if fold_sims:
            sims.append(np.mean(fold_sims))
            gts.append(r.similarity_scale)
    if len(sims) < 5: return None
    rho, p = spearmanr(sims, gts)
    return {"model": name, "domain": dom, "n_pairs": len(sims),
            "spearman_rho": float(rho), "p_value": float(p)}


sp_rows = []
for dom in ("drugs", "weapon"):
    gt_path = gt_files[dom]
    print(f"\n  {dom.upper()} GT pairs from {gt_path.name}")
    r_b = evaluate_spearman("baseline (no filter)", folds_base, dom, gt_path)
    r_f = evaluate_spearman("filtered", folds_filt, dom, gt_path)
    if r_b: sp_rows.append(r_b); print(f"    baseline:  ρ = {r_b['spearman_rho']:.3f}  (n={r_b['n_pairs']}, p={r_b['p_value']:.3g})")
    if r_f: sp_rows.append(r_f); print(f"    filtered:  ρ = {r_f['spearman_rho']:.3f}  (n={r_f['n_pairs']}, p={r_f['p_value']:.3g})")
    if r_b and r_f:
        delta = r_f['spearman_rho'] - r_b['spearman_rho']
        arrow = "↑" if delta > 0 else "↓"
        print(f"    Δρ = {delta:+.3f}{arrow}")

sp_df = pd.DataFrame(sp_rows)
sp_df.to_csv("/tmp/comparison_spearman.csv", index=False)


# ===== 3. Overlap with citation pool =====
print("\n" + "=" * 70)
print(" 3. OVERLAP WITH CITATION POOL")
print("=" * 70)

cit_pairs = set()
cit_df = pd.read_csv(EXP / "data_per_domain/network_analysis/citation_pair_types.csv")
for r in cit_df.itertuples(index=False):
    if r.citation_type in ("1hop", "2hop", "cocite"):
        a, b = sorted([r.verdict_1, r.verdict_2])
        cit_pairs.add((a, b))
print(f"  citation pool: {len(cit_pairs):,} pairs (1hop/2hop/cocite)")


def topk_overlap_with_cit(folds, dom, K=20):
    """For each test query, take top-K cosine neighbors from fold-train and check overlap with citation pool."""
    overlap_counts = []
    for f in range(1, N_FOLDS + 1):
        if (dom, f) not in folds: continue
        ff = folds[(dom, f)]
        emb, v2i = ff["emb"], ff["v2i"]
        train_ids, test_ids = ff["train_ids"], ff["test_ids"]
        train_idx = np.array([v2i[v] for v in train_ids])
        for q in test_ids:
            if q not in v2i: continue
            qi = v2i[q]
            sims = emb[qi] @ emb[train_idx].T
            top_idx = np.argsort(-sims)[:K]
            picked = [train_ids[i] for i in top_idx]
            in_cit = sum(1 for p in picked if tuple(sorted([q, p])) in cit_pairs)
            overlap_counts.append(in_cit / K)
    return float(np.mean(overlap_counts)) if overlap_counts else 0.0


ov_rows = []
for dom in ("drugs", "weapon"):
    print(f"\n  {dom.upper()}")
    for K in K_VALUES:
        b = topk_overlap_with_cit(folds_base, dom, K)
        f = topk_overlap_with_cit(folds_filt, dom, K)
        print(f"    K={K:>2d}:  baseline overlap = {b*100:>5.1f}%   filtered overlap = {f*100:>5.1f}%   delta = {(f-b)*100:+5.1f}pp")
        ov_rows.append({"domain": dom, "K": K, "baseline_overlap": b, "filtered_overlap": f, "delta_pp": (f-b)*100})

ov_df = pd.DataFrame(ov_rows)
ov_df.to_csv("/tmp/comparison_citation_overlap.csv", index=False)

print("\n✓ All comparisons saved to /tmp/comparison_*.csv")
