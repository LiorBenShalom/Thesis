#!/usr/bin/env python3
"""
Ensemble experiment (5-fold CV): citation pool ∪ supervised_topk top-K.

Hypothesis: since citation and supervised pick orthogonal neighbors (~6-15% overlap),
union should improve over each alone.

For each test query in each fold:
  - get citation pool (any 1hop/2hop/cocite, in fold-train)
  - get supervised_topk top-K cosine (in fold-train)
  - UNION them, predict median of all neighbors' (low, high)
  - Compare to median of cit-only and sup-only

Also: weighted ensemble — average of cit-only prediction and sup-only prediction.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
EXP  = ROOT / "experiments"
OUT  = EXP / "results/2_sentencing_range/predictions/cv_5fold_ensemble.csv"

K_VALUES = [3, 5, 10, 20]
N_FOLDS  = 5
CIT_W = {"1hop": 3, "2hop": 2, "cocite": 1, "none": 0}


def cit_strength(t):
    if not isinstance(t, str): return 0
    return max((CIT_W.get(p, 0) for p in t.split(",")), default=0)


def main():
    m = pd.read_csv(EXP / "data_per_domain/master_inventory.csv",
                    usecols=["canonical_id","domain","sentencing_range_low","sentencing_range_high","sentencing_confidence"])
    inset = m[m.domain.isin(["drugs","weapon"]) & m.sentencing_range_low.notna()
              & (m.sentencing_confidence == "גבוהה")].drop_duplicates("canonical_id")
    range_low  = dict(zip(inset.canonical_id, inset.sentencing_range_low))
    range_high = dict(zip(inset.canonical_id, inset.sentencing_range_high))
    domain_of  = dict(zip(inset.canonical_id, inset.domain))

    # Citation graph
    cit = pd.read_csv(EXP / "data_per_domain/network_analysis/citation_pair_types.csv",
                      usecols=["verdict_1","verdict_2","domain","citation_type"])
    cit = cit[cit.citation_type != "none"]
    cit_edges = {q: [] for q in inset.canonical_id}
    for r in cit.itertuples(index=False):
        for src, tgt in [(r.verdict_1, r.verdict_2), (r.verdict_2, r.verdict_1)]:
            if src in cit_edges and tgt in domain_of and domain_of.get(src) == domain_of.get(tgt):
                cit_edges[src].append((tgt, cit_strength(r.citation_type)))

    # Per-fold embeddings
    folds = {}
    for dom in ["drugs","weapon"]:
        for f in range(1, N_FOLDS+1):
            emb = np.load(EXP / f"simcse_outputs/supervised/verdict_embeddings_{dom}_topk_fold{f}.npy").astype(np.float32)
            idx = pd.read_csv(EXP / f"simcse_outputs/supervised/verdict_index_{dom}_topk_fold{f}.csv")
            v2i = dict(zip(idx.verdict, range(len(idx))))
            folds[(dom, f)] = {"emb": emb, "v2i": v2i,
                               "train_ids": idx[idx.split=="train"].verdict.tolist(),
                               "test_ids":  idx[idx.split=="test"].verdict.tolist()}

    rng = np.random.default_rng(42)
    print(f"=== Computing per-query: cit / sup / union / avg(cit,sup) ===")
    rows = []
    for dom in ["drugs","weapon"]:
        for fold in range(1, N_FOLDS+1):
            f = folds[(dom, fold)]
            train_set = set(f["train_ids"])
            train_idx = np.array([f["v2i"][v] for v in f["train_ids"]])
            for q in f["test_ids"]:
                qi = f["v2i"][q]
                true_lo, true_hi = range_low[q], range_high[q]
                # citation neighbors in train (sorted by strength desc, random tie-break)
                cit_n = [(t, s) for t, s in cit_edges.get(q, []) if t in train_set]
                cit_n.sort(key=lambda x: (-x[1], rng.random()))
                cit_only = [t for t, _ in cit_n]
                # supervised top-K
                sims = f["emb"][qi] @ f["emb"][train_idx].T
                sup_order = np.argsort(-sims)
                sup_ids_sorted = [f["train_ids"][i] for i in sup_order]
                for K in K_VALUES:
                    sup_top = sup_ids_sorted[:K]
                    cit_top = cit_only[:K] if len(cit_only) >= K else None
                    union   = list(dict.fromkeys(sup_top + cit_only[:K]))  # union, preserve order
                    # Predictions
                    def med(picks):
                        if not picks: return None, None
                        return (float(np.median([range_low[p]  for p in picks])),
                                float(np.median([range_high[p] for p in picks])))
                    sup_lo, sup_hi = med(sup_top)
                    cit_lo, cit_hi = med(cit_top) if cit_top else (None, None)
                    uni_lo, uni_hi = med(union)
                    avg_lo = (sup_lo + cit_lo)/2 if cit_lo is not None else None
                    avg_hi = (sup_hi + cit_hi)/2 if cit_hi is not None else None
                    rows.append({
                        "qid": q, "domain": dom, "fold": fold, "K": K,
                        "true_lo": true_lo, "true_hi": true_hi,
                        "n_sup_top": len(sup_top), "n_cit_pool": len(cit_only),
                        "n_union": len(union),
                        "sup_lo_err":   abs(sup_lo - true_lo) if sup_lo is not None else None,
                        "sup_hi_err":   abs(sup_hi - true_hi) if sup_hi is not None else None,
                        "cit_lo_err":   abs(cit_lo - true_lo) if cit_lo is not None else None,
                        "cit_hi_err":   abs(cit_hi - true_hi) if cit_hi is not None else None,
                        "union_lo_err": abs(uni_lo - true_lo) if uni_lo is not None else None,
                        "union_hi_err": abs(uni_hi - true_hi) if uni_hi is not None else None,
                        "avg_lo_err":   abs(avg_lo - true_lo) if avg_lo is not None else None,
                        "avg_hi_err":   abs(avg_hi - true_hi) if avg_hi is not None else None,
                    })
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("\n" + "=" * 110)
    print("ENSEMBLE: citation pool + supervised_topk top-K (5-fold CV)")
    print("=" * 110)
    summary = []
    for dom in ["drugs","weapon"]:
        for K in K_VALUES:
            sub = df[(df.domain==dom) & (df.K==K)]
            n_total = len(sub)
            row = {"domain": dom, "K": K, "n_total": n_total}
            for label in ["sup","cit","union","avg"]:
                lo_col, hi_col = f"{label}_lo_err", f"{label}_hi_err"
                ssub = sub.dropna(subset=[lo_col, hi_col])
                if len(ssub):
                    mae_avg = float(np.mean(list(ssub[lo_col]) + list(ssub[hi_col])))
                    row[f"{label}_n"]    = len(ssub)
                    row[f"{label}_cov"]  = round(100*len(ssub)/n_total, 1)
                    row[f"{label}_MAE"]  = round(mae_avg, 2)
                else:
                    row[f"{label}_n"] = 0; row[f"{label}_cov"] = 0; row[f"{label}_MAE"] = None
            summary.append(row)
    sdf = pd.DataFrame(summary)
    print(sdf.to_string(index=False))
    sdf.to_csv(OUT.with_name("cv_5fold_ensemble_summary.csv"), index=False)
    print(f"\n💾 Saved → {OUT}")
    print(f"💾 Saved → {OUT.with_name('cv_5fold_ensemble_summary.csv')}")


if __name__ == "__main__":
    main()
