#!/usr/bin/env python3
"""
Cross-validated K selection for kNN sentencing-range prediction.
Nested 5-fold CV (Hastie, Tibshirani, Friedman 2009): for each fold,
tune K on the inner training folds, evaluate on the held-out test fold.

For each (rep, domain, target):
  Sweep K ∈ {3, 5, 7, 10, 15, 20, 30, 50}
  Returns: best K, test MAE at best K, plus full curve.

This avoids the bias of selecting K on the same data we evaluate on.

Inputs : 4 similarity_scores_*_combined.csv files
Outputs: cv_k_selection.csv (per rep × domain × target × K)
         cv_k_optimal.csv   (best K per cell)
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
EXP = ROOT / "experiments"
DATA_MASTER = ROOT / "innovation_submission/data_master_final/verdicts_clean.csv"
OUT_DIR = EXP / "results/2_sentencing_range/predictions"

REPS = {
    "Hybrid-Full": EXP / "data_per_domain/similarity_scores_combined.csv",
    "Gemini":      EXP / "data_per_domain/similarity_scores_gemini_combined.csv",
    "TF-IDF":      EXP / "data_per_domain/similarity_scores_tfidf_combined.csv",
    "Random-K":    EXP / "data_per_domain/similarity_scores_random_combined.csv",
}
K_GRID = [3, 5, 7, 10, 15, 20]
N_FOLDS = 5
RNG = 42


def softmax_weighted(values, sims, T=10.0):
    z = sims / T; z = z - z.max(); w = np.exp(z); w = w / w.sum()
    return float(np.dot(w, values))


def aggregate(values, sims, dom):
    if dom == "drugs":
        return float(np.median(values))
    return softmax_weighted(values, sims)


def load_targets():
    df = pd.read_csv(DATA_MASTER)
    df = df.dropna(subset=["sentencing_range_low", "sentencing_range_high"])
    df = df.drop_duplicates("canonical_id")
    return df.set_index("canonical_id")[["domain", "sentencing_range_low", "sentencing_range_high"]]


def build_neighbors(sim_csv: Path, targets: pd.DataFrame, k_max: int):
    """Per-query sorted (neighbor, sim) lists, restricted to same-domain
    and capped at k_max for efficiency."""
    sims = pd.read_csv(sim_csv).dropna(subset=["similarity_score"])
    ngh = defaultdict(list)
    for v1, v2, _, s in sims[["verdict_1", "verdict_2", "domain", "similarity_score"]].itertuples(index=False):
        ngh[v1].append((v2, float(s)))
        ngh[v2].append((v1, float(s)))
    # Sort and cap each query's neighbor list to k_max for the same domain.
    out = {}
    for q, ns in ngh.items():
        if q not in targets.index:
            continue
        q_dom = targets.at[q, "domain"]
        if q_dom not in ("drugs", "weapon"):
            continue
        good = [(n, s) for n, s in ns if n in targets.index
                and targets.at[n, "domain"] == q_dom]
        good.sort(key=lambda x: -x[1])
        out[q] = good[:k_max]
    return out


def predict_at_K(neighbors_by_q, targets, K: int, target_col: str):
    """Predict sentencing_range_<low|high> for all queries with ≥K neighbors."""
    rows = []
    for q, good in neighbors_by_q.items():
        if len(good) < K:
            continue
        top = good[:K]
        nb_ids = [n for n, _ in top]
        nb_sims = np.array([s for _, s in top], dtype=float)
        nb_y = targets.loc[nb_ids, target_col].to_numpy(dtype=float)
        q_dom = targets.at[q, "domain"]
        pred = aggregate(nb_y, nb_sims, q_dom)
        actual = float(targets.at[q, target_col])
        rows.append({"verdict": q, "domain": q_dom,
                     "actual": actual, "pred": pred,
                     "err": abs(pred - actual)})
    return pd.DataFrame(rows)


def main():
    targets = load_targets()
    k_max = max(K_GRID)

    rows = []
    optimal_rows = []

    for rep, csv in REPS.items():
        print(f"\n=== {rep} ===")
        # Cap neighbor lists at k_max for memory only — query inclusion below
        # is handled per-K based on its own coverage.
        ngh = build_neighbors(csv, targets, k_max)
        for tgt in ["sentencing_range_low", "sentencing_range_high"]:
            tgt_short = "low" if "low" in tgt else "high"
            for dom in ["drugs", "weapon"]:
                # For CV K-selection we need apples-to-apples comparison →
                # use the query set that has enough neighbors for the LARGEST K.
                dom_queries = [q for q, gs in ngh.items()
                                if len(gs) >= k_max and targets.at[q, "domain"] == dom]
                if len(dom_queries) < N_FOLDS * 10:
                    continue

                kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG)
                test_mae_per_k = {K: [] for K in K_GRID}
                # For NESTED selection: best K per outer fold, then test MAE at that K
                outer_best_k = []
                outer_test_mae = []

                for tr_idx, te_idx in kf.split(dom_queries):
                    train_q = [dom_queries[i] for i in tr_idx]
                    test_q  = [dom_queries[i] for i in te_idx]

                    # Inner: best K on train
                    best_train_mae = np.inf; best_K = K_GRID[0]
                    for K in K_GRID:
                        train_subset = {q: ngh[q] for q in train_q}
                        df_train = predict_at_K(train_subset, targets, K, tgt)
                        mae_train = df_train.err.mean()
                        if mae_train < best_train_mae:
                            best_train_mae = mae_train; best_K = K

                    # Outer: evaluate at best_K on test
                    test_subset = {q: ngh[q] for q in test_q}
                    df_test = predict_at_K(test_subset, targets, best_K, tgt)
                    outer_test_mae.append(df_test.err.mean())
                    outer_best_k.append(best_K)

                    # Also record per-K test MAE for the curve
                    for K in K_GRID:
                        df_K = predict_at_K(test_subset, targets, K, tgt)
                        test_mae_per_k[K].append(df_K.err.mean())

                # Average curve across folds
                for K in K_GRID:
                    rows.append({
                        "rep": rep, "domain": dom, "target": tgt_short, "K": K,
                        "test_MAE_mean": float(np.mean(test_mae_per_k[K])),
                        "test_MAE_std":  float(np.std(test_mae_per_k[K])),
                        "n_test_per_fold": int(len(dom_queries) / N_FOLDS),
                    })

                # Nested CV summary: average test MAE at the per-fold best K
                most_common_k = max(set(outer_best_k), key=outer_best_k.count)
                optimal_rows.append({
                    "rep": rep, "domain": dom, "target": tgt_short,
                    "best_K_per_fold": str(outer_best_k),
                    "best_K_majority": most_common_k,
                    "nested_CV_test_MAE_mean": float(np.mean(outer_test_mae)),
                    "nested_CV_test_MAE_std":  float(np.std(outer_test_mae)),
                    "n_total_queries": len(dom_queries),
                })
                print(f"  {dom:6s}/{tgt_short:4s}: best K per fold = {outer_best_k}, "
                      f"majority = {most_common_k}, "
                      f"test MAE = {np.mean(outer_test_mae):.3f} ± {np.std(outer_test_mae):.3f}")

    # === Additional: MAE @ each K's OWN coverage (no apples-to-apples bias) ===
    own_cov_rows = []
    for rep, csv in REPS.items():
        ngh = build_neighbors(csv, targets, k_max)
        for tgt in ["sentencing_range_low", "sentencing_range_high"]:
            tgt_short = "low" if "low" in tgt else "high"
            for dom in ["drugs", "weapon"]:
                # All queries in this domain (no neighbor-count gate)
                all_q = {q: ngh[q] for q in ngh
                          if targets.at[q, "domain"] == dom}
                for K in K_GRID:
                    # Each K evaluated on queries with ≥K neighbors only
                    df_pred = predict_at_K(all_q, targets, K, tgt)
                    n_dom_total = sum(1 for v, d in zip(targets.index, targets.domain) if d == dom)
                    own_cov_rows.append({
                        "rep": rep, "domain": dom, "target": tgt_short, "K": K,
                        "n_predicted": len(df_pred),
                        "n_dom_total": n_dom_total,
                        "coverage_pct": round(100 * len(df_pred) / n_dom_total, 1),
                        "MAE_full_data": float(df_pred.err.mean()) if len(df_pred) else None,
                    })

    pd.DataFrame(rows).to_csv(OUT_DIR / "cv_k_selection.csv", index=False)
    pd.DataFrame(optimal_rows).to_csv(OUT_DIR / "cv_k_optimal.csv", index=False)
    pd.DataFrame(own_cov_rows).to_csv(OUT_DIR / "k_sweep_own_coverage.csv", index=False)

    print("\n" + "=" * 80)
    print("Optimal K (majority across folds) per (rep, domain, target)")
    print("=" * 80)
    odf = pd.DataFrame(optimal_rows)
    piv = odf.pivot_table(index="rep", columns=["domain", "target"],
                           values="best_K_majority", aggfunc="first")
    print(piv.to_string())

    print("\n" + "=" * 80)
    print("Nested-CV test MAE at best K (unbiased estimate)")
    print("=" * 80)
    piv2 = odf.pivot_table(index="rep", columns=["domain", "target"],
                            values="nested_CV_test_MAE_mean").round(3)
    print(piv2.to_string())

    # Print own-coverage curve
    own_df = pd.DataFrame(own_cov_rows)
    print("\n" + "=" * 80)
    print("MAE at each K's OWN coverage (no shared-Q gate)")
    print("=" * 80)
    for tgt in ["low", "high"]:
        for dom in ["drugs", "weapon"]:
            print(f"\n--- {dom} / {tgt} ---")
            sub = own_df[(own_df.domain==dom) & (own_df.target==tgt)]
            piv = sub.pivot_table(index="rep", columns="K", values="MAE_full_data").round(2)
            piv["coverage_at_K=10(%)"] = sub[sub.K==10].set_index("rep")["coverage_pct"]
            piv["coverage_at_K=20(%)"] = sub[sub.K==20].set_index("rep")["coverage_pct"]
            print(piv.to_string())


if __name__ == "__main__":
    main()
