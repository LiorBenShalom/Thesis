#!/usr/bin/env python3
"""
Cross-validated SIMILARITY-THRESHOLD selection per model for kNN sentencing-range
prediction. Each model picks its own optimal threshold by maximizing test MAE
subject to a minimum coverage constraint.

Approach (5-fold CV):
  - Sweep threshold ∈ percentiles {30, 40, 50, 60, 70, 80} of each model's
    sim score distribution (per domain).
  - For each threshold: include only queries with ≥3 neighbors above threshold;
    aggregate via median (drugs) / softmax (weapon).
  - Inner: pick threshold that minimizes train MAE_high subject to
    train coverage ≥ 30% (avoids degenerate "predict on 5 cases").
  - Outer: report test MAE + coverage at that threshold.

Outputs: cv_threshold_optimal.csv  (per rep × domain × target)
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
PERCENTILES = [30, 40, 50, 60, 70, 80]
MIN_K = 3
MIN_COVERAGE = 0.30
N_FOLDS = 5
RNG = 42


def softmax_weighted(values, sims, T=10.0):
    z = sims / T; z = z - z.max(); w = np.exp(z); w = w / w.sum()
    return float(np.dot(w, values))


def aggregate(values, sims, dom):
    return float(np.median(values)) if dom == "drugs" else softmax_weighted(values, sims)


def load_targets():
    df = pd.read_csv(DATA_MASTER)
    df = df.dropna(subset=["sentencing_range_low", "sentencing_range_high"])
    df = df.drop_duplicates("canonical_id")
    return df.set_index("canonical_id")[["domain", "sentencing_range_low", "sentencing_range_high"]]


def predict_at_threshold(neighbors_by_q, query_set, targets, thr, target_col):
    """For each query in query_set, predict using neighbors with sim ≥ thr.
    Skip queries with <MIN_K eligible neighbors."""
    rows = []
    for q in query_set:
        ns = neighbors_by_q.get(q, [])
        good = [(n, s) for n, s in ns if s >= thr][:50]
        if len(good) < MIN_K:
            continue
        nb_ids = [n for n, _ in good]
        nb_sims = np.array([s for _, s in good], dtype=float)
        nb_y = targets.loc[nb_ids, target_col].to_numpy(dtype=float)
        q_dom = targets.at[q, "domain"]
        pred = aggregate(nb_y, nb_sims, q_dom)
        actual = float(targets.at[q, target_col])
        rows.append({"verdict": q, "actual": actual, "pred": pred,
                     "err": abs(pred - actual)})
    return pd.DataFrame(rows)


def main():
    targets = load_targets()
    print("Loading neighbor lists per rep...")

    rep_data = {}
    for rep, csv in REPS.items():
        sims = pd.read_csv(csv).dropna(subset=["similarity_score"])
        ngh = defaultdict(list)
        for v1, v2, _, s in sims[["verdict_1", "verdict_2", "domain", "similarity_score"]].itertuples(index=False):
            if targets.at[v1, "domain"] if v1 in targets.index else None == \
               targets.at[v2, "domain"] if v2 in targets.index else None:
                pass
            ngh[v1].append((v2, float(s)))
            ngh[v2].append((v1, float(s)))
        # Restrict to same-domain neighbors and sort
        out = {}
        for q, ns in ngh.items():
            if q not in targets.index: continue
            q_dom = targets.at[q, "domain"]
            if q_dom not in ("drugs", "weapon"): continue
            good = [(n, s) for n, s in ns
                    if n in targets.index and targets.at[n, "domain"] == q_dom]
            good.sort(key=lambda x: -x[1])
            out[q] = good
        # Per-domain percentile thresholds for this rep
        per_domain_pct_thr = {}
        for d in ["drugs", "weapon"]:
            scores_d = sims[sims.domain == d].similarity_score.dropna().values
            per_domain_pct_thr[d] = {p: float(np.percentile(scores_d, p))
                                      for p in PERCENTILES}
        rep_data[rep] = {"ngh": out, "thr_by_pct": per_domain_pct_thr}

    rows = []
    for rep, data in rep_data.items():
        print(f"\n=== {rep} ===")
        for tgt in ["sentencing_range_low", "sentencing_range_high"]:
            tgt_short = "low" if "low" in tgt else "high"
            for dom in ["drugs", "weapon"]:
                dom_queries = [q for q in data["ngh"]
                                if targets.at[q, "domain"] == dom]
                kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG)
                outer_results = []
                for tr_idx, te_idx in kf.split(dom_queries):
                    train_q = [dom_queries[i] for i in tr_idx]
                    test_q  = [dom_queries[i] for i in te_idx]

                    # Inner: find best percentile on train (lowest MAE
                    # subject to coverage ≥ MIN_COVERAGE)
                    best_train_mae = np.inf; best_pct = None; best_thr = None
                    for pct in PERCENTILES:
                        thr = data["thr_by_pct"][dom][pct]
                        df_tr = predict_at_threshold(data["ngh"], train_q, targets, thr, tgt)
                        cov_tr = len(df_tr) / len(train_q)
                        if cov_tr < MIN_COVERAGE:
                            continue
                        mae_tr = df_tr.err.mean()
                        if mae_tr < best_train_mae:
                            best_train_mae = mae_tr; best_pct = pct; best_thr = thr
                    if best_pct is None:
                        continue
                    # Outer: test
                    df_te = predict_at_threshold(data["ngh"], test_q, targets, best_thr, tgt)
                    cov_te = len(df_te) / len(test_q)
                    mae_te = df_te.err.mean() if len(df_te) else np.nan
                    outer_results.append({
                        "best_pct": best_pct, "best_thr": best_thr,
                        "test_coverage": cov_te, "test_MAE": mae_te,
                    })

                if not outer_results:
                    continue
                df_o = pd.DataFrame(outer_results)
                rows.append({
                    "rep": rep, "domain": dom, "target": tgt_short,
                    "best_pct_majority": int(df_o.best_pct.mode().iloc[0]),
                    "best_thr_mean": float(df_o.best_thr.mean()),
                    "test_coverage_mean": float(df_o.test_coverage.mean()),
                    "test_coverage_std":  float(df_o.test_coverage.std()),
                    "test_MAE_mean": float(df_o.test_MAE.mean()),
                    "test_MAE_std":  float(df_o.test_MAE.std()),
                    "n_total_queries": len(dom_queries),
                })
                print(f"  {dom:6s}/{tgt_short:4s}: best pct={df_o.best_pct.mode().iloc[0]:>2d}, "
                      f"thr={df_o.best_thr.mean():>6.2f}, "
                      f"coverage={df_o.test_coverage.mean()*100:>4.0f}%, "
                      f"MAE={df_o.test_MAE.mean():.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "cv_threshold_optimal.csv", index=False)

    print("\n" + "=" * 90)
    print("PER-MODEL OPTIMAL THRESHOLD (CV-derived from sentencing task itself)")
    print("=" * 90)
    print()
    print(res[["rep","domain","target","best_pct_majority","best_thr_mean",
                 "test_coverage_mean","test_MAE_mean"]].round(3).to_string(index=False))

    print("\n" + "=" * 90)
    print("PIVOT: Test MAE @ each model's optimal threshold")
    print("=" * 90)
    piv = res.pivot_table(index="rep", columns=["domain","target"],
                            values="test_MAE_mean").round(3)
    print(piv.to_string())

    print("\nCoverage @ optimal threshold:")
    piv_c = res.pivot_table(index="rep", columns=["domain","target"],
                              values="test_coverage_mean").round(3)
    print(piv_c.to_string())


if __name__ == "__main__":
    main()
