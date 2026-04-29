#!/usr/bin/env python3
"""
Cross-validated Risk-Coverage analysis for selective sentencing prediction.
Geifman & El-Yaniv 2017 ("Selective Classification for DNNs") style.

For each target MAE budget τ, asks: "What fraction of queries can the model
predict on (using σ_combined as confidence) while keeping out-of-sample
MAE ≤ τ?" Threshold tuned on train fold, evaluated on held-out test fold.

Inputs: top-K=10 predictions per rep
        (results/2_sentencing_range/predictions/topk10_clean/).
Outputs: cv_risk_coverage.csv — per (rep, domain, target, τ).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

EXP = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/experiments")
PRED_DIR = EXP / "results/2_sentencing_range/predictions/topk10_clean"
OUT_DIR = EXP / "results/2_sentencing_range/predictions"
REPS = ["Hybrid-Full", "Gemini", "TF-IDF", "Random-K"]
RNG = 42


def cv_risk_coverage(df: pd.DataFrame, target: str, tau_list: list[float],
                      n_splits: int = 5, seed: int = RNG) -> pd.DataFrame:
    """For each target MAE τ:
       - On train: find largest sigma threshold s.t. MAE_kept ≤ τ
       - On test:  apply that threshold; report (coverage, MAE)."""
    err_col = f"err_{target}"
    sig = df["sig_combined"].to_numpy()
    err = df[err_col].to_numpy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for tau in tau_list:
        cov_list, mae_list = [], []
        for tr_idx, te_idx in kf.split(err):
            order = np.argsort(sig[tr_idx])
            sorted_err = err[tr_idx][order]
            sorted_sig = sig[tr_idx][order]
            cum_mae = np.cumsum(sorted_err) / np.arange(1, len(sorted_err) + 1)
            ok = np.where(cum_mae <= tau)[0]
            if len(ok) == 0:
                cov_list.append(0.0); mae_list.append(np.nan); continue
            thr = sorted_sig[ok[-1]]
            mask = sig[te_idx] <= thr
            cov_list.append(float(mask.mean()))
            mae_list.append(float(err[te_idx][mask].mean()) if mask.sum() else np.nan)
        rows.append({
            "tau_target": tau,
            "coverage_mean": float(np.mean(cov_list)),
            "coverage_std": float(np.std(cov_list)),
            "MAE_test_mean": float(np.nanmean(mae_list)),
            "MAE_test_std": float(np.nanstd(mae_list)),
        })
    return pd.DataFrame(rows)


def main():
    preds = {r: pd.read_csv(PRED_DIR / f"preds_{r}_topk.csv") for r in REPS}
    for r in preds:
        preds[r]["sig_combined"] = preds[r].sigma_low + preds[r].sigma_high

    rc_all = []
    for r in REPS:
        for dom in ["drugs", "weapon"]:
            sub = preds[r][preds[r].domain == dom]
            for tgt in ["low", "high"]:
                tau_list = [2, 3, 4, 5, 7, 10] if tgt == "low" else [3, 5, 7, 10, 15]
                rc = cv_risk_coverage(sub, tgt, tau_list)
                rc["rep"] = r; rc["domain"] = dom; rc["target"] = tgt
                rc_all.append(rc)
    rc_df = pd.concat(rc_all, ignore_index=True)
    rc_df.to_csv(OUT_DIR / "cv_risk_coverage.csv", index=False)

    print("=" * 80)
    print("CV Risk-Coverage (5-fold; threshold tuned on train, evaluated on test)")
    print("=" * 80)
    for tgt in ["low", "high"]:
        for tau in ([5, 7] if tgt == "high" else [3, 5]):
            sub = rc_df[(rc_df.target == tgt) & (rc_df.tau_target == tau)]
            if sub.empty: continue
            print(f"\nCoverage at MAE_{tgt} ≤ {tau} months (mean across CV folds):")
            piv = sub.pivot_table(index="rep", columns="domain",
                                   values="coverage_mean").round(3)
            print(piv.to_string())
    print(f"\n→ {OUT_DIR/'cv_risk_coverage.csv'}")


if __name__ == "__main__":
    main()
