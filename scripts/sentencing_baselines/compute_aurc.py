#!/usr/bin/env python3
"""
AURC (Area Under Risk-Coverage curve) for sentencing-range prediction.

Methodology (selective prediction, El-Yaniv & Wiener 2010):
  1. For each query, compute prediction confidence = -σ_combined (low spread → high confidence)
  2. Sort queries by confidence (descending)
  3. Sweep coverage from 0 → 100% by accepting queries in confidence order
  4. At each coverage point, compute MAE on accepted queries
  5. AURC = mean MAE across coverage points (lower is better)

Also reports:
  - E-AURC = AURC − optimal AURC (gap from oracle ranker — best-case bound)
  - Coverage@MAE≤τ — what coverage you get at a fixed MAE budget τ
  - MAE@coverage=κ — what MAE you get at a fixed coverage κ

Output: per (rep, domain) table.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

EXP = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/experiments")
OUT_DIR = EXP / "data_per_domain/prediction_results"


def aurc(errors: np.ndarray, confidences: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute AURC. Returns (aurc, coverages, risks)."""
    n = len(errors)
    order = np.argsort(-confidences)  # high confidence first
    sorted_err = errors[order]
    cum = np.cumsum(sorted_err)
    coverages = np.arange(1, n + 1) / n
    risks = cum / np.arange(1, n + 1)  # MAE on the first k accepted
    return float(np.mean(risks)), coverages, risks


def optimal_aurc(errors: np.ndarray) -> float:
    """Oracle: sort by error itself (lowest-error first)."""
    sorted_err = np.sort(errors)
    cum = np.cumsum(sorted_err)
    risks = cum / np.arange(1, len(errors) + 1)
    return float(np.mean(risks))


def coverage_at_risk(errors: np.ndarray, confidences: np.ndarray, tau: float) -> float:
    """Largest coverage where MAE ≤ tau."""
    order = np.argsort(-confidences)
    sorted_err = errors[order]
    cum = np.cumsum(sorted_err)
    risks = cum / np.arange(1, len(errors) + 1)
    ok = np.where(risks <= tau)[0]
    return float((ok[-1] + 1) / len(errors)) if len(ok) else 0.0


def risk_at_coverage(errors: np.ndarray, confidences: np.ndarray, kappa: float) -> float:
    """MAE at a fixed coverage."""
    n = len(errors)
    k = max(1, int(np.round(kappa * n)))
    order = np.argsort(-confidences)
    return float(errors[order[:k]].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=str(EXP / "data_per_domain/prediction_results/qwk_thresholds"))
    ap.add_argument("--reps", nargs="+", default=["Hybrid-Full", "Gemini", "TF-IDF", "Random-K"])
    ap.add_argument("--out-suffix", default="qwk")
    args = ap.parse_args()

    PRED = Path(args.pred_dir)
    rows = []
    curves_rows = []
    for rep in args.reps:
        candidates = [
            PRED / f"preds_{rep}_thr60_allpairs_corrected.csv",
            PRED / f"preds_{rep}_topk.csv",
        ]
        f = next((c for c in candidates if c.exists()), candidates[0])
        if not f.exists():
            print(f"  missing {f}"); continue
        df = pd.read_csv(f)
        df["sig_combined"] = df.sigma_low + df.sigma_high
        # Use -σ_combined as confidence (lower spread = higher confidence)
        for dom, sub in df.groupby("domain"):
            for tgt in ["low", "high"]:
                err = sub[f"err_{tgt}"].to_numpy()
                conf = -sub["sig_combined"].to_numpy()
                a, covs, risks = aurc(err, conf)
                opt = optimal_aurc(err)
                rows.append({
                    "rep": rep, "domain": dom, "target": tgt, "n": len(err),
                    "AURC": a,
                    "optimal_AURC": opt,
                    "E_AURC": a - opt,
                    "MAE_full": float(err.mean()),
                    "Cov@MAE5": coverage_at_risk(err, conf, 5.0),
                    "Cov@MAE7": coverage_at_risk(err, conf, 7.0),
                    "MAE@cov30": risk_at_coverage(err, conf, 0.30),
                    "MAE@cov50": risk_at_coverage(err, conf, 0.50),
                    "MAE@cov70": risk_at_coverage(err, conf, 0.70),
                })
                # save curve points (downsampled to 100 points)
                idx = np.linspace(0, len(covs) - 1, 100).astype(int)
                for i in idx:
                    curves_rows.append({
                        "rep": rep, "domain": dom, "target": tgt,
                        "coverage": float(covs[i]), "risk": float(risks[i]),
                    })

    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(OUT_DIR / f"aurc_{args.out_suffix}.csv", index=False)
    pd.DataFrame(curves_rows).to_csv(OUT_DIR / f"aurc_{args.out_suffix}_curves.csv", index=False)

    print(f"\n=== AURC summary (using {args.pred_dir.split('/')[-1]} predictions, σ_combined as confidence) ===\n")
    pivot = df_metrics.pivot_table(
        index="rep", columns=["domain", "target"],
        values=["AURC", "MAE@cov50", "Cov@MAE5"],
    ).round(3)
    print(pivot.to_string())
    print(f"\nFull → {OUT_DIR/f'aurc_{args.out_suffix}.csv'}")


if __name__ == "__main__":
    main()
