#!/usr/bin/env python3
"""
Aggregate all per-cell CSVs (model × rep × domain) and compute representation-level
significance: does the v6 ordering of representations (manual_fe > H-Full > ...)
hold under the score-only prompt?

Outputs (under multimodel_score_only/analysis/):
  - all_metrics.csv         — per (model, rep, domain) all metrics
  - rep_summary.csv         — mean per (rep, domain) across models
  - sig_rep_comparisons.csv — Wilcoxon+FDR pairwise comparisons between reps
  - REPORT.md               — Hebrew + English summary
"""
from __future__ import annotations
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, average_precision_score, precision_recall_curve, f1_score
from scipy.stats import wilcoxon, spearmanr
from statsmodels.stats.multitest import multipletests

from .config import REPS, MODELS, RESULTS_DIR, OUT_DIR

ANALYSIS_DIR = OUT_DIR / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)


def score_to_scale(s):
    if pd.isna(s): return None
    if s < 25: return 0
    if s < 50: return 1
    if s < 75: return 2
    return 3

def c_index(y_true, y_score):
    n_c = n_d = 0
    for i in range(len(y_true)):
        for j in range(i+1, len(y_true)):
            yi, si = y_true[i], y_score[i]
            yj, sj = y_true[j], y_score[j]
            if yi == yj: continue
            if (yi > yj and si > sj) or (yi < yj and si < sj): n_c += 1
            elif (yi > yj and si < sj) or (yi < yj and si > sj): n_d += 1
    return n_c / (n_c + n_d) if (n_c+n_d) else 0.5

def f1_oracle(y_true, score):
    p, r, _ = precision_recall_curve(y_true, score)
    f1 = 2*p*r / (p+r+1e-12)
    return float(np.nanmax(f1))

def metrics_for(gt: np.ndarray, score: np.ndarray) -> dict:
    mask = ~np.isnan(score)
    gt = gt[mask].astype(int); score = score[mask].astype(float)
    if len(gt) < 5:
        return {"n": int(len(gt))}
    scale = np.array([score_to_scale(v) for v in score])
    bin_strict = (gt >= 3).astype(int)
    bin_lenient = (gt >= 2).astype(int)
    rho, _ = spearmanr(gt, score) if len(set(gt)) > 1 else (np.nan, np.nan)
    return {
        "n": int(len(gt)),
        "QWK_scale": float(cohen_kappa_score(gt, scale, weights="quadratic", labels=[0,1,2,3])),
        "C_index":   float(c_index(gt.tolist(), score.tolist())),
        "Spearman":  float(rho) if not np.isnan(rho) else np.nan,
        "AP_strict": float(average_precision_score(bin_strict, score)) if bin_strict.sum() else np.nan,
        "AP_lenient":float(average_precision_score(bin_lenient, score)) if bin_lenient.sum() else np.nan,
        "F1_strict_oracle":  f1_oracle(bin_strict, score) if bin_strict.sum() else np.nan,
        "F1_lenient_oracle": f1_oracle(bin_lenient, score) if bin_lenient.sum() else np.nan,
        "F1_strict_t50":     float(f1_score(bin_strict, (score>=50).astype(int), zero_division=0)) if bin_strict.sum() else np.nan,
        "F1_lenient_t50":    float(f1_score(bin_lenient, (score>=50).astype(int), zero_division=0)) if bin_lenient.sum() else np.nan,
    }


def load_all_cells() -> pd.DataFrame:
    """Returns long DataFrame: one row per (model, rep, domain, pair)."""
    rows = []
    for csv in sorted(RESULTS_DIR.glob("*.csv")):
        # filename format: {model}__{rep}__{domain}.csv
        stem = csv.stem
        parts = stem.split("__")
        if len(parts) != 3: continue
        model, rep, domain = parts
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        df["model"] = model; df["rep"] = rep; df["domain"] = domain
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def per_cell_metrics(all_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, rep, domain) with all metrics."""
    out = []
    for (m, r, d), sub in all_df.groupby(["model", "rep", "domain"]):
        gt = sub["GT"].astype(int).values
        score = pd.to_numeric(sub["model_score"], errors="coerce").values
        met = metrics_for(gt, score)
        met.update({"model": m, "rep": r, "domain": d})
        out.append(met)
    return pd.DataFrame(out)


def per_rep_summary(cells_df: pd.DataFrame) -> pd.DataFrame:
    """Mean across models per (rep, domain)."""
    metric_cols = [c for c in cells_df.columns if c not in ("model","rep","domain","n")]
    return cells_df.groupby(["rep", "domain"])[metric_cols].mean(numeric_only=True).reset_index()


def rep_pairwise_significance(cells_df: pd.DataFrame, metric: str = "AP_strict") -> pd.DataFrame:
    """For each domain, pairwise Wilcoxon between reps using each model as a paired observation.
    Then FDR-BH correction across the (k choose 2) comparisons."""
    rows = []
    for dom in cells_df["domain"].unique():
        sub = cells_df[cells_df["domain"] == dom]
        rep_to_vals = {}
        for rep in sub["rep"].unique():
            vals = sub[sub["rep"] == rep].set_index("model")[metric]
            rep_to_vals[rep] = vals
        reps = list(rep_to_vals.keys())
        for r1, r2 in combinations(reps, 2):
            v1 = rep_to_vals[r1]; v2 = rep_to_vals[r2]
            common = v1.index.intersection(v2.index)
            if len(common) < 3: continue
            a = v1.loc[common].values; b = v2.loc[common].values
            try:
                stat, p = wilcoxon(a, b, zero_method="zsplit")
            except Exception:
                stat, p = np.nan, np.nan
            rows.append({
                "domain": dom, "rep_a": r1, "rep_b": r2, "metric": metric,
                "n_models": len(common),
                "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
                "diff_a_minus_b": float(np.mean(a) - np.mean(b)),
                "wilcoxon_W": float(stat) if not np.isnan(stat) else None,
                "p_value": float(p) if not np.isnan(p) else None,
            })
    df = pd.DataFrame(rows)
    if df.empty: return df
    # FDR-BH across all comparisons in this metric+domain
    out = []
    for dom in df["domain"].unique():
        sub = df[df["domain"] == dom].copy()
        ps = sub["p_value"].fillna(1.0).values
        rej, p_adj, _, _ = multipletests(ps, method="fdr_bh")
        sub["p_fdr_bh"] = p_adj
        sub["sig_005"] = rej
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def main():
    all_df = load_all_cells()
    if all_df.empty:
        print("No result CSVs found yet")
        return
    print(f"Loaded {all_df['model'].nunique()} models × {all_df['rep'].nunique()} reps × 2 domains  ({len(all_df):,} rows)")

    cells = per_cell_metrics(all_df)
    cells.to_csv(ANALYSIS_DIR / "all_metrics.csv", index=False)
    print(f"✅ all_metrics.csv  ({len(cells)} cells)")

    summ = per_rep_summary(cells)
    summ.to_csv(ANALYSIS_DIR / "rep_summary.csv", index=False)
    print(f"✅ rep_summary.csv")

    sig_rows = []
    for metric in ["AP_strict", "AP_lenient", "QWK_scale", "C_index", "F1_strict_oracle"]:
        sig = rep_pairwise_significance(cells, metric)
        sig_rows.append(sig)
    sig_all = pd.concat(sig_rows, ignore_index=True)
    sig_all.to_csv(ANALYSIS_DIR / "sig_rep_comparisons.csv", index=False)
    print(f"✅ sig_rep_comparisons.csv  ({len(sig_all)} comparisons)")

    # Print quick summary
    print("\n## REP RANKING — mean AP_strict per (rep, domain) across models")
    pivot = summ.pivot_table(index="rep", columns="domain", values="AP_strict").round(3).sort_values("drugs", ascending=False)
    print(pivot.to_string())


if __name__ == "__main__":
    main()
