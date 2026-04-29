#!/usr/bin/env python3
"""
Aggregate per-rep predictions (preds_<rep>_topk.csv) into:
  1. comparison_baselines_topk.csv — one row per (rep, domain, sigma_filter)
  2. wilcoxon_baselines_<dom>_<target>.csv — paired tests Hybrid-Full vs baselines
                                              on the SHARED set of query verdicts
                                              evaluated by both reps.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

EXP = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/experiments")
PRED_DIR = EXP / "data_per_domain/prediction_results/baselines"
OUT_DIR = EXP / "data_per_domain/prediction_results"

REPS = ["Hybrid-Full", "Gemini", "TF-IDF", "Random-K"]
ANCHOR = "Hybrid-Full"


def iou(a_lo, a_hi, p_lo, p_hi):
    lo = np.maximum(a_lo, p_lo)
    hi = np.minimum(a_hi, p_hi)
    inter = np.maximum(0.0, hi - lo)
    union = np.maximum(a_hi, p_hi) - np.minimum(a_lo, p_lo)
    return np.where(union > 0, inter / union, 0.0)


def load_preds() -> dict[str, pd.DataFrame]:
    out = {}
    for r in REPS:
        f = PRED_DIR / f"preds_{r}_topk.csv"
        df = pd.read_csv(f)
        df["rep"] = r
        out[r] = df
    return out


def comparison_table(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for r, df in preds.items():
        for dom, sub in df.groupby("domain"):
            for sig_filter in ["no_sigma", "with_sigma"]:
                ev = sub
                if sig_filter == "with_sigma":
                    q50_lo = sub["sigma_low"].quantile(0.5)
                    q50_hi = sub["sigma_high"].quantile(0.5)
                    ev = sub[(sub["sigma_low"] <= q50_lo) & (sub["sigma_high"] <= q50_hi)]
                if len(ev) == 0:
                    continue
                rows.append({
                    "rep": r,
                    "domain": dom,
                    "sigma": sig_filter,
                    "n": len(ev),
                    "MAE_low":  float(ev["err_low"].mean()),
                    "MAE_high": float(ev["err_high"].mean()),
                    "MedAE_low":  float(ev["err_low"].median()),
                    "MedAE_high": float(ev["err_high"].median()),
                    "IoU": float(ev["iou"].mean()),
                })
    return pd.DataFrame(rows)


def bh_fdr(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR-adjusted p-values."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.array(pvals)[order]
    adj = ranked * n / (np.arange(n) + 1)
    # enforce monotone non-increasing from the largest down
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out.tolist()


def wilcoxon_pairwise(preds: dict[str, pd.DataFrame], domain: str, target: str,
                      sigma_filter: bool) -> pd.DataFrame:
    """For each baseline rep, pair its per-verdict abs errors with the anchor's
    on the SHARED verdict set (post sigma-filter applied per-rep), then Wilcoxon."""
    err_col = f"err_{target}"

    def sigma_eval(df):
        if not sigma_filter:
            return df
        q50_lo = df["sigma_low"].quantile(0.5)
        q50_hi = df["sigma_high"].quantile(0.5)
        return df[(df["sigma_low"] <= q50_lo) & (df["sigma_high"] <= q50_hi)]

    a = preds[ANCHOR]
    a = a[a.domain == domain]
    a = sigma_eval(a)
    a_map = dict(zip(a["verdict"], a[err_col]))

    rows = []
    pvals = []
    for r in REPS:
        if r == ANCHOR:
            continue
        b = preds[r]
        b = b[b.domain == domain]
        b = sigma_eval(b)
        b_map = dict(zip(b["verdict"], b[err_col]))
        shared = sorted(set(a_map.keys()) & set(b_map.keys()))
        if len(shared) < 10:
            rows.append({"baseline": r, "n_shared": len(shared),
                         "mae_anchor": np.nan, "mae_baseline": np.nan,
                         "median_diff": np.nan, "p_raw": np.nan})
            pvals.append(np.nan)
            continue
        a_err = np.array([a_map[v] for v in shared])
        b_err = np.array([b_map[v] for v in shared])
        diff = b_err - a_err  # positive => baseline worse than anchor
        try:
            stat, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            p = np.nan
        rows.append({
            "baseline": r,
            "n_shared": len(shared),
            "mae_anchor":  float(a_err.mean()),
            "mae_baseline": float(b_err.mean()),
            "median_diff": float(np.median(diff)),
            "p_raw": p,
        })
        pvals.append(p)
    df = pd.DataFrame(rows)
    valid = [i for i, p in enumerate(pvals) if not (p is None or np.isnan(p))]
    p_bh = [np.nan] * len(pvals)
    if valid:
        adj = bh_fdr([pvals[i] for i in valid])
        for i, a_p in zip(valid, adj):
            p_bh[i] = a_p
    df["p_bh"] = p_bh
    df["winner"] = np.where(
        df["p_bh"] < 0.05,
        np.where(df["mae_anchor"] < df["mae_baseline"], ANCHOR, df["baseline"]),
        "tie",
    )
    df["domain"] = domain
    df["target"] = target
    df["sigma"] = "with_sigma" if sigma_filter else "no_sigma"
    return df


def main():
    preds = load_preds()
    comp = comparison_table(preds)
    comp_path = OUT_DIR / "comparison_baselines_topk_full85k.csv"
    comp.to_csv(comp_path, index=False)
    print(f"\n{'='*80}\nComparison table -> {comp_path}\n{'='*80}")
    pivot = comp[comp["sigma"] == "with_sigma"].pivot_table(
        index="rep", columns="domain", values=["MAE_low", "MAE_high", "IoU"]
    )
    print(pivot.round(3).to_string())

    print("\n\nWilcoxon (paired, BH-FDR within (domain,target,sigma)):\n")
    all_w = []
    for dom in ["drugs", "weapon"]:
        for tgt in ["low", "high"]:
            for sig in [False, True]:
                w = wilcoxon_pairwise(preds, dom, tgt, sig)
                all_w.append(w)
                tag = f"{dom} / {tgt} / {'with_sigma' if sig else 'no_sigma'}"
                print(f"--- {tag} ---")
                print(w[["baseline", "n_shared", "mae_anchor", "mae_baseline",
                         "median_diff", "p_bh", "winner"]].to_string(index=False))
                print()
    wfull = pd.concat(all_w, ignore_index=True)
    wpath = OUT_DIR / "wilcoxon_baselines_topk_full85k.csv"
    wfull.to_csv(wpath, index=False)
    print(f"All Wilcoxon -> {wpath}")


if __name__ == "__main__":
    main()
