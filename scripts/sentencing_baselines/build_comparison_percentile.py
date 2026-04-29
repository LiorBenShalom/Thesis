#!/usr/bin/env python3
"""
Aggregate per-rep predictions in PERCENTILE mode (all neighbors above per-rep,
per-domain Q50, min_k=3) into a comparison table + Wilcoxon paired tests.
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


def bh_fdr(pvals):
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.array(pvals)[order]
    adj = ranked * n / (np.arange(n) + 1)
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out.tolist()


def main():
    preds = {r: pd.read_csv(PRED_DIR / f"preds_{r}_percentile.csv") for r in REPS}

    # Comparison table
    rows = []
    for r, df in preds.items():
        for dom, sub in df.groupby("domain"):
            for sig in ["no_sigma", "with_sigma"]:
                ev = sub
                if sig == "with_sigma":
                    q50_lo = sub["sigma_low"].quantile(0.5)
                    q50_hi = sub["sigma_high"].quantile(0.5)
                    ev = sub[(sub["sigma_low"] <= q50_lo) & (sub["sigma_high"] <= q50_hi)]
                rows.append({
                    "rep": r, "domain": dom, "sigma": sig, "n": len(ev),
                    "avg_n_neighbors": float(ev["n_neighbors"].mean()) if len(ev) else np.nan,
                    "MAE_low":  float(ev["err_low"].mean()),
                    "MAE_high": float(ev["err_high"].mean()),
                    "MedAE_low":  float(ev["err_low"].median()),
                    "MedAE_high": float(ev["err_high"].median()),
                    "IoU": float(ev["iou"].mean()),
                })
    comp = pd.DataFrame(rows)
    comp_path = OUT_DIR / "comparison_baselines_percentile.csv"
    comp.to_csv(comp_path, index=False)
    print(f"\n=== Comparison (percentile mode) -> {comp_path.name} ===")
    pivot = comp[comp.sigma == "no_sigma"].pivot_table(
        index="rep", columns="domain",
        values=["MAE_low", "MAE_high", "IoU", "n", "avg_n_neighbors"]
    )
    print(pivot.round(3).to_string())

    # Wilcoxon
    print("\n=== Wilcoxon (paired, BH-FDR) ===\n")
    all_w = []
    for dom in ["drugs", "weapon"]:
        for tgt in ["low", "high"]:
            for sig_filter in [False, True]:
                a = preds[ANCHOR]
                a = a[a.domain == dom]
                if sig_filter:
                    q50_lo = a["sigma_low"].quantile(0.5)
                    q50_hi = a["sigma_high"].quantile(0.5)
                    a = a[(a["sigma_low"] <= q50_lo) & (a["sigma_high"] <= q50_hi)]
                a_map = dict(zip(a["verdict"], a[f"err_{tgt}"]))
                rows_w, pvs = [], []
                for r in REPS:
                    if r == ANCHOR:
                        continue
                    b = preds[r]
                    b = b[b.domain == dom]
                    if sig_filter:
                        q50_lo = b["sigma_low"].quantile(0.5)
                        q50_hi = b["sigma_high"].quantile(0.5)
                        b = b[(b["sigma_low"] <= q50_lo) & (b["sigma_high"] <= q50_hi)]
                    b_map = dict(zip(b["verdict"], b[f"err_{tgt}"]))
                    shared = sorted(set(a_map.keys()) & set(b_map.keys()))
                    if len(shared) < 10:
                        rows_w.append({"baseline": r, "n_shared": len(shared)})
                        pvs.append(np.nan); continue
                    ae = np.array([a_map[v] for v in shared])
                    be = np.array([b_map[v] for v in shared])
                    try:
                        _, p = wilcoxon(be - ae, zero_method="wilcox", alternative="two-sided")
                    except ValueError:
                        p = np.nan
                    rows_w.append({
                        "baseline": r, "n_shared": len(shared),
                        "mae_anchor": float(ae.mean()),
                        "mae_baseline": float(be.mean()),
                        "median_diff": float(np.median(be - ae)),
                        "p_raw": p,
                    })
                    pvs.append(p)
                df_w = pd.DataFrame(rows_w)
                valid = [i for i, p in enumerate(pvs) if not (p is None or np.isnan(p))]
                p_bh = [np.nan] * len(pvs)
                if valid:
                    adj = bh_fdr([pvs[i] for i in valid])
                    for i, ap in zip(valid, adj):
                        p_bh[i] = ap
                df_w["p_bh"] = p_bh
                df_w["winner"] = np.where(
                    df_w["p_bh"] < 0.05,
                    np.where(df_w["mae_anchor"] < df_w["mae_baseline"], ANCHOR, df_w["baseline"]),
                    "tie",
                )
                df_w["domain"] = dom; df_w["target"] = tgt
                df_w["sigma"] = "with_sigma" if sig_filter else "no_sigma"
                all_w.append(df_w)
                tag = f"{dom} / {tgt} / {'with_sigma' if sig_filter else 'no_sigma'}"
                print(f"--- {tag} ---")
                print(df_w[["baseline", "n_shared", "mae_anchor", "mae_baseline",
                            "median_diff", "p_bh", "winner"]].to_string(index=False))
                print()

    pd.concat(all_w, ignore_index=True).to_csv(
        OUT_DIR / "wilcoxon_baselines_percentile.csv", index=False
    )
    print(f"All Wilcoxon -> {OUT_DIR/'wilcoxon_baselines_percentile.csv'}")


if __name__ == "__main__":
    main()
