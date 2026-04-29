#!/usr/bin/env python3
"""
Build unified comparison table + Wilcoxon-FDR significance for paper-style runs.
Paired tests: Hybrid-Full vs each baseline on the SHARED query verdict set.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

EXP = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/experiments")
PRED_DIR = EXP / "data_per_domain/prediction_results/paper_style"
OUT_DIR = EXP / "data_per_domain/prediction_results"

# (rep, file)
REPS = [
    ("Hybrid-Full", "preds_Hybrid-Full_thr60_corrected.csv"),
    ("Gemini",      "preds_Gemini_thr60_corrected.csv"),
    ("TF-IDF",      "preds_TF-IDF_thr60_corrected.csv"),
    ("Random-K",    "preds_Random-K_thr35_corrected.csv"),
]
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
    preds = {}
    for r, f in REPS:
        df = pd.read_csv(PRED_DIR / f)
        df["rep"] = r
        df["sig_combined"] = df["sigma_low"] + df["sigma_high"]
        preds[r] = df

    # Unified comparison
    rows = []
    for r, df in preds.items():
        for dom, sub in df.groupby("domain"):
            for sig in ["no_sigma", "with_sigma"]:
                ev = sub
                if sig == "with_sigma":
                    ev = sub[sub["sig_combined"] <= sub["sig_combined"].quantile(0.5)]
                if len(ev) == 0:
                    continue
                rows.append({
                    "rep": r, "domain": dom, "sigma": sig,
                    "n": len(ev), "avg_n_neighbors": float(ev["n_neighbors"].mean()),
                    "MAE_low":  float(ev["err_low"].mean()),
                    "MAE_high": float(ev["err_high"].mean()),
                    "MedAE_low":  float(ev["err_low"].median()),
                    "MedAE_high": float(ev["err_high"].median()),
                    "IoU": float(ev["iou"].mean()),
                })
    comp = pd.DataFrame(rows)
    comp_path = OUT_DIR / "comparison_paper_style.csv"
    comp.to_csv(comp_path, index=False)
    print("=== COMPARISON (paper-style, THR=60-equivalent percentile, weighted_mean, citation-linked, corrected graph) ===\n")
    pivot = comp.pivot_table(index=["rep", "sigma"], columns="domain",
                              values=["n", "MAE_low", "MAE_high", "IoU"]).round(3)
    print(pivot.to_string())

    # Wilcoxon
    print("\n\n=== Wilcoxon (paired, BH-FDR within (domain, target, sigma)) ===\n")
    all_w = []
    for dom in ["drugs", "weapon"]:
        for tgt in ["low", "high"]:
            for sig in [False, True]:
                a = preds[ANCHOR]
                a = a[a.domain == dom]
                if sig:
                    a = a[a["sig_combined"] <= a["sig_combined"].quantile(0.5)]
                a_map = dict(zip(a["verdict"], a[f"err_{tgt}"]))

                rows_w, pvs = [], []
                for r, _ in REPS:
                    if r == ANCHOR:
                        continue
                    b = preds[r]
                    b = b[b.domain == dom]
                    if sig:
                        b = b[b["sig_combined"] <= b["sig_combined"].quantile(0.5)]
                    b_map = dict(zip(b["verdict"], b[f"err_{tgt}"]))
                    shared = sorted(set(a_map) & set(b_map))
                    if len(shared) < 10:
                        rows_w.append({"baseline": r, "n_shared": len(shared)}); pvs.append(np.nan); continue
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
                df_w["sigma"] = "with_sigma" if sig else "no_sigma"
                all_w.append(df_w)
                tag = f"{dom} / {tgt} / {'with_sigma' if sig else 'no_sigma'}"
                print(f"--- {tag} ---")
                print(df_w[["baseline", "n_shared", "mae_anchor", "mae_baseline",
                            "median_diff", "p_bh", "winner"]].to_string(index=False))
                print()
    pd.concat(all_w, ignore_index=True).to_csv(OUT_DIR / "wilcoxon_paper_style.csv", index=False)
    print(f"All Wilcoxon -> {OUT_DIR/'wilcoxon_paper_style.csv'}")


if __name__ == "__main__":
    main()
