#!/usr/bin/env python3
"""K-sweep: run top-K for K ∈ {1,3,5,10,20,50,100} for all 4 reps; plot MAE vs K."""
from __future__ import annotations
import argparse
from pathlib import Path
from collections import defaultdict
import json

import numpy as np
import pandas as pd

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
K_VALUES = [1, 3, 5, 10, 20, 50, 100]
DOMAINS = {"drugs": "median", "weapon": "median"}  # uniform median for fair comparison


def softmax_weighted(values, sims, T=10.0):
    z = sims / T; z = z - z.max(); w = np.exp(z); w = w / w.sum()
    return float(np.dot(w, values))


def aggregate(values, sims, agg):
    if agg == "median": return float(np.median(values))
    if agg == "softmax": return softmax_weighted(values, sims)
    if agg == "mean": return float(np.mean(values))
    raise ValueError(agg)


def iou(a_lo, a_hi, p_lo, p_hi):
    lo = np.maximum(a_lo, p_lo); hi = np.minimum(a_hi, p_hi)
    inter = np.maximum(0, hi - lo); union = np.maximum(a_hi, p_hi) - np.minimum(a_lo, p_lo)
    return np.where(union > 0, inter / union, 0.0)


def load_targets():
    df = pd.read_csv(DATA_MASTER)
    df = df.dropna(subset=["sentencing_range_low","sentencing_range_high"])
    df = df[["canonical_id","domain","sentencing_range_low","sentencing_range_high"]]
    df = df.drop_duplicates("canonical_id")
    return df.set_index("canonical_id")


def run_topk(sim_csv, targets, k_max):
    """Build per-query sorted neighbors once, then evaluate per K."""
    sims = pd.read_csv(sim_csv).dropna(subset=["similarity_score"])
    ngh = defaultdict(list)
    for v1, v2, dom, s in sims[["verdict_1","verdict_2","domain","similarity_score"]].itertuples(index=False):
        ngh[v1].append((v2, float(s)))
        ngh[v2].append((v1, float(s)))
    # sort per query by descending sim
    sorted_ngh = {q: sorted(ns, key=lambda x: -x[1]) for q, ns in ngh.items()}
    return sorted_ngh


def evaluate_at_k(sorted_ngh, targets, K, agg_per_dom):
    rows = []
    for q, ns in sorted_ngh.items():
        if q not in targets.index: continue
        q_dom = targets.at[q, "domain"]
        if q_dom not in agg_per_dom: continue
        good = [(n, s) for n, s in ns if n in targets.index and targets.at[n,"domain"]==q_dom][:K]
        if len(good) < K: continue  # only evaluate verdicts with at least K neighbors
        nb_ids = [n for n,_ in good]
        nb_sims = np.array([s for _,s in good])
        nb_lo = targets.loc[nb_ids,"sentencing_range_low"].to_numpy(dtype=float)
        nb_hi = targets.loc[nb_ids,"sentencing_range_high"].to_numpy(dtype=float)
        agg = agg_per_dom[q_dom]
        pl = aggregate(nb_lo, nb_sims, agg); ph = aggregate(nb_hi, nb_sims, agg)
        rows.append({
            "verdict": q, "domain": q_dom,
            "actual_low": float(targets.at[q,"sentencing_range_low"]),
            "actual_high": float(targets.at[q,"sentencing_range_high"]),
            "pred_low": pl, "pred_high": ph,
            "sigma_low": float(nb_lo.std()), "sigma_high": float(nb_hi.std()),
        })
    df = pd.DataFrame(rows)
    df["err_low"] = (df.pred_low - df.actual_low).abs()
    df["err_high"] = (df.pred_high - df.actual_high).abs()
    df["sig_combined"] = df.sigma_low + df.sigma_high
    df["iou"] = iou(df.actual_low.values, df.actual_high.values,
                    df.pred_low.values, df.pred_high.values)
    return df


def main():
    print("Loading targets...")
    targets = load_targets()

    rows = []
    for rep, csv in REPS.items():
        print(f"\n=== {rep} ===")
        sorted_ngh = run_topk(csv, targets, max(K_VALUES))
        for K in K_VALUES:
            df = evaluate_at_k(sorted_ngh, targets, K, DOMAINS)
            for dom, sub in df.groupby("domain"):
                # no_sigma
                rows.append({
                    "rep": rep, "K": K, "domain": dom, "sigma": "no_sigma", "n": len(sub),
                    "MAE_low": float(sub.err_low.mean()),
                    "MAE_high": float(sub.err_high.mean()),
                    "IoU": float(sub.iou.mean()),
                })
                # with_sigma Q50
                cut = sub.sig_combined.quantile(0.5)
                ev = sub[sub.sig_combined <= cut]
                if len(ev):
                    rows.append({
                        "rep": rep, "K": K, "domain": dom, "sigma": "with_sigma", "n": len(ev),
                        "MAE_low": float(ev.err_low.mean()),
                        "MAE_high": float(ev.err_high.mean()),
                        "IoU": float(ev.iou.mean()),
                    })
            print(f"  K={K}: drugs n={(df.domain=='drugs').sum()}, weapon n={(df.domain=='weapon').sum()}")

    sweep = pd.DataFrame(rows)
    sweep_path = OUT_DIR / "k_sweep_clean.csv"
    sweep.to_csv(sweep_path, index=False)
    print(f"\nSaved → {sweep_path}")

    # Print pivot
    for sig in ["no_sigma", "with_sigma"]:
        for tgt in ["MAE_low", "MAE_high", "IoU"]:
            print(f"\n=== {tgt} ({sig}) ===")
            piv = sweep[sweep.sigma==sig].pivot_table(
                index=["domain","rep"], columns="K", values=tgt
            ).round(3)
            print(piv.to_string())

    # Build extra metric: Coverage@MAE_high≤6 (per K, per rep, per domain)
    # = fraction of queries with err_high ≤ 6
    extra = []
    for rep, csv in REPS.items():
        sorted_ngh = run_topk(csv, targets, max(K_VALUES))
        for K in K_VALUES:
            df = evaluate_at_k(sorted_ngh, targets, K, DOMAINS)
            for dom, sub in df.groupby("domain"):
                cov6 = float((sub.err_high <= 6).mean())
                extra.append({"rep": rep, "K": K, "domain": dom, "Cov@MAE≤6": cov6})
    extra_df = pd.DataFrame(extra)

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        colors = {"Hybrid-Full":"#1f77b4","Gemini":"#ff7f0e","TF-IDF":"#2ca02c","Random-K":"#888888"}
        for i_dom, dom in enumerate(["drugs","weapon"]):
            for j_metric, (metric, ylabel, source) in enumerate([
                ("MAE_low","MAE low (months)","sweep"),
                ("MAE_high","MAE high (months)","sweep"),
                ("IoU","Range IoU","sweep"),
                ("Cov@MAE≤6","Coverage @ MAE≤6 (fraction)","extra"),
            ]):
                ax = axes[i_dom, j_metric]
                for rep in REPS:
                    if source == "sweep":
                        sub = sweep[(sweep.rep==rep) & (sweep.domain==dom) & (sweep.sigma=="no_sigma")].sort_values("K")
                    else:
                        sub = extra_df[(extra_df.rep==rep) & (extra_df.domain==dom)].sort_values("K")
                    ax.plot(sub.K, sub[metric], marker='o', label=rep, color=colors[rep], linewidth=2, markersize=6)
                ax.set_title(f"{dom} — {ylabel}")
                ax.set_xscale("log"); ax.set_xticks(K_VALUES); ax.set_xticklabels(K_VALUES)
                ax.set_xlabel("K (top-K neighbors)")
                ax.set_ylabel(ylabel)
                ax.set_title(f"{dom} — {ylabel}")
                ax.grid(alpha=0.3)
                if i_dom == 0 and j_metric == 0:
                    ax.legend(loc='upper right', fontsize=9)
        plt.tight_layout()
        fig_path = OUT_DIR / "k_sweep_clean.png"
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
        print(f"\nFigure → {fig_path}")
    except ImportError:
        print("matplotlib not available")


if __name__ == "__main__":
    main()
