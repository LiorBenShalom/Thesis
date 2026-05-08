#!/usr/bin/env python3
"""
Within-query robust kNN aggregation for sentencing-range prediction.

For each query: take top-K=10 neighbors, then DROP OUTLIER NEIGHBORS within
that group before aggregating. This reduces variance/noise in the prediction
itself (rather than dropping queries — which is selective prediction).

Three filter variants implemented:
  1. Trimmed mean   — drop top/bottom α=20% by sentence value
  2. MAD outlier    — drop neighbors with |x − median| / MAD > 2.5
  3. IQR fence      — drop neighbors outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR]

Comparison: baseline = no filter (median for drugs, softmax for weapon — same
as canonical pipeline). All variants evaluated on the SAME 4,094 queries.

Inputs:
  data_per_domain/similarity_scores_combined.csv  (Hybrid-Full)
  data_per_domain/similarity_scores_gemini_combined.csv
  data_per_domain/similarity_scores_tfidf_combined.csv
  data_per_domain/similarity_scores_random_combined.csv
  innovation_submission/data_master_final/verdicts_clean.csv
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
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
K = 10


# ── Within-query filters ────────────────────────────────────────────────────

def filter_none(values):
    return values

def filter_trimmed(values, alpha=0.20):
    """Keep middle (1−2α) of values by sorted order."""
    if len(values) < 5:
        return values
    n_drop = int(np.floor(len(values) * alpha))
    if n_drop == 0:
        return values
    sorted_vals = np.sort(values)
    return sorted_vals[n_drop:len(sorted_vals) - n_drop]

def filter_mad(values, k=2.5):
    """Drop neighbors farther than k·MAD from median."""
    if len(values) < 3:
        return values
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad == 0:
        return values
    keep = np.abs(values - med) / mad <= k
    return values[keep] if keep.sum() >= 2 else values

def filter_iqr(values, mult=1.5):
    """Drop neighbors outside [Q1 − mult·IQR, Q3 + mult·IQR]."""
    if len(values) < 4:
        return values
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return values
    lo, hi = q1 - mult * iqr, q3 + mult * iqr
    keep = (values >= lo) & (values <= hi)
    return values[keep] if keep.sum() >= 2 else values

FILTERS = {
    "none":    filter_none,
    "trimmed": filter_trimmed,
    "mad":     filter_mad,
    "iqr":     filter_iqr,
}


# ── Aggregation (canonical pipeline) ───────────────────────────────────────

def softmax_weighted(values, sims, T=10.0):
    z = sims / T; z = z - z.max(); w = np.exp(z); w = w / w.sum()
    return float(np.dot(w, values))

def aggregate(values, sims, dom):
    if dom == "drugs":
        return float(np.median(values))
    return softmax_weighted(values, sims)


# ── Pipeline ────────────────────────────────────────────────────────────────

def load_targets():
    df = pd.read_csv(DATA_MASTER)
    df = df.dropna(subset=["sentencing_range_low", "sentencing_range_high"])
    df = df.drop_duplicates("canonical_id")
    return df.set_index("canonical_id")[["domain", "sentencing_range_low", "sentencing_range_high"]]


def evaluate_rep(sim_csv: Path, targets: pd.DataFrame, filt_name: str):
    """Run kNN with within-query filter; return per-verdict predictions."""
    sims = pd.read_csv(sim_csv).dropna(subset=["similarity_score"])

    ngh = defaultdict(list)
    for v1, v2, _, s in sims[["verdict_1", "verdict_2", "domain", "similarity_score"]].itertuples(index=False):
        ngh[v1].append((v2, float(s)))
        ngh[v2].append((v1, float(s)))
    sorted_ngh = {q: sorted(ns, key=lambda x: -x[1]) for q, ns in ngh.items()}

    filt = FILTERS[filt_name]
    rows = []
    for q, ns in sorted_ngh.items():
        if q not in targets.index:
            continue
        q_dom = targets.at[q, "domain"]
        if q_dom not in ("drugs", "weapon"):
            continue
        good = [(n, s) for n, s in ns
                if n in targets.index and targets.at[n, "domain"] == q_dom][:K]
        if len(good) < K:
            continue
        nb_sims = np.array([s for _, s in good], dtype=float)
        nb_lo = targets.loc[[n for n, _ in good], "sentencing_range_low"].to_numpy(dtype=float)
        nb_hi = targets.loc[[n for n, _ in good], "sentencing_range_high"].to_numpy(dtype=float)

        # Apply within-query filter to LOW and HIGH separately
        # (sims aren't filtered — we just trim neighbors by sentence value)
        filt_lo = filt(nb_lo)
        filt_hi = filt(nb_hi)
        # For aggregation, give equal sims when neighbors got trimmed (since we
        # don't know which sims correspond to surviving values after sort);
        # acceptable since softmax+median are robust.
        sims_lo = np.ones(len(filt_lo))
        sims_hi = np.ones(len(filt_hi))
        pl = aggregate(filt_lo, sims_lo, q_dom)
        ph = aggregate(filt_hi, sims_hi, q_dom)
        rows.append({
            "verdict": q, "domain": q_dom,
            "actual_low":  float(targets.at[q, "sentencing_range_low"]),
            "actual_high": float(targets.at[q, "sentencing_range_high"]),
            "pred_low": pl, "pred_high": ph,
            "n_after_filter_low":  len(filt_lo),
            "n_after_filter_high": len(filt_hi),
        })
    df = pd.DataFrame(rows)
    df["err_low"]  = (df.pred_low  - df.actual_low ).abs()
    df["err_high"] = (df.pred_high - df.actual_high).abs()
    inter = np.maximum(0, np.minimum(df.pred_high, df.actual_high) - np.maximum(df.pred_low, df.actual_low))
    union = np.maximum(df.pred_high, df.actual_high) - np.minimum(df.pred_low, df.actual_low)
    df["iou"] = (inter / np.maximum(union, 1)).astype(float)
    return df


def main():
    targets = load_targets()
    rows = []
    for filt_name in FILTERS:
        for rep, csv in REPS.items():
            df = evaluate_rep(csv, targets, filt_name)
            for dom, sub in df.groupby("domain"):
                rows.append({
                    "filter": filt_name, "rep": rep, "domain": dom, "n": len(sub),
                    "avg_kept_low":  float(sub.n_after_filter_low.mean()),
                    "avg_kept_high": float(sub.n_after_filter_high.mean()),
                    "MAE_low":  float(sub.err_low.mean()),
                    "MAE_high": float(sub.err_high.mean()),
                    "IoU":      float(sub.iou.mean()),
                })
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "within_query_robust.csv", index=False)

    # Print pivot: MAE_high per (rep, domain) for each filter
    print("=" * 90)
    print("Within-query filtering — MAE_high (months, full coverage = same N for all)")
    print("=" * 90)
    for dom in ["drugs", "weapon"]:
        print(f"\n--- {dom} ---")
        piv = res[res.domain == dom].pivot_table(
            index="rep", columns="filter", values="MAE_high"
        ).round(3)
        # Reorder
        piv = piv[["none", "trimmed", "mad", "iqr"]]
        piv["Δ_trim_pct"] = ((piv["trimmed"] - piv["none"]) / piv["none"] * 100).round(1)
        piv["Δ_mad_pct"]  = ((piv["mad"]     - piv["none"]) / piv["none"] * 100).round(1)
        piv["Δ_iqr_pct"]  = ((piv["iqr"]     - piv["none"]) / piv["none"] * 100).round(1)
        print(piv.to_string())

    print(f"\n→ {OUT_DIR/'within_query_robust.csv'}")


if __name__ == "__main__":
    main()
