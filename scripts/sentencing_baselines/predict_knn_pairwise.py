#!/usr/bin/env python3
"""
Universal kNN + selective-prediction pipeline for sentencing-range prediction.

Takes a similarity_scores file (verdict_1, verdict_2, domain, similarity_score)
and produces predicted (low, high) per query verdict via kNN aggregation.

Operating points (matching the headline 3-way comparison):
  drugs:  sim>=40, k>=3, sigma<=Q50,  agg=median
  weapon: sim>=60, k>=3, sigma<=Q50,  agg=softmax-weighted-mean

Outputs:
  - predictions CSV (one row per evaluated query verdict)
  - metrics CSV   (one row per (rep, domain, sigma_filter))
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!")
DATA_MASTER = ROOT / "new_try/innovation_submission/data_master_final/verdicts_clean.csv"

# Canonical operating point per the 3-way headline (commit 87d850e).
DOMAIN_CFG = {
    "drugs":  {"sim_thr": 40.0, "agg": "median"},
    "weapon": {"sim_thr": 60.0, "agg": "softmax"},
}
K_MIN = 3
K_TOP = 3  # top-K mode: take this many most-similar neighbors


def load_targets() -> pd.DataFrame:
    df = pd.read_csv(DATA_MASTER)
    df = df[["canonical_id", "domain", "sentencing_range_low", "sentencing_range_high"]].copy()
    df = df.rename(columns={"canonical_id": "verdict"})
    df = df.dropna(subset=["sentencing_range_low", "sentencing_range_high"])
    # canonical_id may have duplicates (multiple original IDs collapsing) -- keep first
    df = df.drop_duplicates(subset=["verdict"], keep="first")
    return df.set_index("verdict")


def softmax_weighted_mean(values: np.ndarray, sims: np.ndarray, T: float = 10.0) -> float:
    z = sims / T
    z = z - z.max()
    w = np.exp(z)
    w = w / w.sum()
    return float(np.dot(w, values))


def aggregate(values: np.ndarray, sims: np.ndarray, agg: str) -> float:
    if agg == "median":
        return float(np.median(values))
    if agg == "softmax":
        return softmax_weighted_mean(values, sims)
    if agg == "mean":
        return float(np.mean(values))
    if agg == "weighted_mean":
        if sims.sum() <= 0:
            return float(np.mean(values))
        return float(np.dot(sims / sims.sum(), values))
    raise ValueError(f"Unknown agg: {agg}")


def iou_range(a_lo: float, a_hi: float, p_lo: float, p_hi: float) -> float:
    lo = max(a_lo, p_lo)
    hi = min(a_hi, p_hi)
    inter = max(0.0, hi - lo)
    union = max(a_hi, p_hi) - min(a_lo, p_lo)
    return inter / union if union > 0 else 0.0


def predict_for_rep(
    sim_csv: Path,
    rep_label: str,
    targets: pd.DataFrame,
    mode: str = "topk",
    agg_override: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run kNN selective prediction for one representation.

    mode='fixed':  use per-domain sim_thr from DOMAIN_CFG (canonical headline)
    mode='topk':   take K_TOP most-similar neighbors per query, no threshold
    mode='percentile': use Q50 of THIS rep's score distribution per domain
    """
    sims = pd.read_csv(sim_csv, usecols=["verdict_1", "verdict_2", "domain", "similarity_score"])
    sims = sims.dropna(subset=["similarity_score"])

    # Per-rep, per-domain percentile thresholds (Q50)
    if mode == "percentile":
        pct_thr = {
            d: float(g["similarity_score"].quantile(0.5))
            for d, g in sims.groupby("domain")
        }
    else:
        pct_thr = {}

    # Build neighbor lists per query verdict (use both directions: sim is symmetric).
    neighbors: Dict[str, List[Tuple[str, float]]] = {}
    for v1, v2, dom, s in sims.itertuples(index=False):
        neighbors.setdefault(v1, []).append((v2, float(s)))
        neighbors.setdefault(v2, []).append((v1, float(s)))

    rows = []
    for query, nbrs in neighbors.items():
        if query not in targets.index:
            continue
        q_dom = targets.at[query, "domain"]
        if q_dom not in DOMAIN_CFG:
            continue
        cfg = DOMAIN_CFG[q_dom]
        if mode == "fixed":
            sim_thr = cfg["sim_thr"]
        elif mode == "percentile":
            sim_thr = pct_thr[q_dom]
        else:  # topk
            sim_thr = -np.inf

        # Filter: same domain, known target, sim>=thr
        good = []
        for nb, s in nbrs:
            if nb not in targets.index:
                continue
            if targets.at[nb, "domain"] != q_dom:
                continue
            if s < sim_thr:
                continue
            good.append((nb, s))
        good.sort(key=lambda x: -x[1])

        if mode == "topk":
            if len(good) < K_TOP:
                continue
            good = good[:K_TOP]
        else:
            if len(good) < K_MIN:
                continue
        nb_ids = [g[0] for g in good]
        nb_sims = np.array([g[1] for g in good], dtype=float)
        nb_lo = targets.loc[nb_ids, "sentencing_range_low"].to_numpy(dtype=float)
        nb_hi = targets.loc[nb_ids, "sentencing_range_high"].to_numpy(dtype=float)

        sigma_lo = float(np.std(nb_lo))
        sigma_hi = float(np.std(nb_hi))

        agg = agg_override or cfg["agg"]
        pred_lo = aggregate(nb_lo, nb_sims, agg)
        pred_hi = aggregate(nb_hi, nb_sims, agg)

        rows.append({
            "verdict": query,
            "domain": q_dom,
            "rep": rep_label,
            "n_neighbors": len(good),
            "actual_low": float(targets.at[query, "sentencing_range_low"]),
            "actual_high": float(targets.at[query, "sentencing_range_high"]),
            "pred_low": pred_lo,
            "pred_high": pred_hi,
            "sigma_low": sigma_lo,
            "sigma_high": sigma_hi,
            "mean_sim": float(np.mean(nb_sims)),
        })

    preds = pd.DataFrame(rows)
    if preds.empty:
        return preds, pd.DataFrame()

    preds["err_low"]  = (preds["pred_low"]  - preds["actual_low"]).abs()
    preds["err_high"] = (preds["pred_high"] - preds["actual_high"]).abs()
    preds["iou"] = preds.apply(
        lambda r: iou_range(r["actual_low"], r["actual_high"],
                            r["pred_low"], r["pred_high"]),
        axis=1,
    )

    # Metrics: with and without the sigma<=Q50 filter (per-domain Q50)
    metric_rows = []
    for dom, sub in preds.groupby("domain"):
        for sig_filter in ["no_sigma", "with_sigma"]:
            ev = sub
            if sig_filter == "with_sigma":
                q50_lo = sub["sigma_low"].quantile(0.5)
                q50_hi = sub["sigma_high"].quantile(0.5)
                ev = sub[(sub["sigma_low"] <= q50_lo) & (sub["sigma_high"] <= q50_hi)]
            if len(ev) == 0:
                continue
            metric_rows.append({
                "rep": rep_label,
                "domain": dom,
                "sigma": sig_filter,
                "n": len(ev),
                "MAE_low":  float(ev["err_low"].mean()),
                "MAE_high": float(ev["err_high"].mean()),
                "MedAE_low":  float(ev["err_low"].median()),
                "MedAE_high": float(ev["err_high"].median()),
                "IoU": float(ev["iou"].mean()),
            })
    return preds, pd.DataFrame(metric_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-csv", required=True, type=Path)
    ap.add_argument("--rep", required=True, help="Label, e.g. Hybrid-Full or Gemini-Text")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--mode", default="topk", choices=["topk", "fixed", "percentile"])
    ap.add_argument("--agg", default=None, choices=[None, "median", "softmax", "mean", "weighted_mean"])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = load_targets()
    preds, metrics = predict_for_rep(args.sim_csv, args.rep, targets, mode=args.mode, agg_override=args.agg)

    safe = args.rep.replace(" ", "_").replace("/", "_")
    suffix = f"_{args.mode}"
    if args.agg:
        suffix += f"_{args.agg}"
    p_path = args.out_dir / f"preds_{safe}{suffix}.csv"
    m_path = args.out_dir / f"metrics_{safe}{suffix}.csv"
    preds.to_csv(p_path, index=False)
    metrics.to_csv(m_path, index=False)
    print(f"Saved {len(preds)} predictions -> {p_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
