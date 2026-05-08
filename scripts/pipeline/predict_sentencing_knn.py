#!/usr/bin/env python3
"""
KNN-based sentencing range prediction using H-Full features.

For each verdict t:
1. Compute Jaccard similarity to every other verdict in the same domain
2. Take top-K most similar neighbors
3. Aggregate their (low, high) ranges via weighted-mean (weights = similarity scores)
4. Output predicted (low, high) per verdict

Scope: leave-one-out on all 5,191 verdicts in verdicts_clean.csv.
Domain isolation: drugs ↔ drugs only, weapon ↔ weapon only.

Usage:
  predict_sentencing_knn.py --K 3 5 10 20
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!")
DATA_DIR = ROOT / "new_try" / "experiments" / "data" / "sentencing_range"
HFULL_CACHE = DATA_DIR / "hfull_features" / "hybrid_full_cache.json"
CLEAN_CSV = DATA_DIR / "verdicts_clean.csv"
OUT_DIR = DATA_DIR / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Feature signature ----------
SKIP_VALUES = {"", "לא", "אין", "none", "null", "n/a", None, "0"}


def _normalize_value(v) -> List[str]:
    """Normalize a feature value into a list of comparable string tokens."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        out: List[str] = []
        for item in v:
            out.extend(_normalize_value(item))
        return out
    if isinstance(v, dict):
        # Flatten dict — uncommon in our data but handle defensively
        return [f"{k}={vv}" for k, vv in v.items()]
    s = str(v).strip().lower()
    if s in SKIP_VALUES:
        return []
    # Common normalizations
    s = s.replace('"', "").replace("'", "")
    return [s]


def feature_signature(feat: dict) -> Set[str]:
    """Convert a feature dict into a set of (key, value) tokens for Jaccard."""
    sig: Set[str] = set()
    for k, v in feat.items():
        if k.startswith("__"):  # error fields
            continue
        for tok in _normalize_value(v):
            sig.add(f"{k}::{tok}")
    return sig


# ---------- Similarity ----------
def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ---------- Aggregation ----------
def weighted_mean(values: List[float], weights: List[float]) -> float:
    """Weighted mean. If all weights are 0, returns simple mean."""
    if not values:
        return float("nan")
    total_w = sum(weights)
    if total_w <= 0:
        return float(np.mean(values))
    return sum(v * w for v, w in zip(values, weights)) / total_w


# ---------- Main pipeline ----------
def predict(K_list: List[int]) -> pd.DataFrame:
    print(f"📥 Loading H-Full features from {HFULL_CACHE.name}...")
    with open(HFULL_CACHE) as f:
        hfull = json.load(f)

    print(f"📥 Loading clean dataset...")
    clean = pd.read_csv(CLEAN_CSV)
    clean["verdict"] = clean["verdict"].astype(str)

    # Build features map (keep only verdicts with valid hfull entries)
    valid = []
    sigs: Dict[str, Set[str]] = {}
    for _, row in clean.iterrows():
        vid = row["verdict"]
        if vid in hfull and "__error" not in hfull[vid]:
            sigs[vid] = feature_signature(hfull[vid])
            valid.append(row)
    df = pd.DataFrame(valid).reset_index(drop=True)
    print(f"   {len(df):,} verdicts with valid features ({100*len(df)/len(clean):.1f}%)")
    print(f"   avg signature size: {np.mean([len(s) for s in sigs.values()]):.1f} tokens")

    # Index by domain
    by_domain: Dict[str, List[int]] = {"drugs": [], "weapon": []}
    for i, row in df.iterrows():
        if row["domain"] in by_domain:
            by_domain[row["domain"]].append(i)
    print(f"   domain split: drugs={len(by_domain['drugs']):,}  weapon={len(by_domain['weapon']):,}")

    K_max = max(K_list)
    rows_out = []

    for domain, idxs in by_domain.items():
        print(f"\n🔍 Computing similarities — {domain} ({len(idxs):,} verdicts)")
        # Pre-extract sigs and labels for this domain
        d_vids = [df.loc[i, "verdict"] for i in idxs]
        d_sigs = [sigs[v] for v in d_vids]
        d_low = np.array([df.loc[i, "sentencing_range_low"] for i in idxs], dtype=float)
        d_high = np.array([df.loc[i, "sentencing_range_high"] for i in idxs], dtype=float)
        n = len(idxs)

        for i in tqdm(range(n), desc=f"KNN {domain}"):
            sims = np.zeros(n, dtype=float)
            si = d_sigs[i]
            for j in range(n):
                if j == i:
                    continue
                sims[j] = jaccard(si, d_sigs[j])
            # Top-K_max neighbors by similarity (descending)
            order = np.argsort(-sims)
            row = {
                "verdict": d_vids[i],
                "domain": domain,
                "actual_low": d_low[i],
                "actual_high": d_high[i],
            }
            for K in K_list:
                top = order[:K]
                w = sims[top]
                row[f"pred_low_K{K}"] = weighted_mean(list(d_low[top]), list(w))
                row[f"pred_high_K{K}"] = weighted_mean(list(d_high[top]), list(w))
                row[f"mean_sim_K{K}"] = float(np.mean(w))
                row[f"min_sim_K{K}"] = float(np.min(w))
            rows_out.append(row)

    return pd.DataFrame(rows_out)


def metrics(df: pd.DataFrame, K_list: List[int]) -> pd.DataFrame:
    """Compute MAE, RMSE, Pearson, within-Xm for each K, target, and domain."""
    rows = []
    for K in K_list:
        for target in ["low", "high"]:
            actual_col = f"actual_{target}"
            pred_col = f"pred_{target}_K{K}"
            for domain in ["drugs", "weapon", "all"]:
                sub = df if domain == "all" else df[df["domain"] == domain]
                a = sub[actual_col].values
                p = sub[pred_col].values
                err = a - p
                abs_err = np.abs(err)
                row = {
                    "K": K,
                    "target": target,
                    "domain": domain,
                    "n": len(sub),
                    "MAE": float(np.mean(abs_err)),
                    "RMSE": float(np.sqrt(np.mean(err**2))),
                    "MedAE": float(np.median(abs_err)),
                    "Pearson": float(np.corrcoef(a, p)[0, 1]) if len(sub) > 1 else float("nan"),
                    "within_3m": float(np.mean(abs_err <= 3) * 100),
                    "within_6m": float(np.mean(abs_err <= 6) * 100),
                    "within_12m": float(np.mean(abs_err <= 12) * 100),
                    "within_24m": float(np.mean(abs_err <= 24) * 100),
                }
                rows.append(row)
    return pd.DataFrame(rows)


def baseline_domain_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-mean baseline: predict the mean of the same-domain training set."""
    rows = []
    for target in ["low", "high"]:
        actual_col = f"actual_{target}"
        for domain in ["drugs", "weapon", "all"]:
            sub = df if domain == "all" else df[df["domain"] == domain]
            mean_val = sub[actual_col].mean()
            err = sub[actual_col].values - mean_val
            abs_err = np.abs(err)
            rows.append({
                "K": "baseline_mean",
                "target": target,
                "domain": domain,
                "n": len(sub),
                "MAE": float(np.mean(abs_err)),
                "RMSE": float(np.sqrt(np.mean(err**2))),
                "MedAE": float(np.median(abs_err)),
                "Pearson": float("nan"),
                "within_3m": float(np.mean(abs_err <= 3) * 100),
                "within_6m": float(np.mean(abs_err <= 6) * 100),
                "within_12m": float(np.mean(abs_err <= 12) * 100),
                "within_24m": float(np.mean(abs_err <= 24) * 100),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="+", default=[3, 5, 10, 20])
    args = ap.parse_args()

    preds_df = predict(args.K)
    pred_path = OUT_DIR / "knn_hfull_predictions.csv"
    preds_df.to_csv(pred_path, index=False)
    print(f"\n✅ Saved predictions: {pred_path}  ({len(preds_df):,} rows)")

    knn_metrics = metrics(preds_df, args.K)
    base_metrics = baseline_domain_mean(preds_df)
    all_metrics = pd.concat([base_metrics, knn_metrics], ignore_index=True)
    metrics_path = OUT_DIR / "knn_hfull_metrics.csv"
    all_metrics.to_csv(metrics_path, index=False)
    print(f"✅ Saved metrics:     {metrics_path}")

    print("\n" + "="*80)
    print("RESULTS — domain=all")
    print("="*80)
    pivot = all_metrics[all_metrics["domain"] == "all"].pivot_table(
        index=["K", "target"], values=["MAE", "Pearson", "within_6m", "within_12m"], aggfunc="first"
    ).round(2)
    print(pivot.to_string())


if __name__ == "__main__":
    main()
