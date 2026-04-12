#!/usr/bin/env python3
"""
Post-process v6 multimodel outputs under an experiment folder (like FINAL_RESULTS_9_MODELS).

Reads:
  EXPERIMENT_ROOT/<drugs|weapon>/results_*/similarity_database_hybrid_full_gpt_v6score_*_stats.json
  matching *_preds.csv

Writes under EXPERIMENT_ROOT/analysis/:
  - leaderboard_v6.csv
  - pairwise_mcnemar_<domain>.csv (McNemar p-values, BH-FDR)
  - shuffled_baseline_v6.csv (actual F1 vs shuffled baseline, 1000 shuffles)

No API calls.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy import stats

N_SHUFFLES = 1000
GT_COL = "similarity_binary_0"


def fdr_bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    sp = p[order]
    ranks = np.arange(1, n + 1, dtype=float)
    q = sp * n / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(n)
    out[order] = q
    return out


def mcnemar_binom(y_true: np.ndarray, y_a: np.ndarray, y_b: np.ndarray) -> tuple[float, float, int, int]:
    """Two-sided binomial test on discordant pairs (same as run_statistical_tests_final)."""
    n12 = int(np.sum((y_a == y_true) & (y_b != y_true)))
    n21 = int(np.sum((y_a != y_true) & (y_b == y_true)))
    if n12 + n21 == 0:
        return 0.0, 1.0, n12, n21
    r = stats.binomtest(n12, n12 + n21, 0.5, alternative="two-sided")
    return float(r.statistic), float(r.pvalue), n12, n21


def load_v6_run(stats_path: Path) -> dict | None:
    try:
        with open(stats_path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None
    if d.get("method") != "v6_score_multimodel":
        return None
    preds = stats_path.with_name(stats_path.name.replace("_stats.json", "_preds.csv"))
    g = d.get("global_best") or {}
    loo = d.get("loo") or {}
    return {
        "domain": d.get("domain"),
        "model": d.get("model"),
        "representation_id": d.get("representation_id"),
        "csv": d.get("csv"),
        "complete": d.get("complete", False),
        "n_pairs": d.get("n_pairs"),
        "n_valid": d.get("n_valid"),
        "n_failed": d.get("n_failed"),
        "threshold": d.get("best_threshold"),
        "f1_global": g.get("f1"),
        "f1_loo": loo.get("f1"),
        "precision": g.get("precision"),
        "recall": g.get("recall"),
        "stats_path": stats_path,
        "preds_path": preds,
    }


def shuffled_f1_report(y_true: np.ndarray, y_pred: np.ndarray, n_shuffles: int = N_SHUFFLES) -> dict:
    actual = f1_score(y_true, y_pred, zero_division=0)
    shufs = []
    rng = np.random.default_rng(42)
    for _ in range(n_shuffles):
        y_s = rng.permutation(y_pred)
        shufs.append(f1_score(y_true, y_s, zero_division=0))
    shufs = np.array(shufs)
    p_one_sided = float(np.mean(shufs >= actual))
    return {
        "actual_f1": float(actual),
        "shuffled_mean_f1": float(np.mean(shufs)),
        "shuffled_std_f1": float(np.std(shufs)),
        "shuffled_ci_low": float(np.percentile(shufs, 2.5)),
        "shuffled_ci_high": float(np.percentile(shufs, 97.5)),
        "p_vs_shuffled": p_one_sided,
        "n_shuffles": n_shuffles,
    }


def collect_runs(experiment_root: Path) -> list[dict]:
    runs: list[dict] = []
    for domain in ("drugs", "weapon"):
        sub = "results_drugs" if domain == "drugs" else "results_weapon"
        rd = experiment_root / domain / sub
        if not rd.is_dir():
            continue
        for p in sorted(rd.glob("similarity_database_hybrid_full_gpt_v6score_*_stats.json")):
            r = load_v6_run(p)
            if r:
                runs.append(r)
    return runs


def pairwise_table(
    domain: str,
    runs: list[dict],
) -> pd.DataFrame:
    domain_runs = [r for r in runs if r["domain"] == domain and r.get("threshold") is not None]
    # Only models with full preds for fair pairing: need same preds file length
    models_data: dict[str, tuple[pd.DataFrame, float, str]] = {}
    for r in domain_runs:
        m = r["model"]
        if not r["preds_path"].exists():
            continue
        df = pd.read_csv(r["preds_path"])
        thr = float(r["threshold"])
        if GT_COL not in df.columns:
            continue
        models_data[m] = (df, thr, r["preds_path"].name)

    rows = []
    for a, b in combinations(sorted(models_data.keys()), 2):
        df_a, thr_a, _ = models_data[a]
        df_b, thr_b, _ = models_data[b]
        key = ["verdict_1", "verdict_2"]
        merged = df_a[key + [GT_COL, "score"]].merge(
            df_b[key + ["score"]], on=key, suffixes=("_a", "_b")
        )
        if len(merged) == 0:
            continue
        y = merged[GT_COL].astype(int).values
        pa = (pd.to_numeric(merged["score_a"], errors="coerce") >= thr_a).astype(int).values
        pb = (pd.to_numeric(merged["score_b"], errors="coerce") >= thr_b).astype(int).values
        mask = ~(np.isnan(merged["score_a"].astype(float)) | np.isnan(merged["score_b"].astype(float)))
        if np.sum(mask) < 10:
            continue
        y = y[mask]
        pa = pa[mask]
        pb = pb[mask]
        stat, p, n12, n21 = mcnemar_binom(y, pa, pb)
        rows.append(
            {
                "domain": domain,
                "model_a": a,
                "model_b": b,
                "n_pairs": len(y),
                "f1_a": f1_score(y, pa, zero_division=0),
                "f1_b": f1_score(y, pb, zero_division=0),
                "mcnemar_stat": stat,
                "p_value": p,
                "n_a_correct_b_wrong": n12,
                "n_a_wrong_b_correct": n21,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    pv = df["p_value"].values.astype(float)
    df["p_fdr_bh"] = fdr_bh_adjust(pv)
    df["significant_0.05"] = df["p_fdr_bh"] < 0.05
    return df


def shuffled_rows(runs: list[dict]) -> pd.DataFrame:
    out = []
    for r in runs:
        if r.get("threshold") is None:
            continue
        if not r["preds_path"].exists():
            continue
        df = pd.read_csv(r["preds_path"])
        if GT_COL not in df.columns or "score" not in df.columns:
            continue
        thr = float(r["threshold"])
        y = df[GT_COL].astype(int).values
        sc = pd.to_numeric(df["score"], errors="coerce")
        st = df["status"] if "status" in df.columns else None
        ok = sc.notna() & ((st == "ok") if st is not None else pd.Series(True, index=df.index))
        if ok.sum() < 10:
            continue
        y = y[ok.values]
        y_pred = (sc[ok] >= thr).astype(int).values
        rep = shuffled_f1_report(y, y_pred)
        out.append(
            {
                "domain": r["domain"],
                "model": r["model"],
                "n_used": len(y),
                **rep,
            }
        )
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "experiment_root",
        type=str,
        help="Path to experiment root (same as --output-root for v6_score_multimodel_experiment.py)",
    )
    args = ap.parse_args()
    root = Path(args.experiment_root).resolve()
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(root)
    if not runs:
        print(f"No v6 stats found under {root}")
        return

    lb = pd.DataFrame(
        [
            {
                "domain": r["domain"],
                "model": r["model"],
                "representation_id": r["representation_id"],
                "complete": r["complete"],
                "n_valid": r["n_valid"],
                "n_failed": r["n_failed"],
                "threshold": r["threshold"],
                "f1_global": r["f1_global"],
                "f1_loo": r["f1_loo"],
                "precision": r["precision"],
                "recall": r["recall"],
            }
            for r in runs
        ]
    )
    lb = lb.sort_values(["domain", "f1_global"], ascending=[True, False], na_position="last")
    lb_path = out_dir / "leaderboard_v6.csv"
    lb.to_csv(lb_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {lb_path} ({len(lb)} rows)")

    for dom in ("drugs", "weapon"):
        pw = pairwise_table(dom, runs)
        if len(pw):
            ppath = out_dir / f"pairwise_mcnemar_{dom}.csv"
            pw.to_csv(ppath, index=False, encoding="utf-8-sig")
            print(f"Wrote {ppath} ({len(pw)} comparisons)")

    sh = shuffled_rows(runs)
    if len(sh):
        sh_path = out_dir / "shuffled_baseline_v6.csv"
        sh.to_csv(sh_path, index=False, encoding="utf-8-sig")
        print(f"Wrote {sh_path} ({len(sh)} rows)")

    print("Done.")


if __name__ == "__main__":
    main()
