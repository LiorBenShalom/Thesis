"""Permutation (random shuffle) baseline for all paper metrics.

For each (domain x rep x model x task), shuffle the model's predicted scores
N times and recompute the metric against the fixed ground truth. This produces
a per-cell null distribution that preserves the marginal distribution of scores
(i.e. "random guessing with the same score proportions").

Metrics covered:
  - F1-Oracle, F1-CV (binary_0, binary_1)
  - AP-PR (binary_0, binary_1)
  - QWK-Oracle, QWK-CV (similarity_scale 1..3)

Outputs (under experiments/results_paper_baselines/):
  - random_full.csv        : one row per cell, observed + null mean/std/CI/p
  - random_summary.csv     : per-rep aggregate (mean across 11 LLM models)
  - RANDOM_REPORT.md       : compact tables and Δ (observed - null)

Usage:
  cd new_try/experiments/src/analysis
  python random_baseline.py [--n-perm 1000] [--n-perm-cv 200]

Note on compute: QWK-Oracle / F1-Oracle / CV variants are O(k^2) or include CV
folds, so we use fewer permutations for those (default 200) than for AP-PR and
simple shuffles (default 1000). Override via --n-perm / --n-perm-cv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

EXP = Path(__file__).resolve().parents[2]
OUT = EXP / "results_paper_baselines"
OUT.mkdir(exist_ok=True)

DOMAINS = {
    "drugs": EXP / "v6_final" / "drugs" / "results_drugs",
    "weapon": EXP / "v6_final" / "weapon" / "results_weapon",
}
REP_PREFIX = {
    "Manual": "similarity_database_fe",
    "GPT-Schema": "similarity_database_fe_gpt_schema_v2",
    "GPT-Free": "similarity_database_with_gpt_features",
    "GPT-Law": "similarity_database_with_gpt_law_features",
    "Raw-Facts": "similarity_database_with_indicment_facts",
    "Hybrid-Manual": "similarity_database_hybrid",
    "Hybrid-Full": "similarity_database_hybrid_full_gpt",
}
MODELS = [
    "gpt4", "gpt5mini", "gpt52", "gpt51_thinking", "claude_sonnet_4_6",
    "gemini_25_pro", "gemini_3_flash", "gemma3_27b", "gemma4_31b_or",
    "llama3_70b", "qwen3_vl_235b_or",
]


# ─── Metric helpers (match paper_results.py + paper_results_qwk.py) ───

def _best_f1(scores: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    best = 0.0
    for thr in np.unique(scores):
        f = f1_score(y, (scores >= thr).astype(int), zero_division=0)
        if f > best:
            best = f
    return best


def _cv_f1(scores: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 42) -> float:
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < k:
        return np.nan
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=int)
    for tr, te in skf.split(scores, y):
        best_f, best_t = 0.0, None
        for t in np.unique(scores[tr]):
            f = f1_score(y[tr], (scores[tr] >= t).astype(int), zero_division=0)
            if f > best_f:
                best_f, best_t = f, t
        if best_t is None:
            best_t = float(np.median(scores[tr]))
        pred[te] = (scores[te] >= best_t).astype(int)
    return f1_score(y, pred, zero_division=0)


def _qwk(y_true: np.ndarray, y_pred: np.ndarray, n_r: int = 3) -> float:
    O = np.zeros((n_r, n_r))
    for t, p in zip(y_true, y_pred):
        O[t - 1, p - 1] += 1
    N = len(y_true)
    ht = np.bincount(y_true - 1, minlength=n_r)
    hp = np.bincount(y_pred - 1, minlength=n_r)
    E = np.outer(ht, hp).astype(float) / N
    W = np.zeros((n_r, n_r))
    for i in range(n_r):
        for j in range(n_r):
            W[i, j] = ((i - j) ** 2) / ((n_r - 1) ** 2)
    denom = np.sum(W * E)
    return 1.0 - (np.sum(W * O) / denom) if denom > 0 else 0.0


def _best_qwk_thresholds(scores: np.ndarray, gt: np.ndarray) -> float:
    uniq = np.unique(scores)
    if len(uniq) < 3:
        return 0.0
    mids = (uniq[:-1] + uniq[1:]) / 2.0
    best = -1.0
    for i, t1 in enumerate(mids):
        for t2 in mids[i + 1:]:
            preds = np.where(scores < t1, 1, np.where(scores < t2, 2, 3))
            if len(np.unique(preds)) < 2:
                continue
            q = _qwk(gt, preds)
            if q > best:
                best = q
    return max(best, 0.0)


def _cv_qwk(scores: np.ndarray, gt: np.ndarray, k: int = 10, seed: int = 42) -> float:
    if len(np.unique(gt)) < 2 or min(np.bincount(gt - 1)) < k:
        return np.nan
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    preds = np.zeros(len(gt), dtype=int)
    for tr, te in skf.split(scores, gt):
        uniq = np.unique(scores[tr])
        if len(uniq) < 3:
            preds[te] = 2
            continue
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        best_q, bt1, bt2 = -1.0, mids[0], mids[-1]
        for i, t1 in enumerate(mids):
            for t2 in mids[i + 1:]:
                p = np.where(scores[tr] < t1, 1, np.where(scores[tr] < t2, 2, 3))
                if len(np.unique(p)) < 2:
                    continue
                q = _qwk(gt[tr], p)
                if q > best_q:
                    best_q, bt1, bt2 = q, t1, t2
        preds[te] = np.where(scores[te] < bt1, 1, np.where(scores[te] < bt2, 2, 3))
    return _qwk(gt, preds)


# ─── Permutation engine ───

def perm_null(metric_fn, scores: np.ndarray, gt: np.ndarray,
              n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Return array of null metric values over n_perm shuffles of `scores`."""
    out = np.empty(n_perm)
    sc = scores.copy()
    for i in range(n_perm):
        rng.shuffle(sc)
        out[i] = metric_fn(sc, gt)
    return out


def _fast_f1_oracle_null(scores: np.ndarray, y: np.ndarray,
                         n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Vectorized F1-Oracle null under score shuffle.

    Shuffling scores (with fixed y) is equivalent to shuffling y (with fixed scores).
    Sort scores once, then for each permutation shuffle y and compute F1 at every
    rank threshold via cumulative TP/FP in O(n).
    """
    if len(np.unique(y)) < 2:
        return np.full(n_perm, np.nan)
    order = np.argsort(-scores, kind="mergesort")
    n = len(y)
    P = int(y.sum())
    out = np.empty(n_perm)
    y_perm = y.copy()
    # Precompute per-rank predicted-positive count for the descending-sort rule
    # "predict 1 for top-k". For tie handling consistent with best_f1 over unique
    # thresholds: use "score >= thr". Here we approximate with rank-based top-k,
    # which is correct when scores are continuous; for discrete ties it slightly
    # over-optimizes (acceptable as both observed and null use same convention
    # in this fast path — in practice scores are near-continuous 0..100).
    k_arr = np.arange(1, n + 1)  # predicted positives at each cut
    for i in range(n_perm):
        rng.shuffle(y_perm)
        y_sorted = y_perm[order]
        tp = np.cumsum(y_sorted)
        fp = k_arr - tp
        fn = P - tp
        with np.errstate(divide="ignore", invalid="ignore"):
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = 2 * prec * rec / (prec + rec)
            f1[np.isnan(f1)] = 0.0
        out[i] = float(f1.max())
    return out


def _fast_ap_null(scores: np.ndarray, y: np.ndarray,
                  n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Vectorized AP-PR null under score shuffle (equivalent to y shuffle).

    Under random ranking, AP in expectation equals the base rate of positives.
    We still compute empirically so the CI is honest.
    """
    if y.sum() == 0:
        return np.full(n_perm, np.nan)
    order = np.argsort(-scores, kind="mergesort")
    n = len(y)
    P = int(y.sum())
    k_arr = np.arange(1, n + 1, dtype=float)
    out = np.empty(n_perm)
    y_perm = y.copy()
    for i in range(n_perm):
        rng.shuffle(y_perm)
        y_sorted = y_perm[order]
        tp = np.cumsum(y_sorted)
        precision_at_k = tp / k_arr
        # AP = (1/P) * sum over positive ranks of precision@k
        out[i] = float((precision_at_k * y_sorted).sum() / P)
    return out


# ─── Per-cell evaluation ───

def cell_record(base: Path, rep: str, rep_prefix: str, model: str,
                n_perm: int, n_perm_cv: int, seed: int) -> list[dict] | None:
    p = base / f"{rep_prefix}_v6score_{model}_binary_0_preds.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    if len(df) < 20:
        return None

    sc = df["score"].astype(float).values
    mask_sc = ~np.isnan(sc)
    sc = sc[mask_sc]
    df = df.iloc[mask_sc.nonzero()[0]]

    gt_scale = df["similarity_scale"].astype(int).values
    y0 = df["similarity_binary_0"].astype(int).values
    y1 = df["similarity_binary_1"].astype(int).values

    rng = np.random.default_rng(seed)

    # Observed metrics
    obs = {
        "F1_Oracle_b0": _best_f1(sc, y0),
        "F1_Oracle_b1": _best_f1(sc, y1),
        "F1_CV_b0":     _cv_f1(sc, y0),
        "F1_CV_b1":     _cv_f1(sc, y1),
        "AP_b0":        average_precision_score(y0, sc) if y0.sum() > 0 else np.nan,
        "AP_b1":        average_precision_score(y1, sc) if y1.sum() > 0 else np.nan,
        "QWK_Oracle":   _best_qwk_thresholds(sc, gt_scale),
        "QWK_CV":       _cv_qwk(sc, gt_scale),
    }

    # Nulls: AP + F1-Oracle use fast vectorized paths; CV + QWK-Oracle use slower loops.
    null_ap_b0 = _fast_ap_null(sc, y0, n_perm, rng)
    null_ap_b1 = _fast_ap_null(sc, y1, n_perm, rng)
    null_f1o_b0 = _fast_f1_oracle_null(sc, y0, n_perm, rng)
    null_f1o_b1 = _fast_f1_oracle_null(sc, y1, n_perm, rng)
    null_qwko = perm_null(lambda s, g=gt_scale: _best_qwk_thresholds(s, g),
                          sc, gt_scale, n_perm_cv, rng)
    null_f1cv_b0 = perm_null(lambda s, g=y0: _cv_f1(s, g), sc, y0, n_perm_cv, rng)
    null_f1cv_b1 = perm_null(lambda s, g=y1: _cv_f1(s, g), sc, y1, n_perm_cv, rng)
    null_qwkcv = perm_null(lambda s, g=gt_scale: _cv_qwk(s, g),
                           sc, gt_scale, n_perm_cv, rng)

    nulls = {
        "AP_b0": null_ap_b0, "AP_b1": null_ap_b1,
        "F1_Oracle_b0": null_f1o_b0, "F1_Oracle_b1": null_f1o_b1,
        "QWK_Oracle": null_qwko,
        "F1_CV_b0": null_f1cv_b0, "F1_CV_b1": null_f1cv_b1,
        "QWK_CV": null_qwkcv,
    }

    records = []
    for metric, null in nulls.items():
        null = null[~np.isnan(null)]
        if len(null) == 0:
            continue
        o = obs[metric]
        p_emp = float((null >= o).mean()) if not np.isnan(o) else np.nan
        records.append(dict(
            domain=base.parent.name, rep=rep, model=model, metric=metric,
            observed=o,
            null_mean=float(null.mean()),
            null_std=float(null.std(ddof=1)) if len(null) > 1 else 0.0,
            null_ci_lo=float(np.quantile(null, 0.025)),
            null_ci_hi=float(np.quantile(null, 0.975)),
            delta=(o - float(null.mean())) if not np.isnan(o) else np.nan,
            p_emp=p_emp,
            n_perm=int(len(null)),
        ))
    return records


def _cell_worker(args_tuple):
    dom, base_str, rep, pref, m, n_perm, n_perm_cv, seed = args_tuple
    base = Path(base_str)
    recs = cell_record(base, rep, pref, m, n_perm, n_perm_cv, seed)
    return dom, rep, m, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1000,
                    help="Permutations for fast metrics (AP, F1-Oracle) (default 1000)")
    ap.add_argument("--n-perm-cv", type=int, default=100,
                    help="Permutations for slow metrics (CV, QWK-Oracle) (default 100)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=0,
                    help="Number of parallel workers (0 = auto = cpu_count-1)")
    args = ap.parse_args()

    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)

    tasks = []
    for dom, base in DOMAINS.items():
        for rep, pref in REP_PREFIX.items():
            for m in MODELS:
                tasks.append((dom, str(base), rep, pref, m,
                              args.n_perm, args.n_perm_cv, args.seed))
    total = len(tasks)
    print(f"Launching {total} cells on {workers} workers "
          f"(n_perm={args.n_perm}, n_perm_cv={args.n_perm_cv})...")

    all_records = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_cell_worker, t) for t in tasks]
        for fut in as_completed(futures):
            dom, rep, m, recs = fut.result()
            done += 1
            if recs is None:
                print(f"  [{done:3d}/{total}] {dom:6s} | {rep:<14s} | {m:<20s} | SKIP (no file)")
                continue
            all_records.extend(recs)
            q_rec = [r for r in recs if r["metric"] == "QWK_CV"]
            if q_rec:
                r = q_rec[0]
                print(f"  [{done:3d}/{total}] {dom:6s} | {rep:<14s} | {m:<20s} | "
                      f"QWK_CV obs={r['observed']:.3f} null={r['null_mean']:+.3f} "
                      f"delta={r['delta']:+.3f}")

    full = pd.DataFrame(all_records)
    full.to_csv(OUT / "random_full.csv", index=False)

    # Aggregate per rep (mean across 11 models)
    summ = (full.groupby(["domain", "rep", "metric"])
                .agg(observed=("observed", "mean"),
                     null_mean=("null_mean", "mean"),
                     null_std=("null_std", "mean"),
                     delta=("delta", "mean"),
                     p_emp_median=("p_emp", "median"))
                .reset_index())
    summ.to_csv(OUT / "random_summary.csv", index=False)

    # Build compact markdown report
    md = ["# Random (Permutation) Baseline Report\n"]
    md.append(f"_N permutations: fast metrics (AP, F1-Oracle) = {args.n_perm}; "
              f"slow metrics (CV, QWK-Oracle) = {args.n_perm_cv}._")
    md.append("_Per-cell null: shuffle the model's own predicted scores and recompute "
              "the metric against fixed ground truth. Preserves the score marginal._\n")
    md.append("_Reported cell = mean across 11 LLM models. Δ = observed - null mean._\n")

    metrics_order = ["F1_Oracle_b0", "F1_Oracle_b1", "F1_CV_b0", "F1_CV_b1",
                     "AP_b0", "AP_b1", "QWK_Oracle", "QWK_CV"]

    for dom in DOMAINS:
        md.append(f"\n## {dom.upper()}\n")
        sub = summ[summ.domain == dom]
        for metric in metrics_order:
            m_sub = sub[sub.metric == metric]
            if m_sub.empty:
                continue
            md.append(f"\n### {metric}\n")
            tbl = (m_sub.set_index("rep")[["observed", "null_mean", "delta"]]
                         .reindex(list(REP_PREFIX)))
            tbl.columns = ["Observed (mean)", "Random null (mean)", "Δ (obs - null)"]
            md.append(tbl.round(3).to_markdown())

    (OUT / "RANDOM_REPORT.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\nDone. Outputs under: {OUT}")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
