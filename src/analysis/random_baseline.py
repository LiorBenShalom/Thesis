"""Random permutation baseline — advisor's method (GT shuffle, shared per task).

Follows the pattern from new_try/code/calculate_baseline_CORRECT.py:

  Logic:
    - For each (domain, task/metric), shuffle the GROUND TRUTH 1000 times with a
      fixed seed. This creates ONE shared null distribution per (domain, task),
      because the GT is identical across all 77 model/rep cells.
    - For each model's predictions: derive the prediction vector from the same
      scoring convention used by paper_results.py / paper_results_qwk.py
      (Oracle threshold for F1-Oracle and QWK-Oracle; CV-pooled for F1-CV and
      QWK-CV; raw scores for AP-PR).
    - Compare each model's prediction vector against the 1000 shuffled GTs →
      per-cell null F1/AP/QWK distributions → baseline_mean, CI, p_emp.

  Rationale:
    - Shuffling GT preserves the *class proportions* exactly (permutation of a
      vector preserves its marginal distribution).
    - One shared shuffle set per (domain, task) → reproducible, and the same
      null distribution is used for every model/rep (fair comparison).
    - Equivalent under the null to shuffling predictions, but the GT-shuffle
      framing matches the advisor's convention and is the classical
      permutation-test formulation.

Outputs (under experiments/results_paper_baselines/):
  - random_full.csv         : per-cell observed vs null (rep x model x metric)
  - random_per_task.csv     : advisor-style shared baseline (domain x task x metric)
  - random_summary.csv      : aggregated per (domain, rep, metric)
  - RANDOM_REPORT.md        : markdown tables

Usage:
  cd new_try/experiments/src/analysis
  python random_baseline.py [--n-shuffles 1000] [--workers 12] [--seed 42]
"""
from __future__ import annotations

import argparse
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

METRICS_ALL = ["F1_Oracle_b0", "F1_Oracle_b1", "F1_CV_b0", "F1_CV_b1",
               "AP_b0", "AP_b1", "QWK_Oracle", "QWK_CV"]


# ─── Metric helpers (match paper_results.py + paper_results_qwk.py) ───

def best_f1_threshold(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (best_f1, best_threshold). Oracle = threshold chosen on same data."""
    if len(np.unique(y)) < 2:
        return np.nan, float(np.median(scores))
    best_f1, best_t = 0.0, None
    for thr in np.unique(scores):
        f = f1_score(y, (scores >= thr).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(thr)
    if best_t is None:
        best_t = float(np.median(scores))
    return best_f1, best_t


def cv_f1_predictions(scores: np.ndarray, y: np.ndarray,
                      k: int = 5, seed: int = 42) -> np.ndarray | None:
    """Return pooled CV binary predictions (threshold tuned per fold).
    None if CV is not feasible."""
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < k:
        return None
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
    return pred


def best_qwk_thresholds(scores: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    """Return (best_qwk, t1, t2) that optimize QWK mapping scores->1/2/3."""
    uniq = np.unique(scores)
    if len(uniq) < 3:
        return 0.0, 0.0, 50.0
    mids = (uniq[:-1] + uniq[1:]) / 2.0
    best_qwk, best_t1, best_t2 = -1.0, float(mids[0]), float(mids[-1])
    for i, t1 in enumerate(mids):
        for t2 in mids[i + 1:]:
            preds = np.where(scores < t1, 1, np.where(scores < t2, 2, 3))
            if len(np.unique(preds)) < 2:
                continue
            q = _qwk(gt, preds)
            if q > best_qwk:
                best_qwk, best_t1, best_t2 = q, float(t1), float(t2)
    return max(best_qwk, 0.0), best_t1, best_t2


def cv_qwk_predictions(scores: np.ndarray, gt: np.ndarray,
                       k: int = 10, seed: int = 42) -> np.ndarray | None:
    if len(np.unique(gt)) < 2 or min(np.bincount(gt - 1)) < k:
        return None
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
    return preds


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


# ─── Shuffled-GT generator (one shared set per (domain, task)) ───

def generate_shuffled_gts(y_true: np.ndarray, n_shuffles: int, seed: int) -> np.ndarray:
    """Return shape (n_shuffles, len(y_true)) of GT permutations.
    Preserves class proportions exactly (permutation preserves marginal)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    out = np.empty((n_shuffles, n), dtype=y_true.dtype)
    for i in range(n_shuffles):
        out[i] = rng.permutation(y_true)
    return out


# ─── Per-cell: observed metric + null distribution ───

def load_cell(base: Path, rep_prefix: str, model: str) -> dict | None:
    """Load one (rep, model) cell. Returns scores + GTs + binary preds + ordinal preds."""
    p = base / f"{rep_prefix}_v6score_{model}_binary_0_preds.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    if len(df) < 20:
        return None
    sc = df["score"].astype(float).values
    m = ~np.isnan(sc)
    sc = sc[m]
    df = df.iloc[m.nonzero()[0]]
    return dict(
        scores=sc,
        y0=df["similarity_binary_0"].astype(int).values,
        y1=df["similarity_binary_1"].astype(int).values,
        gt_scale=df["similarity_scale"].astype(int).values,
    )


def cell_metrics_and_preds(cell: dict) -> dict:
    """Compute observed metrics and fix prediction vectors (for null evaluation)."""
    sc, y0, y1, gt = cell["scores"], cell["y0"], cell["y1"], cell["gt_scale"]

    f1o_b0, t_b0 = best_f1_threshold(sc, y0)
    f1o_b1, t_b1 = best_f1_threshold(sc, y1)
    pred_f1o_b0 = (sc >= t_b0).astype(int)
    pred_f1o_b1 = (sc >= t_b1).astype(int)

    pred_cv_b0 = cv_f1_predictions(sc, y0)
    pred_cv_b1 = cv_f1_predictions(sc, y1)
    f1cv_b0 = f1_score(y0, pred_cv_b0, zero_division=0) if pred_cv_b0 is not None else np.nan
    f1cv_b1 = f1_score(y1, pred_cv_b1, zero_division=0) if pred_cv_b1 is not None else np.nan

    qwk_o, t1, t2 = best_qwk_thresholds(sc, gt)
    pred_qwko = np.where(sc < t1, 1, np.where(sc < t2, 2, 3))

    pred_qwkcv = cv_qwk_predictions(sc, gt)
    qwk_cv = _qwk(gt, pred_qwkcv) if pred_qwkcv is not None else np.nan

    ap_b0 = average_precision_score(y0, sc) if y0.sum() > 0 else np.nan
    ap_b1 = average_precision_score(y1, sc) if y1.sum() > 0 else np.nan

    return dict(
        observed={
            "F1_Oracle_b0": f1o_b0, "F1_Oracle_b1": f1o_b1,
            "F1_CV_b0": f1cv_b0,   "F1_CV_b1": f1cv_b1,
            "AP_b0": ap_b0,        "AP_b1": ap_b1,
            "QWK_Oracle": qwk_o,   "QWK_CV": qwk_cv,
        },
        preds={
            "F1_Oracle_b0": pred_f1o_b0, "F1_Oracle_b1": pred_f1o_b1,
            "F1_CV_b0": pred_cv_b0,      "F1_CV_b1": pred_cv_b1,
            "AP_b0": sc,                 "AP_b1": sc,       # AP uses raw scores
            "QWK_Oracle": pred_qwko,     "QWK_CV": pred_qwkcv,
        },
    )


def null_distribution(metric: str, preds: np.ndarray,
                      shuf_b0: np.ndarray, shuf_b1: np.ndarray,
                      shuf_scale: np.ndarray) -> np.ndarray:
    """Return (n_shuffles,) null values under shuffled GT.

    shuf_* are the shuffled-GT arrays of shape (n_shuffles, n).
    Predictions are fixed; evaluate the metric on each shuffled GT."""
    if preds is None:
        return np.array([])
    n_shuffles = shuf_b0.shape[0]
    out = np.empty(n_shuffles)

    if metric in ("F1_Oracle_b0", "F1_CV_b0"):
        for i in range(n_shuffles):
            out[i] = f1_score(shuf_b0[i], preds, zero_division=0)
    elif metric in ("F1_Oracle_b1", "F1_CV_b1"):
        for i in range(n_shuffles):
            out[i] = f1_score(shuf_b1[i], preds, zero_division=0)
    elif metric == "AP_b0":
        for i in range(n_shuffles):
            out[i] = average_precision_score(shuf_b0[i], preds) if shuf_b0[i].sum() > 0 else np.nan
    elif metric == "AP_b1":
        for i in range(n_shuffles):
            out[i] = average_precision_score(shuf_b1[i], preds) if shuf_b1[i].sum() > 0 else np.nan
    elif metric in ("QWK_Oracle", "QWK_CV"):
        for i in range(n_shuffles):
            out[i] = _qwk(shuf_scale[i], preds)
    return out


# ─── Parallel per-cell worker ───

def _cell_worker(args):
    """Per-cell null evaluation.

    Each cell shuffles its OWN GT (length = n valid pairs for this cell),
    which handles the case where different (rep, model) combinations have
    slightly different valid-pair counts (e.g. status=='ok' filtering).
    All cells in a domain use the same seed stream for reproducibility.
    """
    (dom, base_str, rep, pref, model, n_shuffles, seed) = args
    base = Path(base_str)
    cell = load_cell(base, pref, model)
    if cell is None:
        return dom, rep, model, None

    mp = cell_metrics_and_preds(cell)
    # Per-cell shuffles (same seed → reproducible; length matches cell's n).
    shuf_b0    = generate_shuffled_gts(cell["y0"],       n_shuffles, seed)
    shuf_b1    = generate_shuffled_gts(cell["y1"],       n_shuffles, seed + 1)
    shuf_scale = generate_shuffled_gts(cell["gt_scale"], n_shuffles, seed + 2)

    rows = []
    for metric in METRICS_ALL:
        obs = mp["observed"][metric]
        preds = mp["preds"][metric]
        null = null_distribution(metric, preds, shuf_b0, shuf_b1, shuf_scale)
        null = null[~np.isnan(null)]
        if len(null) == 0:
            continue
        p_emp = float((null >= obs).mean()) if not np.isnan(obs) else np.nan
        rows.append(dict(
            domain=dom, rep=rep, model=model, metric=metric,
            n=int(len(cell["scores"])),
            observed=float(obs) if not np.isnan(obs) else np.nan,
            baseline_mean=float(null.mean()),
            baseline_std=float(null.std(ddof=1)) if len(null) > 1 else 0.0,
            baseline_ci_lo=float(np.quantile(null, 0.025)),
            baseline_ci_hi=float(np.quantile(null, 0.975)),
            improvement=(float(obs) - float(null.mean())) if not np.isnan(obs) else np.nan,
            p_value=p_emp,
            significantly_better=bool(p_emp < 0.05) if not np.isnan(p_emp) else False,
            n_shuffles=int(len(null)),
        ))
    return dom, rep, model, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shuffles", type=int, default=1000,
                    help="Number of GT permutations (default 1000, matches advisor's script)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel workers (0 = auto)")
    args = ap.parse_args()

    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)

    # ─── Step 1: log GT proportions for reference ───
    print(f"Method: shuffle GT {args.n_shuffles} times per cell (advisor's method).")
    print(f"Preserves class proportions (permutation has same marginal).\n")
    print("Canonical GT proportions per domain (from Manual+gpt4):")
    for dom, base in DOMAINS.items():
        ref_path = base / f"{REP_PREFIX['Manual']}_v6score_gpt4_binary_0_preds.csv"
        if not ref_path.exists():
            continue
        df = pd.read_csv(ref_path)
        if "status" in df.columns:
            df = df[df["status"] == "ok"]
        y0 = df["similarity_binary_0"].astype(int).values
        y1 = df["similarity_binary_1"].astype(int).values
        scale = df["similarity_scale"].astype(int).values
        print(f"  {dom}: n={len(df)}, p(b0=1)={y0.mean():.3f}, "
              f"p(b1=1)={y1.mean():.3f}, scale dist={np.bincount(scale)[1:].tolist()}")

    # ─── Step 2: per-cell null evaluation (parallel, per-cell shuffles) ───
    tasks = []
    for dom, base in DOMAINS.items():
        for rep, pref in REP_PREFIX.items():
            for m in MODELS:
                tasks.append((dom, str(base), rep, pref, m,
                              args.n_shuffles, args.seed))
    total = len(tasks)
    print(f"\nStep 2: evaluating {total} cells on {workers} workers")

    all_rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_cell_worker, t) for t in tasks]
        for fut in as_completed(futures):
            dom, rep, m, rows = fut.result()
            done += 1
            if rows is None:
                print(f"  [{done:3d}/{total}] {dom:6s} | {rep:<14s} | {m:<20s} | SKIP")
                continue
            all_rows.extend(rows)
            q = [r for r in rows if r["metric"] == "QWK_CV"]
            if q:
                r = q[0]
                sig = "✓" if r["significantly_better"] else "✗"
                print(f"  [{done:3d}/{total}] {dom:6s} | {rep:<14s} | {m:<20s} | "
                      f"QWK-CV obs={r['observed']:.3f} null={r['baseline_mean']:+.3f} "
                      f"p={r['p_value']:.3f} {sig}")

    full = pd.DataFrame(all_rows)
    full.to_csv(OUT / "random_full.csv", index=False)

    # ─── Step 3: advisor-style shared-baseline table (domain x task/metric) ───
    # Since the shuffled GTs are SHARED, the null distribution of most metrics
    # only depends on (domain, metric). We report the shared-null summary too.
    # For F1/AP we can analytically compute "average over model preds" =
    # simply aggregate the baseline_mean across all rep×model cells.
    per_task = (full.groupby(["domain", "metric"])
                     .agg(baseline_mean=("baseline_mean", "mean"),
                          baseline_std=("baseline_mean", "std"),
                          baseline_ci_lo=("baseline_ci_lo", "mean"),
                          baseline_ci_hi=("baseline_ci_hi", "mean"),
                          observed_mean=("observed", "mean"),
                          observed_best=("observed", "max"),
                          n_cells=("observed", "count"),
                          frac_sig=("significantly_better", "mean"))
                     .reset_index())
    per_task.to_csv(OUT / "random_per_task.csv", index=False)

    # ─── Step 4: per-rep summary ───
    summ = (full.groupby(["domain", "rep", "metric"])
                .agg(observed=("observed", "mean"),
                     baseline_mean=("baseline_mean", "mean"),
                     improvement=("improvement", "mean"),
                     p_median=("p_value", "median"),
                     frac_sig=("significantly_better", "mean"))
                .reset_index())
    summ.to_csv(OUT / "random_summary.csv", index=False)

    # ─── Step 5: markdown report ───
    md = ["# Random Permutation Baseline — Advisor's Method\n"]
    md.append(f"_Method: shuffle GROUND TRUTH {args.n_shuffles} times per (domain, task) "
              "with fixed seed. One shared shuffle set per (domain, task), used across "
              "all 77 rep x model cells. Shuffling preserves class proportions exactly._\n")
    md.append("_Equivalent under the null to shuffling predictions; the GT-shuffle framing "
              "matches the canonical permutation-test formulation (see "
              "`new_try/code/calculate_baseline_CORRECT.py`)._\n")

    md.append("\n## 1. Shared baseline per (domain, metric)\n")
    md.append("_One number per (domain, metric) — the null distribution mean, "
              "averaged across rep x model cells that share the same shuffled GT. "
              "`frac_sig` = fraction of cells whose p-value < 0.05._\n")
    for dom in DOMAINS:
        sub = per_task[per_task.domain == dom].copy()
        sub = sub.set_index("metric").reindex(METRICS_ALL)
        sub = sub.rename(columns={"baseline_mean": "Random baseline (mean)",
                                  "baseline_ci_lo": "CI-lo",
                                  "baseline_ci_hi": "CI-hi",
                                  "observed_mean": "Observed (avg across 77 cells)",
                                  "observed_best": "Observed (best cell)",
                                  "frac_sig": "Frac cells p<0.05"})[[
            "Random baseline (mean)", "CI-lo", "CI-hi",
            "Observed (avg across 77 cells)", "Observed (best cell)",
            "Frac cells p<0.05",
        ]]
        md.append(f"\n### {dom.upper()}\n")
        md.append(sub.round(3).to_markdown())

    md.append("\n\n## 2. Per-rep summary — mean across 11 models\n")
    for dom in DOMAINS:
        md.append(f"\n### {dom.upper()}\n")
        for metric in METRICS_ALL:
            sub = summ[(summ.domain == dom) & (summ.metric == metric)].copy()
            if sub.empty:
                continue
            sub = sub.set_index("rep").reindex(list(REP_PREFIX))
            sub = sub[["observed", "baseline_mean", "improvement", "frac_sig"]].rename(
                columns={"observed": "Observed",
                         "baseline_mean": "Random null",
                         "improvement": "Δ (obs - null)",
                         "frac_sig": "Frac cells p<0.05"})
            md.append(f"\n**{metric}**\n")
            md.append(sub.round(3).to_markdown())

    (OUT / "RANDOM_REPORT.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\nDone. Outputs under: {OUT}")
    for p in sorted(OUT.iterdir()):
        if p.is_file():
            print(f"  {p.name}")


if __name__ == "__main__":
    main()
