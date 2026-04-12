#!/usr/bin/env python3
"""
Full statistical analysis for v6_full_matrix vs optional baseline experiment (e.g. 9-model hybrid_full_gpt run).

Outputs (under EXPERIMENT_ROOT/analysis_full/):
  - master_runs.csv — every complete v6 stats row (+ AP when computable)
  - summary_representation_domain_task.csv — mean/std F1 per rep × domain × task
  - wilcoxon_representation_pairs.csv — paired Wilcoxon on F1 differences (like thesis slides)
  - mcnemar_representation_pairs_by_cell.csv — McNemar for each (rep_a, rep_b, domain, task, model)
  - shuffled_baseline_all_runs.csv — F1 vs label-shuffled baseline (1000 shuffles)
  - friedman_representation_<domain>_<task>.csv — Friedman test across representations (complete cases only)
  - vs_baseline_hybrid_full_gpt.csv — ΔF1 vs BASELINE_ROOT for overlapping models (hybrid_full_gpt only)

Requires: pandas, numpy, scipy, sklearn (same as v6_experiment_report.py).
No API calls.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parents[2] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

GT_COL = "similarity_binary_0"
N_SHUFFLE = 1000


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
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
    n12 = int(np.sum((y_a == y_true) & (y_b != y_true)))
    n21 = int(np.sum((y_a != y_true) & (y_b == y_true)))
    if n12 + n21 == 0:
        return 0.0, 1.0, n12, n21
    r = stats.binomtest(n12, n12 + n21, 0.5, alternative="two-sided")
    return float(r.statistic), float(r.pvalue), n12, n21


def load_stats_json(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if d.get("method") != "v6_score_multimodel":
        return None
    return d


def ap_from_preds(preds: Path, task: str) -> float | None:
    if not preds.exists():
        return None
    label = "similarity_binary_0" if task == "binary_0" else "similarity_binary_1"
    y, s = [], []
    with preds.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("status") != "ok":
                continue
            try:
                y.append(int(row[label]))
                s.append(float(row["score"]))
            except (ValueError, KeyError):
                continue
    if len(y) < 2 or len(set(y)) < 2:
        return None
    return float(average_precision_score(y, s))


def collect_master(experiment_root: Path) -> pd.DataFrame:
    rows = []
    for domain in ("drugs", "weapon"):
        sub = experiment_root / domain / (f"results_{domain}")
        if not sub.is_dir():
            continue
        for p in sorted(sub.glob("*_stats.json")):
            if "_binary_" not in p.name:
                continue
            d = load_stats_json(p)
            if not d or not d.get("complete"):
                continue
            task = d.get("task") or ("binary_0" if "binary_0" in p.name else "binary_1")
            g = d.get("global_best") or {}
            preds = Path(d.get("preds_csv") or "")
            if not preds.is_absolute():
                preds = sub / preds.name
            if not preds.exists():
                preds = p.with_name(p.name.replace("_stats.json", "_preds.csv"))
            ap = ap_from_preds(preds, task) if preds.exists() else None
            rows.append(
                {
                    "domain": d.get("domain"),
                    "task": task,
                    "model": d.get("model"),
                    "representation_id": d.get("representation_id"),
                    "f1": g.get("f1"),
                    "precision": g.get("precision"),
                    "recall": g.get("recall"),
                    "threshold": d.get("best_threshold"),
                    "n_valid": d.get("n_valid"),
                    "n_failed": d.get("n_failed"),
                    "ap_pr": ap,
                    "stats_path": str(p),
                    "preds_path": str(preds),
                }
            )
    return pd.DataFrame(rows)


def shuffled_report(y_true: np.ndarray, y_pred: np.ndarray, n_shuffles: int = N_SHUFFLE) -> dict:
    actual = f1_score(y_true, y_pred, zero_division=0)
    rng = np.random.default_rng(42)
    shufs = []
    for _ in range(n_shuffles):
        y_s = rng.permutation(y_pred)
        shufs.append(f1_score(y_true, y_s, zero_division=0))
    shufs = np.array(shufs)
    return {
        "f1_actual": float(actual),
        "shuf_mean": float(np.mean(shufs)),
        "shuf_std": float(np.std(shufs)),
        "p_vs_shuffled": float(np.mean(shufs >= actual)),
        "n": len(y_true),
    }


def preds_binary_labels(preds: Path, threshold: float, task: str) -> pd.DataFrame | None:
    if not preds.exists():
        return None
    label = "similarity_binary_0" if task == "binary_0" else "similarity_binary_1"
    df = pd.read_csv(preds)
    if not {"verdict_1", "verdict_2", "score", label}.issubset(df.columns):
        return None
    st = df["status"] if "status" in df.columns else None
    sc = pd.to_numeric(df["score"], errors="coerce")
    ok = sc.notna() & ((st == "ok") if st is not None else pd.Series(True, index=df.index))
    df = df.loc[ok].copy()
    df["_pred"] = (sc[ok] >= float(threshold)).astype(int)
    df["_y"] = df[label].astype(int)
    return df[["verdict_1", "verdict_2", "_y", "_pred"]]


def wilcoxon_rep_pairs(master: pd.DataFrame) -> pd.DataFrame:
    """Paired F1 differences across (model, domain, task) for each representation pair."""
    reps = sorted(master["representation_id"].unique())
    rows = []
    key = ["model", "domain", "task"]
    for a, b in combinations(reps, 2):
        ma = master[master["representation_id"] == a].set_index(key)["f1"]
        mb = master[master["representation_id"] == b].set_index(key)["f1"]
        common = ma.index.intersection(mb.index)
        if len(common) < 5:
            continue
        da = ma.loc[common].values.astype(float)
        db = mb.loc[common].values.astype(float)
        diff = da - db
        try:
            w = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
            stat, p = float(w.statistic), float(w.pvalue)
        except Exception:
            stat, p = np.nan, np.nan
        mean_d = float(np.mean(diff))
        std_d = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
        cohen_d = mean_d / std_d if std_d > 1e-9 else np.nan
        wins_a = int(np.sum(diff > 0))
        wins_b = int(np.sum(diff < 0))
        rows.append(
            {
                "rep_a": a,
                "rep_b": b,
                "n_pairs": len(common),
                "mean_diff_f1_a_minus_b": mean_d,
                "cohen_d_paired": cohen_d,
                "wins_a": wins_a,
                "wins_b": wins_b,
                "wilcoxon_stat": stat,
                "p_value": p,
            }
        )
    df = pd.DataFrame(rows)
    if len(df):
        df["p_fdr_bh"] = fdr_bh(df["p_value"].values)
    return df


def mcnemar_all_pairs(master: pd.DataFrame) -> pd.DataFrame:
    """
    McNemar for each representation pair, per (domain, task, model).
    """
    rows = []
    # index master for quick lookup of threshold + preds
    lut = {}
    for _, r in master.iterrows():
        lut[(r["domain"], r["task"], r["model"], r["representation_id"])] = r

    reps = sorted(master["representation_id"].unique())
    domains = sorted(master["domain"].unique())
    tasks = sorted(master["task"].unique())
    models = sorted(master["model"].unique())

    for dom in domains:
        for task in tasks:
            for model in models:
                for a, b in combinations(reps, 2):
                    ka = (dom, task, model, a)
                    kb = (dom, task, model, b)
                    if ka not in lut or kb not in lut:
                        continue
                    ra, rb = lut[ka], lut[kb]
                    pa = Path(ra["preds_path"])
                    pb = Path(rb["preds_path"])
                    if not pa.exists() or not pb.exists():
                        continue
                    dfa = preds_binary_labels(pa, float(ra["threshold"]), task)
                    dfb = preds_binary_labels(pb, float(rb["threshold"]), task)
                    if dfa is None or dfb is None:
                        continue
                    m = dfa.merge(dfb, on=["verdict_1", "verdict_2"], suffixes=("_a", "_b"))
                    if len(m) < 10:
                        continue
                    if not (m["_y_a"].astype(int) == m["_y_b"].astype(int)).all():
                        continue
                    y = m["_y_a"].astype(int).values
                    p1 = m["_pred_a"].astype(int).values
                    p2 = m["_pred_b"].astype(int).values
                    _, p, n12, n21 = mcnemar_binom(y, p1, p2)
                    rows.append(
                        {
                            "domain": dom,
                            "task": task,
                            "model": model,
                            "rep_a": a,
                            "rep_b": b,
                            "n_pairs": len(m),
                            "n12_a_right_b_wrong": n12,
                            "n21_a_wrong_b_right": n21,
                            "mcnemar_p": p,
                        }
                    )
    df = pd.DataFrame(rows)
    if len(df):
        df["p_fdr_bh"] = fdr_bh(df["mcnemar_p"].values)
    return df


def friedman_one(master: pd.DataFrame, domain: str, task: str) -> dict | None:
    models = sorted(master["model"].unique())
    reps = sorted(master["representation_id"].unique())
    mat = []
    use_models = []
    for m in models:
        row = []
        ok = True
        for rep in reps:
            sub = master[
                (master["domain"] == domain)
                & (master["task"] == task)
                & (master["model"] == m)
                & (master["representation_id"] == rep)
            ]
            if len(sub) != 1:
                ok = False
                break
            row.append(float(sub.iloc[0]["f1"]))
        if ok:
            mat.append(row)
            use_models.append(m)
    if len(mat) < 3:
        return None
    mat = np.array(mat)
    try:
        stat, p = stats.friedmanchisquare(*[mat[:, j] for j in range(mat.shape[1])])
    except Exception:
        return None
    return {
        "domain": domain,
        "task": task,
        "n_models": len(use_models),
        "n_reps": len(reps),
        "friedman_stat": float(stat),
        "p_value": float(p),
        "representations": ",".join(reps),
    }


def compare_baseline(full_master: pd.DataFrame, baseline_root: Path, rep: str = "hybrid_full_gpt") -> pd.DataFrame:
    bm = collect_master(baseline_root)
    bm = bm[(bm["representation_id"] == rep) & bm["task"].isin(["binary_0", "binary_1"])]
    bm = bm.set_index(["domain", "task", "model"])["f1"]
    cur = full_master[(full_master["representation_id"] == rep)].set_index(["domain", "task", "model"])["f1"]
    rows = []
    for idx in cur.index:
        if idx not in bm.index:
            continue
        rows.append(
            {
                "domain": idx[0],
                "task": idx[1],
                "model": idx[2],
                "f1_full_matrix": float(cur.loc[idx]),
                "f1_baseline": float(bm.loc[idx]),
                "delta_f1": float(cur.loc[idx]) - float(bm.loc[idx]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "experiment_root",
        type=str,
        help="Path to v6_full_matrix (or any v6 --output-root folder)",
    )
    ap.add_argument(
        "--baseline",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1] / "v6_hybrid_full_gpt_score_multimodel"
        ),
        help="Earlier experiment folder to compare hybrid_full_gpt F1 (9-model style run).",
    )
    ap.add_argument(
        "--regen-tables",
        action="store_true",
        help="Run regenerate_v6_tables.py first (same folder).",
    )
    args = ap.parse_args()
    root = Path(args.experiment_root).resolve()
    out = root / "analysis_full"
    out.mkdir(parents=True, exist_ok=True)

    if args.regen_tables:
        import subprocess

        reg = root / "regenerate_v6_tables.py"
        if reg.is_file():
            subprocess.run([sys.executable, str(reg)], cwd=str(root), check=False)

    print("Collecting complete v6 runs...")
    master = collect_master(root)
    if master.empty:
        print("No complete runs found.")
        return
    master = master.sort_values(["domain", "task", "representation_id", "model"])
    master.to_csv(out / "master_runs.csv", index=False, encoding="utf-8-sig")

    summ = (
        master.groupby(["representation_id", "domain", "task"])
        .agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"), n=("f1", "count"))
        .reset_index()
    )
    summ.to_csv(out / "summary_representation_domain_task.csv", index=False, encoding="utf-8-sig")

    print("Wilcoxon representation pairs...")
    wdf = wilcoxon_rep_pairs(master)
    if len(wdf):
        wdf.to_csv(out / "wilcoxon_representation_pairs.csv", index=False, encoding="utf-8-sig")

    print("McNemar (all rep pairs × domain × task × model)...")
    mcn = mcnemar_all_pairs(master)
    if len(mcn):
        mcn.to_csv(out / "mcnemar_representation_pairs_by_cell.csv", index=False, encoding="utf-8-sig")

    print("Shuffled baseline per run...")
    sh_rows = []
    for _, r in master.iterrows():
        preds = Path(r["preds_path"])
        if not preds.exists():
            continue
        task = r["task"]
        label = "similarity_binary_0" if task == "binary_0" else "similarity_binary_1"
        df = pd.read_csv(preds)
        if "status" in df.columns:
            df = df[df["status"] == "ok"]
        sc = pd.to_numeric(df["score"], errors="coerce")
        y = df[label].astype(int).values
        thr = float(r["threshold"])
        ok = sc.notna()
        y = y[ok.values]
        y_pred = (sc[ok] >= thr).astype(int).values
        if len(y) < 10:
            continue
        rep = shuffled_report(y, y_pred)
        sh_rows.append({**r.to_dict(), **rep})
    pd.DataFrame(sh_rows).to_csv(out / "shuffled_baseline_all_runs.csv", index=False, encoding="utf-8-sig")

    fr_rows = []
    for dom in ("drugs", "weapon"):
        for task in ("binary_0", "binary_1"):
            fr = friedman_one(master, dom, task)
            if fr:
                fr_rows.append(fr)
    if fr_rows:
        pd.DataFrame(fr_rows).to_csv(out / "friedman_by_domain_task.csv", index=False, encoding="utf-8-sig")

    bl = Path(args.baseline)
    if bl.is_dir():
        print(f"Comparing hybrid_full_gpt to baseline {bl}...")
        cmpdf = compare_baseline(master, bl)
        if len(cmpdf):
            cmpdf.to_csv(out / "vs_baseline_hybrid_full_gpt.csv", index=False, encoding="utf-8-sig")

    # README stub for thesis
    readme = out / "README_analysis_full.md"
    readme.write_text(
        f"""# v6 full matrix — statistical analysis

Generated from `{root.name}`.

## Files
- `master_runs.csv` — one row per complete (domain, task, model, representation).
- `summary_representation_domain_task.csv` — mean/std F1 per representation slice.
- `wilcoxon_representation_pairs.csv` — paired Wilcoxon on F1 differences (same model/domain/task).
- `mcnemar_representation_pairs_by_cell.csv` — McNemar discordant pairs per cell.
- `shuffled_baseline_all_runs.csv` — F1 vs label-shuffled predictions (p_vs_shuffled).
- `friedman_by_domain_task.csv` — Friedman test across representations (complete grid only).
- `vs_baseline_hybrid_full_gpt.csv` — ΔF1 vs baseline folder for `hybrid_full_gpt`.

Interpret FDR columns as exploratory when many tests are run.
""",
        encoding="utf-8",
    )
    print(f"Done. Wrote under {out}/")


if __name__ == "__main__":
    main()
