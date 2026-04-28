"""3-way confusion matrix per representation (similarity_scale 1/2/3).

For each (domain, rep, model) we fit two thresholds with the same QWK
optimiser as paper_results_qwk._find_best_thresholds, build a 3x3 matrix
on that model's pairs (drugs: 100 pairs, weapon: 141 pairs), and compute
per-class recall + the 1<->3 off-diagonal rate. We then report mean ± std
across the N=13 final panel — the unit is the model, not the pooled tally.

The N=13 panel — all under v6_final/ (unified):
  9 ORIGINAL (minus llama3_70b + gemma3_27b)
  4 candidates (mistral_large_or, deepseek_r1_or, claude_haiku_4_5, kimi_k26_or)

Outputs (under experiments/results_paper/confusion_3way/):
  - cm_<domain>_<rep>.csv       3x3 confusion matrix (rows = GT, cols = pred)
  - summary.csv                 one row per (domain, rep) with recall_1/2/3 + off-diag
  - REPORT.md                   markdown tables matching the paper discussion
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score

EXP = Path(__file__).resolve().parents[2]
OUT = EXP/"results_paper"/"confusion_3way"
OUT.mkdir(parents=True, exist_ok=True)

# All 13 panel models live under v6_final (unified after pilot merge)
PROD_BASE = EXP/"v6_final"

def base_for(domain: str, model: str) -> Path:
    return PROD_BASE/domain/f"results_{domain}"

DOMAINS = ["drugs", "weapon"]
REP_PREFIX = {
    "Manual":        "similarity_database_fe",
    "Hybrid-Full":   "similarity_database_hybrid_full_gpt",
    "GPT-Schema":    "similarity_database_fe_gpt_schema_v2",
    "Hybrid-Manual": "similarity_database_hybrid",
    "GPT-Free":      "similarity_database_with_gpt_features",
    "Raw-Facts":     "similarity_database_with_indicment_facts",
    "GPT-Law":       "similarity_database_with_gpt_law_features",
}
# N=13 final panel
MODELS = [
    # 9 ORIGINAL (v6_final, after dropping llama3_70b and gemma3_27b)
    "gpt4", "gpt5mini", "gpt52", "gpt51_thinking", "claude_sonnet_4_6",
    "gemini_25_pro", "gemini_3_flash", "gemma4_31b_or", "qwen3_vl_235b_or",
    # 4 candidates (v6_pilot_5models)
    "mistral_large_or", "deepseek_r1_or",
    "claude_haiku_4_5", "kimi_k26_or",
]


def best_qwk_thresholds(scores: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Mirror paper_results_qwk._find_best_thresholds: grid-search t1<t2 on
    midpoints between consecutive unique scores, maximise QWK on 1/2/3."""
    uniq = np.unique(scores)
    if len(uniq) < 3:
        return 0.0, 50.0
    mids = (uniq[:-1] + uniq[1:]) / 2.0
    best_q, best_t1, best_t2 = -1.0, float(mids[0]), float(mids[-1])
    for i, t1 in enumerate(mids):
        for t2 in mids[i + 1:]:
            preds = np.where(scores < t1, 1, np.where(scores < t2, 2, 3))
            if len(np.unique(preds)) < 2:
                continue
            q = cohen_kappa_score(gt, preds, weights="quadratic")
            if q > best_q:
                best_q, best_t1, best_t2 = q, float(t1), float(t2)
    return best_t1, best_t2


def collect_predictions(domain: str, prefix: str) -> pd.DataFrame:
    """For each model, fit QWK-optimal thresholds on its own predictions and
    emit (similarity_scale, predicted_tier) rows. Pool across models."""
    frames = []
    for m in MODELS:
        p = base_for(domain, m)/f"{prefix}_v6score_{m}_binary_0_preds.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "status" in df.columns:
            df = df[df["status"] == "ok"]
        df = df.dropna(subset=["similarity_scale", "score"])
        if len(df) < 20:
            continue
        gt = df["similarity_scale"].astype(int).values
        sc = df["score"].astype(float).values
        t_low, t_high = best_qwk_thresholds(sc, gt)
        pred = np.where(sc < t_low, 1, np.where(sc < t_high, 2, 3))
        frames.append(pd.DataFrame({
            "similarity_scale": gt,
            "pred": pred,
            "model": m,
            "t_low": t_low,
            "t_high": t_high,
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def confusion_3x3(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=int)
    for gt in (1, 2, 3):
        for pr in (1, 2, 3):
            cm[gt-1, pr-1] = int(((y == gt) & (pred == pr)).sum())
    return cm


def summarize(cm: np.ndarray) -> dict:
    row_sums = cm.sum(axis=1)
    recall = np.divide(np.diag(cm), row_sums, out=np.zeros(3, float), where=row_sums>0)
    total = cm.sum()
    off_13 = (cm[0, 2] + cm[2, 0]) / total if total else np.nan
    return dict(n=int(total),
                recall_1=float(recall[0]), recall_2=float(recall[1]), recall_3=float(recall[2]),
                off_diag_1_3=float(off_13))


rows = []
per_model_rows = []
for dom in DOMAINS:
    for rep, pref in REP_PREFIX.items():
        pooled = collect_predictions(dom, pref)
        if pooled.empty:
            continue
        # pooled CM saved for inspection (n = pairs * n_models)
        cm_pool = confusion_3x3(pooled["similarity_scale"].values,
                                pooled["pred"].values)
        pd.DataFrame(cm_pool,
                     index=[f"GT={i}" for i in (1,2,3)],
                     columns=[f"pred={i}" for i in (1,2,3)]
                     ).to_csv(OUT/f"cm_{dom}_{rep.replace('-','_')}.csv")

        # per-model rates -> mean/std -> the unit is the model
        per_model = []
        for m, g in pooled.groupby("model"):
            cm_m = confusion_3x3(g["similarity_scale"].values, g["pred"].values)
            s_m = summarize(cm_m)
            s_m.update(dict(domain=dom, rep=rep, model=m,
                            t_low=float(g["t_low"].iloc[0]),
                            t_high=float(g["t_high"].iloc[0])))
            per_model.append(s_m)
            per_model_rows.append(s_m)
        df_m = pd.DataFrame(per_model)
        rows.append(dict(
            domain=dom, rep=rep,
            n_pairs=int(df_m["n"].iloc[0]),
            n_models=len(df_m),
            recall_1_mean=df_m["recall_1"].mean(),
            recall_1_std=df_m["recall_1"].std(),
            recall_2_mean=df_m["recall_2"].mean(),
            recall_2_std=df_m["recall_2"].std(),
            recall_3_mean=df_m["recall_3"].mean(),
            recall_3_std=df_m["recall_3"].std(),
            off_diag_1_3_mean=df_m["off_diag_1_3"].mean(),
            off_diag_1_3_std=df_m["off_diag_1_3"].std(),
            t_low_mean=df_m["t_low"].mean(),
            t_high_mean=df_m["t_high"].mean(),
        ))

summary = pd.DataFrame(rows)
summary.to_csv(OUT/"summary.csv", index=False)
pd.DataFrame(per_model_rows).to_csv(OUT/"per_model.csv", index=False)


# ---- markdown report ----
def fmt_table(df: pd.DataFrame) -> str:
    df = df.sort_values("off_diag_1_3_mean").reset_index(drop=True)
    lines = ["| rep | recall_1 | recall_2 | recall_3 | off-diag 1↔3 (mean ± std) |",
             "|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['rep']} "
            f"| {r['recall_1_mean']:.2f} ± {r['recall_1_std']:.2f} "
            f"| {r['recall_2_mean']:.2f} ± {r['recall_2_std']:.2f} "
            f"| {r['recall_3_mean']:.2f} ± {r['recall_3_std']:.2f} "
            f"| {r['off_diag_1_3_mean']*100:.2f}% ± {r['off_diag_1_3_std']*100:.2f}% |"
        )
    return "\n".join(lines)

md = ["# Confusion matrix — 3-way (similarity_scale 1/2/3)",
      "Mean ± std across 14 models; the unit is the model.\n"]
for dom in DOMAINS:
    sub = summary[summary["domain"] == dom]
    n_pairs = int(sub["n_pairs"].iloc[0])
    n_models = int(sub["n_models"].iloc[0])
    md.append(f"## {dom.capitalize()} (n_pairs={n_pairs}, n_models={n_models})\n")
    md.append(fmt_table(sub))
    md.append("")
(OUT/"REPORT.md").write_text("\n".join(md), encoding="utf-8")

print(f"wrote {OUT}")
print(summary.to_string(index=False))
