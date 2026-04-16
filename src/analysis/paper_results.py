"""Paper-grade results: full F1 + AP-PR matrices, significance tables, per-domain reports.

Outputs (under experiments/results_paper/):
  - full_results.csv          (7 reps × 11 models × 2 domains × 2 tasks × [F1, AP-PR])
  - summary_mean_f1.csv       (rep × domain × task, mean ± std across models)
  - summary_mean_ap.csv       (same for AP-PR)
  - significance_f1.csv       (pairwise rep vs rep, Wilcoxon Bonferroni/FDR)
  - significance_ap.csv
  - heatmap_wins.csv          (rep × model: 1 if rep is best for that model)
  - REPORT.md                 (paper-ready markdown with all tables)

Usage:
  python paper_results.py
"""
from __future__ import annotations
import json, itertools
from pathlib import Path
from typing import Iterable
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

EXP = Path(__file__).resolve().parents[2]
OUT = EXP/"results_paper"
OUT.mkdir(exist_ok=True)

DOMAINS = {
    "drugs":  EXP/"v6_final"/"drugs"/"results_drugs",
    "weapon": EXP/"v6_final"/"weapon"/"results_weapon",
}
REP_PREFIX = {
    "Manual":        "similarity_database_fe",
    "GPT-Schema":    "similarity_database_fe_gpt_schema_v2",
    "GPT-Free":      "similarity_database_with_gpt_features",
    "GPT-Law":       "similarity_database_with_gpt_law_features",
    "Raw-Facts":     "similarity_database_with_indicment_facts",
    "Hybrid-Manual": "similarity_database_hybrid",
    "Hybrid-Full":   "similarity_database_hybrid_full_gpt",
}
MODELS = ["gpt4","gpt5mini","gpt52","gpt51_thinking","claude_sonnet_4_6",
          "gemini_25_pro","gemini_3_flash","gemma3_27b","gemma4_31b_or",
          "llama3_70b","qwen3_vl_235b_or"]
TASKS = [0, 1]

def best_f1(scores: np.ndarray, y: np.ndarray) -> float:
    """Oracle best-threshold F1 (threshold picked on same data — upper bound)."""
    if len(np.unique(y)) < 2: return np.nan
    best = 0.0
    for thr in np.unique(scores):
        best = max(best, f1_score(y, (scores >= thr).astype(int), zero_division=0))
    return best

def cv_f1(scores: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 42) -> float:
    """Generalization F1: k-fold stratified CV. Tune threshold on train folds,
    predict on held-out fold, pool all held-out predictions, compute F1."""
    if len(np.unique(y)) < 2: return np.nan
    if min(np.bincount(y)) < k: return np.nan
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=int)
    for tr, te in skf.split(scores, y):
        best_f = 0.0; best_t = None
        for t in np.unique(scores[tr]):
            f = f1_score(y[tr], (scores[tr] >= t).astype(int), zero_division=0)
            if f > best_f: best_f = f; best_t = t
        if best_t is None: best_t = np.median(scores[tr])
        pred[te] = (scores[te] >= best_t).astype(int)
    return f1_score(y, pred, zero_division=0)

def cell_metrics(base: Path, rep_prefix: str, model: str, task: int) -> tuple[float,float,float]:
    p = base/f"{rep_prefix}_v6score_{model}_binary_0_preds.csv"
    if not p.exists(): return np.nan, np.nan, np.nan
    df = pd.read_csv(p)
    df = df[df["status"] == "ok"] if "status" in df.columns else df
    if len(df) < 20: return np.nan, np.nan, np.nan
    y = df[f"similarity_binary_{task}"].astype(int).values
    sc = df["score"].astype(float).values
    f1_oracle = best_f1(sc, y)
    f1_cv = cv_f1(sc, y)
    ap = average_precision_score(y, sc) if y.sum() > 0 else np.nan
    return f1_oracle, f1_cv, ap

def cohens_dz(d: np.ndarray) -> float:
    d = d[~np.isnan(d)]
    if len(d) < 2 or d.std(ddof=1) == 0: return np.nan
    return d.mean()/d.std(ddof=1)

def eff_label(dz: float) -> str:
    if np.isnan(dz): return "—"
    a = abs(dz)
    return "Negligible" if a<0.2 else "Small" if a<0.5 else "Medium" if a<0.8 else "Large"

# ---------- 1. Full metrics matrix ----------
rows = []
for dom, base in DOMAINS.items():
    for rep, pref in REP_PREFIX.items():
        for m in MODELS:
            for t in TASKS:
                f1o, f1cv, ap = cell_metrics(base, pref, m, t)
                rows.append(dict(domain=dom, rep=rep, model=m, task=t,
                                 F1=f1o, F1_CV=f1cv, AP_PR=ap))
full = pd.DataFrame(rows)
full.to_csv(OUT/"full_results.csv", index=False)

# ---------- 2. Summary mean±std across models ----------
def summarize(metric: str) -> pd.DataFrame:
    g = full.groupby(["domain","task","rep"])[metric].agg(["mean","std","count"]).reset_index()
    g["cell"] = g["mean"].map(lambda x: f"{x:.3f}") + " ± " + g["std"].map(lambda x: f"{x:.3f}")
    pivot = g.pivot_table(index="rep", columns=["domain","task"], values="cell", aggfunc="first")
    pivot = pivot.reindex(list(REP_PREFIX))
    return pivot

summarize("F1").to_csv(OUT/"summary_mean_f1.csv")
summarize("F1_CV").to_csv(OUT/"summary_mean_f1_cv.csv")
summarize("AP_PR").to_csv(OUT/"summary_mean_ap.csv")

# ---------- 3. Pairwise significance (Wilcoxon, one-sided A>B) ----------
def pairwise_sig(domain: str, task: int, metric: str) -> pd.DataFrame:
    sub = full[(full.domain==domain)&(full.task==task)]
    wide = sub.pivot(index="model", columns="rep", values=metric).reindex(columns=list(REP_PREFIX))
    recs = []
    for a, b in itertools.permutations(REP_PREFIX.keys(), 2):
        A = wide[a].values; B = wide[b].values
        mask = ~(np.isnan(A)|np.isnan(B)); d = A[mask]-B[mask]
        if len(d) < 3 or np.all(d==0):
            recs.append(dict(A=a, B=b, n=len(d), delta=np.nan, dz=np.nan, wins=0, p=np.nan)); continue
        try: p = float(wilcoxon(d, alternative="greater", zero_method="wilcox").pvalue)
        except Exception: p = np.nan
        recs.append(dict(A=a, B=b, n=len(d),
                         delta=float(d.mean()), dz=cohens_dz(d),
                         wins=int((d>0).sum()), p=p))
    df = pd.DataFrame(recs)
    # Correct within (A): 6 tests per focus rep
    for a in REP_PREFIX:
        idx = df[df.A==a].index
        pv = df.loc[idx,"p"].values
        mask = ~np.isnan(pv)
        if mask.sum()==0: continue
        _, fdr, _, _ = multipletests(pv[mask], alpha=0.05, method="fdr_bh")
        bonf = np.minimum(pv[mask]*mask.sum(), 1.0)
        df.loc[idx[mask], "p_fdr"]  = fdr
        df.loc[idx[mask], "p_bonf"] = bonf
    df["effect"] = df["dz"].map(eff_label)
    df["sig_fdr"]  = (df["p_fdr"]  < 0.05).map({True:"✓", False:""})
    df["sig_bonf"] = (df["p_bonf"] < 0.05).map({True:"✓", False:""})
    df["domain"] = domain; df["task"] = task; df["metric"] = metric
    return df

sig_f1    = pd.concat([pairwise_sig(d, t, "F1")    for d in DOMAINS for t in TASKS], ignore_index=True)
sig_f1_cv = pd.concat([pairwise_sig(d, t, "F1_CV") for d in DOMAINS for t in TASKS], ignore_index=True)
sig_ap    = pd.concat([pairwise_sig(d, t, "AP_PR") for d in DOMAINS for t in TASKS], ignore_index=True)
sig_f1.to_csv(OUT/"significance_f1.csv", index=False)
sig_f1_cv.to_csv(OUT/"significance_f1_cv.csv", index=False)
sig_ap.to_csv(OUT/"significance_ap.csv", index=False)

# ---------- 4. Wins per rep (how many (model,task,domain) it's top-1 for each metric) ----------
def wins(metric: str) -> pd.DataFrame:
    w = full.dropna(subset=[metric]).copy()
    w["rank"] = w.groupby(["domain","task","model"])[metric].rank(method="min", ascending=False)
    tbl = w[w["rank"]==1].groupby(["domain","task","rep"]).size().reset_index(name="wins")
    pivot = tbl.pivot_table(index="rep", columns=["domain","task"], values="wins", fill_value=0)
    return pivot.reindex(list(REP_PREFIX), fill_value=0)

wins("F1").to_csv(OUT/"wins_f1.csv")
wins("AP_PR").to_csv(OUT/"wins_ap.csv")

# ---------- 5. REPORT.md ----------
def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown()

def focus_table(sig: pd.DataFrame, focus: str, domain: str, task: int) -> pd.DataFrame:
    s = sig[(sig.A==focus)&(sig.domain==domain)&(sig.task==task)].copy()
    s = s[["B","delta","dz","effect","wins","n","p","p_fdr","p_bonf","sig_fdr","sig_bonf"]]
    s.columns = ["vs","Δ","Cohen's dz","Effect","Wins","n","raw p","FDR p","Bonf p","FDR ✓","Bonf ✓"]
    for c in ["Δ","Cohen's dz"]: s[c] = s[c].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "—")
    for c in ["raw p","FDR p","Bonf p"]: s[c] = s[c].map(lambda x: f"{x:.4f}" if pd.notna(x) and x>=1e-4 else ("<0.0001" if pd.notna(x) else "—"))
    return s.reset_index(drop=True)

md = []
md.append("# Thesis Results — Paper-grade Report\n")
md.append("_7 representations × 11 models × 2 domains (drugs, weapon) × 2 binary tasks (b0, b1)._")
md.append("_Metrics:_")
md.append("_  (a) **F1-Oracle**: best threshold picked on same data (upper bound)._")
md.append("_  (b) **F1-CV**: 5-fold stratified CV — threshold tuned on train folds, F1 computed on pooled held-out predictions (generalization estimate)._")
md.append("_  (c) **AP-PR**: average precision, threshold-free (primary)._")
md.append("_Statistical test: one-sided Wilcoxon signed-rank across 11 models, corrected within each focus rep (6 comparisons) via Bonferroni + Benjamini-Hochberg FDR, α=0.05._\n")

md.append("## 1. Summary — mean F1 (Oracle) across 11 models\n")
md.append(md_table(summarize("F1")))
md.append("\n## 2. Summary — mean F1 (5-fold CV) across 11 models\n")
md.append(md_table(summarize("F1_CV")))
md.append("\n## 3. Summary — mean AP-PR across 11 models\n")
md.append(md_table(summarize("AP_PR")))
md.append("\n## 4. Wins (top-1 rep per model)\n")
md.append("### F1 (Oracle)\n"+md_table(wins("F1")))
md.append("\n### F1 (CV)\n"+md_table(wins("F1_CV")))
md.append("\n### AP-PR\n"+md_table(wins("AP_PR")))

md.append("\n## 5. Significance — focused tables (focus = Manual)\n")
for dom in DOMAINS:
    for t in TASKS:
        focus = "Manual"
        md.append(f"\n### {dom.upper()} — Task b{t}")
        md.append(f"\n**F1 (Oracle)**\n"+md_table(focus_table(sig_f1, focus, dom, t)))
        md.append(f"\n**F1 (5-fold CV)**\n"+md_table(focus_table(sig_f1_cv, focus, dom, t)))
        md.append(f"\n**AP-PR**\n"+md_table(focus_table(sig_ap, focus, dom, t)))

md.append("\n## 6. Full per-cell results\n")
md.append("See `full_results.csv` (one row per rep × model × domain × task, with F1, F1_CV, AP_PR).\n")

(OUT/"REPORT.md").write_text("\n".join(md), encoding="utf-8")

print(f"Done. Outputs under: {OUT}")
for p in sorted(OUT.iterdir()): print(f"  {p.name}")
