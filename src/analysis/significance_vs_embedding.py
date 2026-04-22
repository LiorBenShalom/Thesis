"""Statistical significance tests: LLM-rep vs. best embedding baseline.

For each (domain, rep, metric), we have 11 LLM-model observed scores and
ONE best-embedding value (the max across all 21 embedding cells:
3 text-embedding models + 7 reps x 3 = 21). We test whether the 11-model
distribution is significantly greater than that single baseline.

Test: one-sample Wilcoxon signed-rank, one-sided (H1: median > baseline),
plus a one-sample t-test for reference. Corrections: Benjamini-Hochberg FDR
and Bonferroni across 28 QWK cells (7 reps x 2 domains x 2 QWK variants).

Also emits significance vs. random null (pre-computed in random_full.csv).

Outputs (under experiments/results_paper_baselines/):
  - significance_vs_embedding.csv  — raw / FDR / Bonferroni p-values per cell
  - significance_summary.md        — human-readable narrative

Usage:
  cd new_try/experiments/src/analysis
  python significance_vs_embedding.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

EXP = Path(__file__).resolve().parents[2]
OUT = EXP / "results_paper_baselines"

METRICS = ["QWK_Oracle", "QWK_CV",
           "F1_Oracle_b0", "F1_Oracle_b1",
           "F1_CV_b0", "F1_CV_b1",
           "AP_b0", "AP_b1"]

rand = pd.read_csv(OUT / "random_full.csv")
emb_text = pd.read_csv(OUT / "emb_full.csv")
emb_reps = pd.read_csv(OUT / "emb_reps_full.csv")

# Build best-embedding reference per (domain, metric)
et = emb_text.copy()
et["src"] = "Text+" + et["model_display"]
et = et.rename(columns={"value": "v"})[["domain", "metric", "v", "src"]]
er = emb_reps.copy()
er["src"] = "Emb-on-" + er["rep"] + "+" + er["emb_model_display"]
er = er.rename(columns={"value": "v"})[["domain", "metric", "v", "src"]]
emb_all = pd.concat([et, er])
best_per_dm = emb_all.loc[emb_all.groupby(["domain", "metric"])["v"].idxmax()]
best_map = {(r.domain, r.metric): (r.v, r.src) for _, r in best_per_dm.iterrows()}

# Per (domain, rep, metric): collect 11 LLM scores and test vs best embedding
rows = []
for (dom, rep, metric), grp in rand.groupby(["domain", "rep", "metric"]):
    if metric not in METRICS:
        continue
    obs = grp["observed"].dropna().values
    if len(obs) < 3:
        continue
    emb_val, emb_src = best_map[(dom, metric)]
    d = obs - emb_val
    if np.all(d == 0):
        p_w = np.nan
    else:
        try:
            _, p_w = stats.wilcoxon(d, alternative="greater", zero_method="wilcox")
        except Exception:
            p_w = np.nan
    # For reference, median random-null p (vs random baseline) — already in grp
    p_random = float(grp["p_value"].median())
    rows.append(dict(
        domain=dom, rep=rep, metric=metric, n_llm_models=len(obs),
        llm_mean=float(obs.mean()),
        llm_min=float(obs.min()),
        llm_max=float(obs.max()),
        best_emb=float(emb_val),
        best_src=emb_src,
        gap_vs_emb=float(obs.mean() - emb_val),
        p_vs_emb=float(p_w) if not np.isnan(p_w) else np.nan,
        p_vs_random=p_random,
    ))

df = pd.DataFrame(rows)

# FDR + Bonferroni corrections, applied WITHIN each metric family (one per metric
# type). The paper-cited correction should match what's done in paper_results.py
# (per-focus-rep FDR). Here we correct across all cells of each metric family
# separately, which is more defensible than a single giant family.
for metric in METRICS:
    sub = df[df.metric == metric]
    if sub.empty:
        continue
    idx = sub.index
    pv = df.loc[idx, "p_vs_emb"].values
    mask = ~np.isnan(pv)
    if mask.sum() == 0:
        continue
    _, fdr, _, _ = multipletests(pv[mask], alpha=0.05, method="fdr_bh")
    bonf = np.minimum(pv[mask] * mask.sum(), 1.0)
    df.loc[idx[mask], "p_vs_emb_fdr"] = fdr
    df.loc[idx[mask], "p_vs_emb_bonf"] = bonf

df["sig_vs_emb_raw"]  = df.p_vs_emb      < 0.05
df["sig_vs_emb_fdr"]  = df.p_vs_emb_fdr  < 0.05
df["sig_vs_emb_bonf"] = df.p_vs_emb_bonf < 0.05
df["sig_vs_random"]   = df.p_vs_random   < 0.05

df = df.sort_values(["domain", "metric", "gap_vs_emb"])
df.to_csv(OUT / "significance_vs_embedding.csv", index=False)

# ─── Markdown summary ───
md = ["# Statistical Significance — LLM-rep vs. Embedding Baselines\n"]
md.append("_Test: one-sample Wilcoxon signed-rank, one-sided (H1: LLM-rep median > best embedding)._")
md.append("_N=11 LLM models per cell. Best embedding = max across 24 embedding cells "
          "(4 text-embedding models + 7 reps × 4 = 28)._")
md.append("_FDR / Bonferroni applied within each of 8 metric families (14 cells each)._\n")

# Summary counts by metric
md.append("## 1. Overall pass-rates by metric (14 cells = 7 reps × 2 domains)\n")
rows_summary = []
for metric in METRICS:
    sub = df[df.metric == metric]
    if sub.empty:
        continue
    rows_summary.append(dict(
        metric=metric,
        cells=len(sub),
        pass_raw=int(sub.sig_vs_emb_raw.sum()),
        pass_fdr=int(sub.sig_vs_emb_fdr.sum()),
        pass_bonf=int(sub.sig_vs_emb_bonf.sum()),
        pass_vs_random=int(sub.sig_vs_random.sum()),
    ))
md.append(pd.DataFrame(rows_summary).set_index("metric").to_markdown())

# Detailed QWK tables (most important)
md.append("\n## 2. Detailed QWK significance (per cell)\n")
md.append("_One row per (domain, rep), ordered by gap vs. best embedding._\n")
for metric in ["QWK_CV", "QWK_Oracle"]:
    md.append(f"\n### {metric}\n")
    sub = df[df.metric == metric].copy()
    sub = sub[["domain", "rep", "llm_mean", "best_emb", "best_src",
               "gap_vs_emb", "p_vs_emb", "p_vs_emb_fdr", "p_vs_emb_bonf",
               "sig_vs_emb_raw", "sig_vs_emb_fdr", "sig_vs_emb_bonf"]]
    sub.columns = ["Domain", "Rep", "LLM mean", "Best emb", "Source",
                   "Gap", "p(raw)", "p(FDR)", "p(Bonf)",
                   "Raw<.05", "FDR<.05", "Bonf<.05"]
    for c in ["LLM mean", "Best emb", "Gap"]:
        sub[c] = sub[c].map(lambda x: f"{x:.3f}")
    for c in ["p(raw)", "p(FDR)", "p(Bonf)"]:
        sub[c] = sub[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    for c in ["Raw<.05", "FDR<.05", "Bonf<.05"]:
        sub[c] = sub[c].map({True: "✓", False: ""})
    md.append(sub.to_markdown(index=False))

# Non-significant cells
md.append("\n## 3. Non-significant cells (LLM NOT significantly > best embedding, raw p≥0.05)\n")
nonsig = df[~df.sig_vs_emb_raw].copy()
if len(nonsig) == 0:
    md.append("_All cells significant — LLM beats embedding everywhere._")
else:
    md.append(f"_{len(nonsig)} / {len(df)} cells._\n")
    ns = nonsig[["domain", "rep", "metric", "llm_mean", "best_emb",
                 "gap_vs_emb", "p_vs_emb"]].copy()
    ns.columns = ["Domain", "Rep", "Metric", "LLM mean", "Best emb",
                  "Gap", "p(raw)"]
    for c in ["LLM mean", "Best emb", "Gap"]:
        ns[c] = ns[c].map(lambda x: f"{x:.3f}")
    ns["p(raw)"] = ns["p(raw)"].map(lambda x: f"{x:.4f}")
    md.append(ns.to_markdown(index=False))

# Vs random baseline (sanity check)
md.append("\n\n## 4. Vs. Random Baseline (per cell Wilcoxon from permutation test)\n")
md.append("_Permutation-test p-value from random_baseline.py (shuffle GT 1000x)._\n")
summary_rand = df.groupby("metric").agg(
    total=("sig_vs_random", "size"),
    sig_vs_random=("sig_vs_random", "sum"),
).reset_index()
md.append(summary_rand.to_markdown(index=False))

(OUT / "significance_summary.md").write_text("\n".join(md), encoding="utf-8")

print(f"Wrote:")
print(f"  {OUT/'significance_vs_embedding.csv'}")
print(f"  {OUT/'significance_summary.md'}")
print()
print("Headlines (QWK only, 14 cells):")
qwk = df[df.metric.isin(['QWK_Oracle', 'QWK_CV'])]
print(f"  Raw p<0.05:  {qwk.sig_vs_emb_raw.sum()}/14")
print(f"  FDR<0.05:    {qwk.sig_vs_emb_fdr.sum()}/14")
print(f"  Bonf<0.05:   {qwk.sig_vs_emb_bonf.sum()}/14")
print(f"  Vs random:   {qwk.sig_vs_random.sum()}/14 (all should pass)")
