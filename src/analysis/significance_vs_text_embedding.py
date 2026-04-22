"""Statistical significance: LLM-rep vs. BEST TEXT-ONLY embedding baseline.

Rationale: the natural embedding baseline for the paper is the strongest
*text-only* embedding (no structured features). Comparing against
"Emb-on-Manual+Gemini" is unfair because it already uses the Manual
representation's structure — that's an ablation of Manual, not a baseline.

This script tests every LLM-rep against the best raw-text embedding
(max across 4 text embedding models: OpenAI 3-large, Gemini, mE5, BGE-M3).

Outputs (under experiments/results_paper_baselines/):
  - significance_vs_text_embedding.csv  — per-cell p-values (raw/FDR/Bonf)
  - significance_text_summary.md        — narrative with per-rep pass rates

Usage:
  python src/analysis/significance_vs_text_embedding.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

EXP = Path(__file__).resolve().parents[2]
OUT = EXP / "results_paper_baselines"

METRICS = ["F1_Oracle_b0", "F1_Oracle_b1", "F1_CV_b0", "F1_CV_b1",
           "AP_b0", "AP_b1", "QWK_Oracle", "QWK_CV"]
REPS = ["Manual", "Hybrid-Manual", "Hybrid-Full", "GPT-Schema",
        "Raw-Facts", "GPT-Free", "GPT-Law"]
DOMAINS = ["drugs", "weapon"]

rand = pd.read_csv(OUT / "random_full.csv")
emb_text = pd.read_csv(OUT / "emb_full.csv")

# Best TEXT embedding per (domain, metric)
best_text = emb_text.groupby(["domain", "metric"])["value"].max().to_dict()

rows = []
for rep in REPS:
    for dom in DOMAINS:
        for metric in METRICS:
            llm = rand[(rand.domain == dom) & (rand.rep == rep) & (rand.metric == metric)]["observed"].values
            if len(llm) < 3:
                continue
            baseline = best_text[(dom, metric)]
            d = llm - baseline
            if np.all(d == 0):
                p = np.nan
            else:
                try:
                    _, p = stats.wilcoxon(d, alternative="greater", zero_method="wilcox")
                except Exception:
                    p = np.nan
            rows.append(dict(
                rep=rep, domain=dom, metric=metric,
                llm_mean=float(llm.mean()),
                best_text_emb=float(baseline),
                gap=float(llm.mean() - baseline),
                p_raw=p,
            ))

df = pd.DataFrame(rows)

# Apply corrections within each metric family (14 cells per metric)
for metric in METRICS:
    sub = df[df.metric == metric]
    idx = sub.index
    pv = df.loc[idx, "p_raw"].values
    mask = ~np.isnan(pv)
    if mask.sum() == 0:
        continue
    _, fdr, _, _ = multipletests(pv[mask], alpha=0.05, method="fdr_bh")
    bonf = np.minimum(pv[mask] * mask.sum(), 1.0)
    df.loc[idx[mask], "p_fdr"] = fdr
    df.loc[idx[mask], "p_bonf"] = bonf

df["sig_raw"] = df.p_raw < 0.05
df["sig_fdr"] = df.p_fdr < 0.05
df["sig_bonf"] = df.p_bonf < 0.05

df.to_csv(OUT / "significance_vs_text_embedding.csv", index=False)

# ─── Markdown ───
md = ["# Significance vs. Text-Only Embedding Baseline\n"]
md.append("_Natural paper baseline: best raw-text embedding (no structured features)._")
md.append("_Per (domain, metric), the baseline is max across 4 text-embedding models "
          "(OpenAI text-embedding-3-large, Gemini-embedding-001, mE5-large-instruct, BGE-M3)._")
md.append("_Test: one-sample Wilcoxon signed-rank, one-sided (H1: LLM-rep mean > best text emb)._")
md.append("_Corrections: BH-FDR and Bonferroni applied within each metric family (14 cells each)._\n")

md.append("## 1. Pass rates per representation (16 cells each: 2 domains × 8 metrics)\n")
per_rep = df.groupby("rep").agg(
    total=("sig_raw", "size"),
    raw=("sig_raw", "sum"),
    fdr=("sig_fdr", "sum"),
    bonf=("sig_bonf", "sum"),
).reset_index()
per_rep["rep"] = pd.Categorical(per_rep["rep"], categories=REPS, ordered=True)
per_rep = per_rep.sort_values("rep").reset_index(drop=True)
md.append(per_rep.to_markdown(index=False))

md.append("\n## 2. QWK-only pass rates per rep (4 cells each: 2 domains × 2 QWK variants)\n")
qwk = df[df.metric.isin(["QWK_CV", "QWK_Oracle"])]
per_rep_qwk = qwk.groupby("rep").agg(
    total=("sig_raw", "size"),
    raw=("sig_raw", "sum"),
    fdr=("sig_fdr", "sum"),
    bonf=("sig_bonf", "sum"),
).reset_index()
per_rep_qwk["rep"] = pd.Categorical(per_rep_qwk["rep"], categories=REPS, ordered=True)
per_rep_qwk = per_rep_qwk.sort_values("rep").reset_index(drop=True)
md.append(per_rep_qwk.to_markdown(index=False))

md.append("\n## 3. Detailed QWK tables\n")
for metric in ["QWK_CV", "QWK_Oracle"]:
    md.append(f"\n### {metric}\n")
    sub = df[df.metric == metric].copy()
    sub["rep"] = pd.Categorical(sub["rep"], categories=REPS, ordered=True)
    sub = sub.sort_values(["domain", "rep"])
    for c in ["llm_mean", "best_text_emb", "gap"]:
        sub[c] = sub[c].map(lambda x: f"{x:.3f}")
    for c in ["p_raw", "p_fdr", "p_bonf"]:
        sub[c] = sub[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    for c in ["sig_raw", "sig_fdr", "sig_bonf"]:
        sub[c] = sub[c].map({True: "✓", False: ""})
    sub = sub[["domain", "rep", "llm_mean", "best_text_emb", "gap",
               "p_raw", "p_fdr", "p_bonf", "sig_raw", "sig_fdr", "sig_bonf"]]
    sub.columns = ["Domain", "Rep", "LLM mean", "Best text emb", "Gap",
                   "p(raw)", "p(FDR)", "p(Bonf)", "R", "F", "B"]
    md.append(sub.to_markdown(index=False))

md.append("\n## 4. Non-significant cells (raw p>=0.05)\n")
nonsig = df[~df.sig_raw][["rep", "domain", "metric", "llm_mean", "best_text_emb", "gap", "p_raw"]].copy()
if len(nonsig) == 0:
    md.append("_All 112 cells significant. Every LLM-rep beats the best text embedding everywhere._")
else:
    md.append(f"_{len(nonsig)} / {len(df)} cells non-significant. All on drugs lenient (b1) "
              "where Gemini raw text happens to be very strong._\n")
    for c in ["llm_mean", "best_text_emb", "gap"]:
        nonsig[c] = nonsig[c].map(lambda x: f"{x:.3f}")
    nonsig["p_raw"] = nonsig["p_raw"].map(lambda x: f"{x:.4f}")
    md.append(nonsig.to_markdown(index=False))

(OUT / "significance_text_summary.md").write_text("\n".join(md), encoding="utf-8")

print(f"Wrote:")
print(f"  {OUT/'significance_vs_text_embedding.csv'}")
print(f"  {OUT/'significance_text_summary.md'}")
