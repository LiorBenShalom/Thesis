"""Combined baseline report: observed reps vs. random null vs. embedding baselines.

Reads:
  - results_paper_baselines/random_full.csv   (from random_baseline.py)
  - results_paper_baselines/emb_full.csv      (from embedding_baseline.py)

Produces:
  - results_paper_baselines/BASELINES_REPORT.md
  - results_paper_baselines/comparison_table.csv
  - results_paper_baselines/headline_plot.png  (bar chart: Random < Embeddings < Best LLM rep)

Usage:
  cd new_try/experiments/src/analysis
  python baseline_report.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[2]
OUT = EXP / "results_paper_baselines"

METRIC_LABELS = {
    "F1_Oracle_b0": "F1-Oracle (b0 strict)",
    "F1_Oracle_b1": "F1-Oracle (b1 lenient)",
    "F1_CV_b0":     "F1-CV (b0 strict)",
    "F1_CV_b1":     "F1-CV (b1 lenient)",
    "AP_b0":        "AP-PR (b0 strict)",
    "AP_b1":        "AP-PR (b1 lenient)",
    "QWK_Oracle":   "QWK-Oracle",
    "QWK_CV":       "QWK-CV (10-fold)",
}
METRICS_ORDER = list(METRIC_LABELS.keys())

REP_ORDER = ["Manual", "Hybrid-Manual", "Hybrid-Full", "GPT-Schema",
             "Raw-Facts", "GPT-Free", "GPT-Law"]


def build():
    rand = pd.read_csv(OUT / "random_full.csv")
    emb = pd.read_csv(OUT / "emb_full.csv")

    # ─── 1. Build comparison table: per (domain, metric), one row per category ───
    # Categories: Random (mean across 77 cells), each embedding model, each rep (mean across 11 models)
    rows = []

    # Random: mean null across all reps×models
    for dom, sub in rand.groupby("domain"):
        for metric, m_sub in sub.groupby("metric"):
            rows.append(dict(
                domain=dom, metric=metric, category="Random (null mean)",
                value=float(m_sub["null_mean"].mean()),
                detail=f"CI~[{m_sub['null_ci_lo'].mean():.3f}, "
                       f"{m_sub['null_ci_hi'].mean():.3f}]",
            ))
        for rep, r_sub in sub.groupby("rep"):
            for metric, m_sub in r_sub.groupby("metric"):
                rows.append(dict(
                    domain=dom, metric=metric, category=f"LLM-rep: {rep}",
                    value=float(m_sub["observed"].mean()),
                    detail=f"n={len(m_sub)} models",
                ))

    # Embedding: one row per model × domain × metric
    for _, r in emb.iterrows():
        rows.append(dict(
            domain=r["domain"], metric=r["metric"],
            category=f"Embedding: {r['model_display']}",
            value=float(r["value"]),
            detail="cosine(emb_1, emb_2)",
        ))

    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "comparison_table.csv", index=False)

    # ─── 2. Build markdown report ───
    md = ["# Baselines — Paper-ready Report\n"]
    md.append("_Compares three layers for each (domain, metric):_")
    md.append("_  1. **Random null** — permutation mean across 1000 shuffles per cell (77 cells avg)._")
    md.append("_  2. **Embedding baselines** — cosine similarity of verdict-fact embeddings "
              "(OpenAI 3-large, mE5-large-instruct, BGE-M3)._")
    md.append("_  3. **LLM reps** — scores averaged across 11 LLM models (from the main experiment)._\n")
    md.append("_**Bottom line:** random and embedding are baselines; LLM reps (especially Manual) "
              "should clearly dominate._\n")

    def _pivot(domain: str) -> pd.DataFrame:
        sub = comp[comp.domain == domain].copy()
        # Category order: random → embeddings → LLM reps (in REP_ORDER)
        emb_cats = sorted(c for c in sub.category.unique() if c.startswith("Embedding"))
        llm_cats = [f"LLM-rep: {r}" for r in REP_ORDER if f"LLM-rep: {r}" in sub.category.unique()]
        cat_order = ["Random (null mean)"] + emb_cats + llm_cats

        p = sub.pivot_table(index="category", columns="metric", values="value", aggfunc="first")
        p = p.reindex(index=cat_order, columns=METRICS_ORDER)
        return p

    for domain in ["drugs", "weapon"]:
        md.append(f"\n## {domain.upper()}\n")
        p = _pivot(domain)
        # Nicer column headers
        p.columns = [METRIC_LABELS[c] for c in p.columns]
        md.append(p.round(3).to_markdown())

    # ─── 3. Significance vs. null (one-tailed empirical p-value per rep×model) ───
    # Fraction of cells where the LLM-rep's observed beats its null CI-hi
    md.append("\n## Significance vs. Random Null — cells where observed > 97.5% null CI-hi\n")
    rand_sig = (rand.assign(beats=lambda d: d["observed"] > d["null_ci_hi"])
                    .groupby(["domain", "metric", "rep"])["beats"]
                    .mean().reset_index(name="frac_cells_beats_null"))
    pivot = rand_sig.pivot_table(index=["domain", "rep"], columns="metric",
                                 values="frac_cells_beats_null", aggfunc="first")
    pivot = pivot.reindex(columns=METRICS_ORDER)
    pivot.columns = [METRIC_LABELS[c] for c in pivot.columns]
    md.append(pivot.round(2).to_markdown())

    # ─── 4. Headline plot ───
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        headline_metric = "QWK_CV"
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        fig.suptitle(f"Baselines vs. LLM representations — {METRIC_LABELS[headline_metric]}",
                     fontsize=13, fontweight="bold")

        for ax, domain in zip(axes, ["drugs", "weapon"]):
            sub = comp[(comp.domain == domain) & (comp.metric == headline_metric)].copy()
            emb_cats = sorted(c for c in sub.category.unique() if c.startswith("Embedding"))
            llm_cats = [f"LLM-rep: {r}" for r in REP_ORDER if f"LLM-rep: {r}" in sub.category.unique()]
            order = ["Random (null mean)"] + emb_cats + llm_cats
            vals = [float(sub[sub.category == c]["value"].iloc[0]) for c in order]
            colors = (["#bdbdbd"] + ["#64b5f6"] * len(emb_cats)
                      + ["#2e7d32" if c == "LLM-rep: Manual" else "#81c784" for c in llm_cats])
            # Short labels for x axis
            short = {f"LLM-rep: {r}": r for r in REP_ORDER}
            short.update({c: c.replace("Embedding: ", "emb:") for c in emb_cats})
            short["Random (null mean)"] = "Random"
            xlabs = [short[c] for c in order]

            ax.bar(range(len(order)), vals, color=colors, edgecolor="white")
            for i, v in enumerate(vals):
                ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(xlabs, rotation=40, ha="right", fontsize=9)
            ax.set_title(domain.upper())
            ax.set_ylabel(METRIC_LABELS[headline_metric])
            ax.set_ylim(min(0.0, min(vals) - 0.05), 1.0)
            ax.grid(axis="y", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        plt.savefig(OUT / "headline_plot.png", bbox_inches="tight", dpi=150)
        plt.close()
        md.append("\n![Baselines headline plot](headline_plot.png)\n")
    except Exception as e:
        print(f"  (Skipping plot: {e})")

    (OUT / "BASELINES_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {OUT/'BASELINES_REPORT.md'} and {OUT/'comparison_table.csv'}")


if __name__ == "__main__":
    build()
