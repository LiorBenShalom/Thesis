"""Combined baseline report: observed reps vs. random null vs. embedding baselines.

Reads:
  - results_paper_baselines/random_full.csv      (from random_baseline.py)
  - results_paper_baselines/emb_full.csv         (from embedding_baseline.py —
                                                  raw-facts embedding only)
  - results_paper_baselines/emb_reps_full.csv    (from embedding_all_reps.py —
                                                  embedding applied to every rep)

Produces:
  - results_paper_baselines/BASELINES_REPORT.md
  - results_paper_baselines/comparison_table.csv
  - results_paper_baselines/headline_plot.png
  - results_paper_baselines/ablation_rep_vs_emb.png (if emb_reps available)

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
    emb_reps_path = OUT / "emb_reps_full.csv"
    emb_reps = pd.read_csv(emb_reps_path) if emb_reps_path.exists() else None

    # ─── 1. Build comparison table ───
    rows = []

    # Random: shared baseline per (domain, metric) — advisor's format
    per_task_path = OUT / "random_per_task.csv"
    if per_task_path.exists():
        per_task = pd.read_csv(per_task_path)
        for _, r in per_task.iterrows():
            rows.append(dict(
                domain=r["domain"], metric=r["metric"],
                category="Random (null mean)",
                value=float(r["baseline_mean"]),
                detail=f"CI~[{r['baseline_ci_lo']:.3f}, {r['baseline_ci_hi']:.3f}]",
            ))
    else:
        for dom, sub in rand.groupby("domain"):
            for metric, m_sub in sub.groupby("metric"):
                rows.append(dict(
                    domain=dom, metric=metric, category="Random (null mean)",
                    value=float(m_sub["baseline_mean"].mean()),
                    detail=f"CI~[{m_sub['baseline_ci_lo'].mean():.3f}, "
                           f"{m_sub['baseline_ci_hi'].mean():.3f}]",
                ))

    # LLM-reps: observed mean across 11 models (from the main experiment)
    for dom, sub in rand.groupby("domain"):
        for rep, r_sub in sub.groupby("rep"):
            for metric, m_sub in r_sub.groupby("metric"):
                rows.append(dict(
                    domain=dom, metric=metric, category=f"LLM-rep: {rep}",
                    value=float(m_sub["observed"].mean()),
                    detail=f"n={len(m_sub)} models",
                ))

    # Text-embedding baseline (one per embedding model, on raw facts only)
    for _, r in emb.iterrows():
        rows.append(dict(
            domain=r["domain"], metric=r["metric"],
            category=f"Embedding-Text: {r['model_display']}",
            value=float(r["value"]),
            detail="cosine(emb(raw_text_V1), emb(raw_text_V2))",
        ))

    # Embedding-on-reps: one per (rep, embedding_model); also aggregate mean across emb models
    if emb_reps is not None:
        for _, r in emb_reps.iterrows():
            rows.append(dict(
                domain=r["domain"], metric=r["metric"],
                category=f"Emb-on-{r['rep']}: {r['emb_model_display']}",
                value=float(r["value"]),
                detail=f"cosine on {r['rep']} features",
            ))
        # Mean across the 3 embedding models per (rep, domain, metric)
        emb_reps_mean = (emb_reps.groupby(["domain", "rep", "metric"])["value"]
                                 .mean().reset_index())
        for _, r in emb_reps_mean.iterrows():
            rows.append(dict(
                domain=r["domain"], metric=r["metric"],
                category=f"Emb-on-{r['rep']}: mean",
                value=float(r["value"]),
                detail="mean across 3 embedding models",
            ))

    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "comparison_table.csv", index=False)

    # ─── 2. Markdown report ───
    md = ["# Baselines — Paper-ready Report\n"]
    md.append("_Compares several layers for each (domain, metric):_\n")
    md.append("_  1. **Random null** — 1000 GT-shuffles per cell (advisor's method). "
              "Preserves class proportions exactly._")
    md.append("_  2. **Embedding-Text** — cosine similarity of full verdict-text embeddings "
              "(OpenAI 3-large, mE5-large-instruct, BGE-M3). Non-structured baseline._")
    if emb_reps is not None:
        md.append("_  3. **Emb-on-rep** — embedding of the rep's *structured feature vector* + "
                  "cosine. Isolates structure vs. LLM reasoning contributions._")
    md.append("_  4. **LLM-rep** — main experiment: rep + LLM scoring (mean across 11 models)._\n")

    md.append("## 1. Headline comparison — mean across aggregation\n")

    def _pivot(domain: str) -> pd.DataFrame:
        sub = comp[comp.domain == domain].copy()
        base_cats = ["Random (null mean)"]
        text_emb_cats = sorted(c for c in sub.category.unique() if c.startswith("Embedding-Text"))
        rep_emb_cats = sorted(c for c in sub.category.unique()
                              if c.startswith("Emb-on-") and c.endswith(": mean"))
        llm_cats = [f"LLM-rep: {r}" for r in REP_ORDER if f"LLM-rep: {r}" in sub.category.unique()]
        cat_order = base_cats + text_emb_cats + rep_emb_cats + llm_cats

        p = sub.pivot_table(index="category", columns="metric", values="value", aggfunc="first")
        p = p.reindex(index=cat_order, columns=METRICS_ORDER)
        return p

    for domain in ["drugs", "weapon"]:
        md.append(f"\n### {domain.upper()}\n")
        p = _pivot(domain)
        p.columns = [METRIC_LABELS[c] for c in p.columns]
        md.append(p.round(3).to_markdown())

    # ─── 3. Per-rep ablation: LLM vs Embedding-on-rep ───
    if emb_reps is not None:
        md.append("\n\n## 2. Ablation — Rep+LLM vs. Rep+Embedding (QWK-CV)\n")
        md.append("_Gap isolates the contribution of LLM reasoning on top of a given "
                  "structured representation._\n")
        for domain in ["drugs", "weapon"]:
            md.append(f"\n### {domain.upper()}\n")
            llm_vals = {}
            for rep in REP_ORDER:
                sub = rand[(rand.domain == domain) & (rand.rep == rep) &
                           (rand.metric == "QWK_CV")]
                if not sub.empty:
                    llm_vals[rep] = float(sub["observed"].mean())
            emb_rep_mean = (emb_reps[(emb_reps.domain == domain) &
                                     (emb_reps.metric == "QWK_CV")]
                             .groupby("rep")["value"].mean().to_dict())
            tbl_rows = []
            for rep in REP_ORDER:
                if rep in llm_vals and rep in emb_rep_mean:
                    tbl_rows.append(dict(
                        rep=rep,
                        LLM_avg=llm_vals[rep],
                        Emb_avg=emb_rep_mean[rep],
                        Gap=llm_vals[rep] - emb_rep_mean[rep],
                    ))
            tbl = pd.DataFrame(tbl_rows).set_index("rep")
            md.append(tbl.round(3).to_markdown())

    # ─── 4. Significance fraction vs. null (per-rep) ───
    md.append("\n\n## 3. Fraction of cells where LLM-rep beats 97.5% null CI-hi\n")
    rand_sig = (rand.assign(beats=lambda d: d["observed"] > d["baseline_ci_hi"])
                    .groupby(["domain", "metric", "rep"])["beats"]
                    .mean().reset_index(name="frac_cells_beats_null"))
    pivot = rand_sig.pivot_table(index=["domain", "rep"], columns="metric",
                                 values="frac_cells_beats_null", aggfunc="first")
    pivot = pivot.reindex(columns=METRICS_ORDER)
    pivot.columns = [METRIC_LABELS[c] for c in pivot.columns]
    md.append(pivot.round(2).to_markdown())

    # ─── 5. Plots ───
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Plot A: headline QWK-CV bar chart
        headline_metric = "QWK_CV"
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        fig.suptitle(f"Baselines vs. LLM representations — {METRIC_LABELS[headline_metric]}",
                     fontsize=13, fontweight="bold")
        for ax, domain in zip(axes, ["drugs", "weapon"]):
            sub = comp[(comp.domain == domain) & (comp.metric == headline_metric)].copy()
            base_cats = ["Random (null mean)"]
            text_emb_cats = sorted(c for c in sub.category.unique() if c.startswith("Embedding-Text"))
            rep_emb_mean_cats = sorted(c for c in sub.category.unique()
                                        if c.startswith("Emb-on-") and c.endswith(": mean"))
            llm_cats = [f"LLM-rep: {r}" for r in REP_ORDER if f"LLM-rep: {r}" in sub.category.unique()]
            order = base_cats + text_emb_cats + rep_emb_mean_cats + llm_cats
            vals = [float(sub[sub.category == c]["value"].iloc[0]) for c in order]
            colors = (["#bdbdbd"] + ["#64b5f6"] * len(text_emb_cats)
                      + ["#ffb74d"] * len(rep_emb_mean_cats)
                      + ["#2e7d32" if c == "LLM-rep: Manual" else "#81c784" for c in llm_cats])
            short = {f"LLM-rep: {r}": r for r in REP_ORDER}
            short.update({c: c.replace("Embedding-Text: ", "txt:") for c in text_emb_cats})
            short.update({c: c.replace("Emb-on-", "emb:").replace(": mean", "") for c in rep_emb_mean_cats})
            short["Random (null mean)"] = "Random"
            xlabs = [short[c] for c in order]
            ax.bar(range(len(order)), vals, color=colors, edgecolor="white")
            for i, v in enumerate(vals):
                ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(xlabs, rotation=40, ha="right", fontsize=8)
            ax.set_title(domain.upper())
            ax.set_ylabel(METRIC_LABELS[headline_metric])
            ax.set_ylim(min(0.0, min(vals) - 0.05), 1.0)
            ax.grid(axis="y", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(OUT / "headline_plot.png", bbox_inches="tight", dpi=150)
        plt.close()

        # Plot B: LLM-vs-Emb per rep (ablation)
        if emb_reps is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
            fig.suptitle("Per-rep ablation — Rep+Emb (structure only) vs Rep+LLM (full pipeline)",
                         fontsize=13, fontweight="bold")
            for ax, domain in zip(axes, ["drugs", "weapon"]):
                llm_vals = {}
                for rep in REP_ORDER:
                    sub = rand[(rand.domain == domain) & (rand.rep == rep) &
                               (rand.metric == "QWK_CV")]
                    if not sub.empty:
                        llm_vals[rep] = float(sub["observed"].mean())
                emb_vals = (emb_reps[(emb_reps.domain == domain) &
                                     (emb_reps.metric == "QWK_CV")]
                             .groupby("rep")["value"].mean().to_dict())
                reps = [r for r in REP_ORDER if r in llm_vals and r in emb_vals]
                x = np.arange(len(reps))
                w = 0.38
                ax.bar(x - w/2, [emb_vals[r] for r in reps], w,
                       color="#ffb74d", edgecolor="white", label="Rep + Embedding (mean of 3)")
                ax.bar(x + w/2, [llm_vals[r] for r in reps], w,
                       color="#2e7d32", edgecolor="white", label="Rep + LLM (mean of 11)")
                for i, r in enumerate(reps):
                    ax.text(i - w/2, emb_vals[r] + 0.01, f"{emb_vals[r]:.2f}", ha="center",
                            fontsize=7, fontweight="bold")
                    ax.text(i + w/2, llm_vals[r] + 0.01, f"{llm_vals[r]:.2f}", ha="center",
                            fontsize=7, fontweight="bold")
                ax.set_xticks(x)
                ax.set_xticklabels(reps, rotation=30, ha="right", fontsize=9)
                ax.set_title(domain.upper())
                ax.set_ylabel("QWK-CV (10-fold)")
                ax.set_ylim(0, 1.0)
                ax.grid(axis="y", alpha=0.3)
                ax.legend(loc="lower right", fontsize=8)
                ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            plt.savefig(OUT / "ablation_rep_vs_emb.png", bbox_inches="tight", dpi=150)
            plt.close()

        md.append("\n![Baselines headline plot](headline_plot.png)\n")
        if emb_reps is not None:
            md.append("\n![Per-rep ablation](ablation_rep_vs_emb.png)\n")
    except Exception as e:
        print(f"  (Skipping plots: {e})")

    (OUT / "BASELINES_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {OUT/'BASELINES_REPORT.md'} and {OUT/'comparison_table.csv'}")


if __name__ == "__main__":
    build()
