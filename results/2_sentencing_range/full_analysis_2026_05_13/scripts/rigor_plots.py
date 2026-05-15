"""Rigorous plots: forest plot of MAE with CIs, paired differences, quartile MAE."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/tmp/thesis_plots")
OUT.mkdir(exist_ok=True)

mae = pd.read_csv("/tmp/rigor_mae_with_ci.csv")
paired = pd.read_csv("/tmp/rigor_paired_diffs.csv")
quart = pd.read_csv("/tmp/rigor_quartile_ci.csv")
year = pd.read_csv("/tmp/rigor_year_cluster.csv")

# Order methods by MAE
METHOD_ORDER = [
    "global_median",
    "offense_matched_random",
    "tfidf_ridge",
    "bm25",
    "random_llm",
    "citation_llm",
    "sup_only",
    "sup_llm",
    "llm_best",
]
METHOD_LABELS = {
    "global_median": "Global median (no sim)",
    "offense_matched_random": "Offense-matched random",
    "tfidf_ridge": "TF-IDF + Ridge",
    "bm25": "BM25 retrieval",
    "random_llm": "Random + LLM",
    "citation_llm": "Citation + LLM",
    "sup_only": "Supervised alone",
    "sup_llm": "Supervised + LLM",
    "llm_best": "LLM-best (upper bound)",
}
COLORS = ["#888", "#aaa", "#cc6677", "#ddaa44", "#e69f00", "#56b4e9", "#009e73", "#117733", "#000"]


# ============ FOREST PLOT — MAE per method with CI ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plots = [
    (axes[0,0], "drugs", "mae_lo", "mae_lo_ci_low", "mae_lo_ci_hi", "Drugs — MAE-low"),
    (axes[0,1], "drugs", "mae_hi", "mae_hi_ci_low", "mae_hi_ci_hi", "Drugs — MAE-high"),
    (axes[1,0], "weapon", "mae_lo", "mae_lo_ci_low", "mae_lo_ci_hi", "Weapon — MAE-low"),
    (axes[1,1], "weapon", "mae_hi", "mae_hi_ci_low", "mae_hi_ci_hi", "Weapon — MAE-high"),
]
for ax, dom, col_m, col_l, col_h, title in plots:
    sub = mae[mae.domain == dom].set_index("method").loc[METHOD_ORDER].reset_index()
    y = np.arange(len(sub))
    means = sub[col_m].values
    los = sub[col_l].values
    his = sub[col_h].values
    err_low = means - los
    err_high = his - means
    ax.errorbar(means, y, xerr=[err_low, err_high], fmt='o', ecolor='gray', capsize=4,
                color="#117733", markersize=10, markerfacecolor="white", markeredgewidth=2)
    for i, mn in enumerate(means):
        ax.text(mn, y[i] + 0.25, f"{mn:.2f}", ha="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABELS[m] for m in sub.method], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("MAE (months) — 95% CI")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="x")
plt.suptitle("MAE per method with bootstrap 95% confidence intervals (B=2000)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "plot_rigor_forest_mae.png", dpi=140)
plt.close()
print("✅ plot_rigor_forest_mae.png")


# ============ PAIRED DIFFERENCES — sup_llm as reference ============
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, dom in zip(axes, ("drugs", "weapon")):
    sub = paired[(paired.domain == dom) & (paired.A == "sup_llm")].copy()
    sub = sub.iloc[::-1]  # reverse for plot
    y = np.arange(len(sub))
    means = sub.mean_diff_A_minus_B.values
    los = sub.ci_low.values
    his = sub.ci_hi.values
    err_low = means - los
    err_high = his - means
    colors = ['#117733' if (l > 0 or h < 0) else '#cc6677' for l, h in zip(los, his)]
    ax.errorbar(means, y, xerr=[err_low, err_high], fmt='o', ecolor='gray', capsize=4,
                markersize=10, markerfacecolor='white', markeredgewidth=2)
    for i, (mn, l, h) in enumerate(zip(means, los, his)):
        sig = "***" if h < 0 else ("ns" if (l < 0 and h > 0) else "")
        ax.text(mn, y[i] + 0.25, f"{mn:+.2f} {sig}", ha="center", fontsize=9)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(["vs " + b for b in sub.B], fontsize=10)
    ax.set_xlabel("Δ MAE: sup_llm − baseline (negative = sup_llm is better)")
    ax.set_title(f"{dom.upper()}: paired differences vs sup_llm")
    ax.grid(alpha=0.3, axis="x")
plt.suptitle("Paired bootstrap differences (sup_llm − baseline) — 95% CI", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "plot_rigor_paired_diffs.png", dpi=140)
plt.close()
print("✅ plot_rigor_paired_diffs.png")


# ============ QUARTILE MAE with CI ============
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
methods_to_plot = ["sup_only", "sup_llm", "tfidf_ridge", "llm_best"]
colors = ["#888888", "#117733", "#cc6677", "#000000"]
for ax, dom in zip(axes, ("drugs", "weapon")):
    for method, color in zip(methods_to_plot, colors):
        sub = quart[(quart.domain == dom) & (quart.method == method)].sort_values("quartile")
        if len(sub) == 0: continue
        x = np.arange(len(sub))
        means = sub.avg_mae.values
        los = sub.ci_low.values
        his = sub.ci_hi.values
        ax.errorbar(x + (methods_to_plot.index(method) - 1.5) * 0.15, means,
                    yerr=[means-los, his-means], fmt='o-', capsize=4,
                    color=color, label=METHOD_LABELS[method], markersize=8, linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(sub.quartile.tolist())
    ax.set_xlabel("Quartile of true sentence (Q1=light, Q4=severe)")
    ax.set_ylabel("Average MAE (months)")
    ax.set_title(f"{dom.upper()}: MAE within quartile")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
plt.suptitle("Per-quartile MAE with bootstrap 95% CI", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "plot_rigor_quartile.png", dpi=140)
plt.close()
print("✅ plot_rigor_quartile.png")


# ============ YEAR-CLUSTER vs PER-QUERY CI WIDTH ============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, dom in zip(axes, ("drugs", "weapon")):
    sub_y = year[year.domain == dom]
    # also pull per-query CIs (compute from MAE table)
    sub_m = mae[mae.domain == dom]
    methods_plot = ["sup_only", "sup_llm", "tfidf_ridge", "llm_best"]
    rows = []
    for method in methods_plot:
        y_row = sub_y[sub_y.method == method]
        m_row = sub_m[sub_m.method == method]
        if len(y_row) == 0 or len(m_row) == 0: continue
        y_row = y_row.iloc[0]; m_row = m_row.iloc[0]
        # avg of low and high CIs from per-query
        pq_low_low, pq_low_hi = m_row.mae_lo_ci_low, m_row.mae_lo_ci_hi
        pq_hi_low, pq_hi_hi = m_row.mae_hi_ci_low, m_row.mae_hi_ci_hi
        pq_width = ((pq_low_hi - pq_low_low) + (pq_hi_hi - pq_hi_low)) / 2
        yr_width = y_row.ci_width
        rows.append({"method": method, "per_query_CI_width": pq_width, "year_cluster_CI_width": yr_width})
    df_w = pd.DataFrame(rows)
    x = np.arange(len(df_w))
    width = 0.35
    ax.bar(x - width/2, df_w.per_query_CI_width, width, label="Per-query bootstrap", color="#117733")
    ax.bar(x + width/2, df_w.year_cluster_CI_width, width, label="Year-clustered bootstrap", color="#cc6677")
    for i, (pq, yc) in enumerate(zip(df_w.per_query_CI_width, df_w.year_cluster_CI_width)):
        ax.text(i - width/2, pq + 0.05, f"{pq:.2f}", ha="center", fontsize=8)
        ax.text(i + width/2, yc + 0.05, f"{yc:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in df_w.method], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("CI width (months)")
    ax.set_title(f"{dom.upper()}: Per-query vs Year-clustered CI width")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
plt.suptitle("Year-cluster CIs are WIDER → indicates temporal effects in data", fontsize=12)
plt.tight_layout()
plt.savefig(OUT / "plot_rigor_year_cluster.png", dpi=140)
plt.close()
print("✅ plot_rigor_year_cluster.png")

print(f"\nAll rigor plots saved to {OUT}")
