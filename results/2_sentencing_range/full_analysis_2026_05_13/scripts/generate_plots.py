"""
Generate the key plots for the thesis:
  1. K sweep — MAE vs K, all filters
  2. Source-set sweep — MAE vs source-set fraction
  3. min_k sweep — MAE vs coverage tradeoff
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("/tmp/thesis_plots")
OUT_DIR.mkdir(exist_ok=True)

df_K = pd.read_csv("/tmp/sweep_K.csv")
df_src = pd.read_csv("/tmp/sweep_source.csv")
df_mk = pd.read_csv("/tmp/sweep_min_k.csv")

FILTER_COLORS = {
    "global_median": "#888888",
    "random": "#cc6677", "random_llm": "#882255",
    "citation_all": "#88ccee", "citation_all_llm": "#3377aa",
    "supervised": "#ddcc77", "supervised_llm": "#aa8822",
    "llm_top": "#117733",
}
FILTER_LABELS = {
    "global_median": "Global median",
    "random": "Random",
    "random_llm": "Random + LLM",
    "citation_all": "Citation",
    "citation_all_llm": "Citation + LLM",
    "supervised": "Supervised",
    "supervised_llm": "Supervised + LLM",
    "llm_top": "LLM-best (oracle)",
}


# ========= PLOT 1: K SWEEP =========
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (dom, metric, title) in zip(
    axes.flatten(),
    [("drugs","mae_lo","Drugs — MAE-low"),
     ("drugs","mae_hi","Drugs — MAE-high"),
     ("weapon","mae_lo","Weapon — MAE-low"),
     ("weapon","mae_hi","Weapon — MAE-high")],
):
    for filter_name in FILTER_LABELS:
        sub = df_K[(df_K["filter"] == filter_name) & (df_K.domain == dom)].sort_values("K")
        if len(sub) == 0: continue
        ax.plot(sub.K, sub[metric], "-o",
                color=FILTER_COLORS[filter_name],
                label=FILTER_LABELS[filter_name], markersize=5, linewidth=2)
    ax.set_xlabel("K (neighbors used for median)")
    ax.set_ylabel(f"{metric.replace('_',' ').upper()} (months)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
plt.suptitle("MAE vs K (number of neighbors) — all filter strategies", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_K_sweep.png", dpi=140)
plt.close()
print("✅ saved plot_K_sweep.png")


# ========= PLOT 2: SOURCE-SET SWEEP =========
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (dom, metric, title) in zip(
    axes.flatten(),
    [("drugs","mae_lo","Drugs — MAE-low"),
     ("drugs","mae_hi","Drugs — MAE-high"),
     ("weapon","mae_lo","Weapon — MAE-low"),
     ("weapon","mae_hi","Weapon — MAE-high")],
):
    for filter_name in FILTER_LABELS:
        sub = df_src[(df_src["filter"] == filter_name) & (df_src.domain == dom)].sort_values("source_frac")
        if len(sub) == 0: continue
        ax.plot(sub.source_frac * 100, sub[metric], "-o",
                color=FILTER_COLORS[filter_name],
                label=FILTER_LABELS[filter_name], markersize=5, linewidth=2)
    ax.set_xlabel("Source-set fraction (% of train pool)")
    ax.set_ylabel(f"{metric.replace('_',' ').upper()} (months)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
plt.suptitle("MAE vs source-set size (subsampling train pool)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_source_sweep.png", dpi=140)
plt.close()
print("✅ saved plot_source_sweep.png")


# ========= PLOT 3: MIN_K — COVERAGE/MAE TRADEOFF =========
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, dom in zip(axes, ("drugs", "weapon")):
    for filter_name in FILTER_LABELS:
        sub = df_mk[(df_mk["filter"] == filter_name) & (df_mk.domain == dom)].sort_values("min_k")
        if len(sub) == 0: continue
        ax.plot(sub.coverage * 100, sub.mae_lo, "-o",
                color=FILTER_COLORS[filter_name],
                label=FILTER_LABELS[filter_name], markersize=8, linewidth=2)
        # annotate min_k values
        for _, r in sub.iterrows():
            ax.annotate(f"k≥{int(r.min_k)}", (r.coverage*100, r.mae_lo),
                        fontsize=7, alpha=0.6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Coverage (% of test queries with prediction)")
    ax.set_ylabel("MAE-low (months)")
    ax.set_title(f"{dom.upper()}: coverage vs MAE-low tradeoff (varying min_k)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
plt.suptitle("Coverage vs MAE — increasing min_k filters out queries with few candidates", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_min_k_coverage.png", dpi=140)
plt.close()
print("✅ saved plot_min_k_coverage.png")


# ========= PLOT 4: THE THESIS HEADLINE CHART =========
# Grouped bars: 3 categories (global, sup, sup+llm, oracle) × 4 metrics
labels = ["Drugs MAE-lo", "Drugs MAE-hi", "Weapon MAE-lo", "Weapon MAE-hi"]
configs_to_show = [
    ("global_median",     "Global median (no sim)"),
    ("supervised",        "Supervised filter alone"),
    ("supervised_llm",    "Supervised + LLM rerank"),
    ("llm_top",           "LLM-best (oracle)"),
]
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(labels))
width = 0.20
colors = ["#888888", "#ddcc77", "#aa8822", "#117733"]
for i, (filter_name, lbl) in enumerate(configs_to_show):
    vals = []
    sub = df_K[(df_K["filter"] == filter_name) & (df_K.K == 10)]
    for dom, metric in [("drugs","mae_lo"),("drugs","mae_hi"),
                        ("weapon","mae_lo"),("weapon","mae_hi")]:
        v = sub[sub.domain == dom][metric].iloc[0]
        vals.append(v)
    bars = ax.bar(x + i*width - width*1.5, vals, width, label=lbl, color=colors[i])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.15, f"{v:.1f}",
                ha="center", fontsize=8, color="black")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("MAE (months)")
ax.set_title("Thesis headline: similarity model helps sentencing range prediction (K=10)")
ax.legend(loc="upper left", fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_headline.png", dpi=140)
plt.close()
print("✅ saved plot_headline.png")


print(f"\nAll plots saved to {OUT_DIR}/")
for p in sorted(OUT_DIR.glob("*.png")):
    print(f"  {p.name}  ({p.stat().st_size/1024:.0f} KB)")
