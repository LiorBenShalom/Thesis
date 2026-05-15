"""Plot the THESIS HEADLINE: richer pool → better LLM performance (monotonic)."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/tmp/thesis_plots")
OUT.mkdir(exist_ok=True)

df = pd.read_csv("/tmp/sweep_pool_size.csv")

# Order pool_size correctly
order = ["10", "20", "50", "100", "200", "500", "1000", "all"]
df["pool_idx"] = df.pool_size.map({p: i for i, p in enumerate(order)})
df = df.sort_values("pool_idx")

# 2x2 panel: MAE-lo and MAE-hi for each domain
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
plots = [
    (axes[0,0], "drugs", "mae_lo", "Drugs — MAE-low (months)"),
    (axes[0,1], "drugs", "mae_hi", "Drugs — MAE-high (months)"),
    (axes[1,0], "weapon", "mae_lo", "Weapon — MAE-low (months)"),
    (axes[1,1], "weapon", "mae_hi", "Weapon — MAE-high (months)"),
]
for ax, dom, metric, title in plots:
    sub = df[df.domain == dom].sort_values("pool_idx")
    x = range(len(sub))
    y = sub[metric].values
    ax.plot(x, y, "-o", color="#117733", markersize=10, linewidth=3, label="Sup top-N → LLM rerank → top-10")
    # annotate values
    for i, val in enumerate(y):
        ax.annotate(f"{val:.2f}", (i, val),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    # horizontal line for global median as ref
    glob_med_ref = {("drugs","mae_lo"): 8.43, ("drugs","mae_hi"): 14.08,
                    ("weapon","mae_lo"): 16.67, ("weapon","mae_hi"): 25.46}[(dom, metric)]
    ax.axhline(glob_med_ref, color="#888", linestyle="--", alpha=0.7,
               label=f"Global median: {glob_med_ref:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(sub.pool_size.tolist())
    ax.set_xlabel("Pool size — # candidates given to LLM")
    ax.set_ylabel("MAE (months)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")

plt.suptitle("Richer candidate pool → better LLM picks → lower MAE\n"
             "(Pool=10 = supervised cosine alone, no LLM. Pool=all = LLM-best from existing pool)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "plot_pool_richness.png", dpi=140)
plt.close()
print(f"✅ saved {OUT / 'plot_pool_richness.png'}")


# Single-panel version (cleaner for abstract)
fig, ax = plt.subplots(figsize=(11, 6))
for dom, color, marker in [("drugs", "#117733", "o"), ("weapon", "#cc6677", "s")]:
    sub = df[df.domain == dom].sort_values("pool_idx")
    # use average MAE (low+high)/2 for single line
    sub["avg_mae"] = (sub.mae_lo + sub.mae_hi) / 2
    x = range(len(sub))
    y = sub.avg_mae.values
    ax.plot(x, y, f"-{marker}", color=color, markersize=12, linewidth=3,
            label=f"{dom.upper()} (avg of MAE-low + MAE-high)")
    for i, val in enumerate(y):
        ax.annotate(f"{val:.1f}", (i, val), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10, color=color, fontweight="bold")

# global median references
ax.axhline((8.43+14.08)/2, color="#117733", linestyle=":", alpha=0.5, label="Drugs global median: 11.26")
ax.axhline((16.67+25.46)/2, color="#cc6677", linestyle=":", alpha=0.5, label="Weapon global median: 21.06")
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order)
ax.set_xlabel("Pool size — # candidates given to LLM", fontsize=12)
ax.set_ylabel("Average MAE (months)", fontsize=12)
ax.set_title("LLM does its best work when given a richer candidate pool", fontsize=13, fontweight="bold")
ax.grid(alpha=0.3)
ax.legend(fontsize=10, loc="upper right")
plt.tight_layout()
plt.savefig(OUT / "plot_pool_richness_headline.png", dpi=140)
plt.close()
print(f"✅ saved {OUT / 'plot_pool_richness_headline.png'}")
