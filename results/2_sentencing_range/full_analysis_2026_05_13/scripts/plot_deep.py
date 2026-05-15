"""Plot the deep analysis results."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/tmp/thesis_plots")
OUT.mkdir(exist_ok=True)

POOL_ORDER = ["10", "20", "50", "100", "200", "500", "1000", "all"]
POOL_NUM = {p: int(p) if p != "all" else 1500 for p in POOL_ORDER}


# ===== PLOT 1: RECALL — supervised pool as a funnel for LLM-best =====
df = pd.read_csv("/tmp/deep_recall.csv")
df["pool_idx"] = df.pool_size.map({p: i for i, p in enumerate(POOL_ORDER)})
df = df.sort_values("pool_idx")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, dom in zip(axes, ("drugs", "weapon")):
    for K_oracle, color in zip([10, 20, 50], ["#117733", "#3377aa", "#cc6677"]):
        sub = df[(df.domain == dom) & (df.K_oracle == K_oracle)].sort_values("pool_idx")
        ax.plot(range(len(sub)), sub.recall * 100, "-o",
                color=color, markersize=8, linewidth=2.5,
                label=f"K_oracle = {K_oracle}")
    ax.set_xticks(range(len(POOL_ORDER)))
    ax.set_xticklabels(POOL_ORDER)
    ax.set_xlabel("Supervised pool size (top-N by cosine)")
    ax.set_ylabel("Recall (% of LLM-oracle top-K inside pool)")
    ax.set_title(f"{dom.upper()}: supervised pool ⊃ LLM-best cases")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 105)
plt.suptitle("How well does the supervised pool ‘capture’ the LLM-best candidates?",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "plot_deep_recall.png", dpi=140)
plt.close()
print("✅ plot_deep_recall.png")


# ===== PLOT 2: POOL QUALITY — concentration of LLM signal in narrow pools =====
df = pd.read_csv("/tmp/deep_pool_quality.csv")
df["pool_idx"] = df.pool_size.map({p: i for i, p in enumerate(POOL_ORDER)})
df = df.sort_values("pool_idx")

fig, ax = plt.subplots(figsize=(11, 5))
for dom, color, marker in [("drugs", "#117733", "o"), ("weapon", "#cc6677", "s")]:
    sub = df[df.domain == dom].sort_values("pool_idx")
    ax.plot(range(len(sub)), sub.mean_llm_in_pool, f"-{marker}",
            color=color, markersize=10, linewidth=3, label=f"{dom.upper()}")
    for i, v in enumerate(sub.mean_llm_in_pool):
        ax.annotate(f"{v:.0f}", (i, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color=color)
ax.set_xticks(range(len(POOL_ORDER)))
ax.set_xticklabels(POOL_ORDER)
ax.set_xlabel("Supervised pool size")
ax.set_ylabel("Mean LLM-similarity score of pool members")
ax.set_title("Narrower supervised pool = higher mean LLM-similarity (concentration of signal)")
ax.grid(alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "plot_deep_pool_quality.png", dpi=140)
plt.close()
print("✅ plot_deep_pool_quality.png")


# ===== PLOT 3: COST-QUALITY PARETO =====
df = pd.read_csv("/tmp/deep_pareto.csv")
df["pool_idx"] = df.pool_size.map({p: i for i, p in enumerate(POOL_ORDER)})
df = df.sort_values("pool_idx")

fig, ax = plt.subplots(figsize=(11, 6))
for dom, color, marker in [("drugs", "#117733", "o"), ("weapon", "#cc6677", "s")]:
    sub = df[df.domain == dom].sort_values("cost_usd")
    ax.plot(sub.cost_usd, sub.avg_mae, f"-{marker}",
            color=color, markersize=12, linewidth=3, label=f"{dom.upper()}")
    for _, r in sub.iterrows():
        ax.annotate(f"pool={r.pool_size}",
                    (r.cost_usd, r.avg_mae),
                    textcoords="offset points", xytext=(8, 8), fontsize=8, color=color)
ax.set_xscale("log")
ax.set_xlabel("LLM scoring cost (USD)", fontsize=12)
ax.set_ylabel("Average MAE (months)", fontsize=12)
ax.set_title("Cost-Quality Pareto frontier: $ spent vs MAE achieved", fontsize=13, fontweight="bold")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "plot_deep_pareto.png", dpi=140)
plt.close()
print("✅ plot_deep_pareto.png")


# ===== PLOT 4: PER-QUARTILE MAE =====
df = pd.read_csv("/tmp/deep_quartile.csv")
df["pool_idx"] = df.pool_size.map({p: i for i, p in enumerate(POOL_ORDER)})
df = df.sort_values("pool_idx")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, dom in zip(axes, ("drugs", "weapon")):
    for quartile, color in zip(["Q1", "Q2", "Q3", "Q4"],
                                ["#117733", "#3377aa", "#ddcc77", "#cc6677"]):
        sub = df[(df.domain == dom) & (df.quartile == quartile)].sort_values("pool_idx")
        ax.plot(range(len(sub)), sub.avg_mae, "-o",
                color=color, markersize=8, linewidth=2.5,
                label=f"{quartile} (n={int(sub.n.iloc[0])})")
    ax.set_xticks(range(len(POOL_ORDER)))
    ax.set_xticklabels(POOL_ORDER)
    ax.set_xlabel("Pool size")
    ax.set_ylabel("Avg MAE (months)")
    ax.set_title(f"{dom.upper()}: MAE by true sentence quartile")
    ax.grid(alpha=0.3)
    ax.legend()
plt.suptitle("Per-quartile MAE — defends against ‘median regressor’ critique",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "plot_deep_quartile.png", dpi=140)
plt.close()
print("✅ plot_deep_quartile.png")


# ===== PLOT 5: CALIBRATION =====
df = pd.read_csv("/tmp/deep_calibration.csv")
df["pool_idx"] = df.pool_size.map({p: i for i, p in enumerate(POOL_ORDER)})
df = df.sort_values("pool_idx")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, dom in zip(axes, ("drugs", "weapon")):
    sub = df[df.domain == dom].sort_values("pool_idx")
    x = range(len(sub))
    ax.plot(x, sub.low_within_6mo * 100, "-o", color="#117733", markersize=8, linewidth=2.5,
            label="MAE-low ≤ 6 months")
    ax.plot(x, sub.high_within_6mo * 100, "-o", color="#cc6677", markersize=8, linewidth=2.5,
            label="MAE-high ≤ 6 months")
    ax.plot(x, sub.both_within_6mo * 100, "-o", color="#3377aa", markersize=8, linewidth=2.5,
            label="Both within 6 months")
    ax.set_xticks(x)
    ax.set_xticklabels(POOL_ORDER)
    ax.set_xlabel("Pool size")
    ax.set_ylabel("% of test queries")
    ax.set_title(f"{dom.upper()}: calibration (prediction within 6 months)")
    ax.grid(alpha=0.3)
    ax.legend()
plt.suptitle("Prediction calibration — within 6-month tolerance",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "plot_deep_calibration.png", dpi=140)
plt.close()
print("✅ plot_deep_calibration.png")


# ===== PLOT 6: HYBRID POOL =====
df = pd.read_csv("/tmp/deep_hybrid.csv")
df["avg_mae"] = (df.mae_lo + df.mae_hi) / 2
df = df.sort_values(["domain", "pool_base"])

# Compare to plain supervised at same pool sizes
df_sup = pd.read_csv("/tmp/sweep_pool_size.csv")
df_sup = df_sup[df_sup.pool_size.isin(["50","100","200","500"])].copy()
df_sup["pool_base"] = df_sup.pool_size.astype(int)
df_sup["avg_mae"] = (df_sup.mae_lo + df_sup.mae_hi) / 2

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, dom in zip(axes, ("drugs", "weapon")):
    sub_h = df[df.domain == dom].sort_values("pool_base")
    sub_s = df_sup[df_sup.domain == dom].sort_values("pool_base")
    ax.plot(sub_s.pool_base, sub_s.avg_mae, "-o", color="#888888", markersize=10, linewidth=2.5,
            label="Supervised alone")
    ax.plot(sub_h.pool_base, sub_h.avg_mae, "-s", color="#117733", markersize=10, linewidth=2.5,
            label="Supervised ∪ Citation")
    ax.set_xlabel("Supervised pool size")
    ax.set_ylabel("Avg MAE (months)")
    ax.set_title(f"{dom.upper()}: hybrid pool — sup ∪ citation")
    ax.grid(alpha=0.3)
    ax.legend()
plt.suptitle("Hybrid pool: combining supervised + citation candidates",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "plot_deep_hybrid.png", dpi=140)
plt.close()
print("✅ plot_deep_hybrid.png")


print("\nAll plots saved to", OUT)
for p in sorted(OUT.glob("*.png")):
    print(f"  {p.name}  ({p.stat().st_size/1024:.0f} KB)")
