"""
Plotting recipes for the raw data — use this as a starting point for your own plots.

The raw data file: data/rigor_raw_per_query_K.csv

Schema (one row per query × method × K):
    query              — verdict ID
    domain             — drugs / weapon
    fold               — 1-5 (CV fold)
    year               — verdict year (1984-2024)
    method             — global_median / tfidf_ridge / sup_only / sup_llm / llm_best /
                         random_llm / citation_llm / bm25 / offense_matched_random
    K                  — number of neighbors used (None for global_median / tfidf_ridge)
    true_lo, true_hi   — ground truth sentence range
    pred_lo, pred_hi   — predicted (low, high)
    err_lo, err_hi     — |pred - true| absolute errors
    n_actual           — actual number of picked verdicts (may be < K for sparse methods)
    neighbors          — JSON list of picked verdict IDs
    mean_llm_in_picked — mean LLM score of the picked neighbors
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Load raw data ===
df = pd.read_csv("/tmp/rigor_raw_per_query_K.csv")
df["avg_err"] = (df.err_lo + df.err_hi) / 2

print(f"Total records: {len(df):,}")
print(f"Methods: {sorted(df.method.unique())}")
print(f"K values: {sorted(df[df.K.notna()].K.unique())}")
print(f"Domains: {sorted(df.domain.unique())}")

# ============ RECIPE 1: MAE vs K, per method, per domain ============
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
methods = ["sup_only", "sup_llm", "llm_best", "bm25", "citation_llm", "random_llm"]
colors = {"sup_only":"#888","sup_llm":"#117733","llm_best":"#000",
          "bm25":"#cc6677","citation_llm":"#3377aa","random_llm":"#ddaa44"}
for ax, dom in zip(axes, ("drugs", "weapon")):
    for method in methods:
        sub = df[(df.domain == dom) & (df.method == method)].dropna(subset=["K"])
        agg = sub.groupby("K")["avg_err"].mean().reset_index()
        ax.plot(agg.K, agg.avg_err, "-o", color=colors[method], label=method, linewidth=2, markersize=8)
    ax.set_xlabel("K")
    ax.set_ylabel("Avg MAE (months)")
    ax.set_title(f"{dom.upper()}: MAE vs K")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/tmp/example_plot_1_mae_vs_K.png", dpi=140)
plt.close()
print("✅ example_plot_1_mae_vs_K.png")


# ============ RECIPE 2: Distribution of errors at K=10 ============
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, dom in zip(axes, ("drugs", "weapon")):
    sub = df[(df.domain == dom) & (df.K == 10) & (df.method.isin(["sup_llm", "bm25", "llm_best"]))]
    for method, color in [("bm25","#cc6677"),("sup_llm","#117733"),("llm_best","#000")]:
        m_sub = sub[sub.method == method]
        ax.hist(m_sub.avg_err, bins=50, alpha=0.5, label=method, color=color)
    ax.set_xlabel("Avg MAE per query (months)")
    ax.set_ylabel("# queries")
    ax.set_title(f"{dom.upper()}: Error distribution @ K=10")
    ax.set_xlim(0, 60 if dom == "drugs" else 100)
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/tmp/example_plot_2_error_distribution.png", dpi=140)
plt.close()
print("✅ example_plot_2_error_distribution.png")


# ============ RECIPE 3: Scatter — predicted vs true, color by method ============
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for col_i, method in enumerate(["sup_only", "sup_llm", "llm_best"]):
    for row_i, (target, ylabel) in enumerate([("low","MAE-low"),("high","MAE-high")]):
        ax = axes[row_i][col_i]
        sub = df[(df.K == 10) & (df.method == method) & (df.domain == "drugs")]
        ax.scatter(sub[f"true_{target}"], sub[f"pred_{target}"],
                   alpha=0.3, s=8, color="#117733")
        # diagonal
        mx = max(sub[f"true_{target}"].max(), sub[f"pred_{target}"].max())
        ax.plot([0, mx], [0, mx], 'k--', alpha=0.4, linewidth=1)
        ax.set_xlabel(f"True {target}_months")
        ax.set_ylabel(f"Pred {target}_months")
        ax.set_title(f"DRUGS — {method} @ K=10 — {target}")
        ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/tmp/example_plot_3_scatter_pred_vs_true.png", dpi=140)
plt.close()
print("✅ example_plot_3_scatter_pred_vs_true.png")


# ============ RECIPE 4: MAE by year (line plot) ============
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, dom in zip(axes, ("drugs", "weapon")):
    for method, color in [("sup_llm","#117733"),("bm25","#cc6677"),("llm_best","#000")]:
        sub = df[(df.domain == dom) & (df.K == 10) & (df.method == method)].dropna(subset=["year"])
        # only show years with >= 50 queries
        agg = sub.groupby("year").agg(n=("avg_err","size"), mae=("avg_err","mean")).reset_index()
        agg = agg[agg.n >= 50]
        ax.plot(agg.year, agg.mae, "-o", color=color, label=f"{method} (n>=50)", markersize=6)
    ax.set_xlabel("Year")
    ax.set_ylabel("MAE")
    ax.set_title(f"{dom.upper()}: MAE by year @ K=10")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/tmp/example_plot_4_mae_by_year.png", dpi=140)
plt.close()
print("✅ example_plot_4_mae_by_year.png")


# ============ RECIPE 5: Neighbor LLM-score → error (scatter) ============
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, dom in zip(axes, ("drugs", "weapon")):
    sub = df[(df.domain == dom) & (df.K == 10) & (df.method == "sup_llm")].dropna(subset=["mean_llm_in_picked"])
    ax.scatter(sub.mean_llm_in_picked, sub.avg_err, alpha=0.3, s=8, color="#117733")
    # bin and mean
    bins = pd.cut(sub.mean_llm_in_picked, bins=np.linspace(0, 100, 11))
    binned = sub.groupby(bins, observed=True).agg(n=("avg_err","size"), mae=("avg_err","mean")).reset_index()
    binned["bin_mid"] = binned["mean_llm_in_picked"].apply(lambda x: x.mid)
    ax.plot(binned.bin_mid, binned.mae, "-o", color="red", linewidth=3, markersize=10,
            label="Binned mean MAE")
    ax.set_xlabel("Mean LLM score of picked neighbors (top-10)")
    ax.set_ylabel("Per-query MAE (months)")
    ax.set_title(f"{dom.upper()}: confidence calibration (sup_llm K=10)")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/tmp/example_plot_5_confidence_vs_error.png", dpi=140)
plt.close()
print("✅ example_plot_5_confidence_vs_error.png")


print("\n--- Quick recipes for your own plots ---")
print("""
# Filter to a subset:
sub = df[(df.domain == "drugs") & (df.K == 10) & (df.method == "sup_llm")]

# MAE per method:
df.groupby(["domain","method","K"])["avg_err"].agg(["mean","std","count"])

# Calibration — is true within ±X months?
sub["within_6"] = (sub.err_lo <= 6) & (sub.err_hi <= 6)
sub.groupby("method")["within_6"].mean()

# Distance from median analysis:
sub["dist_from_median"] = abs((sub.true_lo + sub.true_hi)/2 - median_per_domain[dom])

# Which neighbors does method X use for query Q?
import json
neighbors = json.loads(df[(df.query==Q) & (df.method=="sup_llm") & (df.K==10)].neighbors.iloc[0])

# Bootstrap CI on MAE:
errs = sub["avg_err"].values
boots = [errs[np.random.choice(len(errs), len(errs))].mean() for _ in range(2000)]
print(f"95% CI: [{np.percentile(boots, 2.5):.2f}, {np.percentile(boots, 97.5):.2f}]")
""")
