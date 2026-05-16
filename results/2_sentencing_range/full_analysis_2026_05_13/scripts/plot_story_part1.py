"""
Plot for Part 1 — The reality of sentencing variance.

Combines all 3 panels into one figure:
  - Panel A: LLM bucket vs sentencing gap
  - Panel B: Citation type vs sentencing gap
  - Panel C: Random baseline reference line
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/tmp/thesis_plots")
OUT.mkdir(exist_ok=True)

llm_df = pd.read_csv("/tmp/story_llm_gaps.csv")
cit_df = pd.read_csv("/tmp/story_citation_gaps.csv")

# Random baseline (recomputed from master_inventory for reference)
m = pd.read_csv("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/experiments/data_per_domain/master_inventory.csv",
                usecols=["canonical_id","domain","sentencing_range_low","sentencing_range_high","sentencing_confidence"])
m = m[m.domain.isin(["drugs","weapon"]) & m.sentencing_range_low.notna() & (m.sentencing_confidence == "גבוהה")].drop_duplicates("canonical_id")

random_baseline = {}  # EXACT mean |Δ| over ALL C(n,2) pairs (Gini mean difference)
for dom in ("drugs", "weapon"):
    sub = m[m.domain == dom]
    lows = sub.sentencing_range_low.values.astype(float)
    highs = sub.sentencing_range_high.values.astype(float)
    n = len(lows)
    total_pairs = n * (n - 1) // 2
    exact_lo = np.abs(lows[:, None] - lows[None, :]).sum() / 2.0 / total_pairs
    exact_hi = np.abs(highs[:, None] - highs[None, :]).sum() / 2.0 / total_pairs
    random_baseline[dom] = (exact_lo, exact_hi)

# --- Figure: 4 panels (2 rows × 2 domains) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Bucket setup
LLM_BUCKETS = [("0-24",0,25),("25-49",25,50),("50-74",50,75),("75-89",75,90),("90-100",90,101)]
LLM_COLORS = ["#cc6677","#ddaa44","#88aabb","#3377aa","#117733"]

CIT_TYPES = [("1hop","#117733"),("2hop","#3377aa"),("cocite","#ddaa44"),("none","#cc6677")]

for col_i, dom in enumerate(["drugs", "weapon"]):
    # --- Top row: LLM bucket
    ax = axes[0][col_i]
    sub_dom = llm_df[llm_df.domain == dom]
    bucket_data = []
    for label, lo, hi in LLM_BUCKETS:
        b = sub_dom[(sub_dom.llm_score >= lo) & (sub_dom.llm_score < hi)]
        if len(b) > 0:
            bucket_data.append((label, len(b), b.d_lo.mean(), b.d_hi.mean()))
    x = np.arange(len(bucket_data))
    width = 0.35
    means_lo = [r[2] for r in bucket_data]
    means_hi = [r[3] for r in bucket_data]
    ax.bar(x - width/2, means_lo, width, label="|Δlow|", color="#3377aa")
    ax.bar(x + width/2, means_hi, width, label="|Δhigh|", color="#cc6677")
    # Random baseline line
    rand_lo, rand_hi = random_baseline[dom]
    ax.axhline(rand_lo, color="#3377aa", linestyle="--", alpha=0.6, label=f"Random |Δlow|={rand_lo:.1f}")
    ax.axhline(rand_hi, color="#cc6677", linestyle="--", alpha=0.6, label=f"Random |Δhigh|={rand_hi:.1f}")
    # Annotate values
    for i, (lbl, n, mlo, mhi) in enumerate(bucket_data):
        ax.text(i - width/2, mlo + 0.5, f"{mlo:.1f}\n(n={n})", ha="center", fontsize=8)
        ax.text(i + width/2, mhi + 0.5, f"{mhi:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in bucket_data])
    ax.set_xlabel("LLM similarity bucket")
    ax.set_ylabel("|Δ sentencing| (months)")
    ax.set_title(f"{dom.upper()}: LLM score → sentencing gap")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # --- Bottom row: Citation type
    ax = axes[1][col_i]
    sub_dom = cit_df[cit_df.domain == dom]
    cit_data = []
    for ct, _ in CIT_TYPES:
        c = sub_dom[sub_dom.cit_type == ct]
        if len(c) > 0:
            cit_data.append((ct, len(c), c.d_lo.mean(), c.d_hi.mean()))
    x = np.arange(len(cit_data))
    means_lo = [r[2] for r in cit_data]
    means_hi = [r[3] for r in cit_data]
    ax.bar(x - width/2, means_lo, width, label="|Δlow|", color="#3377aa")
    ax.bar(x + width/2, means_hi, width, label="|Δhigh|", color="#cc6677")
    ax.axhline(rand_lo, color="#3377aa", linestyle="--", alpha=0.6, label=f"Random |Δlow|={rand_lo:.1f}")
    ax.axhline(rand_hi, color="#cc6677", linestyle="--", alpha=0.6, label=f"Random |Δhigh|={rand_hi:.1f}")
    for i, (lbl, n, mlo, mhi) in enumerate(cit_data):
        ax.text(i - width/2, mlo + 0.5, f"{mlo:.1f}\n(n={n})", ha="center", fontsize=8)
        ax.text(i + width/2, mhi + 0.5, f"{mhi:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in cit_data])
    ax.set_xlabel("Citation type")
    ax.set_ylabel("|Δ sentencing| (months)")
    ax.set_title(f"{dom.upper()}: Citation → sentencing gap")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

plt.suptitle("Sentencing variance as function of similarity (LLM + Citation) vs Random baseline",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "plot_story_part1_variance.png", dpi=140)
plt.close()
print(f"✅ {OUT / 'plot_story_part1_variance.png'}")
