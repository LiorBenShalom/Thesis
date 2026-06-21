#!/usr/bin/env python3
"""
Baseline Reality, case-study view: how noisy is sentencing *within a single case's
own citation network*? For each test verdict we take the cases in its citation
network (1hop+2hop+cocite, the citation_llm candidate set) and measure the spread of
their sentencing ranges. Even cases the legal system links as related disagree by
tens of months.

Outputs:
  data/within_network_spread.csv            per-case spread stats
  data/within_network_example_neighbors.csv neighbor ranges for the 2 plotted cases
  plots/plot_baseline_noise_within_network.png
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[5]
FILT = ROOT / "simcse_cuda_bundle/outputs_supervised_filtered"
EXP  = ROOT / "experiments"
DATA = Path(__file__).resolve().parents[1] / "data"
PLOTS = Path(__file__).resolve().parents[1] / "plots"

sup = pd.read_csv(ROOT / "simcse_cuda_bundle/data/supervised_data.csv"); sup["verdict"] = sup.verdict.astype(str)
LO = dict(zip(sup.verdict, sup.sentencing_range_low)); HI = dict(zip(sup.verdict, sup.sentencing_range_high))

cit = pd.read_csv(EXP / "data_per_domain/network_analysis/citation_pair_types.csv")
adj = defaultdict(set)
for r in cit.itertuples(index=False):
    if str(r.citation_type) in ("1hop", "2hop", "cocite"):
        a, b = str(r.verdict_1), str(r.verdict_2); adj[a].add(b); adj[b].add(a)

folds = {}
for dom in ("drugs", "weapon"):
    for f in range(1, 6):
        idx = pd.read_csv(FILT / f"verdict_index_{dom}_topk_fold{f}_offenseFiltered.csv"); idx["verdict"] = idx.verdict.astype(str)
        for q in idx.loc[idx.split == "test", "verdict"]:
            folds[q] = (dom, set(idx.loc[idx.split == "train", "verdict"]))

rows = []
for q, (dom, train) in folds.items():
    if q not in LO or pd.isna(LO[q]): continue
    nb = [n for n in adj.get(q, set()) & train if n in LO and pd.notna(LO[n])]
    if len(nb) < 5: continue
    los = np.array([LO[n] for n in nb], float)
    rows.append(dict(query=q, domain=dom, true_low=LO[q], n=len(nb),
                     nb_min=los.min(), nb_max=los.max(), nb_median=np.median(los),
                     nb_std=los.std(), nb_iqr=np.percentile(los, 75) - np.percentile(los, 25),
                     nb_spread=los.max() - los.min()))
df = pd.DataFrame(rows); df.to_csv(DATA / "within_network_spread.csv", index=False)
print("Aggregate within-network spread of neighbor sentence (low), months:")
for dom in ("drugs", "weapon"):
    s = df[df.domain == dom]
    print(f"  {dom}: n={len(s)}  median std={s.nb_std.median():.1f}  median IQR={s.nb_iqr.median():.0f}  median(max-min)={s.nb_spread.median():.0f}")

# two example cases (rich neighbor count, large spread)
# typical cases (near each domain's MEDIAN within-network spread), not the extremes
EXAMPLES = [("תפ_34581-05-24", "drugs"), ("תפ_69925-03-23", "weapon")]
ex_rows = []
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (q, dom) in zip(axes, EXAMPLES):
    train = folds[q][1]
    nb = [n for n in adj.get(q, set()) & train if n in LO and pd.notna(LO[n])]
    nb = sorted(nb, key=lambda n: LO[n])
    for i, n in enumerate(nb):
        ax.plot([LO[n], HI[n]], [i, i], color="#9aa0a6", lw=2, alpha=.7,
                zorder=1, solid_capstyle="round")
        ex_rows.append(dict(query=q, neighbor=n, low=LO[n], high=HI[n]))
    ax.axvspan(LO[q], HI[q], color="#c62828", alpha=.18, zorder=0)
    ax.axvline(LO[q], color="#c62828", lw=2, label=f"this case: {LO[q]:.0f}-{HI[q]:.0f}m")
    los = np.array([LO[n] for n in nb])
    ax.set_title(f"{q} ({dom})\n{len(nb)} cases in its citation network — their 'low' spans {los.min():.0f}-{los.max():.0f} months",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("sentencing range (months)"); ax.set_ylabel("neighbor cases (sorted by low)")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=.25, axis="x")
fig.suptitle("Baseline reality: even within ONE case's citation network, sentences are wildly spread",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(PLOTS / "plot_baseline_noise_within_network.png", dpi=140, bbox_inches="tight")
pd.DataFrame(ex_rows).to_csv(DATA / "within_network_example_neighbors.csv", index=False)
print("\nwrote plot_baseline_noise_within_network.png + 2 CSVs")
