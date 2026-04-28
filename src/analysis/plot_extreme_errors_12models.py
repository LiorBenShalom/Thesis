"""Same figure as plot_extreme_errors but on 12 models (drop weakest two:
llama3_70b, gemma3_27b). Adds significance brackets vs Manual using
Wilcoxon signed-rank with FDR-BH correction across the 6 comparisons."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

EXP = Path(__file__).resolve().parents[2]
SRC = EXP/"results_paper"/"confusion_3way"/"per_model.csv"
OUT = EXP/"results_paper"/"confusion_3way"/"fig_extreme_errors_12models.png"

EXCLUDE = {"llama3_70b", "gemma3_27b"}
ORDER = ["Manual", "GPT-Schema", "Hybrid-Manual", "Hybrid-Full",
         "GPT-Free", "Raw-Facts", "GPT-Law"]
TIER = {
    "Manual":        ("#3a8a3a", "Tier 1: Manual"),
    "GPT-Schema":    ("#f7e3a1", "Tier 2 (structured)"),
    "Hybrid-Manual": ("#f7e3a1", "Tier 2 (structured)"),
    "Hybrid-Full":   ("#f7e3a1", "Tier 2 (structured)"),
    "GPT-Free":      ("#c0392b", "Tier 3 (unstructured)"),
    "Raw-Facts":     ("#c0392b", "Tier 3 (unstructured)"),
    "GPT-Law":       ("#c0392b", "Tier 3 (unstructured)"),
}

df = pd.read_csv(SRC)
df = df[~df["model"].isin(EXCLUDE)].copy()
df["pct"] = df["off_diag_1_3"] * 100


def stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def manual_vs_others_sig(sub: pd.DataFrame) -> dict[str, str]:
    pivot = sub.pivot(index="model", columns="rep", values="pct")
    others = [r for r in ORDER if r != "Manual"]
    raw = []
    for r in others:
        d = pivot["Manual"].values - pivot[r].values
        try:
            _, p = wilcoxon(d, zero_method="wilcox")
        except ValueError:
            p = 1.0
        raw.append(p)
    _, fdr, _, _ = multipletests(raw, method="fdr_bh")
    return {r: stars(p) for r, p in zip(others, fdr)}


fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
for ax, dom, title in zip(axes, ["drugs", "weapon"], ["Drugs", "Weapons"]):
    sub = df[df.domain == dom]
    agg = (sub.groupby("rep")["pct"].agg(["mean", "std"])
                .reindex(ORDER).reset_index())
    sig = manual_vs_others_sig(sub)
    sig["Manual"] = ""
    colors = [TIER[r][0] for r in agg["rep"]]

    bars = ax.bar(agg["rep"], agg["mean"], yerr=agg["std"], color=colors,
                  edgecolor="black", linewidth=0.6, capsize=4,
                  error_kw=dict(ecolor="#444", lw=1))

    ymax = (agg["mean"] + agg["std"]).max()
    for b, r, m, s in zip(bars, agg["rep"], agg["mean"], agg["std"]):
        label = f"{m:.1f}%"
        sig_mark = sig[r]
        if sig_mark and sig_mark != "ns":
            label += f" {sig_mark}"
        ax.text(b.get_x() + b.get_width()/2, m + s + ymax*0.02,
                label, ha="center", va="bottom", fontsize=9)

    n_pairs = int(sub["n"].iloc[0])
    n_models = sub["model"].nunique()
    ax.set_title(f"{title}  (n={n_pairs} pairs, {n_models} models)",
                 fontweight="bold")
    ax.set_ylabel("Extreme (1↔3) error rate (%)")
    ax.set_ylim(0, ymax * 1.30)
    ax.tick_params(axis="x", labelrotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

seen = {}
for r in ORDER:
    c, lbl = TIER[r]
    seen.setdefault(lbl, c)
handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="black", lw=0.6)
           for c in seen.values()]
fig.legend(handles, list(seen.keys()), loc="upper center",
           ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Extreme prediction errors (1↔3), 12 models — *** /** /* = "
             "FDR-corrected Wilcoxon vs Manual",
             y=1.08, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"wrote {OUT}")
