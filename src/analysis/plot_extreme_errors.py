"""Bar chart in the paper-figure-3 style: extreme 1<->3 error rate per
representation, two domains side-by-side, bars tinted by tier
(Manual = green, structured = pale yellow, unstructured = red).
Reads results_paper/confusion_3way/summary.csv."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parents[2]
SRC = EXP/"results_paper"/"confusion_3way"/"summary.csv"
OUT = EXP/"results_paper"/"confusion_3way"/"fig_extreme_errors.png"

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
df["pct"] = df["off_diag_1_3_mean"] * 100
df["err"] = df["off_diag_1_3_std"] * 100

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
for ax, dom, title in zip(axes, ["drugs", "weapon"], ["Drugs", "Weapons"]):
    sub = (df[df.domain == dom]
           .set_index("rep").loc[ORDER].reset_index())
    colors = [TIER[r][0] for r in sub["rep"]]
    bars = ax.bar(sub["rep"], sub["pct"], yerr=sub["err"],
                  color=colors, edgecolor="black", linewidth=0.6,
                  capsize=4, error_kw=dict(ecolor="#444", lw=1))
    n_pairs = int(sub["n_pairs"].iloc[0])
    n_models = int(sub["n_models"].iloc[0])
    for b, v, e in zip(bars, sub["pct"], sub["err"]):
        ax.text(b.get_x() + b.get_width()/2, v + e + (sub["pct"].max()*0.02),
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{title}  (n={n_pairs} pairs, {n_models} models)",
                 fontweight="bold")
    ax.set_ylabel("Extreme (1↔3) error rate (%)")
    ax.set_ylim(0, (sub["pct"].max() + sub["err"].max()) * 1.22)
    ax.tick_params(axis="x", labelrotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# single shared legend
seen = {}
for r in ORDER:
    c, lbl = TIER[r]
    if lbl not in seen:
        seen[lbl] = c
handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="black", lw=0.6)
           for c in seen.values()]
fig.legend(handles, list(seen.keys()), loc="upper center",
           ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Extreme prediction errors (1↔3): structured representations make far fewer",
             y=1.08, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"wrote {OUT}")
