"""Generate updated paper figures for N=13 panel (= 9 ORIG + Mistral + DeepSeek + Haiku + Kimi):
  1. fig_extreme_errors_n13.png  — bar chart of 1↔3 error rate by rep, mean ± std across models
  2. fig_cld_qwk_n13.png         — boxplot with Compact Letter Display for QWK Oracle

CLD algorithm: pairwise Wilcoxon (paired across models) with BH-FDR correction at α=0.05.
Reps sharing a letter are NOT significantly different.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

EXP = Path(__file__).resolve().parents[2]
ROOT = EXP/"v6_final"  # all 13 panel models live here after pilot merge
OUT = EXP/"results_paper"/"confusion_3way"
OUT.mkdir(parents=True, exist_ok=True)
OUT_QWK = EXP/"results_paper_qwk"
OUT_QWK.mkdir(parents=True, exist_ok=True)

PANEL = [
    "gpt4","gpt5mini","gpt52","gpt51_thinking","claude_sonnet_4_6",
    "gemini_25_pro","gemini_3_flash","gemma4_31b_or","qwen3_vl_235b_or",
    "mistral_large_or","deepseek_r1_or","claude_haiku_4_5","kimi_k26_or",
]
REPS = [
    ("Raw-Facts",     "similarity_database_with_indicment_facts"),
    ("Manual",        "similarity_database_fe"),
    ("GPT-Schema",    "similarity_database_fe_gpt_schema_v2"),
    ("GPT-Free",      "similarity_database_with_gpt_features"),
    ("GPT-Law",       "similarity_database_with_gpt_law_features"),
    ("Hybrid-Manual", "similarity_database_hybrid"),
    ("Hybrid-Full",   "similarity_database_hybrid_full_gpt"),
]
T1, T2, T3 = ["Manual"], ["GPT-Schema","Hybrid-Manual","Hybrid-Full"], ["Raw-Facts","GPT-Free","GPT-Law"]
TIER = {**{r:"T1" for r in T1}, **{r:"T2" for r in T2}, **{r:"T3" for r in T3}}

def base_for(dom, m):
    return ROOT/dom/f"results_{dom}"

def best_qwk(scores, gt):
    uniq = np.unique(scores)
    if len(uniq) < 3: return np.nan, 0, 0
    mids = (uniq[:-1] + uniq[1:]) / 2
    bq, bt = -1, (mids[0], mids[-1])
    for i, t1 in enumerate(mids):
        for t2 in mids[i+1:]:
            pred = np.where(scores < t1, 1, np.where(scores < t2, 2, 3))
            if len(np.unique(pred)) < 2: continue
            q = cohen_kappa_score(gt, pred, weights="quadratic")
            if q > bq: bq, bt = q, (t1, t2)
    return bq, bt[0], bt[1]

def cell(dom, m, prefix):
    p = base_for(dom, m)/f"{prefix}_v6score_{m}_binary_0_preds.csv"
    if not p.exists(): return None
    df = pd.read_csv(p)
    if "status" in df.columns: df = df[df.status == "ok"]
    df = df.dropna(subset=["similarity_scale", "score"])
    if len(df) < 50: return None
    gt = df.similarity_scale.astype(int).values
    sc = df.score.astype(float).values
    q, t1, t2 = best_qwk(sc, gt)
    pred = np.where(sc < t1, 1, np.where(sc < t2, 2, 3))
    ext = 100.0 * (((gt==1)&(pred==3)).sum() + ((gt==3)&(pred==1)).sum()) / len(gt)
    return q, ext

# Collect data
rows = []
for m in PANEL:
    for rep, prefix in REPS:
        for dom in ["drugs", "weapon"]:
            r = cell(dom, m, prefix)
            if r: rows.append(dict(model=m, rep=rep, dom=dom, qwk=r[0], ext=r[1]))
df = pd.DataFrame(rows)


def cld_letters(metric_pivot: pd.DataFrame, higher_is_better: bool) -> dict[str, str]:
    """Compact Letter Display: pairwise Wilcoxon with FDR-BH correction.
    Returns {rep: letters}. Reps sharing a letter are NOT significantly different."""
    reps = list(metric_pivot.columns)
    means = metric_pivot.mean()
    order = means.sort_values(ascending=not higher_is_better).index.tolist()
    n = len(order)
    pmatrix = np.ones((n, n))
    for i in range(n):
        for j in range(i+1, n):
            common = metric_pivot[[order[i], order[j]]].dropna()
            if len(common) < 3: continue
            try:
                _, p = wilcoxon(common[order[i]] - common[order[j]], zero_method="wilcox")
            except (ValueError, TypeError):
                p = 1.0
            pmatrix[i, j] = pmatrix[j, i] = p
    # FDR-BH on the (n choose 2) tests
    upper = [(i, j) for i in range(n) for j in range(i+1, n)]
    pvals = [pmatrix[i, j] for i, j in upper]
    _, pfdr, _, _ = multipletests(pvals, method="fdr_bh", alpha=0.05)
    sig = np.zeros((n, n), bool)
    for k, (i, j) in enumerate(upper):
        sig[i, j] = sig[j, i] = pfdr[k] < 0.05
    # Greedy CLD letter assignment
    groups = []  # each group: list of rep indices, all pairwise NOT significantly different
    for i in range(n):
        placed = False
        for g in groups:
            if all(not sig[i, j] for j in g):
                g.append(i); placed = True
                # don't break — must add to ALL eligible groups
        if not placed:
            groups.append([i])
        else:
            # ensure rep is in all groups it belongs to
            for g in groups:
                if i not in g and all(not sig[i, j] for j in g):
                    g.append(i)
    letters = {order[i]: "" for i in range(n)}
    for li, g in enumerate(groups):
        ch = chr(ord('a') + li)
        for i in g:
            if ch not in letters[order[i]]:
                letters[order[i]] += ch
    return letters


# ============ FIG 1: Extreme errors bar chart ============
ORDER_BAR = ["Manual","GPT-Schema","Hybrid-Manual","Hybrid-Full","GPT-Free","Raw-Facts","GPT-Law"]
TIER_COLOR = {"Manual": ("#3a8a3a", "Tier 1: Manual"),
              "GPT-Schema": ("#f7e3a1", "Tier 2 (structured)"),
              "Hybrid-Manual": ("#f7e3a1", "Tier 2 (structured)"),
              "Hybrid-Full": ("#f7e3a1", "Tier 2 (structured)"),
              "GPT-Free": ("#c0392b", "Tier 3 (unstructured)"),
              "Raw-Facts": ("#c0392b", "Tier 3 (unstructured)"),
              "GPT-Law": ("#c0392b", "Tier 3 (unstructured)")}

def stars_from_p(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
for ax, dom, title in zip(axes, ["drugs", "weapon"], ["Drugs", "Weapons"]):
    sub = df[df.dom == dom]
    pivot = sub.pivot_table(index="model", columns="rep", values="ext")
    # per-rep mean/std
    g = sub.groupby("rep")["ext"].agg(["mean", "std"]).reindex(ORDER_BAR).reset_index()
    # Manual vs each — paired Wilcoxon FDR-BH
    others = [r for r in ORDER_BAR if r != "Manual"]
    pvals = []
    for r in others:
        common = pivot[["Manual", r]].dropna()
        try: _, p = wilcoxon(common["Manual"] - common[r], zero_method="wilcox")
        except (ValueError, TypeError): p = 1.0
        pvals.append(p)
    _, pfdr, _, _ = multipletests(pvals, method="fdr_bh")
    sig = dict(zip(others, [stars_from_p(p) for p in pfdr])); sig["Manual"] = ""

    colors = [TIER_COLOR[r][0] for r in g["rep"]]
    bars = ax.bar(g["rep"], g["mean"], yerr=g["std"], color=colors,
                  edgecolor="black", linewidth=0.6, capsize=4,
                  error_kw=dict(ecolor="#444", lw=1))
    ymax = (g["mean"] + g["std"]).max()
    for b, r, m, s in zip(bars, g["rep"], g["mean"], g["std"]):
        lbl = f"{m:.1f}%"
        if sig[r]: lbl += f" {sig[r]}"
        ax.text(b.get_x() + b.get_width()/2, m + s + ymax*0.02, lbl,
                ha="center", va="bottom", fontsize=9)
    n_pairs = {"drugs": 100, "weapon": 141}[dom]
    ax.set_title(f"{title}  (n={n_pairs} pairs, 13 models)", fontweight="bold")
    ax.set_ylabel("Extreme (1↔3) error rate (%)")
    ax.set_ylim(0, ymax * 1.30)
    ax.tick_params(axis="x", labelrotation=30)
    for lbl in ax.get_xticklabels(): lbl.set_horizontalalignment("right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

seen = {}
for r in ORDER_BAR:
    c, lbl = TIER_COLOR[r]; seen.setdefault(lbl, c)
handles = [plt.Rectangle((0,0),1,1,color=c,ec="black",lw=0.6) for c in seen.values()]
fig.legend(handles, list(seen.keys()), loc="upper center", ncol=3,
           frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Extreme prediction errors (1↔3), N=13 panel — *** /** /* = FDR Wilcoxon vs Manual",
             y=1.08, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT/"fig_extreme_errors_n13.png", dpi=180, bbox_inches="tight")
print(f"wrote {OUT/'fig_extreme_errors_n13.png'}")
plt.close()


# ============ FIG 2: CLD boxplot for QWK ============
TIER_BOX_COLOR = {"T1": "#1f4e79", "T2": "#a5c8e1", "T3": "#d9d9d9"}
TIER_LABEL = {"T1": "Tier 1: Manual",
              "T2": "Tier 2: GPT-Schema, Hybrids",
              "T3": "Tier 3: Raw-Facts, GPT-Free, GPT-Law"}

fig, axes = plt.subplots(1, 2, figsize=(13, 6.0))
for ax, dom, title in zip(axes, ["drugs", "weapon"], ["drugs", "weapon"]):
    sub = df[df.dom == dom]
    pivot = sub.pivot_table(index="model", columns="rep", values="qwk")
    means = pivot.mean()
    order = means.sort_values(ascending=False).index.tolist()
    letters = cld_letters(pivot, higher_is_better=True)

    data = [pivot[r].dropna().values for r in order]
    bp = ax.boxplot(data, positions=range(len(order)), widths=0.6,
                    patch_artist=True, medianprops=dict(color="black", lw=1.5))
    for patch, rep in zip(bp["boxes"], order):
        patch.set_facecolor(TIER_BOX_COLOR[TIER[rep]])
        patch.set_edgecolor("black"); patch.set_linewidth(0.8)

    # scatter individual model points
    for i, rep in enumerate(order):
        vals = pivot[rep].dropna().values
        ax.scatter(np.repeat(i, len(vals)) + np.random.uniform(-0.12, 0.12, len(vals)),
                   vals, s=18, color="#1f4e79", alpha=0.5, edgecolor="none")

    # CLD letters above each box
    ymax = pivot.max().max()
    ymin = pivot.min().min()
    for i, rep in enumerate(order):
        ax.text(i, pivot[rep].max() + (ymax-ymin)*0.04, letters[rep],
                ha="center", fontsize=14, fontweight="bold")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("QWK (Oracle)")
    ax.set_title(f"{title}  (n=13 models)", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(ymin - (ymax-ymin)*0.08, ymax + (ymax-ymin)*0.18)

# legend — placed BELOW the plots so it doesn't obscure either subplot
handles = [plt.Rectangle((0,0),1,1,color=TIER_BOX_COLOR[t],ec="black",lw=0.8)
           for t in ["T1","T2","T3"]]
labels = [TIER_LABEL[t] for t in ["T1","T2","T3"]]
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
           ncol=3, frameon=False)
fig.suptitle("QWK (Oracle) — Compact Letter Display.  Reps sharing a letter are NOT significantly different "
             "(Wilcoxon two-sided, BH-FDR α=0.05).",
             fontsize=11)
plt.tight_layout(rect=[0, 0.05, 1, 0.96])
plt.savefig(OUT_QWK/"fig_cld_qwk_oracle_n13.png", dpi=180, bbox_inches="tight")
print(f"wrote {OUT_QWK/'fig_cld_qwk_oracle_n13.png'}")
plt.close()
