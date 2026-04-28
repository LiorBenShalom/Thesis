"""Paper-grade ordinal evaluation on 1-3 similarity scale.

Three separate metrics:
  1. QWK — Quadratic Weighted Kappa (Oracle + 5-fold CV)
  2. Significance — Manual vs. each other representation (Wilcoxon)
  3. C-index — Ordinal concordance, threshold-free

Usage:
  cd new_try/experiments/src/analysis
  python paper_results_qwk.py
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

EXP = Path(__file__).resolve().parents[2]
OUT = EXP / "results_paper_qwk"
OUT.mkdir(exist_ok=True)

DOMAINS = {
    "drugs": EXP / "v6_final" / "drugs" / "results_drugs",
    "weapon": EXP / "v6_final" / "weapon" / "results_weapon",
}
REP_PREFIX = {
    "Manual": "similarity_database_fe",
    "GPT-Schema": "similarity_database_fe_gpt_schema_v2",
    "GPT-Free": "similarity_database_with_gpt_features",
    "GPT-Law": "similarity_database_with_gpt_law_features",
    "Raw-Facts": "similarity_database_with_indicment_facts",
    "Hybrid-Manual": "similarity_database_hybrid",
    "Hybrid-Full": "similarity_database_hybrid_full_gpt",
}
REP_ORDER = list(REP_PREFIX.keys())
MODELS = [
    "gpt4", "gpt5mini", "gpt52", "gpt51_thinking", "claude_sonnet_4_6",
    "gemini_25_pro", "gemini_3_flash", "gemma3_27b", "gemma4_31b_or",
    "llama3_70b", "qwen3_vl_235b_or",
]
MODEL_DISPLAY = {
    "gpt4": "GPT-4", "gpt5mini": "GPT-5-Mini", "gpt52": "GPT-5.2",
    "gpt51_thinking": "GPT-5.1", "claude_sonnet_4_6": "Claude 4.6",
    "gemini_25_pro": "Gemini 2.5 Pro", "gemini_3_flash": "Gemini 3 Flash",
    "gemma3_27b": "Gemma3-27B", "gemma4_31b_or": "Gemma4-31B",
    "llama3_70b": "Llama3-70B", "qwen3_vl_235b_or": "Qwen3-235B",
}


# ═══════════════════════════════════════════════════════════════════════
#  1. QWK helpers
# ═══════════════════════════════════════════════════════════════════════

def _qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Quadratic Weighted Kappa for ordinal ratings 1-3."""
    n_r = 3
    O = np.zeros((n_r, n_r))
    for t, p in zip(y_true, y_pred):
        O[t - 1, p - 1] += 1
    N = len(y_true)
    ht = np.bincount(y_true - 1, minlength=n_r)
    hp = np.bincount(y_pred - 1, minlength=n_r)
    E = np.outer(ht, hp).astype(float) / N
    W = np.zeros((n_r, n_r))
    for i in range(n_r):
        for j in range(n_r):
            W[i, j] = ((i - j) ** 2) / ((n_r - 1) ** 2)
    denom = np.sum(W * E)
    return 1.0 - (np.sum(W * O) / denom) if denom > 0 else 0.0


def _find_best_thresholds(scores: np.ndarray, gt: np.ndarray):
    """Find (qwk, t1, t2) maximising QWK when mapping scores -> 1/2/3."""
    uniq = np.unique(scores)
    if len(uniq) < 3:
        return 0.0, 0.0, 50.0
    mids = (uniq[:-1] + uniq[1:]) / 2.0
    best_qwk, best_t1, best_t2 = -1.0, mids[0], mids[-1]
    for i, t1 in enumerate(mids):
        for t2 in mids[i + 1:]:
            preds = np.where(scores < t1, 1, np.where(scores < t2, 2, 3))
            if len(np.unique(preds)) < 2:
                continue
            q = _qwk(gt, preds)
            if q > best_qwk:
                best_qwk, best_t1, best_t2 = q, t1, t2
    return best_qwk, best_t1, best_t2


def _cv_qwk(scores: np.ndarray, gt: np.ndarray, k: int = 10, seed: int = 42) -> float:
    """10-fold stratified CV for QWK: tune 2 thresholds on train, predict on test, pool."""
    if len(np.unique(gt)) < 2:
        return np.nan
    if min(np.bincount(gt - 1)) < k:
        return np.nan
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    preds = np.zeros(len(gt), dtype=int)
    for tr, te in skf.split(scores, gt):
        _, t1, t2 = _find_best_thresholds(scores[tr], gt[tr])
        preds[te] = np.where(scores[te] < t1, 1, np.where(scores[te] < t2, 2, 3))
    return _qwk(gt, preds)


# ═══════════════════════════════════════════════════════════════════════
#  2. C-index helper
# ═══════════════════════════════════════════════════════════════════════

def _ordinal_c_index(gt: np.ndarray, scores: np.ndarray) -> float:
    """Pairwise concordance: P(score_i > score_j | GT_i > GT_j). Threshold-free."""
    n = len(gt)
    concordant = 0
    discordant = 0
    tied = 0
    for i in range(n):
        for j in range(i + 1, n):
            if gt[i] == gt[j]:
                continue
            hi, lo = (i, j) if gt[i] > gt[j] else (j, i)
            if scores[hi] > scores[lo]:
                concordant += 1
            elif scores[hi] < scores[lo]:
                discordant += 1
            else:
                tied += 1
    total = concordant + discordant + tied
    return (concordant + 0.5 * tied) / total if total > 0 else np.nan


# ═══════════════════════════════════════════════════════════════════════
#  3. Per-cell computation
# ═══════════════════════════════════════════════════════════════════════

def cell_metrics(base: Path, rep_prefix: str, model: str):
    """Returns (qwk_oracle, qwk_cv, c_index, n) or NaNs."""
    nans = (np.nan, np.nan, np.nan, 0)
    p = base / f"{rep_prefix}_v6score_{model}_binary_0_preds.csv"
    if not p.exists():
        return nans
    df = pd.read_csv(p)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    if len(df) < 20 or "similarity_scale" not in df.columns:
        return nans
    gt = df["similarity_scale"].astype(int).values
    sc = df["score"].astype(float).values
    mask = ~np.isnan(sc)
    gt, sc = gt[mask], sc[mask]
    if len(sc) < 20:
        return nans

    qwk_oracle, _, _ = _find_best_thresholds(sc, gt)
    qwk_cv = _cv_qwk(sc, gt)
    ci = _ordinal_c_index(gt, sc)
    return qwk_oracle, qwk_cv, ci, len(sc)


# ═══════════════════════════════════════════════════════════════════════
#  4. Significance
# ═══════════════════════════════════════════════════════════════════════

def cohens_dz(d: np.ndarray) -> float:
    d = d[~np.isnan(d)]
    if len(d) < 2 or d.std(ddof=1) == 0:
        return np.nan
    return d.mean() / d.std(ddof=1)


def eff_label(dz: float) -> str:
    if np.isnan(dz):
        return "-"
    a = abs(dz)
    if a < 0.2: return "Negligible"
    if a < 0.5: return "Small"
    if a < 0.8: return "Medium"
    return "Large"


def pairwise_sig(full_df: pd.DataFrame, domain: str, metric: str) -> pd.DataFrame:
    sub = full_df[full_df.domain == domain]
    wide = sub.pivot(index="model", columns="rep", values=metric).reindex(columns=REP_ORDER)
    recs = []
    for a, b in itertools.permutations(REP_ORDER, 2):
        A, B = wide[a].values, wide[b].values
        mask = ~(np.isnan(A) | np.isnan(B))
        d = A[mask] - B[mask]
        if len(d) < 3 or np.all(d == 0):
            recs.append(dict(A=a, B=b, n=len(d), delta=np.nan, dz=np.nan, wins=0, p=np.nan))
            continue
        try:
            p = float(wilcoxon(d, alternative="greater", zero_method="wilcox").pvalue)
        except Exception:
            p = np.nan
        recs.append(dict(A=a, B=b, n=len(d), delta=float(d.mean()),
                         dz=cohens_dz(d), wins=int((d > 0).sum()), p=p))
    df = pd.DataFrame(recs)
    for a in REP_ORDER:
        idx = df[df.A == a].index
        pv = df.loc[idx, "p"].values
        m = ~np.isnan(pv)
        if m.sum() == 0:
            continue
        _, fdr, _, _ = multipletests(pv[m], alpha=0.05, method="fdr_bh")
        bonf = np.minimum(pv[m] * m.sum(), 1.0)
        df.loc[idx[m], "p_fdr"] = fdr
        df.loc[idx[m], "p_bonf"] = bonf
    df["effect"] = df["dz"].map(eff_label)
    df["sig_fdr"] = (df["p_fdr"] < 0.05).map({True: "Y", False: ""})
    df["sig_bonf"] = (df["p_bonf"] < 0.05).map({True: "Y", False: ""})
    df["domain"] = domain
    df["metric"] = metric
    return df


# ═══════════════════════════════════════════════════════════════════════
#  5. Visualization
# ═══════════════════════════════════════════════════════════════════════

def make_plots(full: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "Arial"
    matplotlib.rcParams["figure.dpi"] = 150

    REP_SHORT_MAP = {"Manual": "Manual", "GPT-Schema": "GPT-Sch", "Hybrid-Full": "Hyb-Full",
                      "Hybrid-Manual": "Hyb-Man", "Raw-Facts": "Facts", "GPT-Free": "GPT-Free", "GPT-Law": "GPT-Law"}
    COLOR_MAP = {"Manual": "#2196F3", "GPT-Schema": "#4CAF50", "Hybrid-Full": "#FF9800",
                 "Hybrid-Manual": "#9C27B0", "Raw-Facts": "#795548", "GPT-Free": "#E91E63", "GPT-Law": "#607D8B"}

    def _sorted_bar(ax, full_df, domain, metric, ylabel, ylim):
        """Bar chart sorted by mean (descending)."""
        sub = full_df[full_df["domain"] == domain]
        means = sub.groupby("rep")[metric].mean()
        stds = sub.groupby("rep")[metric].std()
        order = means.sort_values(ascending=False).index.tolist()
        m_vals = means.reindex(order).values
        s_vals = stds.reindex(order).values
        colors = [COLOR_MAP[r] for r in order]
        labels = [REP_SHORT_MAP[r] for r in order]
        ax.bar(range(len(order)), m_vals, yerr=s_vals, color=colors,
               edgecolor="white", capsize=4, error_kw={"linewidth": 1})
        for i, (m, s) in enumerate(zip(m_vals, s_vals)):
            ax.text(i, m + s + 0.005, f"{m:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(domain.upper(), fontsize=13, fontweight="bold")
        ax.set_ylim(ylim)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── Fig 1: QWK-Oracle bar chart (sorted) ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("QWK (Oracle) by Representation", fontsize=14, fontweight="bold")
    for col, domain in enumerate(["drugs", "weapon"]):
        _sorted_bar(axes[col], full, domain, "QWK_Oracle", "QWK (Oracle)", (0.60, 0.95))
    plt.tight_layout()
    plt.savefig(OUT / "fig1_qwk_oracle.png", bbox_inches="tight")
    plt.close()

    # ── Fig 2: QWK-CV bar chart (sorted) ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("QWK (10-Fold CV) by Representation", fontsize=14, fontweight="bold")
    for col, domain in enumerate(["drugs", "weapon"]):
        _sorted_bar(axes[col], full, domain, "QWK_CV", "QWK (10-Fold CV)", (0.55, 0.95))
    plt.tight_layout()
    plt.savefig(OUT / "fig2_qwk_cv.png", bbox_inches="tight")
    plt.close()

    # ── Fig 3: C-index bar chart (sorted) ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Ordinal C-index (Threshold-Free) by Representation", fontsize=14, fontweight="bold")
    for col, domain in enumerate(["drugs", "weapon"]):
        _sorted_bar(axes[col], full, domain, "C_index", "C-index", (0.78, 0.96))
    plt.tight_layout()
    plt.savefig(OUT / "fig3_c_index.png", bbox_inches="tight")
    plt.close()

    def _sorted_heatmap(full_df, metric, title, cmap, vmin, vmax, thresh_white, label, filename):
        """Heatmap with columns sorted by mean metric (descending), rows sorted by mean across reps."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(title, fontsize=14, fontweight="bold")
        for col, domain in enumerate(["drugs", "weapon"]):
            ax = axes[col]
            sub = full_df[full_df["domain"] == domain]
            pivot = sub.pivot(index="model", columns="rep", values=metric)
            # Sort columns (reps) by mean descending
            col_order = pivot.mean().sort_values(ascending=False).index.tolist()
            # Sort rows (models) by mean descending
            row_order = pivot.mean(axis=1).sort_values(ascending=False).index.tolist()
            pivot = pivot.reindex(index=row_order, columns=col_order)
            col_labels = [REP_SHORT_MAP[r] for r in col_order]
            row_labels = [MODEL_DISPLAY.get(m, m) for m in row_order]
            im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(col_order)))
            ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=9)
            ax.set_yticks(range(len(row_order)))
            ax.set_yticklabels(row_labels, fontsize=9)
            ax.set_title(domain.upper(), fontsize=13, fontweight="bold")
            for i in range(len(row_order)):
                for j in range(len(col_order)):
                    v = pivot.values[i, j]
                    if not np.isnan(v):
                        c = "white" if v > thresh_white else "black"
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=c, fontweight="bold")
        fig.colorbar(im, ax=axes, shrink=0.6, label=label, pad=0.02)
        plt.savefig(OUT / filename, bbox_inches="tight")
        plt.close()

    _sorted_heatmap(full, "QWK_Oracle", "QWK (Oracle): Models x Representations",
                    "YlOrRd", 0.55, 0.95, 0.8, "QWK (Oracle)", "fig4_heatmap_qwk_oracle.png")
    _sorted_heatmap(full, "QWK_CV", "QWK (10-Fold CV): Models x Representations",
                    "YlOrRd", 0.55, 0.95, 0.8, "QWK (CV)", "fig5_heatmap_qwk_cv.png")
    _sorted_heatmap(full, "C_index", "Ordinal C-index: Models x Representations",
                    "YlGn", 0.78, 0.95, 0.9, "C-index", "fig6_heatmap_c_index.png")

    # ── Fig 7/8/9: Pairwise significance heatmaps (all reps vs all reps) ──
    def _sig_heatmap(full_df, metric, title, filename):
        """Pairwise significance matrix: row A vs col B.
        Cell = mean delta (A-B). Green = A significantly better, white = not sig."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        for col, domain in enumerate(["drugs", "weapon"]):
            ax = axes[col]
            sub = full_df[full_df["domain"] == domain]
            wide = sub.pivot(index="model", columns="rep", values=metric)
            # Sort reps by mean
            rep_order = wide.mean().sort_values(ascending=False).index.tolist()
            n = len(rep_order)

            # Build delta matrix and p-value matrix
            delta_mat = np.full((n, n), np.nan)
            sig_mat = np.full((n, n), False)
            for i, a in enumerate(rep_order):
                for j, b in enumerate(rep_order):
                    if i == j:
                        delta_mat[i, j] = 0.0
                        continue
                    A, B = wide[a].values, wide[b].values
                    mask = ~(np.isnan(A) | np.isnan(B))
                    d = A[mask] - B[mask]
                    delta_mat[i, j] = d.mean() if len(d) > 0 else 0.0
                    if len(d) >= 3 and not np.all(d == 0):
                        try:
                            p = wilcoxon(d, alternative="greater", zero_method="wilcox").pvalue
                            sig_mat[i, j] = p < 0.05
                        except Exception:
                            pass

            # Color: green where row > col (positive delta), red where row < col
            im = ax.imshow(delta_mat, cmap="RdYlGn", aspect="auto",
                           vmin=-0.12, vmax=0.12)
            labels = [REP_SHORT_MAP[r] for r in rep_order]
            ax.set_xticks(range(n))
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
            ax.set_yticks(range(n))
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_title(domain.upper(), fontsize=13, fontweight="bold")
            ax.set_xlabel("B (column)", fontsize=9)
            ax.set_ylabel("A (row)", fontsize=9)

            for i in range(n):
                for j in range(n):
                    if i == j:
                        ax.text(j, i, "-", ha="center", va="center", fontsize=8, color="gray")
                        continue
                    v = delta_mat[i, j]
                    star = "*" if sig_mat[i, j] else ""
                    color = "black" if abs(v) < 0.08 else "white"
                    ax.text(j, i, f"{v:+.3f}{star}", ha="center", va="center",
                            fontsize=6.5, color=color, fontweight="bold" if star else "normal")

        fig.colorbar(im, ax=axes, shrink=0.6, label="Delta (A - B)", pad=0.02)
        fig.text(0.5, 0.01, "* = significant (Wilcoxon p < 0.05, uncorrected)",
                 ha="center", fontsize=9, style="italic")
        plt.savefig(OUT / filename, bbox_inches="tight")
        plt.close()

    _sig_heatmap(full, "QWK_Oracle", "Pairwise Significance: QWK (Oracle)", "fig7_sig_qwk_oracle.png")
    _sig_heatmap(full, "QWK_CV", "Pairwise Significance: QWK (10-Fold CV)", "fig8_sig_qwk_cv.png")
    _sig_heatmap(full, "C_index", "Pairwise Significance: C-index", "fig9_sig_c_index.png")

    print("  All figures saved.")


# ═══════════════════════════════════════════════════════════════════════
#  6. Report
# ═══════════════════════════════════════════════════════════════════════

def build_report(full: pd.DataFrame, sig_qwk_oracle, sig_qwk_cv, sig_ci):
    def summarize(metric):
        g = full.groupby(["domain", "rep"])[metric].agg(["mean", "std"]).reset_index()
        g["cell"] = g["mean"].map(lambda x: f"{x:.3f}") + " +/- " + g["std"].map(lambda x: f"{x:.3f}")
        pivot = g.pivot_table(index="rep", columns="domain", values="cell", aggfunc="first")
        # Sort by overall mean (across domains)
        means = full.groupby("rep")[metric].mean()
        sort_order = means.sort_values(ascending=False).index.tolist()
        return pivot.reindex(sort_order)

    def wins(metric):
        w = full.dropna(subset=[metric]).copy()
        w["rank"] = w.groupby(["domain", "model"])[metric].rank(method="min", ascending=False)
        tbl = w[w["rank"] == 1].groupby(["domain", "rep"]).size().reset_index(name="wins")
        pivot = tbl.pivot_table(index="rep", columns="domain", values="wins", fill_value=0)
        # Sort by total wins
        pivot["_total"] = pivot.sum(axis=1)
        pivot = pivot.sort_values("_total", ascending=False).drop(columns="_total")
        return pivot

    def focus_table(sig, focus, domain):
        s = sig[(sig.A == focus) & (sig.domain == domain)].copy()
        s = s[["B", "delta", "dz", "effect", "wins", "n", "p", "p_fdr", "p_bonf", "sig_fdr", "sig_bonf"]]
        s.columns = ["vs", "Delta", "Cohen dz", "Effect", "Wins", "n", "raw p", "FDR p", "Bonf p", "FDR", "Bonf"]
        for c in ["Delta", "Cohen dz"]:
            s[c] = s[c].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "-")
        for c in ["raw p", "FDR p", "Bonf p"]:
            s[c] = s[c].map(lambda x: f"{x:.4f}" if pd.notna(x) and x >= 1e-4 else ("<.0001" if pd.notna(x) else "-"))
        return s.reset_index(drop=True)

    def best_model_table(metric):
        rows = []
        for dom in DOMAINS:
            sub = full[(full.domain == dom)].dropna(subset=[metric])
            best = sub.loc[sub.groupby("rep")[metric].idxmax()][["rep", "model", metric]]
            best["model"] = best["model"].map(lambda m: MODEL_DISPLAY.get(m, m))
            best["domain"] = dom
            rows.append(best)
        df = pd.concat(rows)
        return df.pivot_table(index="rep", columns="domain", values=[metric, "model"], aggfunc="first").reindex(REP_ORDER)

    md = []

    # ── Part 1: QWK ──
    md.append("# Part 1: QWK (Quadratic Weighted Kappa)\n")
    md.append("_Measures classification quality of model scores vs. GT on the 1-3 ordinal scale._\n")

    md.append("## 1a. QWK (Oracle) — mean across 11 models\n")
    md.append(summarize("QWK_Oracle").to_markdown())

    md.append("\n## 1b. QWK (10-Fold CV) — mean across 11 models\n")
    md.append(summarize("QWK_CV").to_markdown())

    md.append("\n## 1c. Wins (top-1 rep per model)\n")
    md.append("### QWK (Oracle)\n" + wins("QWK_Oracle").to_markdown())
    md.append("\n### QWK (10-Fold CV)\n" + wins("QWK_CV").to_markdown())

    # ── Part 2: Significance ──
    md.append("\n\n# Part 2: Statistical Significance\n")
    md.append("_One-sided Wilcoxon signed-rank across 11 models, Bonferroni + FDR correction._\n")
    md.append("_See fig7/fig8/fig9 for pairwise heatmaps (all reps vs all reps)._\n")

    # Focus tables: each rep vs all others
    sort_order = full.groupby("rep")["QWK_CV"].mean().sort_values(ascending=False).index.tolist()
    for dom in DOMAINS:
        md.append(f"\n## {dom.upper()}\n")
        for focus in sort_order:
            md.append(f"\n### {focus} vs. others\n")
            md.append(f"**QWK (CV)**\n" + focus_table(sig_qwk_cv, focus, dom).to_markdown())
            md.append(f"\n**C-index**\n" + focus_table(sig_ci, focus, dom).to_markdown())
            md.append("")

    # ── Part 3: C-index ──
    md.append("\n\n# Part 3: C-index (Threshold-Free)\n")
    md.append("_Ordinal concordance: P(score_i > score_j | GT_i > GT_j). Analogous to AUC for ordinal scales._\n")

    md.append("## 3a. Mean C-index across 11 models\n")
    md.append(summarize("C_index").to_markdown())

    md.append("\n## 3b. Wins\n")
    md.append(wins("C_index").to_markdown())

    # ── Part 4: Best model per rep ──
    md.append("\n\n# Part 4: Best Model per Representation\n")
    for dom in DOMAINS:
        md.append(f"\n## {dom.upper()}\n")
        sub = full[full.domain == dom].dropna(subset=["QWK_CV"])
        best = sub.loc[sub.groupby("rep")["QWK_CV"].idxmax()][["rep", "model", "QWK_Oracle", "QWK_CV", "C_index"]]
        best["model"] = best["model"].map(lambda m: MODEL_DISPLAY.get(m, m))
        best = best.set_index("rep").reindex(REP_ORDER)
        md.append(best.to_markdown())

    (OUT / "REPORT_QWK.md").write_text("\n".join(md), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Computing QWK (Oracle + 5-fold CV) and C-index...")
    print(f"  {len(DOMAINS)} domains x {len(REP_PREFIX)} reps x {len(MODELS)} models = {len(DOMAINS)*len(REP_PREFIX)*len(MODELS)} cells\n")

    rows = []
    total = len(DOMAINS) * len(REP_PREFIX) * len(MODELS)
    done = 0
    for dom, base in DOMAINS.items():
        for rep, pref in REP_PREFIX.items():
            for m in MODELS:
                qo, qcv, ci, n = cell_metrics(base, pref, m)
                rows.append(dict(domain=dom, rep=rep, model=m,
                                 QWK_Oracle=qo, QWK_CV=qcv, C_index=ci, n=n))
                done += 1
                if not np.isnan(qo):
                    print(f"  [{done:3d}/{total}] {dom:6s} | {rep:<15s} | {MODEL_DISPLAY.get(m,m):<18s} | "
                          f"QWK-O={qo:.3f}  QWK-CV={qcv:.3f}  C={ci:.3f}")

    full = pd.DataFrame(rows)
    full.to_csv(OUT / "full_results_qwk.csv", index=False)

    # ── Tables ──
    def save_summary(metric, filename):
        g = full.groupby(["domain", "rep"])[metric].agg(["mean", "std"]).reset_index()
        g["cell"] = g["mean"].map(lambda x: f"{x:.3f}") + " +/- " + g["std"].map(lambda x: f"{x:.3f}")
        pivot = g.pivot_table(index="rep", columns="domain", values="cell", aggfunc="first")
        means = full.groupby("rep")[metric].mean()
        sort_order = means.sort_values(ascending=False).index.tolist()
        pivot = pivot.reindex(sort_order)
        pivot.to_csv(OUT / filename)

    save_summary("QWK_Oracle", "summary_qwk_oracle.csv")
    save_summary("QWK_CV", "summary_qwk_cv.csv")
    save_summary("C_index", "summary_c_index.csv")

    # ── Significance ──
    sig_qwk_oracle = pd.concat([pairwise_sig(full, d, "QWK_Oracle") for d in DOMAINS], ignore_index=True)
    sig_qwk_cv = pd.concat([pairwise_sig(full, d, "QWK_CV") for d in DOMAINS], ignore_index=True)
    sig_ci = pd.concat([pairwise_sig(full, d, "C_index") for d in DOMAINS], ignore_index=True)
    sig_qwk_oracle.to_csv(OUT / "significance_qwk_oracle.csv", index=False)
    sig_qwk_cv.to_csv(OUT / "significance_qwk_cv.csv", index=False)
    sig_ci.to_csv(OUT / "significance_c_index.csv", index=False)

    # ── Report ──
    build_report(full, sig_qwk_oracle, sig_qwk_cv, sig_ci)

    # ── Plots ──
    print("\nGenerating figures...")
    make_plots(full)

    print(f"\nDone. Outputs:")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name}")
