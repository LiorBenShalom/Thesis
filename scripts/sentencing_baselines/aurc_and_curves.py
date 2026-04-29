#!/usr/bin/env python3
"""
AURC (Area Under Risk-Coverage curve) + Risk-Coverage plots for the 4 reps.

El-Yaniv & Wiener (NeurIPS 2010) — selective prediction foundational metric.
Lower AURC = the model knows better which queries to be confident about.

Inputs : top-K=10 predictions per rep
         (results/2_sentencing_range/predictions/topk10_clean/).
Outputs: aurc_topk10_clean.csv, risk_coverage_curves.png.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EXP = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/experiments")
PRED_DIR = EXP / "results/2_sentencing_range/predictions/topk10_clean"
OUT_DIR = EXP / "results/2_sentencing_range/predictions"
REPS = ["Hybrid-Full", "Gemini", "TF-IDF", "Random-K"]
COLORS = {"Hybrid-Full": "#1f77b4", "Gemini": "#ff7f0e",
          "TF-IDF": "#2ca02c", "Random-K": "#888888"}


def main():
    preds = {r: pd.read_csv(PRED_DIR / f"preds_{r}_topk.csv") for r in REPS}
    for r in preds:
        preds[r]["sig_combined"] = preds[r].sigma_low + preds[r].sigma_high

    # ─── AURC table ───
    rows = []
    for r in REPS:
        for dom in ["drugs", "weapon"]:
            sub = preds[r][preds[r].domain == dom]
            for tgt in ["low", "high"]:
                err = sub[f"err_{tgt}"].to_numpy()
                conf = -sub.sig_combined.to_numpy()       # high σ ⇒ low conf
                order = np.argsort(-conf)                 # most-confident first
                cum = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
                aurc = float(np.mean(cum))
                rows.append({"rep": r, "domain": dom, "target": tgt,
                             "n": len(sub), "AURC": aurc,
                             "MAE_full_coverage": float(err.mean())})
    df_a = pd.DataFrame(rows)
    df_a.to_csv(OUT_DIR / "aurc_topk10_clean.csv", index=False)
    print("=" * 70)
    print("AURC (lower is better) — 5-fold not needed; this is on full data")
    print("=" * 70)
    piv = df_a.pivot_table(index="rep", columns=["domain", "target"], values="AURC").round(3)
    print(piv.to_string())

    # ─── Risk-Coverage plot ───
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for j_dom, dom in enumerate(["drugs", "weapon"]):
        for i_tgt, tgt in enumerate(["low", "high"]):
            ax = axes[j_dom, i_tgt]
            for r in REPS:
                sub = preds[r][preds[r].domain == dom]
                err = sub[f"err_{tgt}"].to_numpy()
                conf = -sub.sig_combined.to_numpy()
                order = np.argsort(-conf)
                cum = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
                covs = np.arange(1, len(err) + 1) / len(err)
                ax.plot(covs, cum, label=r, color=COLORS[r], linewidth=2)
            ax.set_xlabel("Coverage (fraction of queries)")
            ax.set_ylabel(f"MAE_{tgt} (months)")
            ax.set_title(f"{dom} — {tgt}")
            ax.grid(alpha=0.3)
            if j_dom == 0 and i_tgt == 0:
                ax.legend(loc="upper left", fontsize=10)
    plt.suptitle("Risk-Coverage curves (σ as confidence)", y=1.02)
    plt.tight_layout()
    fig_path = OUT_DIR / "risk_coverage_curves.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"\n→ {OUT_DIR/'aurc_topk10_clean.csv'}")
    print(f"→ {fig_path}")


if __name__ == "__main__":
    main()
