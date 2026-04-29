#!/usr/bin/env python3
"""
Build final paper-ready summary across all experimental configurations:
  1. Per-source ablation (orig vs +internal vs +external vs +both)
  2. 4-way comparison (H-Full vs Gemini vs TF-IDF vs Random) under paper-style
  3. Top-3 vs Paper-style (weighted_mean) comparison
  4. With/without σ-filter

Outputs:
  data_per_domain/prediction_results/FINAL_SUMMARY.csv  — single tidy file
  data_per_domain/prediction_results/FINAL_SUMMARY.md   — readable report
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

EXP = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/experiments")
PRED = EXP / "data_per_domain/prediction_results"
PS = PRED / "paper_style"
TOPK = PRED / "baselines"


def load_metrics(pred_csv: Path, label: str, mode: str):
    df = pd.read_csv(pred_csv)
    df["sig_combined"] = df["sigma_low"] + df["sigma_high"]
    rows = []
    for dom, sub in df.groupby("domain"):
        for sig in ["no_sigma", "with_sigma"]:
            ev = sub
            if sig == "with_sigma":
                if mode == "paper_style":
                    ev = sub[sub["sig_combined"] <= sub["sig_combined"].quantile(0.5)]
                else:  # topk uses marginal Q50
                    ev = sub[(sub["sigma_low"] <= sub["sigma_low"].quantile(0.5)) &
                             (sub["sigma_high"] <= sub["sigma_high"].quantile(0.5))]
            if len(ev) == 0: continue
            rows.append({
                "config": label, "mode": mode, "domain": dom, "sigma": sig,
                "n": len(ev),
                "avg_neighbors": float(ev["n_neighbors"].mean()),
                "MAE_low":  float(ev["err_low"].mean()),
                "MAE_high": float(ev["err_high"].mean()),
                "IoU": float(ev["iou"].mean()),
            })
    return rows


configs = [
    # Per-source ablation (paper-style, H-Full only)
    ("HF orig (85K, buggy graph)",   PS / "preds_HF_orig_thr60_corrected.csv",  "paper_style"),
    ("HF +internal_corrected",        PS / "preds_HF_orig_plus_internal_thr60_corrected.csv", "paper_style"),
    ("HF +external_cocite",           PS / "preds_HF_orig_plus_external_thr60_corrected.csv", "paper_style"),
    ("HF +both (combined 144K)",      PS / "preds_HF_orig_plus_both_thr60_corrected.csv",     "paper_style"),
    # 4-way on combined 144K (paper-style)
    ("HF (paper-style)",              PS / "preds_Hybrid-Full_thr60_corrected.csv",  "paper_style"),
    ("Gemini (paper-style)",          PS / "preds_Gemini_thr60_corrected.csv",       "paper_style"),
    ("TF-IDF (paper-style)",          PS / "preds_TF-IDF_thr60_corrected.csv",       "paper_style"),
    ("Random-K (paper-style)",        PS / "preds_Random-K_thr35_corrected.csv",     "paper_style"),
    # 4-way on 85K (top-3, my earlier mode)
    ("HF (top-3, 85K)",               TOPK / "preds_Hybrid-Full_topk.csv", "topk"),
    ("Gemini (top-3, 85K)",           TOPK / "preds_Gemini_topk.csv",      "topk"),
    ("TF-IDF (top-3, 85K)",           TOPK / "preds_TF-IDF_topk.csv",      "topk"),
    ("Random-K (top-3, 85K)",         TOPK / "preds_Random-K_topk.csv",    "topk"),
]

all_rows = []
for label, p, mode in configs:
    if not p.exists():
        print(f"  ⚠️  missing: {p}")
        continue
    all_rows.extend(load_metrics(p, label, mode))

df = pd.DataFrame(all_rows)
df.to_csv(PRED / "FINAL_SUMMARY.csv", index=False)
print(f"saved CSV: {PRED/'FINAL_SUMMARY.csv'}")

# Build readable markdown
lines = ["# Final summary — sentencing-range prediction\n"]

lines.append("## Per-source ablation (paper-style pipeline)\n")
lines.append("All Hybrid-Full, citation-linked filter (with corrected canonical normalization), THR=60, k≥3, weighted_mean aggregation, +σ-filter at Q50 of σ_combined.\n\n")
sub = df[df["config"].isin(["HF orig (85K, buggy graph)", "HF +internal_corrected",
                          "HF +external_cocite", "HF +both (combined 144K)"])]
piv = sub[sub["sigma"] == "with_sigma"].pivot_table(
    index="config", columns="domain",
    values=["n", "MAE_low", "MAE_high", "IoU"],
).round(3)
lines.append(piv.to_markdown())
lines.append("\n")

lines.append("\n## 4-way comparison: paper-style on combined 144K\n")
lines.append("Same pipeline as ablation. Each rep uses percentile-equivalent THR (so same fraction of pairs kept).\n\n")
sub = df[df["mode"] == "paper_style"]
sub = sub[sub["config"].str.contains("paper-style", na=False)]
for sig in ["no_sigma", "with_sigma"]:
    lines.append(f"### {sig}\n")
    piv = sub[sub["sigma"] == sig].pivot_table(
        index="config", columns="domain",
        values=["n", "MAE_low", "MAE_high", "IoU"],
    ).round(3)
    lines.append(piv.to_markdown())
    lines.append("\n")

lines.append("\n## 4-way comparison: top-3 mode on original 85K\n")
lines.append("Top-3 nearest neighbors per query, no sim threshold; agg=median (drugs) / softmax (weapon).\n\n")
sub = df[df["mode"] == "topk"]
for sig in ["no_sigma", "with_sigma"]:
    lines.append(f"### {sig}\n")
    piv = sub[sub["sigma"] == sig].pivot_table(
        index="config", columns="domain",
        values=["n", "MAE_low", "MAE_high", "IoU"],
    ).round(3)
    lines.append(piv.to_markdown())
    lines.append("\n")

with open(PRED / "FINAL_SUMMARY.md", "w") as f:
    f.write("\n".join(lines))
print(f"saved MD: {PRED/'FINAL_SUMMARY.md'}")

# Also print readable
print("\n" + "="*100)
print("PER-SOURCE ABLATION (paper-style, +σ filter)")
print("="*100)
sub = df[df["config"].isin(["HF orig (85K, buggy graph)", "HF +internal_corrected",
                          "HF +external_cocite", "HF +both (combined 144K)"])]
piv = sub[sub["sigma"] == "with_sigma"].pivot_table(
    index="config", columns="domain",
    values=["n", "MAE_low", "MAE_high", "IoU"],
).round(3)
print(piv.to_string())

print("\n\n" + "="*100)
print("4-WAY COMPARISON — paper-style (weighted_mean, THR=60-equiv, citation-linked, +σ)")
print("="*100)
sub = df[df["mode"] == "paper_style"]
sub = sub[sub["config"].str.contains("\\(paper-style\\)", na=False)]
piv = sub[sub["sigma"] == "with_sigma"].pivot_table(
    index="config", columns="domain",
    values=["n", "MAE_low", "MAE_high", "IoU"],
).round(3)
print(piv.to_string())

print("\n\n" + "="*100)
print("4-WAY COMPARISON — top-3 mode (median/softmax) on original 85K, +σ")
print("="*100)
sub = df[df["mode"] == "topk"]
piv = sub[sub["sigma"] == "with_sigma"].pivot_table(
    index="config", columns="domain",
    values=["n", "MAE_low", "MAE_high", "IoU"],
).round(3)
print(piv.to_string())
