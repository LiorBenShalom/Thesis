#!/usr/bin/env python3
"""
Compare score_only vs with_explanation across all models on all metrics.

Outputs:
  experiments/explainability_annotation/score_only_comparison/
    metrics_table.csv        — wide table: model × prompt × domain × metric
    metrics_summary.md       — paper-ready Hebrew + English markdown report
    significance.csv         — Wilcoxon test on per-pair scores per model
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, average_precision_score, f1_score, precision_recall_curve
from scipy.stats import spearmanr, wilcoxon

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!")
HF = ROOT / "new_try/experiments/explainability_annotation/hybrid_full"
OUT = ROOT / "new_try/experiments/explainability_annotation/score_only_comparison"
OUT.mkdir(exist_ok=True)

MODELS = ["gpt4", "claude_sonnet_4_6", "gemma4_31b_or"]

def score_to_scale(s):
    if pd.isna(s): return None
    if s < 25: return 0
    if s < 50: return 1
    if s < 75: return 2
    return 3

def c_index(y_true, y_score):
    n_c = n_d = 0
    arr = list(zip(y_true, y_score))
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            yi, si = arr[i]; yj, sj = arr[j]
            if yi == yj: continue
            if (yi > yj and si > sj) or (yi < yj and si < sj): n_c += 1
            elif (yi > yj and si < sj) or (yi < yj and si > sj): n_d += 1
    return n_c / (n_c + n_d) if (n_c+n_d) else 0.5

def f1_oracle(y_true: np.ndarray, score: np.ndarray) -> float:
    """Best F1 over all thresholds on the same data (upper bound)."""
    p, r, t = precision_recall_curve(y_true, score)
    f1 = 2*p*r / (p+r+1e-12)
    return float(np.nanmax(f1))


def metrics_for(gt, score):
    score = np.asarray(score, dtype=float)
    gt = np.asarray(gt, dtype=int)
    mask = ~np.isnan(score)
    gt = gt[mask]; score = score[mask]
    scale = np.array([score_to_scale(v) for v in score])
    rho, _ = spearmanr(gt, score)
    bin_strict = (gt >= 3).astype(int)
    bin_lenient = (gt >= 2).astype(int)
    return {
        "n": int(len(gt)),
        "QWK_scale":    float(cohen_kappa_score(gt, scale, weights="quadratic", labels=[0,1,2,3])),
        "C_index":      float(c_index(gt, score)),
        "Spearman":     float(rho) if not np.isnan(rho) else np.nan,
        "AP_strict":    float(average_precision_score(bin_strict, score)) if bin_strict.any() else np.nan,
        "AP_lenient":   float(average_precision_score(bin_lenient, score)) if bin_lenient.any() else np.nan,
        "F1_strict_oracle":  f1_oracle(bin_strict, score) if bin_strict.any() else np.nan,
        "F1_lenient_oracle": f1_oracle(bin_lenient, score) if bin_lenient.any() else np.nan,
        # F1 with default threshold = score>=50 (matches v6 SIMILARITY_SCORE>=50 = relevant)
        "F1_strict_t50":     float(f1_score(bin_strict, (score>=50).astype(int), zero_division=0)) if bin_strict.any() else np.nan,
        "F1_lenient_t50":    float(f1_score(bin_lenient, (score>=50).astype(int), zero_division=0)) if bin_lenient.any() else np.nan,
    }


def load_with_expl(domain, model):
    df = pd.read_csv(HF / f"explainability_{domain}_{model}.csv")
    return df[["verdict_1","verdict_2","GT","model_score"]].rename(columns={"model_score":"score"})

def load_score_only(domain, model):
    df = pd.read_csv(HF / f"score_only_{domain}_{model}.csv")
    return df[["verdict_1","verdict_2","GT","model_score"]].rename(columns={"model_score":"score"})


def main():
    rows = []
    sig_rows = []
    for model in MODELS:
        for domain in ["drugs", "weapon"]:
            try:
                e = load_with_expl(domain, model)
                s = load_score_only(domain, model)
            except FileNotFoundError as exc:
                print(f"  ⚠️  missing file for {model}/{domain}: {exc}")
                continue
            # Align by pair
            e = e.set_index(["verdict_1","verdict_2"])
            s = s.set_index(["verdict_1","verdict_2"])
            common = e.index.intersection(s.index)
            e = e.loc[common]; s = s.loc[common]
            gt = e["GT"].astype(int).values
            es = pd.to_numeric(e["score"], errors="coerce")
            ss = pd.to_numeric(s["score"], errors="coerce")
            mask = es.notna() & ss.notna()
            gt = gt[mask.values]; es = es[mask].values; ss = ss[mask].values

            m_e = metrics_for(gt, es); m_e["model"] = model; m_e["domain"] = domain; m_e["prompt"] = "with_expl"
            m_s = metrics_for(gt, ss); m_s["model"] = model; m_s["domain"] = domain; m_s["prompt"] = "score_only"
            rows += [m_e, m_s]

            # Wilcoxon: paired comparison of per-pair scores (within-model sanity)
            try:
                stat, p = wilcoxon(es, ss, zero_method="zsplit")
                sig_rows.append({
                    "model": model, "domain": domain,
                    "n": int(len(es)),
                    "wilcoxon_W": float(stat), "wilcoxon_p": float(p),
                    "median_diff_score_only_minus_with_expl": float(np.median(ss - es)),
                    "mean_diff": float(np.mean(ss - es)),
                })
            except Exception as exc:
                sig_rows.append({"model": model, "domain": domain, "n": int(len(es)),
                                 "wilcoxon_W": None, "wilcoxon_p": None,
                                 "median_diff_score_only_minus_with_expl": float(np.median(ss - es)),
                                 "mean_diff": float(np.mean(ss - es))})

    df = pd.DataFrame(rows)[["model","domain","prompt","n","QWK_scale","C_index","Spearman","AP_strict","AP_lenient"]]
    df.to_csv(OUT / "metrics_table.csv", index=False)
    print(f"\n✅ saved {OUT / 'metrics_table.csv'}")

    sig = pd.DataFrame(sig_rows)
    sig.to_csv(OUT / "significance.csv", index=False)
    print(f"✅ saved {OUT / 'significance.csv'}")

    # Pretty print + comparison report
    md = ["# Score-Only vs With-Explanation — Multi-Model Comparison\n",
          f"GT pairs: 100 drugs + 141 weapon. Same V6 prompt; only line `1. ניתוח קצר ...` removed.\n"]
    for domain in ["drugs", "weapon"]:
        md.append(f"\n## {domain.upper()}\n")
        sub = df[df["domain"]==domain].copy()
        wide = sub.pivot_table(index="model", columns="prompt", values=["QWK_scale","C_index","AP_strict","AP_lenient","Spearman"])
        # Build per-metric pretty rows
        for metric in ["QWK_scale","C_index","AP_strict","AP_lenient","Spearman"]:
            md.append(f"\n### {metric}\n")
            md.append(f"| model | with_expl | score_only | Δ |")
            md.append(f"|---|---|---|---|")
            for model in MODELS:
                e_v = sub[(sub["model"]==model)&(sub["prompt"]=="with_expl")][metric]
                s_v = sub[(sub["model"]==model)&(sub["prompt"]=="score_only")][metric]
                if len(e_v) and len(s_v):
                    e_val = e_v.values[0]; s_val = s_v.values[0]
                    sign = "🟢" if s_val >= e_val else "🔴"
                    md.append(f"| {model} | {e_val:.3f} | {s_val:.3f} | {s_val-e_val:+.3f} {sign} |")
        # Significance
        md.append(f"\n### Wilcoxon (per-pair score paired test)\n")
        md.append(f"| model | n | W | p | median Δ | mean Δ |")
        md.append(f"|---|---|---|---|---|---|")
        for model in MODELS:
            r = sig[(sig["model"]==model)&(sig["domain"]==domain)]
            if len(r):
                rr = r.iloc[0]
                p = rr["wilcoxon_p"]
                p_str = f"{p:.4f}" + (" ⚠️ p<0.05" if p is not None and p<0.05 else "")
                md.append(f"| {model} | {rr['n']} | {rr['wilcoxon_W']:.0f} | {p_str} | {rr['median_diff_score_only_minus_with_expl']:+.1f} | {rr['mean_diff']:+.2f} |")

    (OUT / "metrics_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"✅ saved {OUT / 'metrics_summary.md'}")

    # Pretty stdout
    print("\n" + "="*90)
    print("METRICS TABLE (wide)")
    print("="*90)
    pivot = df.pivot_table(
        index=["model","domain"], columns="prompt",
        values=["QWK_scale","C_index","AP_strict","Spearman"]
    ).round(3)
    print(pivot.to_string())


if __name__ == "__main__":
    main()
