#!/usr/bin/env python3
"""Side-by-side comparison: QWK t1 (lenient — scale 1↔2) vs t2 (strict — scale 2↔3) thresholds."""
from pathlib import Path
import pandas as pd, numpy as np

EXP = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/experiments")
T1 = EXP / "data_per_domain/prediction_results/qwk_t1_thresholds"
T2 = EXP / "data_per_domain/prediction_results/qwk_thresholds"
OUT = EXP / "data_per_domain/prediction_results"
REPS = ["Hybrid-Full", "Gemini", "TF-IDF", "Random-K"]


def load_metrics(pred_csv, label, thr_label):
    df = pd.read_csv(pred_csv)
    df["sig_combined"] = df.sigma_low + df.sigma_high
    rows = []
    for dom, sub in df.groupby("domain"):
        for sig in ["no_sigma", "with_sigma"]:
            ev = sub if sig == "no_sigma" else sub[sub.sig_combined <= sub.sig_combined.quantile(0.5)]
            if len(ev) == 0: continue
            rows.append({
                "rep": label, "thr": thr_label, "domain": dom, "sigma": sig,
                "n": len(ev), "avg_neighbors": float(ev.n_neighbors.mean()),
                "MAE_low": float(ev.err_low.mean()),
                "MAE_high": float(ev.err_high.mean()),
                "MedAE_low": float(ev.err_low.median()),
                "MedAE_high": float(ev.err_high.median()),
                "IoU": float(ev.iou.mean()),
            })
    return rows


all_rows = []
for r in REPS:
    f1 = T1 / f"preds_{r}_thr60_allpairs_corrected.csv"
    f2 = T2 / f"preds_{r}_thr60_allpairs_corrected.csv"
    if f1.exists(): all_rows += load_metrics(f1, r, "t1_lenient")
    if f2.exists(): all_rows += load_metrics(f2, r, "t2_strict")

df = pd.DataFrame(all_rows)
df.to_csv(OUT / "comparison_t1_vs_t2.csv", index=False)
print(f"Saved → {OUT/'comparison_t1_vs_t2.csv'}\n")

print("="*100)
print("COMPARISON: t1 (lenient, scale 1↔2) vs t2 (strict, scale 2↔3) — no_sigma")
print("="*100)
sub = df[df.sigma == "no_sigma"]
piv = sub.pivot_table(
    index="rep", columns=["thr", "domain"],
    values=["n", "MAE_low", "MAE_high", "IoU"],
).round(3)
print(piv.to_string())

print("\n" + "="*100)
print("COMPARISON: t1 vs t2 — with_sigma")
print("="*100)
sub = df[df.sigma == "with_sigma"]
piv = sub.pivot_table(
    index="rep", columns=["thr", "domain"],
    values=["n", "MAE_low", "MAE_high", "IoU"],
).round(3)
print(piv.to_string())

print("\n\n=== ΔCoverage and ΔMAE: t1 → t2 (loss when tightening) ===\n")
for sig in ["no_sigma", "with_sigma"]:
    print(f"\n--- {sig} ---")
    print(f"{'Rep':<14} {'Domain':<7} {'n_t1':>5} {'n_t2':>5} {'Δn':>6}  {'MAE_lo_t1':>10} {'MAE_lo_t2':>10} {'Δ_lo':>6}  {'MAE_hi_t1':>10} {'MAE_hi_t2':>10} {'Δ_hi':>6}")
    for r in REPS:
        for d in ["drugs", "weapon"]:
            t1 = df[(df.rep == r) & (df.thr == "t1_lenient") & (df.sigma == sig) & (df.domain == d)]
            t2 = df[(df.rep == r) & (df.thr == "t2_strict") & (df.sigma == sig) & (df.domain == d)]
            if t1.empty or t2.empty: continue
            t1, t2 = t1.iloc[0], t2.iloc[0]
            print(f"  {r:<14} {d:<7} {int(t1.n):>5} {int(t2.n):>5} {int(t2.n - t1.n):>+6}  "
                  f"{t1.MAE_low:>10.2f} {t2.MAE_low:>10.2f} {t2.MAE_low - t1.MAE_low:>+6.2f}  "
                  f"{t1.MAE_high:>10.2f} {t2.MAE_high:>10.2f} {t2.MAE_high - t1.MAE_high:>+6.2f}")
