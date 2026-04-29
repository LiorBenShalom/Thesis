#!/usr/bin/env python3
"""Final comparison using QWK-Oracle t2 thresholds. Includes:
   - all-queries comparison (each rep on its own valid set)
   - shared-queries Wilcoxon (paired) — H-Full vs each baseline on the SAME query set."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

EXP = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/experiments")
PRED_DIR = EXP / "data_per_domain/prediction_results/f1_thresholds"
OUT_DIR = EXP / "data_per_domain/prediction_results"
REPS = ["Hybrid-Full","Gemini","TF-IDF","Random-K"]
ANCHOR = "Hybrid-Full"


def bh_fdr(pvals):
    n = len(pvals); order = np.argsort(pvals)
    ranked = np.array(pvals)[order]
    adj = ranked * n / (np.arange(n) + 1)
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    out = np.empty(n); out[order] = np.minimum(adj, 1.0)
    return out.tolist()


def main():
    preds = {}
    for r in REPS:
        f = PRED_DIR / f"preds_{r}_thr60_allpairs_corrected.csv"
        df = pd.read_csv(f); df["sig_combined"] = df.sigma_low + df.sigma_high
        preds[r] = df

    # 1) Self-evaluation (each rep on its own valid query set)
    rows = []
    for r, df in preds.items():
        for dom, sub in df.groupby("domain"):
            for sig in ["no_sigma","with_sigma"]:
                ev = sub if sig == "no_sigma" else sub[sub.sig_combined <= sub.sig_combined.quantile(0.5)]
                if len(ev) == 0: continue
                rows.append({"rep": r, "domain": dom, "sigma": sig, "n": len(ev),
                             "avg_neighbors": float(ev.n_neighbors.mean()),
                             "MAE_low": float(ev.err_low.mean()),
                             "MAE_high": float(ev.err_high.mean()),
                             "MedAE_low": float(ev.err_low.median()),
                             "MedAE_high": float(ev.err_high.median()),
                             "IoU": float(ev.iou.mean())})
    comp_self = pd.DataFrame(rows)
    comp_self.to_csv(OUT_DIR / "comparison_f1_self.csv", index=False)
    print("=== SELF (each rep on its own valid set, QWK t2 + σ) ===\n")
    pivot = comp_self.pivot_table(index=["rep","sigma"], columns="domain",
                                   values=["n","MAE_low","MAE_high","IoU"]).round(3)
    print(pivot.to_string())

    # 2) Shared-query comparison: H-Full vs each baseline on the INTERSECTION of valid queries
    print("\n\n=== SHARED-QUERY pairwise comparison (Wilcoxon, BH-FDR within domain×target×σ) ===\n")
    all_w = []
    for dom in ["drugs","weapon"]:
        for tgt in ["low","high"]:
            for sig in [False, True]:
                a = preds[ANCHOR]
                a = a[a.domain == dom]
                if sig: a = a[a.sig_combined <= a.sig_combined.quantile(0.5)]
                a_map = dict(zip(a.verdict, a[f"err_{tgt}"]))
                rows_w, pvs = [], []
                for r in REPS:
                    if r == ANCHOR: continue
                    b = preds[r]; b = b[b.domain == dom]
                    if sig: b = b[b.sig_combined <= b.sig_combined.quantile(0.5)]
                    b_map = dict(zip(b.verdict, b[f"err_{tgt}"]))
                    shared = sorted(set(a_map) & set(b_map))
                    if len(shared) < 10:
                        rows_w.append({"baseline": r, "n_shared": len(shared)}); pvs.append(np.nan); continue
                    ae = np.array([a_map[v] for v in shared])
                    be = np.array([b_map[v] for v in shared])
                    try:
                        _, p = wilcoxon(be - ae, zero_method="wilcox", alternative="two-sided")
                    except ValueError:
                        p = np.nan
                    rows_w.append({"baseline": r, "n_shared": len(shared),
                                   "mae_anchor": float(ae.mean()),
                                   "mae_baseline": float(be.mean()),
                                   "median_diff": float(np.median(be - ae)),
                                   "p_raw": p})
                    pvs.append(p)
                df_w = pd.DataFrame(rows_w)
                valid = [i for i, p in enumerate(pvs) if not (p is None or np.isnan(p))]
                p_bh = [np.nan]*len(pvs)
                if valid:
                    adj = bh_fdr([pvs[i] for i in valid])
                    for i, ap in zip(valid, adj): p_bh[i] = ap
                df_w["p_bh"] = p_bh
                df_w["winner"] = np.where(df_w["p_bh"] < 0.05,
                                           np.where(df_w.mae_anchor < df_w.mae_baseline, ANCHOR, df_w.baseline),
                                           "tie")
                df_w["domain"] = dom; df_w["target"] = tgt
                df_w["sigma"] = "with_sigma" if sig else "no_sigma"
                all_w.append(df_w)
                print(f"--- {dom}/{tgt}/{'with_sigma' if sig else 'no_sigma'} ---")
                print(df_w[["baseline","n_shared","mae_anchor","mae_baseline","median_diff","p_bh","winner"]].to_string(index=False))
                print()
    pd.concat(all_w, ignore_index=True).to_csv(OUT_DIR / "wilcoxon_f1.csv", index=False)
    print(f"\nAll Wilcoxon → {OUT_DIR/'wilcoxon_f1.csv'}")


if __name__ == "__main__":
    main()
