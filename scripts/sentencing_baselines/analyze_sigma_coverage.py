#!/usr/bin/env python3
"""
Analyze the extended 5-fold predictions to address advisor feedback:

  (1) Baseline at 100% coverage:
        • Error distributions per method (mean, median, P75, P95)
        • Per-source ablation: isolated + leave-one-out → which source contributes what?

  (2) σ-Confidence Risk-Coverage with UNIFIED source-case subset:
        For each coverage level C ∈ {10,15,...,100}%:
          • Rank cases by a single σ criterion
          • Pick top-C% most confident
          • Evaluate ALL 4 methods on that SAME subset
        Three σ-ranking variants:
          A. all-sources σ_combined  (σ_lo+σ_hi of all method's K=10 neighbors)
          B. average σ across the 4 methods
          C. full-pool σ (across ALL LLM-similars, "natural variance" baseline)

  (3) Diagnose all-sources dominance:
        • Mean |Δyear| between target and neighbors per method (leakage proxy)
        • Aggregation-only baseline: median of (sup, sup_llm, cit, all) predictions —
          does naive averaging match all-sources?

Input : results/2_sentencing_range/predictions/cv_5fold_extended.csv
Output: results/2_sentencing_range/predictions/sigma_coverage_analysis/
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

EXP = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/experiments")
IN  = EXP / "results/2_sentencing_range/predictions/cv_5fold_extended.csv"
OUT_DIR = EXP / "results/2_sentencing_range/predictions/sigma_coverage_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CORE_METHODS = ["sup", "sup_llm", "cit", "all"]
ABLATION_METHODS = ["simcse_only", "sup_only", "fold5_only",
                    "all_no_cit", "all_no_simcse", "all_no_sup", "all_no_fold5"]
COVERAGE_LEVELS = [100, 90, 80, 70, 60, 50, 40, 30, 20, 15, 10]


def mae_lo(df, m):
    s = df[f"{m}_lo_err"].dropna()
    return float(s.mean()) if len(s) else None, len(s)


def mae_hi(df, m):
    s = df[f"{m}_hi_err"].dropna()
    return float(s.mean()) if len(s) else None, len(s)


def mae_avg(df, m):
    lo = df[f"{m}_lo_err"].dropna(); hi = df[f"{m}_hi_err"].dropna()
    both = lo.tolist() + hi.tolist()
    return float(np.mean(both)) if both else None


# ============================================================
# (1) BASELINE AT 100% COVERAGE
# ============================================================
def baseline_at_100(df: pd.DataFrame) -> pd.DataFrame:
    """Per-method error stats at full coverage (no σ filter)."""
    rows = []
    for dom in ["drugs", "weapon"]:
        sub = df[df.domain == dom]
        for m in CORE_METHODS + ABLATION_METHODS:
            lo = sub[f"{m}_lo_err"].dropna().to_numpy()
            hi = sub[f"{m}_hi_err"].dropna().to_numpy()
            both = np.concatenate([lo, hi]) if len(lo) else np.array([])
            if len(both) == 0:
                continue
            rows.append({
                "domain": dom, "method": m, "n_cases": len(lo),
                "MAE_lo_mean": round(float(lo.mean()), 2),
                "MAE_hi_mean": round(float(hi.mean()), 2),
                "MAE_avg_mean": round(float(both.mean()), 2),
                "MAE_lo_median": round(float(np.median(lo)), 2),
                "MAE_hi_median": round(float(np.median(hi)), 2),
                "MAE_lo_P75": round(float(np.percentile(lo, 75)), 2),
                "MAE_hi_P75": round(float(np.percentile(hi, 75)), 2),
                "MAE_lo_P95": round(float(np.percentile(lo, 95)), 2),
                "MAE_hi_P95": round(float(np.percentile(hi, 95)), 2),
            })
    return pd.DataFrame(rows)


def ablation_lift(df: pd.DataFrame) -> pd.DataFrame:
    """How does each source contribute to all-sources? Compare all_no_X vs all."""
    rows = []
    for dom in ["drugs", "weapon"]:
        sub = df[df.domain == dom]
        all_mae = mae_avg(sub, "all")
        for src in ["cit", "simcse", "sup", "fold5"]:
            without_mae = mae_avg(sub, f"all_no_{src}")
            isolated_mae = mae_avg(sub, "cit") if src == "cit" else mae_avg(sub, f"{src}_only")
            rows.append({
                "domain": dom, "source": src,
                "isolated_MAE": round(isolated_mae, 2) if isolated_mae else None,
                "all_no_X_MAE": round(without_mae, 2) if without_mae else None,
                "all_MAE": round(all_mae, 2),
                "lift_when_removed": round(without_mae - all_mae, 2) if (without_mae and all_mae) else None,
                "interpretation": (
                    "HARMFUL (removing helps)" if (without_mae and all_mae and without_mae < all_mae - 0.05)
                    else "USEFUL (removing hurts)" if (without_mae and all_mae and without_mae > all_mae + 0.05)
                    else "NEUTRAL")
            })
    return pd.DataFrame(rows)


# ============================================================
# (2) σ-COVERAGE WITH UNIFIED SOURCE-CASE SUBSET
# ============================================================
def sigma_coverage_unified(df: pd.DataFrame, sigma_col: str, label: str) -> pd.DataFrame:
    """For each coverage level, pick top-C% by `sigma_col` ascending,
       then compute MAE_lo / MAE_hi for every method on that SAME subset.
       Returns long-format DataFrame."""
    rows = []
    for dom in ["drugs", "weapon"]:
        sub = df[df.domain == dom].copy()
        sub = sub.dropna(subset=[sigma_col])
        sub = sub.sort_values(sigma_col, ascending=True).reset_index(drop=True)
        n_total = len(sub)
        for cov in COVERAGE_LEVELS:
            n_keep = int(round(n_total * cov / 100))
            kept = sub.iloc[:n_keep]
            for m in CORE_METHODS:
                lo = kept[f"{m}_lo_err"].dropna()
                hi = kept[f"{m}_hi_err"].dropna()
                # mark "—" if not every kept case has a prediction (e.g. cit may have gaps)
                full_coverage = (len(lo) == len(kept)) and (len(hi) == len(kept))
                rows.append({
                    "ranking": label, "domain": dom, "coverage_pct": cov,
                    "n_kept": n_keep, "method": m,
                    "n_valid": int(min(len(lo), len(hi))),
                    "MAE_lo": round(float(lo.mean()), 2) if len(lo) else None,
                    "MAE_hi": round(float(hi.mean()), 2) if len(hi) else None,
                    "MAE_avg": round(float(np.concatenate([lo, hi]).mean()), 2) if len(lo) else None,
                    "complete_coverage": full_coverage,
                })
    return pd.DataFrame(rows)


def pivot_for_screenshot(rc_long: pd.DataFrame, target: str = "lo") -> dict:
    """Pivot to the screenshot format: rows=coverage, cols=method, per (domain,ranking)."""
    out = {}
    val_col = f"MAE_{target}"
    for (ranking, dom), g in rc_long.groupby(["ranking", "domain"]):
        piv = g.pivot_table(index="coverage_pct", columns="method", values=val_col)
        piv = piv.reindex(COVERAGE_LEVELS)
        ordered = [c for c in CORE_METHODS if c in piv.columns]
        piv = piv[ordered]
        # also build a parallel mask of "complete" coverage
        mask = g.pivot_table(index="coverage_pct", columns="method", values="complete_coverage")
        mask = mask.reindex(COVERAGE_LEVELS)[ordered]
        # blank out values where coverage incomplete (screenshot's "—")
        piv_display = piv.where(mask.fillna(False).astype(bool))
        out[(ranking, dom, target)] = piv_display
    return out


# ============================================================
# (3) DIAGNOSIS: leakage + aggregation
# ============================================================
def year_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Mean |Δyear| between target case and chosen neighbors, per method.
       Lower Δyear → neighbors come from more similar time period → possible leakage."""
    # Need year_of map for all canonical IDs — build from inventory.
    inv = pd.read_csv(EXP / "data_per_domain/master_inventory.csv",
                      usecols=["canonical_id", "year"]).drop_duplicates("canonical_id")
    year_of = dict(zip(inv.canonical_id, inv.year))

    rows = []
    for dom in ["drugs", "weapon"]:
        sub = df[df.domain == dom]
        for m in CORE_METHODS + ["simcse_only", "sup_only", "fold5_only"]:
            col = f"{m}_neighbors"
            if col not in sub.columns: continue
            deltas = []
            for r in sub.itertuples(index=False):
                y_q = year_of.get(r.qid)
                nbrs = getattr(r, col)
                if pd.isna(nbrs) or y_q is None or pd.isna(y_q):
                    continue
                ys = [year_of.get(n) for n in nbrs.split(";") if year_of.get(n) is not None]
                if not ys: continue
                deltas.append(float(np.mean([abs(y - y_q) for y in ys])))
            if deltas:
                rows.append({
                    "domain": dom, "method": m, "n_cases": len(deltas),
                    "mean_abs_year_diff": round(float(np.mean(deltas)), 2),
                    "median_abs_year_diff": round(float(np.median(deltas)), 2),
                    "P25_abs_year_diff": round(float(np.percentile(deltas, 25)), 2),
                })
    return pd.DataFrame(rows)


def aggregation_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Does naive median-of-method-predictions match all-sources?
       If yes, all-sources is essentially an ensemble effect."""
    rows = []
    for dom in ["drugs", "weapon"]:
        sub = df[df.domain == dom].copy()
        # collect lo/hi predictions across the 4 core methods
        pred_lo_cols = [f"{m}_pred_lo" for m in CORE_METHODS if f"{m}_pred_lo" in sub.columns]
        pred_hi_cols = [f"{m}_pred_hi" for m in CORE_METHODS if f"{m}_pred_hi" in sub.columns]
        # naive median ensemble — ignore NaNs (per-row median of available methods)
        sub["ens_lo"] = sub[pred_lo_cols].median(axis=1, skipna=True)
        sub["ens_hi"] = sub[pred_hi_cols].median(axis=1, skipna=True)
        sub["ens_lo_err"] = (sub["ens_lo"] - sub["true_lo"]).abs()
        sub["ens_hi_err"] = (sub["ens_hi"] - sub["true_hi"]).abs()

        all_mae = mae_avg(sub, "all")
        ens_lo = sub["ens_lo_err"].dropna()
        ens_hi = sub["ens_hi_err"].dropna()
        ens_mae = float(np.concatenate([ens_lo, ens_hi]).mean()) if len(ens_lo) else None
        rows.append({
            "domain": dom, "all_MAE_avg": round(all_mae, 2),
            "ensemble_median_MAE_avg": round(ens_mae, 2) if ens_mae else None,
            "delta": round(ens_mae - all_mae, 2) if (ens_mae and all_mae) else None,
            "interpretation": (
                "all-sources matches ensemble (mostly aggregation effect)"
                if (ens_mae and all_mae and abs(ens_mae - all_mae) < 0.2)
                else "all-sources does more than ensemble (true LLM contribution)"
                if (ens_mae and all_mae and ens_mae > all_mae + 0.2) else "ambiguous"
            ),
        })
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"=== Loading {IN.name} ===")
    df = pd.read_csv(IN)
    # Build the 3 σ-ranking columns
    df["sig_combined_all"] = df["all_sig_lo"] + df["all_sig_hi"]
    sig_cols = [f"{m}_sig_lo" for m in CORE_METHODS] + [f"{m}_sig_hi" for m in CORE_METHODS]
    df["sig_combined_avg4"] = df[sig_cols].mean(axis=1, skipna=True)
    df["sig_combined_fullpool"] = df["full_pool_sig_lo"] + df["full_pool_sig_hi"]
    print(f"  rows: {len(df):,}   cols: {df.shape[1]}")

    # (1) Baseline
    print("\n" + "=" * 90)
    print("(1) BASELINE @ 100% COVERAGE — Error distributions per method")
    print("=" * 90)
    bl = baseline_at_100(df)
    bl.to_csv(OUT_DIR / "01_baseline_100pct.csv", index=False)
    for dom in ["drugs", "weapon"]:
        print(f"\n  {dom.upper()}")
        print(bl[bl.domain == dom][["method","n_cases","MAE_lo_mean","MAE_hi_mean","MAE_avg_mean",
                                     "MAE_lo_median","MAE_lo_P75","MAE_lo_P95"]].to_string(index=False))

    print("\n" + "=" * 90)
    print("(1b) PER-SOURCE ABLATION — what does each source contribute to all-sources?")
    print("=" * 90)
    ab = ablation_lift(df)
    ab.to_csv(OUT_DIR / "02_ablation.csv", index=False)
    print(ab.to_string(index=False))

    # (2) σ-coverage, 3 variants
    print("\n" + "=" * 90)
    print("(2) σ-COVERAGE with UNIFIED source-case subset — 3 ranking variants")
    print("=" * 90)
    all_rc = []
    for label, col in [("A_all_sources_sigma", "sig_combined_all"),
                       ("B_avg4_sigma",        "sig_combined_avg4"),
                       ("C_full_pool_sigma",   "sig_combined_fullpool")]:
        rc = sigma_coverage_unified(df, col, label)
        rc.to_csv(OUT_DIR / f"03_sigma_coverage_{label}.csv", index=False)
        all_rc.append(rc)
    rc_long = pd.concat(all_rc, ignore_index=True)
    rc_long.to_csv(OUT_DIR / "03_sigma_coverage_all_variants.csv", index=False)

    # Print compact pivots for MAE_lo
    pivs = pivot_for_screenshot(rc_long, target="lo")
    for (ranking, dom, _), piv in pivs.items():
        if "drugs" in dom or "weapon" in dom:
            print(f"\n  {ranking}  ·  {dom}  ·  MAE_lo")
            print(piv.round(2).to_string())

    print("\n  Note: blanks = method cannot reach this coverage on this subset (analogue of '—')")

    # (3) Diagnosis
    print("\n" + "=" * 90)
    print("(3a) YEAR-LEAKAGE PROXY — mean |Δyear| between target and chosen neighbors")
    print("=" * 90)
    yl = year_leakage(df)
    yl.to_csv(OUT_DIR / "04_year_leakage.csv", index=False)
    for dom in ["drugs", "weapon"]:
        print(f"\n  {dom.upper()}")
        print(yl[yl.domain == dom].to_string(index=False))

    print("\n" + "=" * 90)
    print("(3b) AGGREGATION BASELINE — naive median-ensemble vs all-sources")
    print("=" * 90)
    ag = aggregation_baseline(df)
    ag.to_csv(OUT_DIR / "05_aggregation_baseline.csv", index=False)
    print(ag.to_string(index=False))

    print(f"\n💾 All outputs → {OUT_DIR}/")


if __name__ == "__main__":
    main()
