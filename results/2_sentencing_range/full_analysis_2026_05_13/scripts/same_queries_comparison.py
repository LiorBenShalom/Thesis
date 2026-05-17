"""
Same-queries comparison: all methods evaluated on the SAME queries that
citation+LLM could predict, per min_k cohort (1/3/10) on the 4,432 corpus.

Proves citation+LLM doesn't win merely by getting "easy" queries — when every
method is restricted to the identical query set, the ranking holds.

Inputs : data/rigor_per_query_errors.csv (canonical per-query errs, 4,432)
         data/citation_llm_query_cohorts.csv
Outputs: data/same_queries_comparison.csv         (per min_k×domain×method)
         data/same_queries_paired_cit_vs_sup.csv  (paired cit_llm vs sup_llm)
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

D = Path(__file__).resolve().parent.parent / "data"
err = pd.read_csv(D / "rigor_per_query_errors.csv")
err["query"] = err["query"].astype(str)
coh = pd.read_csv(D / "citation_llm_query_cohorts.csv")
coh["query"] = coh["query"].astype(str)

METHODS = ["global_median", "offense_matched_random", "tfidf_ridge", "bm25",
           "random_llm", "sup_only", "citation_llm", "sup_llm", "llm_best"]
MIN_KS  = [1, 3, 10]


def boot_ci(a, B=2000, seed=42):
    a = np.asarray(a, float)
    if len(a) == 0: return (np.nan, "—")
    rng = np.random.default_rng(seed)
    bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return (round(a.mean(), 3), f"[{lo:.2f},{hi:.2f}]")


rows, paired = [], []
for mk in MIN_KS:
    for dom in ("drugs", "weapon"):
        cq = set(coh[(coh.domain == dom) &
                     (coh.n_cit_llm_neighbours_capped >= mk)]["query"])
        sub = err[(err.domain == dom) & (err["query"].isin(cq))]
        for me in METHODS:
            e = sub[sub.method == me]
            mlo, clo = boot_ci(e.err_lo); mhi, chi = boot_ci(e.err_hi)
            rows.append({"min_k": mk, "domain": dom, "method": me,
                         "cohort_size": len(cq), "n_queries": len(e),
                         "mae_lo": mlo, "mae_lo_ci": clo,
                         "mae_hi": mhi, "mae_hi_ci": chi})
        # paired citation_llm vs sup_llm on EXACT same queries (both predicted)
        a = sub[sub.method == "citation_llm"][["query", "err_lo"]].rename(
            columns={"err_lo": "cit"})
        b = sub[sub.method == "sup_llm"][["query", "err_lo"]].rename(
            columns={"err_lo": "sup"})
        mg = a.merge(b, on="query")
        if len(mg) >= 10:
            d = mg.cit.values - mg.sup.values            # neg = citation better
            rng = np.random.default_rng(42)
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            try:
                w = wilcoxon(mg.cit, mg.sup).pvalue
            except Exception:
                w = np.nan
            paired.append({"min_k": mk, "domain": dom, "n_paired": len(mg),
                           "mean_diff_cit_minus_sup": round(d.mean(), 3),
                           "ci_low": round(lo, 3), "ci_hi": round(hi, 3),
                           "wilcoxon_p": w,
                           "sig_95ci": "✓" if hi < 0 or lo > 0 else "✗"})

cmp = pd.DataFrame(rows)
cmp.to_csv(D / "same_queries_comparison.csv", index=False)
pp = pd.DataFrame(paired)
pp.to_csv(D / "same_queries_paired_cit_vs_sup.csv", index=False)

for mk in MIN_KS:
    print(f"\n================  min_k = {mk}  ================")
    for dom in ("drugs", "weapon"):
        c = cmp[(cmp.min_k == mk) & (cmp.domain == dom)].sort_values("mae_lo")
        cs = c.cohort_size.iloc[0]
        print(f"\n  {dom}  (cohort = {cs} queries)")
        print(f"  {'method':22s} {'n':>5s} {'MAE-lo':>7s} {'[95% CI]':>14s} {'MAE-hi':>7s}")
        for _, r in c.iterrows():
            print(f"  {r.method:22s} {r.n_queries:5d} {r.mae_lo:7.2f} "
                  f"{r.mae_lo_ci:>14s} {r.mae_hi:7.2f}")
print("\n=== paired: citation_llm − sup_llm (same queries, neg = cit better) ===")
print(pp.to_string(index=False))
print(f"\n✅ wrote same_queries_comparison.csv + same_queries_paired_cit_vs_sup.csv")
