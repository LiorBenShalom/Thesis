"""
Phase B — All statistical analyses from per-query errors:
  1. MAE per method (mean ± bootstrap 95% CI)
  2. Paired bootstrap CI on differences between methods
  3. Paired Wilcoxon test between methods
  4. Within-quartile MAE with significance
  5. Year-clustered bootstrap (using year as cluster)
  6. Error analysis: top 30 hardest queries by sup_llm
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

df = pd.read_csv("/tmp/rigor_per_query_errors.csv")
print(f"Loaded {len(df):,} (query, method) error records")
print(f"Methods: {sorted(df.method.unique())}")
print(f"Domains: {sorted(df.domain.unique())}")

# Define avg_err = (err_lo + err_hi) / 2 for paired tests
df["avg_err"] = (df.err_lo + df.err_hi) / 2

METHOD_ORDER = [
    "global_median",
    "random_llm",
    "offense_matched_random",
    "tfidf_ridge",
    "bm25",
    "citation_llm",
    "sup_only",
    "sup_llm",
    "llm_best",
]


# ============ 1. MAE per method with bootstrap 95% CI ============
print("\n" + "="*90)
print(" 1. MAE per method (bootstrap 95% CI, B=2000)")
print("="*90)
B = 2000
rng = np.random.default_rng(42)

mae_rows = []
for dom in ("drugs", "weapon"):
    for method in METHOD_ORDER:
        sub = df[(df.domain == dom) & (df.method == method)]
        if len(sub) == 0: continue
        errs_lo = sub.err_lo.values
        errs_hi = sub.err_hi.values

        # Bootstrap CI on means
        n = len(errs_lo)
        boot_lo = []
        boot_hi = []
        for _ in range(B):
            idx = rng.integers(0, n, n)
            boot_lo.append(errs_lo[idx].mean())
            boot_hi.append(errs_hi[idx].mean())
        ci_lo_25, ci_lo_975 = np.percentile(boot_lo, [2.5, 97.5])
        ci_hi_25, ci_hi_975 = np.percentile(boot_hi, [2.5, 97.5])

        mae_rows.append({
            "domain": dom, "method": method, "n": n,
            "mae_lo": errs_lo.mean(),
            "mae_lo_ci_low": ci_lo_25, "mae_lo_ci_hi": ci_lo_975,
            "mae_hi": errs_hi.mean(),
            "mae_hi_ci_low": ci_hi_25, "mae_hi_ci_hi": ci_hi_975,
        })

mae_df = pd.DataFrame(mae_rows)
mae_df.to_csv("/tmp/rigor_mae_with_ci.csv", index=False)

print(f"\n{'method':28s} {'dom':6s} {'n':>5s} {'MAE-lo [95% CI]':>26s} {'MAE-hi [95% CI]':>26s}")
print("-"*100)
for _, r in mae_df.iterrows():
    print(f"{r.method:28s} {r.domain:6s} {r.n:>5d} "
          f"{r.mae_lo:>6.2f} [{r.mae_lo_ci_low:.2f}, {r.mae_lo_ci_hi:.2f}]  "
          f"{r.mae_hi:>6.2f} [{r.mae_hi_ci_low:.2f}, {r.mae_hi_ci_hi:.2f}]")


# ============ 2. Paired bootstrap on differences ============
print("\n" + "="*90)
print(" 2. Paired bootstrap CI: method A vs method B (Δ MAE, B=2000)")
print("="*90)
B = 2000
rng = np.random.default_rng(123)

# Pivot: query × method
pairs_to_test = [
    ("sup_llm", "sup_only"),       # does LLM rerank help?
    ("sup_llm", "tfidf_ridge"),    # do we beat TF-IDF baseline?
    ("sup_llm", "bm25"),           # do we beat BM25?
    ("sup_llm", "offense_matched_random"),  # do we beat offense-matching?
    ("sup_llm", "random_llm"),     # is filter better than random?
    ("sup_llm", "citation_llm"),   # vs citation
    ("llm_best", "sup_llm"),       # ceiling vs ours
    ("sup_only", "tfidf_ridge"),   # does our embedding beat TF-IDF without LLM?
    ("sup_only", "bm25"),
    ("sup_only", "offense_matched_random"),
]

paired_rows = []
for dom in ("drugs", "weapon"):
    sub_dom = df[df.domain == dom]
    # pivot to query × method
    pivot = sub_dom.pivot_table(index="query", columns="method", values="avg_err", aggfunc="first")
    for a, b in pairs_to_test:
        if a not in pivot.columns or b not in pivot.columns: continue
        valid = pivot[[a, b]].dropna()
        if len(valid) < 20: continue
        diffs = valid[a].values - valid[b].values
        boot_diffs = []
        n = len(diffs)
        for _ in range(B):
            idx = rng.integers(0, n, n)
            boot_diffs.append(diffs[idx].mean())
        ci_low, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
        # Wilcoxon
        try:
            w_stat, w_p = wilcoxon(diffs)
        except Exception:
            w_p = 1.0
        sig = "✓" if (ci_low < 0 and ci_hi < 0) or (ci_low > 0 and ci_hi > 0) else "✗"
        paired_rows.append({
            "domain": dom, "A": a, "B": b, "n_pairs": n,
            "mean_diff_A_minus_B": diffs.mean(),
            "ci_low": ci_low, "ci_hi": ci_hi,
            "wilcoxon_p": w_p,
            "significant_95ci": sig,
        })

paired_df = pd.DataFrame(paired_rows)
paired_df.to_csv("/tmp/rigor_paired_diffs.csv", index=False)

print(f"\n{'A vs B':40s} {'dom':6s} {'n':>4s} {'Δ(A-B)':>8s} {'95% CI':>22s} {'p_W':>10s} {'sig':>4s}")
print("-"*110)
for _, r in paired_df.iterrows():
    label = f"{r.A} vs {r.B}"
    print(f"{label:40s} {r.domain:6s} {r.n_pairs:>4d} {r.mean_diff_A_minus_B:>+7.3f}  "
          f"[{r.ci_low:>+6.3f}, {r.ci_hi:>+6.3f}] {r.wilcoxon_p:>10.3g} {r.significant_95ci:>4s}")


# ============ 3. Within-quartile MAE with CI ============
print("\n" + "="*90)
print(" 3. Within-quartile MAE (bootstrap CI)")
print("="*90)

# Add quartile column
m = pd.read_csv(Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!") /
                "new_try/experiments/data_per_domain/master_inventory.csv",
                usecols=["canonical_id","domain","sentencing_range_low","sentencing_range_high"])
m = m.dropna(subset=["sentencing_range_low"]).drop_duplicates("canonical_id")
mid = {row.canonical_id: (row.sentencing_range_low + row.sentencing_range_high) / 2
       for row in m.itertuples(index=False)}
q_boundaries = {}
for dom in ("drugs", "weapon"):
    sub = m[m.domain == dom]
    mids = (sub.sentencing_range_low + sub.sentencing_range_high) / 2
    q_boundaries[dom] = [np.percentile(mids, q) for q in [25, 50, 75]]

def get_quartile(q, dom):
    m_q = mid.get(q)
    if m_q is None: return None
    bounds = q_boundaries[dom]
    if m_q < bounds[0]: return "Q1"
    if m_q < bounds[1]: return "Q2"
    if m_q < bounds[2]: return "Q3"
    return "Q4"

df["quartile"] = df.apply(lambda r: get_quartile(r["query"], r.domain), axis=1)

quart_rows = []
B = 1000
rng = np.random.default_rng(7)
for dom in ("drugs", "weapon"):
    for method in ["sup_only", "sup_llm", "tfidf_ridge", "llm_best"]:
        for q_label in ["Q1", "Q2", "Q3", "Q4"]:
            sub = df[(df.domain == dom) & (df.method == method) & (df.quartile == q_label)]
            if len(sub) < 10: continue
            errs = sub.avg_err.values
            n = len(errs)
            boot = [errs[rng.integers(0, n, n)].mean() for _ in range(B)]
            ci_low, ci_hi = np.percentile(boot, [2.5, 97.5])
            quart_rows.append({
                "domain": dom, "method": method, "quartile": q_label, "n": n,
                "avg_mae": errs.mean(), "ci_low": ci_low, "ci_hi": ci_hi
            })

quart_df = pd.DataFrame(quart_rows)
quart_df.to_csv("/tmp/rigor_quartile_ci.csv", index=False)
print(f"\n{'method':15s} {'dom':6s} {'qrt':4s} {'n':>4s} {'avg_MAE':>8s} {'95% CI':>20s}")
print("-"*70)
for _, r in quart_df.iterrows():
    print(f"{r.method:15s} {r.domain:6s} {r.quartile:4s} {r.n:>4d} {r.avg_mae:>7.2f} "
          f"[{r.ci_low:.2f}, {r.ci_hi:.2f}]")


# ============ 4. Year-clustered bootstrap ============
print("\n" + "="*90)
print(" 4. Year-clustered bootstrap — resample by year")
print("="*90)
B = 1000
year_rows = []
rng = np.random.default_rng(11)
for dom in ("drugs", "weapon"):
    for method in ["sup_only", "sup_llm", "tfidf_ridge", "llm_best"]:
        sub = df[(df.domain == dom) & (df.method == method)].dropna(subset=["year"])
        if len(sub) == 0: continue
        years_unique = sub.year.unique()
        boots = []
        for _ in range(B):
            sampled_years = rng.choice(years_unique, size=len(years_unique), replace=True)
            errs = []
            for y in sampled_years:
                errs.extend(sub[sub.year == y].avg_err.values)
            if errs:
                boots.append(np.mean(errs))
        if boots:
            ci_low, ci_hi = np.percentile(boots, [2.5, 97.5])
            year_rows.append({
                "domain": dom, "method": method, "n_years": len(years_unique), "n_queries": len(sub),
                "avg_mae_overall": sub.avg_err.mean(),
                "year_cluster_ci_low": ci_low, "year_cluster_ci_hi": ci_hi,
                "ci_width": ci_hi - ci_low
            })

year_df = pd.DataFrame(year_rows)
year_df.to_csv("/tmp/rigor_year_cluster.csv", index=False)
print(f"\n{'method':15s} {'dom':6s} {'n_y':>4s} {'n':>5s} {'avg_MAE':>8s} {'year-cluster CI':>22s} {'width':>7s}")
print("-"*90)
for _, r in year_df.iterrows():
    print(f"{r.method:15s} {r.domain:6s} {r.n_years:>4d} {r.n_queries:>5d} {r.avg_mae_overall:>7.2f} "
          f"[{r.year_cluster_ci_low:.2f}, {r.year_cluster_ci_hi:.2f}] {r.ci_width:>6.2f}")


# ============ 5. Error analysis — top hardest cases for sup+LLM ============
print("\n" + "="*90)
print(" 5. Hardest cases (top error under sup_llm)")
print("="*90)
sup_llm_df = df[df.method == "sup_llm"].copy()
sup_llm_df["err_total"] = sup_llm_df.err_lo + sup_llm_df.err_hi
hardest = sup_llm_df.sort_values("err_total", ascending=False).head(30)
hardest.to_csv("/tmp/rigor_hardest_cases.csv", index=False)
print(f"\n{'query':25s} {'dom':6s} {'year':>5s} {'true':>14s} {'err_lo':>7s} {'err_hi':>7s}")
print("-"*80)
for _, r in hardest.head(20).iterrows():
    print(f"{str(r['query']):25s} {r.domain:6s} {r.year if r.year else 'NA':>5} "
          f"          {r.err_lo:>6.1f}  {r.err_hi:>6.1f}")

print("\n✅ All Phase B outputs saved to /tmp/rigor_*.csv")
