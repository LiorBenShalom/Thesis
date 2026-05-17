"""
Citation+LLM — min_k=1 (Option A) vs min_k=K (Option B), single 4,432 corpus.

Replicates rigor_phase_a.py's citation_llm logic EXACTLY (same master_inventory
filter, same 6 LLM-score sources, same citation_pair_types, same filtered folds,
same eligibility `q in rng_lo & v_to_text & v2i`, same median_pred) so results
are consistent with the canonical rigor_mae_with_ci.csv.

Outputs (data/):
  citation_llm_neighbour_stats.csv     mean/median/std K_used, pct<10, hist_1..10
  citation_llm_K_histogram_drugs.csv   k_used,n_queries
  citation_llm_K_histogram_weapon.csv
  citation_llm_min_k_sweep_4432.csv    domain,K,coverage,n_pred,n_total,mae_*[_ci]
"""
from pathlib import Path
import json, numpy as np, pandas as pd

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
EXP  = ROOT / "experiments"
FILTERED_DIR = ROOT / "simcse_cuda_bundle/outputs_supervised_filtered"
OUT = Path(__file__).resolve().parent.parent / "data"
N_FOLDS = 5
CAP = 10                          # rigor K cap
MINK_SET = [1, 3, 5, 10, 15, 20]

sup = pd.read_csv(ROOT / "simcse_cuda_bundle/data/supervised_data.csv")
v_to_text = dict(zip(sup.verdict, sup.indictment_facts))

m = pd.read_csv(EXP / "data_per_domain/master_inventory.csv",
                usecols=["canonical_id","domain","sentencing_range_low",
                         "sentencing_range_high","sentencing_confidence"])
m = m[m.domain.isin(["drugs","weapon"]) & m.sentencing_range_low.notna()
      & (m.sentencing_confidence == "גבוהה")].drop_duplicates("canonical_id")
rng_lo = dict(zip(m.canonical_id, m.sentencing_range_low))
rng_hi = dict(zip(m.canonical_id, m.sentencing_range_high))

llm_scores = {}
for path in [
    EXP / "data_per_domain/similarity_scores_combined.csv",
    EXP / "data_per_domain/similarity_batch_5fold/results/similarity_scores_5fold.csv",
    EXP / "data_per_domain/similarity_batch_simcse/results/similarity_scores_simcse.csv",
    EXP / "data_per_domain/similarity_batch_supervised/results/similarity_scores_supervised.csv",
    EXP / "data_per_domain/similarity_batch_5fold_v2/results/similarity_scores_5fold_v2.csv",
    EXP / "data_per_domain/similarity_batch_filtered/results/similarity_scores_filtered.csv",
]:
    if not path.exists(): continue
    df = pd.read_csv(path)
    for r in df.itertuples(index=False):
        if pd.notna(r.similarity_score):
            a, b = sorted([r.verdict_1, r.verdict_2])
            llm_scores[(a, b)] = float(r.similarity_score)

cit_pairs = set()
cd = pd.read_csv(EXP / "data_per_domain/network_analysis/citation_pair_types.csv")
for r in cd.itertuples(index=False):
    if r.citation_type in ("1hop", "2hop", "cocite"):
        cit_pairs.add(tuple(sorted([r.verdict_1, r.verdict_2])))

folds = {}
for dom in ("drugs", "weapon"):
    for f in range(1, N_FOLDS + 1):
        ip = FILTERED_DIR / f"verdict_index_{dom}_topk_fold{f}_offenseFiltered.csv"
        if not ip.exists(): continue
        idx = pd.read_csv(ip)
        folds[(dom, f)] = {"v2i": set(idx.verdict),
                           "train_ids": idx[idx.split == "train"].verdict.tolist(),
                           "test_ids":  idx[idx.split == "test"].verdict.tolist()}

# ---- per-query citation+LLM, replicating rigor exactly ----
# rec: domain -> list of dicts {q, n_used (capped, ∩rng_lo), err_lo, err_hi (Option-A pred),
#                               valid_scored_sorted: [(cand, score)] for Option B}
per_dom = {"drugs": [], "weapon": []}
n_total = {"drugs": 0, "weapon": 0}

for (dom, fid), ff in folds.items():
    train_set = set(ff["train_ids"])
    for q in ff["test_ids"]:
        if q not in rng_lo or q not in v_to_text or q not in ff["v2i"]:
            continue
        n_total[dom] += 1
        true_lo, true_hi = rng_lo[q], rng_hi[q]
        cit_cands = [t for t in train_set if t != q
                     and tuple(sorted([q, t])) in cit_pairs]
        scored = [(c, llm_scores.get(tuple(sorted([q, c])))) for c in cit_cands]
        scored = [(c, s) for c, s in scored if s is not None]
        scored.sort(key=lambda x: -x[1])
        # Option A: rigor — top-CAP then keep those with valid range
        picked = [c for c, _ in scored[:CAP]]
        valid = [p for p in picked if p in rng_lo]
        rec = {"q": q, "covered": len(valid) > 0, "n_used": len(valid)}
        if valid:
            plo = float(np.median([rng_lo[p] for p in valid]))
            phi = float(np.median([rng_hi[p] for p in valid]))
            rec["err_lo"] = abs(plo - true_lo)
            rec["err_hi"] = abs(phi - true_hi)
        # Option B needs the full LLM+range-valid candidate list, score-sorted
        rec["pool"] = [(c, s) for c, s in scored if c in rng_lo]   # already sorted desc
        rec["true"] = (true_lo, true_hi)
        per_dom[dom].append(rec)


def boot_ci(errs, B=2000, seed=42):
    a = np.asarray(errs, float)
    if len(a) == 0: return (np.nan, "—")
    rng = np.random.default_rng(seed)
    bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return (a.mean(), f"[{lo:.2f},{hi:.2f}]")


# ---- Option A: neighbour stats + histogram ----
stat_rows, hist_files = [], {}
for dom in ("drugs", "weapon"):
    recs = per_dom[dom]
    cov = [r for r in recs if r["covered"]]
    ku = np.array([r["n_used"] for r in cov])           # 1..CAP
    hist = {k: int((ku == k).sum()) for k in range(1, CAP + 1)}
    row = {"domain": dom,
           "n_total": n_total[dom], "n_covered": len(cov),
           "coverage": round(len(cov) / n_total[dom], 4),
           "mean_K_used": round(ku.mean(), 3),
           "median_K_used": float(np.median(ku)),
           "std_K_used": round(ku.std(ddof=0), 3),
           "pct_with_lt_10": round((ku < CAP).mean(), 4)}
    for k in range(1, CAP + 1): row[f"hist_{k}"] = hist[k]
    stat_rows.append(row)
    pd.DataFrame({"k_used": list(range(1, CAP + 1)),
                  "n_queries": [hist[k] for k in range(1, CAP + 1)]}
                 ).to_csv(OUT / f"citation_llm_K_histogram_{dom}.csv", index=False)
pd.DataFrame(stat_rows).to_csv(OUT / "citation_llm_neighbour_stats.csv", index=False)

# ---- Option B: min_k = K sweep ----
sweep = []
for dom in ("drugs", "weapon"):
    recs = per_dom[dom]
    for K in MINK_SET:
        elo, ehi = [], []
        for r in recs:
            pool = r["pool"]
            if len(pool) >= K:
                top = [c for c, _ in pool[:K]]
                tl, th = r["true"]
                elo.append(abs(float(np.median([rng_lo[c] for c in top])) - tl))
                ehi.append(abs(float(np.median([rng_hi[c] for c in top])) - th))
        mlo, clo = boot_ci(elo); mhi, chi = boot_ci(ehi)
        sweep.append({"domain": dom, "K": K,
                      "n_total": n_total[dom], "n_pred": len(elo),
                      "coverage": round(len(elo) / n_total[dom], 4),
                      "mae_lo": round(mlo, 3) if elo else None, "mae_lo_ci": clo,
                      "mae_hi": round(mhi, 3) if ehi else None, "mae_hi_ci": chi})
pd.DataFrame(sweep).to_csv(OUT / "citation_llm_min_k_sweep_4432.csv", index=False)

# ---- report + verification ----
print("=== n_total (expect drugs 2713 / weapon 1719) ===")
print(n_total)
print("\n=== Option A — neighbour stats ===")
print(pd.DataFrame(stat_rows)[["domain","n_total","n_covered","coverage",
      "mean_K_used","median_K_used","std_K_used","pct_with_lt_10"]].to_string(index=False))
print("\n=== Option B — min_k=K sweep ===")
print(pd.DataFrame(sweep).to_string(index=False))
print("\n=== VERIFY: Option B min_k=1 coverage == Option A coverage ===")
for dom in ("drugs", "weapon"):
    a = next(r for r in stat_rows if r["domain"] == dom)["coverage"]
    b = next(s for s in sweep if s["domain"] == dom and s["K"] == 1)["coverage"]
    print(f"  {dom}: OptA={a}  OptB(min_k=1)={b}  {'✓' if abs(a-b)<1e-9 else '✗ MISMATCH'}")
print(f"\n✅ wrote 4 files to {OUT}")
