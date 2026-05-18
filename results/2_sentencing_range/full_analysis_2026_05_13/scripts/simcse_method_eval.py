"""
SimCSE as its own method on 4,432 — simcse_only + simcse_llm.

Replicates rigor_phase_a's sup_only / sup_llm logic EXACTLY but on the
holdout-correct SimCSE 5-fold embeddings (outputs_simcse_5fold). Same
eligibility (q in rng_lo & v_to_text & v2i), same K=10, TOP_POOL=100,
same median_pred, same llm_scores pool. SimCSE test == rigor test (split
read from the filtered folds), so numbers are directly comparable to the
bottom-line. Also measures LLM-pool coverage of SimCSE top-100 pairs
(=> size of any supplemental scoring batch needed for full SimCSE+LLM).

Outputs: data/simcse_per_query_errors.csv
         data/simcse_method_summary.csv
"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
EXP  = ROOT / "experiments"
SIM_DIR = ROOT / "simcse_cuda_bundle/outputs_simcse_5fold"
OUT = Path(__file__).resolve().parent.parent / "data"
N_FOLDS, K, TOP_POOL = 5, 10, 100

sup = pd.read_csv(ROOT / "simcse_cuda_bundle/data/supervised_data.csv")
v_to_text = dict(zip(sup.verdict, sup.indictment_facts))
m = pd.read_csv(EXP / "data_per_domain/master_inventory.csv",
                usecols=["canonical_id","domain","sentencing_range_low",
                         "sentencing_range_high","sentencing_confidence","year"])
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
    for r in pd.read_csv(path).itertuples(index=False):
        if pd.notna(r.similarity_score):
            llm_scores[tuple(sorted([r.verdict_1, r.verdict_2]))] = float(r.similarity_score)


def median_pred(picked):
    valid = [p for p in picked if p in rng_lo]
    if not valid: return (None, None)
    return (float(np.median([rng_lo[p] for p in valid])),
            float(np.median([rng_hi[p] for p in valid])))


def boot_ci(a, B=2000, seed=42):
    a = np.asarray(a, float)
    if len(a) == 0: return (np.nan, "—")
    rng = np.random.default_rng(seed)
    bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return (round(a.mean(), 3), f"[{lo:.2f},{hi:.2f}]")


rows = []
n_total = {"drugs": 0, "weapon": 0}
simcse_top100_pairs = set()           # for coverage measurement
for dom in ("drugs", "weapon"):
    for f in range(1, N_FOLDS + 1):
        ep = SIM_DIR / f"verdict_embeddings_simcse_{dom}_fold{f}.npy"
        ip = SIM_DIR / f"verdict_index_simcse_{dom}_fold{f}.csv"
        if not ep.exists(): continue
        emb = np.load(ep); idx = pd.read_csv(ip)
        idx["verdict"] = idx.verdict.astype(str)
        v2i = {v: i for i, v in enumerate(idx.verdict)}
        train_ids = idx[idx.split == "train"].verdict.tolist()
        test_ids  = idx[idx.split == "test"].verdict.tolist()
        tarr = np.array([v2i[v] for v in train_ids])
        for q in test_ids:
            if q not in rng_lo or q not in v_to_text or q not in v2i:
                continue
            n_total[dom] += 1
            tl, th = rng_lo[q], rng_hi[q]
            qi = v2i[q]
            order = np.argsort(-(emb[qi] @ emb[tarr].T))
            # simcse_only
            top10 = [train_ids[i] for i in order[:K]]
            plo, phi = median_pred(top10)
            if plo is not None:
                rows.append({"query": q, "domain": dom, "fold": f,
                             "method": "simcse_only",
                             "err_lo": abs(plo-tl), "err_hi": abs(phi-th)})
            # simcse_llm: top-100 -> LLM rerank -> top-10
            pool = [train_ids[i] for i in order[:TOP_POOL]]
            for c in pool:
                simcse_top100_pairs.add(tuple(sorted([q, c])))
            scored = [(c, llm_scores.get(tuple(sorted([q, c])))) for c in pool]
            scored = [(c, s) for c, s in scored if s is not None]
            scored.sort(key=lambda x: -x[1])
            picked = [c for c, _ in scored[:K]]
            plo, phi = median_pred(picked)
            if plo is not None:
                rows.append({"query": q, "domain": dom, "fold": f,
                             "method": "simcse_llm",
                             "err_lo": abs(plo-tl), "err_hi": abs(phi-th)})

df = pd.DataFrame(rows)
df.to_csv(OUT / "simcse_per_query_errors.csv", index=False)

# coverage of SimCSE top-100 pairs in the existing LLM pool
have = sum(1 for p in simcse_top100_pairs if p in llm_scores)
gap = len(simcse_top100_pairs) - have

summ = []
for dom in ("drugs", "weapon"):
    for me in ("simcse_only", "simcse_llm"):
        e = df[(df.domain == dom) & (df.method == me)]
        mlo, clo = boot_ci(e.err_lo); mhi, chi = boot_ci(e.err_hi)
        summ.append({"domain": dom, "method": me,
                     "n_total": n_total[dom], "n_pred": len(e),
                     "coverage": round(len(e)/n_total[dom], 4),
                     "mae_lo": mlo, "mae_lo_ci": clo,
                     "mae_hi": mhi, "mae_hi_ci": chi})
sm = pd.DataFrame(summ)
sm.to_csv(OUT / "simcse_method_summary.csv", index=False)

print("=== n_total (expect drugs 2713 / weapon 1719) ===", n_total)
print("\n=== SimCSE method MAE (4,432, vs canonical bottom-line) ===")
print(sm.to_string(index=False))
print("\n  canonical refs (rigor): sup_only d5.89/w13.85 · sup_llm d5.69/w13.03")
print("                          citation_llm d5.26/w12.35 · llm_best d4.97/w11.85")
print(f"\n=== SimCSE top-100 LLM-pool coverage ===")
print(f"  unique SimCSE top-100 (q,c) pairs : {len(simcse_top100_pairs):,}")
print(f"  already LLM-scored               : {have:,} ({have/len(simcse_top100_pairs)*100:.1f}%)")
print(f"  NOT scored (supplemental batch)  : {gap:,} ({gap/len(simcse_top100_pairs)*100:.1f}%)")
print(f"\n✅ wrote simcse_per_query_errors.csv + simcse_method_summary.csv")
