"""
M9 (LLM-best) pick composition on the single 4,432 corpus.

For every eligible test query (q in rng_lo & v_to_text & v2i — exactly
rigor_phase_a), among queries with >=10 valid LLM-scored train neighbours:
  - M9 picks = top-10 by LLM score among train with score AND in rng_lo
  - supervised ranking = np.argsort(-(emb[qi] @ emb[train_idx].T))  (rigor-identical)
  - citation link = (q,c) in cit_pairs {1hop,2hop,cocite,...}
Classify each pick: in_sup100 / has_cit / sup_rank; per-query counts.

Output: data/m9_picks_composition_4432.csv
"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
EXP  = ROOT / "experiments"
FILTERED_DIR = ROOT / "simcse_cuda_bundle/outputs_supervised_filtered"   # same as rigor
OUT = Path(__file__).resolve().parent.parent / "data"
N_FOLDS, K, TOP_POOL = 5, 10, 100

sup = pd.read_csv(ROOT / "simcse_cuda_bundle/data/supervised_data.csv")
v_to_text = dict(zip(sup.verdict, sup.indictment_facts))

m = pd.read_csv(EXP / "data_per_domain/master_inventory.csv",
                usecols=["canonical_id","domain","sentencing_range_low",
                         "sentencing_confidence"])
m = m[m.domain.isin(["drugs","weapon"]) & m.sentencing_range_low.notna()
      & (m.sentencing_confidence == "גבוהה")].drop_duplicates("canonical_id")
rng_lo = set(m.canonical_id)

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

rows = []
n_total = {"drugs": 0, "weapon": 0}
n_ge10  = {"drugs": 0, "weapon": 0}

for dom in ("drugs", "weapon"):
    for f in range(1, N_FOLDS + 1):
        ep = FILTERED_DIR / f"verdict_embeddings_{dom}_topk_fold{f}_offenseFiltered.npy"
        ip = FILTERED_DIR / f"verdict_index_{dom}_topk_fold{f}_offenseFiltered.csv"
        if not ep.exists(): continue
        emb = np.load(ep)
        idx = pd.read_csv(ip)
        v2i = {v: i for i, v in enumerate(idx.verdict)}
        train_ids = idx[idx.split == "train"].verdict.tolist()
        test_ids  = idx[idx.split == "test"].verdict.tolist()
        train_idx_arr = np.array([v2i[v] for v in train_ids])
        for q in test_ids:
            if q not in rng_lo or q not in v_to_text or q not in v2i:
                continue
            n_total[dom] += 1
            qi = v2i[q]
            sims = emb[qi] @ emb[train_idx_arr].T
            order = np.argsort(-sims)                       # rigor-identical
            sup_rank = {train_ids[i]: r + 1 for r, i in enumerate(order)}
            sup100 = set(train_ids[i] for i in order[:TOP_POOL])
            # M9 picks: top-10 by LLM score among train with score AND rng_lo
            cand = [(t, llm_scores.get(tuple(sorted([q, t]))))
                    for t in train_ids if t != q]
            cand = [(t, s) for t, s in cand if s is not None and t in rng_lo]
            cand.sort(key=lambda x: -x[1])
            if len(cand) < K:
                continue
            n_ge10[dom] += 1
            picks = [t for t, _ in cand[:K]]
            ranks = [sup_rank[c] for c in picks]
            in100 = [c in sup100 for c in picks]
            hcit  = [tuple(sorted([q, c])) in cit_pairs for c in picks]
            rows.append({
                "domain": dom, "query": q,
                "n_in_sup100": int(sum(in100)),
                "n_has_cit":   int(sum(hcit)),
                "n_in_both":   int(sum(a and b for a, b in zip(in100, hcit))),
                "n_in_neither":int(sum((not a) and (not b)
                                       for a, b in zip(in100, hcit))),
                "avg_sup_rank": round(float(np.mean(ranks)), 2),
                "median_sup_rank": float(np.median(ranks)),
                "n_picks_rank_101_200":   int(sum(101 <= r <= 200 for r in ranks)),
                "n_picks_rank_201_500":   int(sum(201 <= r <= 500 for r in ranks)),
                "n_picks_rank_501_1000":  int(sum(501 <= r <= 1000 for r in ranks)),
                "n_picks_rank_above_1000":int(sum(r > 1000 for r in ranks)),
            })

df = pd.DataFrame(rows)
df.to_csv(OUT / "m9_picks_composition_4432.csv", index=False)

print("=== VERIFY n_total (expect drugs 2713 / weapon 1719) ===")
print("n_total:", n_total, "| n_with_>=10_valid_llm:", n_ge10)
for dom in ("drugs", "weapon"):
    d = df[df.domain == dom]
    N = len(d)
    below = d.n_picks_rank_101_200 + d.n_picks_rank_201_500 + \
            d.n_picks_rank_501_1000 + d.n_picks_rank_above_1000
    tot = N * 10
    print(f"\nDRUGS aggregate" if dom == "drugs" else "\nWEAPON aggregate",
          f"(out of 10 M9 picks, mean across n={N} queries):")
    print(f"  In sup top-100:    {d.n_in_sup100.mean():.2f}  ({d.n_in_sup100.sum()/tot*100:.0f}%)")
    print(f"  Has citation link: {d.n_has_cit.mean():.2f}  ({d.n_has_cit.sum()/tot*100:.0f}%)")
    print(f"  In BOTH:           {d.n_in_both.mean():.2f}  ({d.n_in_both.sum()/tot*100:.0f}%)")
    print(f"  In NEITHER:        {d.n_in_neither.mean():.2f}  ({d.n_in_neither.sum()/tot*100:.0f}%)")
    print(f"  Avg sup rank of picks:    {d.avg_sup_rank.mean():.0f}")
    print(f"  Median sup rank (median of per-query medians): {d.median_sup_rank.median():.0f}")
    print(f"  Below-100 picks distribution (of {int(below.sum())} below-100 picks):")
    b = below.sum()
    if b:
        print(f"    rank 101-200:  {d.n_picks_rank_101_200.sum()/b*100:.0f}%")
        print(f"    rank 201-500:  {d.n_picks_rank_201_500.sum()/b*100:.0f}%")
        print(f"    rank 501-1000: {d.n_picks_rank_501_1000.sum()/b*100:.0f}%")
        print(f"    rank > 1000:   {d.n_picks_rank_above_1000.sum()/b*100:.0f}%")
print(f"\n✅ wrote {OUT/'m9_picks_composition_4432.csv'}  ({len(df)} rows)")
