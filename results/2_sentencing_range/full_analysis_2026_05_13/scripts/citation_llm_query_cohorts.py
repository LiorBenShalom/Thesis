"""
Per-query citation+LLM neighbour counts for same-queries cohort analysis.

One row per eligible test query (full universe = 2,713 drugs + 1,719 weapon,
same as rigor_phase_a). Neighbour = citation pair (1hop/2hop/cocite) that ALSO
has an LLM similarity score AND a valid sentencing range (in rng_lo).
Replicates rigor_phase_a / citation_llm_min_k_analysis logic exactly.

Output: data/citation_llm_query_cohorts.csv
  domain, query, n_cit_llm_neighbours (= capped at 10),
  n_cit_llm_neighbours_capped, n_cit_llm_neighbours_uncapped
"""
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
EXP  = ROOT / "experiments"
FILTERED_DIR = ROOT / "simcse_cuda_bundle/outputs_supervised_filtered"
OUT = Path(__file__).resolve().parent.parent / "data"
N_FOLDS, CAP = 5, 10

sup = pd.read_csv(ROOT / "simcse_cuda_bundle/data/supervised_data.csv")
v_to_text = dict(zip(sup.verdict, sup.indictment_facts))

m = pd.read_csv(EXP / "data_per_domain/master_inventory.csv",
                usecols=["canonical_id","domain","sentencing_range_low",
                         "sentencing_confidence"])
m = m[m.domain.isin(["drugs","weapon"]) & m.sentencing_range_low.notna()
      & (m.sentencing_confidence == "גבוהה")].drop_duplicates("canonical_id")
rng_lo = set(m.canonical_id)

llm_pairs = set()
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
            llm_pairs.add(tuple(sorted([r.verdict_1, r.verdict_2])))

cit_pairs = set()
cd = pd.read_csv(EXP / "data_per_domain/network_analysis/citation_pair_types.csv")
for r in cd.itertuples(index=False):
    if r.citation_type in ("1hop", "2hop", "cocite"):
        cit_pairs.add(tuple(sorted([r.verdict_1, r.verdict_2])))

rows = []
for dom in ("drugs", "weapon"):
    for f in range(1, N_FOLDS + 1):
        ip = FILTERED_DIR / f"verdict_index_{dom}_topk_fold{f}_offenseFiltered.csv"
        if not ip.exists(): continue
        idx = pd.read_csv(ip)
        v2i = set(idx.verdict)
        train_set = set(idx[idx.split == "train"].verdict)
        for q in idx[idx.split == "test"].verdict:
            if q not in rng_lo or q not in v_to_text or q not in v2i:
                continue
            n = 0
            for t in train_set:
                if t == q: continue
                p = tuple(sorted([q, t]))
                if p in cit_pairs and p in llm_pairs and t in rng_lo:
                    n += 1
            rows.append({"domain": dom, "query": q,
                         "n_cit_llm_neighbours": min(n, CAP),
                         "n_cit_llm_neighbours_capped": min(n, CAP),
                         "n_cit_llm_neighbours_uncapped": n})

out = pd.DataFrame(rows)
out.to_csv(OUT / "citation_llm_query_cohorts.csv", index=False)

print("rows:", len(out), "| by domain:", out.domain.value_counts().to_dict())
print("\ncohort sizes (n_cit_llm_neighbours_capped >= min_k):")
for dom in ("drugs", "weapon"):
    d = out[out.domain == dom]
    line = f"  {dom} (N={len(d)}):"
    for mk in (1, 3, 5, 10):
        line += f"  min_k={mk}: {int((d.n_cit_llm_neighbours_capped >= mk).sum())}"
    print(line)
print("\nuncapped distribution (max neighbours seen):")
for dom in ("drugs", "weapon"):
    d = out[out.domain == dom].n_cit_llm_neighbours_uncapped
    print(f"  {dom}: max={d.max()} mean={d.mean():.2f} median={d.median():.0f} "
          f"| >10: {(d>10).sum()} >20: {(d>20).sum()} >50: {(d>50).sum()}")
print(f"\nwrote {OUT/'citation_llm_query_cohorts.csv'}")
