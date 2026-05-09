#!/usr/bin/env python3
"""
2×2 filter × scorer comparison for sentencing-range kNN.

                │  no LLM (filter rank)  │  + LLM (rerank within filter)
────────────────┼────────────────────────┼─────────────────────────────
citation filter │  cell 1                │  cell 2
SimCSE  filter  │  cell 3                │  cell 4

Research question: does LLM scoring add MORE value for SimCSE-filtered
candidates than for citation-filtered ones?
("the worse the filter, the more the scorer matters")

Setup (apples-to-apples):
- Each filter takes its TOP-20 candidates per query (citation: by strength
  1hop=3 > 2hop=2 > cocite=1; SimCSE: by cosine).
- "no LLM" mode: equal-weight median of top-K (K ∈ {3,5,10,20})
- "+ LLM"  mode: rerank the top-20 by gpt-4.1 score, take top-K, equal-weight median.
- LLM_lift = MAE(no_llm) - MAE(+LLM)  →  positive = LLM helped.

Output: results/2_sentencing_range/predictions/filter_2x2_with_llm.csv
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
EXP  = ROOT / "experiments"

EMB        = EXP / "simcse_outputs/verdict_embeddings.npy"
IDX        = EXP / "simcse_outputs/verdict_index.csv"
CIT        = EXP / "data_per_domain/network_analysis/citation_pair_types.csv"
LLM_OLD    = EXP / "data_per_domain/similarity_scores_combined.csv"          # 140K citation pairs
LLM_NEW    = EXP / "data_per_domain/similarity_batch_simcse/results/similarity_scores_simcse.csv"  # 49K SimCSE pairs
MST        = EXP / "data_per_domain/master_inventory.csv"
OUT        = EXP / "results/2_sentencing_range/predictions/filter_2x2_with_llm.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

K_VALUES   = [3, 5, 10, 20]
TOP_POOL   = 20                                  # candidate pool per filter (then rerank)
CIT_WEIGHT = {"1hop": 3, "2hop": 2, "cocite": 1, "none": 0}


def cit_strength(citation_type: str) -> int:
    if not isinstance(citation_type, str): return 0
    return max((CIT_WEIGHT.get(p, 0) for p in citation_type.split(",")), default=0)


def load_data():
    m = pd.read_csv(MST, usecols=["canonical_id","domain","sentencing_range_low",
                                  "sentencing_range_high","sentencing_confidence"])
    inset = m[m.domain.isin(["drugs","weapon"])
              & m.sentencing_range_low.notna()
              & (m.sentencing_confidence == "גבוהה")].drop_duplicates("canonical_id")

    emb = np.load(EMB).astype(np.float32)
    idx = pd.read_csv(IDX)
    v2i = {v: i for i, v in enumerate(idx.verdict)}

    inset = inset[inset.canonical_id.isin(v2i)].reset_index(drop=True)
    print(f"in_set ∩ embedding: {len(inset):,}  ({inset.domain.value_counts().to_dict()})")

    range_low  = dict(zip(inset.canonical_id, inset.sentencing_range_low))
    range_high = dict(zip(inset.canonical_id, inset.sentencing_range_high))
    domain_of  = dict(zip(inset.canonical_id, inset.domain))
    in_set_ids = set(inset.canonical_id)

    cit = pd.read_csv(CIT, usecols=["verdict_1","verdict_2","domain","citation_type"])
    cit = cit[cit.citation_type != "none"].copy()
    cit["strength"] = cit.citation_type.map(cit_strength)
    print(f"citation positive pairs: {len(cit):,}")

    cit_neighbors = {q: [] for q in inset.canonical_id}
    for r in cit.itertuples(index=False):
        for src, tgt in [(r.verdict_1, r.verdict_2), (r.verdict_2, r.verdict_1)]:
            if src in cit_neighbors and tgt in in_set_ids and domain_of.get(src) == domain_of.get(tgt):
                cit_neighbors[src].append((tgt, r.strength))

    # ---- LLM scores (combine both files) ----
    llm_pairs = {}
    for path, label in [(LLM_OLD, "old-140K"), (LLM_NEW, "new-simcse")]:
        df = pd.read_csv(path)
        for r in df.itertuples(index=False):
            a, b = sorted([r.verdict_1, r.verdict_2])
            if pd.notna(r.similarity_score):
                llm_pairs[(a, b)] = float(r.similarity_score)
        print(f"  loaded {len(df):,} from {label}")
    print(f"  combined unique LLM-scored pairs: {len(llm_pairs):,}")

    return inset, emb, v2i, in_set_ids, range_low, range_high, domain_of, cit_neighbors, llm_pairs


def predict_median(neighbors, range_low, range_high):
    """Returns (pred_low, pred_high, sigma_low, sigma_high, n_neighbors)."""
    if not neighbors: return None, None, None, None, 0
    los = np.array([range_low[n]  for n in neighbors], dtype=float)
    his = np.array([range_high[n] for n in neighbors], dtype=float)
    return (float(np.median(los)), float(np.median(his)),
            float(los.std()), float(his.std()), len(neighbors))


def cit_top_pool(qid, cit_neighbors, top_pool, rng):
    cands = cit_neighbors.get(qid, [])
    if not cands: return []
    keys = [(-s, rng.random()) for _, s in cands]
    order = sorted(range(len(cands)), key=lambda i: keys[i])
    return [cands[i][0] for i in order[:top_pool]]


def sim_top_pool(qid, top_pool, emb, v2i, in_set_ids, domain_of):
    qi = v2i[qid]
    qd = domain_of[qid]
    cand_ids = [v for v in in_set_ids if domain_of.get(v) == qd and v != qid]
    cand_idx = np.array([v2i[v] for v in cand_ids])
    sims = emb[qi] @ emb[cand_idx].T
    order = np.argsort(-sims)[:top_pool]
    return [cand_ids[i] for i in order]


def rerank_by_llm(qid, pool, llm_pairs):
    """Sort pool by LLM score (descending). Drop neighbors without LLM score."""
    scored = [(n, llm_pairs.get(tuple(sorted([qid, n])))) for n in pool]
    scored = [(n, s) for n, s in scored if s is not None]
    scored.sort(key=lambda x: -x[1])
    return [n for n, _ in scored]


def main():
    inset, emb, v2i, in_set_ids, range_low, range_high, domain_of, cit_neighbors, llm_pairs = load_data()
    rng = np.random.default_rng(42)

    # ---- Pre-compute pools per query (TOP_POOL each filter) ----
    print(f"\nComputing top-{TOP_POOL} pools per query...")
    pools = {}
    for q in inset.canonical_id:
        pools[q] = {
            "cit": cit_top_pool(q, cit_neighbors, TOP_POOL, rng),
            "sim": sim_top_pool(q, TOP_POOL, emb, v2i, in_set_ids, domain_of),
        }
    print(f"  done")

    # For each query × filter × scorer-mode, store per-query info:
    #   (lo_err, hi_err, sig_combined, n_neighbors)
    # Then aggregate three ways: full, with_sigma (Q50), with_minK.
    all_data = {}   # (filter, scorer_mode, K) → {qid: dict}
    for k in K_VALUES:
        for filter_name, key in [("citation", "cit"), ("simcse", "sim")]:
            no_llm, with_llm = {}, {}
            for q in inset.canonical_id:
                pool = pools[q][key]
                # no LLM
                p_lo, p_hi, s_lo, s_hi, n_used = predict_median(pool[:k], range_low, range_high)
                if p_lo is not None:
                    no_llm[q] = {"lo_err": abs(p_lo-range_low[q]),
                                 "hi_err": abs(p_hi-range_high[q]),
                                 "sig_combined": s_lo + s_hi,
                                 "n_used": n_used}
                # + LLM
                reranked = rerank_by_llm(q, pool, llm_pairs)
                if reranked:
                    p_lo, p_hi, s_lo, s_hi, n_used = predict_median(reranked[:k], range_low, range_high)
                    if p_lo is not None:
                        with_llm[q] = {"lo_err": abs(p_lo-range_low[q]),
                                       "hi_err": abs(p_hi-range_high[q]),
                                       "sig_combined": s_lo + s_hi,
                                       "n_used": n_used}
            all_data[(filter_name, "noLLM",   k)] = no_llm
            all_data[(filter_name, "withLLM", k)] = with_llm

    # ---- Aggregate per (eval_mode, domain, K, filter) ----
    # eval_modes:
    #   full      — predict for every query that has a pool (no filter)
    #   sigma_q50 — keep only bottom 50% by sig_combined (per filter×scorer×K×domain cell)
    #   min_K     — require at least K neighbors used
    #   sigma_q50 + min_K — both filters
    rows = []
    for eval_mode in ["full", "sigma_q50", "min_K", "sigma_q50+min_K"]:
        for k in K_VALUES:
            for dom in ["drugs", "weapon"]:
                dom_qs = set(inset[inset.domain == dom].canonical_id)

                for filter_name in ["citation", "simcse"]:
                    no_d  = all_data[(filter_name, "noLLM",   k)]
                    yes_d = all_data[(filter_name, "withLLM", k)]

                    def filter_subset(d):
                        sub = {q: r for q, r in d.items() if q in dom_qs}
                        if "min_K" in eval_mode:
                            sub = {q: r for q, r in sub.items() if r["n_used"] >= k}
                        if "sigma_q50" in eval_mode and len(sub) >= 2:
                            cut = float(np.median([r["sig_combined"] for r in sub.values()]))
                            sub = {q: r for q, r in sub.items() if r["sig_combined"] <= cut}
                        return sub

                    no_sub  = filter_subset(no_d)
                    yes_sub = filter_subset(yes_d)

                    no_mae  = np.mean([r["lo_err"] for r in no_sub.values()]
                                    + [r["hi_err"] for r in no_sub.values()]) if no_sub else None
                    yes_mae = np.mean([r["lo_err"] for r in yes_sub.values()]
                                    + [r["hi_err"] for r in yes_sub.values()]) if yes_sub else None

                    rows.append({
                        "eval_mode": eval_mode,
                        "filter":    filter_name,
                        "domain":    dom,
                        "K":         k,
                        "n_total_dom": len(dom_qs),
                        "n_eval_noLLM":   len(no_sub),
                        "n_eval_withLLM": len(yes_sub),
                        "coverage_pct":   round(100 * len(no_sub) / len(dom_qs), 1) if dom_qs else 0,
                        "MAE_avg_noLLM":   round(no_mae, 2)  if no_mae  is not None else None,
                        "MAE_avg_withLLM": round(yes_mae, 2) if yes_mae is not None else None,
                        "LLM_lift_months": round(no_mae - yes_mae, 2) if (no_mae is not None and yes_mae is not None) else None,
                        "LLM_lift_pct":    round(100*(no_mae - yes_mae)/no_mae, 1) if (no_mae and yes_mae) else None,
                    })

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    for em in ["full", "min_K", "sigma_q50", "sigma_q50+min_K"]:
        sub = df[df.eval_mode == em]
        print("\n" + "="*100)
        print(f"EVAL MODE: {em.upper()}")
        print("="*100)
        # Show a compact view: K=5 only for brevity, both filters
        comp = sub[sub.K.isin([5, 10])]
        print(comp.to_string(index=False))

        # LLM lift comparison
        piv = sub.pivot_table(index=["domain","K"], columns="filter",
                              values="LLM_lift_pct").reset_index()
        piv.columns.name = None
        if "simcse" in piv.columns and "citation" in piv.columns:
            piv["MORE_LLM_LIFT"] = piv.apply(
                lambda r: "simcse" if r["simcse"] > r["citation"] else
                          "citation" if r["citation"] > r["simcse"] else "tie", axis=1)
            print(f"\n  -- LLM LIFT % ({em}) --")
            print(piv.to_string(index=False))

    print(f"\n💾 Saved → {OUT}")


if __name__ == "__main__":
    main()
