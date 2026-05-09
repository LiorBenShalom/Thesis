#!/usr/bin/env python3
"""
4-filter comparison for sentencing-range kNN.

Filters:
  - citation         : 1hop=3, 2hop=2, cocite=1 (graph-based, static)
  - simcse           : unsupervised SimCSE on indictment-facts
  - supervised_thr   : supervised contrastive, mode=threshold (|Δ| ≤ 6 mo)
  - supervised_topk  : supervised contrastive, mode=top-K (each anchor's K closest)

For each: 4 eval modes (full / +min_K / +sigma_q50 / +both) × 2 scorers (noLLM, +LLM)

Output: results/2_sentencing_range/predictions/filter_4way.csv
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
EXP  = ROOT / "experiments"

EMB_SIM   = EXP / "simcse_outputs/verdict_embeddings.npy"
IDX_SIM   = EXP / "simcse_outputs/verdict_index.csv"
SUP = {
    "thr":  {"drugs":  EXP / "simcse_outputs/supervised/verdict_embeddings_drugs.npy",
             "weapon": EXP / "simcse_outputs/supervised/verdict_embeddings_weapon.npy",
             "idx_d":  EXP / "simcse_outputs/supervised/verdict_index_drugs.csv",
             "idx_w":  EXP / "simcse_outputs/supervised/verdict_index_weapon.csv"},
    "topk": {"drugs":  EXP / "simcse_outputs/supervised/verdict_embeddings_drugs_topk.npy",
             "weapon": EXP / "simcse_outputs/supervised/verdict_embeddings_weapon_topk.npy",
             "idx_d":  EXP / "simcse_outputs/supervised/verdict_index_drugs_topk.csv",
             "idx_w":  EXP / "simcse_outputs/supervised/verdict_index_weapon_topk.csv"},
}
CIT       = EXP / "data_per_domain/network_analysis/citation_pair_types.csv"
LLM_OLD   = EXP / "data_per_domain/similarity_scores_combined.csv"
LLM_NEW   = EXP / "data_per_domain/similarity_batch_simcse/results/similarity_scores_simcse.csv"
LLM_SUP   = EXP / "data_per_domain/similarity_batch_supervised/results/similarity_scores_supervised.csv"
MST       = EXP / "data_per_domain/master_inventory.csv"

OUT = EXP / "results/2_sentencing_range/predictions/filter_4way.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

K_VALUES   = [3, 5, 10, 20]
TOP_POOL   = 20
CIT_WEIGHT = {"1hop": 3, "2hop": 2, "cocite": 1, "none": 0}


def cit_strength(citation_type):
    if not isinstance(citation_type, str): return 0
    return max((CIT_WEIGHT.get(p, 0) for p in citation_type.split(",")), default=0)


def load_supervised(variant):
    """variant = 'thr' or 'topk'."""
    cfg = SUP[variant]
    out = {}
    for dom, ep, ip in [("drugs", cfg["drugs"], cfg["idx_d"]),
                        ("weapon", cfg["weapon"], cfg["idx_w"])]:
        emb = np.load(ep).astype(np.float32)
        idx = pd.read_csv(ip)
        out[dom] = {"emb": emb,
                    "v2i": dict(zip(idx.verdict, range(len(idx)))),
                    "split": dict(zip(idx.verdict, idx.split)),
                    "ids": idx.verdict.tolist()}
    return out


def load_data():
    m = pd.read_csv(MST, usecols=["canonical_id","domain","sentencing_range_low",
                                  "sentencing_range_high","sentencing_confidence"])
    inset = m[m.domain.isin(["drugs","weapon"])
              & m.sentencing_range_low.notna()
              & (m.sentencing_confidence == "גבוהה")].drop_duplicates("canonical_id")

    emb_sim = np.load(EMB_SIM).astype(np.float32)
    idx_sim = pd.read_csv(IDX_SIM)
    sim_v2i = {v: i for i, v in enumerate(idx_sim.verdict)}

    sup_thr  = load_supervised("thr")
    sup_topk = load_supervised("topk")

    # Use the same train/test split (deterministic seed=42 → both variants identical)
    keep_ids = set(idx_sim.verdict)
    inset = inset[inset.canonical_id.isin(keep_ids)].reset_index(drop=True)
    inset = inset[inset.canonical_id.isin(set(sup_thr["drugs"]["ids"]) | set(sup_thr["weapon"]["ids"]))]

    range_low  = dict(zip(inset.canonical_id, inset.sentencing_range_low))
    range_high = dict(zip(inset.canonical_id, inset.sentencing_range_high))
    domain_of  = dict(zip(inset.canonical_id, inset.domain))

    split_of = {}
    for dom in ["drugs","weapon"]:
        for v, s in sup_thr[dom]["split"].items(): split_of[v] = s
    print(f"  in_set: {len(inset):,}  ({inset.domain.value_counts().to_dict()})")

    cit = pd.read_csv(CIT, usecols=["verdict_1","verdict_2","domain","citation_type"])
    cit = cit[cit.citation_type != "none"].copy()
    cit["strength"] = cit.citation_type.map(cit_strength)
    cit_neighbors = {q: [] for q in inset.canonical_id}
    for r in cit.itertuples(index=False):
        for src, tgt in [(r.verdict_1, r.verdict_2), (r.verdict_2, r.verdict_1)]:
            if src in cit_neighbors and tgt in domain_of and domain_of.get(src) == domain_of.get(tgt):
                cit_neighbors[src].append((tgt, r.strength))

    llm_pairs = {}
    for path, label in [(LLM_OLD, "old-140K"), (LLM_NEW, "new-simcse"), (LLM_SUP, "new-supervised")]:
        df = pd.read_csv(path)
        for r in df.itertuples(index=False):
            a, b = sorted([r.verdict_1, r.verdict_2])
            if pd.notna(r.similarity_score):
                llm_pairs[(a, b)] = float(r.similarity_score)
        print(f"  loaded {len(df):,} from {label}")
    print(f"  combined LLM-scored pairs: {len(llm_pairs):,}")

    return (inset, range_low, range_high, domain_of, split_of,
            emb_sim, sim_v2i, sup_thr, sup_topk, cit_neighbors, llm_pairs)


def predict_median(neighbors, range_low, range_high):
    if not neighbors: return None, None, None, None, 0
    los = np.array([range_low[n]  for n in neighbors], dtype=float)
    his = np.array([range_high[n] for n in neighbors], dtype=float)
    return (float(np.median(los)), float(np.median(his)),
            float(los.std()), float(his.std()), len(neighbors))


def cit_top_pool(qid, cit_neighbors, train_ids, top_pool, rng):
    cands = [(t, s) for t, s in cit_neighbors.get(qid, []) if t in train_ids]
    if not cands: return []
    keys = [(-s, rng.random()) for _, s in cands]
    order = sorted(range(len(cands)), key=lambda i: keys[i])
    return [cands[i][0] for i in order[:top_pool]]


def emb_top_pool(qid, emb, v2i, train_ids_dom, top_pool):
    if qid not in v2i: return []
    qi = v2i[qid]
    cand_ids = [v for v in train_ids_dom if v in v2i and v != qid]
    if not cand_ids: return []
    cand_idx = np.array([v2i[v] for v in cand_ids])
    sims = emb[qi] @ emb[cand_idx].T
    order = np.argsort(-sims)[:top_pool]
    return [cand_ids[i] for i in order]


def rerank_by_llm(qid, pool, llm_pairs):
    scored = [(n, llm_pairs.get(tuple(sorted([qid, n])))) for n in pool]
    scored = [(n, s) for n, s in scored if s is not None]
    scored.sort(key=lambda x: -x[1])
    return [n for n, _ in scored]


def main():
    print("=== Loading data ===")
    (inset, range_low, range_high, domain_of, split_of,
     emb_sim, sim_v2i, sup_thr, sup_topk, cit_neighbors, llm_pairs) = load_data()
    rng = np.random.default_rng(42)

    test_qs = {dom: [v for v in inset[inset.domain==dom].canonical_id if split_of.get(v) == "test"]
               for dom in ["drugs","weapon"]}
    train_ids_per_dom = {dom: set(v for v in inset[inset.domain==dom].canonical_id if split_of.get(v) == "train")
                         for dom in ["drugs","weapon"]}

    print(f"\n=== Computing pools per test query (TOP_POOL={TOP_POOL}) ===")
    pools = {}
    for dom in ["drugs", "weapon"]:
        train_dom = train_ids_per_dom[dom]
        for q in test_qs[dom]:
            pools[q] = {
                "cit":            cit_top_pool(q, cit_neighbors, train_dom, TOP_POOL, rng),
                "sim":            emb_top_pool(q, emb_sim, sim_v2i, train_dom, TOP_POOL),
                "supervised_thr": emb_top_pool(q, sup_thr[dom]["emb"], sup_thr[dom]["v2i"], train_dom, TOP_POOL),
                "supervised_topk":emb_top_pool(q, sup_topk[dom]["emb"], sup_topk[dom]["v2i"], train_dom, TOP_POOL),
            }

    all_data = {}
    for k in K_VALUES:
        for fname in ["citation","simcse","supervised_thr","supervised_topk"]:
            key = {"citation":"cit","simcse":"sim",
                   "supervised_thr":"supervised_thr","supervised_topk":"supervised_topk"}[fname]
            no_llm, with_llm = {}, {}
            for q in pools:
                pool = pools[q][key]
                p_lo, p_hi, s_lo, s_hi, n_used = predict_median(pool[:k], range_low, range_high)
                if p_lo is not None:
                    no_llm[q] = {"lo_err": abs(p_lo-range_low[q]),
                                 "hi_err": abs(p_hi-range_high[q]),
                                 "sig_combined": s_lo + s_hi,
                                 "n_used": n_used}
                reranked = rerank_by_llm(q, pool, llm_pairs)
                if reranked:
                    p_lo, p_hi, s_lo, s_hi, n_used = predict_median(reranked[:k], range_low, range_high)
                    if p_lo is not None:
                        with_llm[q] = {"lo_err": abs(p_lo-range_low[q]),
                                       "hi_err": abs(p_hi-range_high[q]),
                                       "sig_combined": s_lo + s_hi,
                                       "n_used": n_used}
            all_data[(fname, "noLLM",   k)] = no_llm
            all_data[(fname, "withLLM", k)] = with_llm

    rows = []
    for em in ["full", "min_K", "sigma_q50", "sigma_q50+min_K"]:
        for k in K_VALUES:
            for dom in ["drugs", "weapon"]:
                dom_qs = set(test_qs[dom])
                for fname in ["citation","simcse","supervised_thr","supervised_topk"]:
                    no_d  = all_data[(fname, "noLLM",   k)]
                    yes_d = all_data[(fname, "withLLM", k)]

                    def filter_subset(d):
                        sub = {q: r for q, r in d.items() if q in dom_qs}
                        if "min_K" in em:
                            sub = {q: r for q, r in sub.items() if r["n_used"] >= k}
                        if "sigma_q50" in em and len(sub) >= 2:
                            cut = float(np.median([r["sig_combined"] for r in sub.values()]))
                            sub = {q: r for q, r in sub.items() if r["sig_combined"] <= cut}
                        return sub

                    no_sub = filter_subset(no_d); yes_sub = filter_subset(yes_d)
                    no_mae  = np.mean([r["lo_err"] for r in no_sub.values()] + [r["hi_err"] for r in no_sub.values()]) if no_sub else None
                    yes_mae = np.mean([r["lo_err"] for r in yes_sub.values()] + [r["hi_err"] for r in yes_sub.values()]) if yes_sub else None
                    rows.append({
                        "eval_mode": em, "filter": fname, "domain": dom, "K": k,
                        "n_test_dom": len(dom_qs),
                        "n_eval_noLLM": len(no_sub), "n_eval_withLLM": len(yes_sub),
                        "coverage_pct": round(100*len(no_sub)/len(dom_qs), 1) if dom_qs else 0,
                        "MAE_avg_noLLM":   round(no_mae, 2)  if no_mae  is not None else None,
                        "MAE_avg_withLLM": round(yes_mae, 2) if yes_mae is not None else None,
                        "LLM_lift_pct":    round(100*(no_mae - yes_mae)/no_mae, 1) if (no_mae and yes_mae) else None,
                    })

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    for em in ["full", "sigma_q50", "sigma_q50+min_K"]:
        sub = df[df.eval_mode == em]
        print(f"\n{'='*120}\nEVAL MODE: {em.upper()}\n{'='*120}")
        for sc in ["noLLM", "withLLM"]:
            piv = sub.pivot_table(index=["domain","K"], columns="filter",
                                  values=f"MAE_avg_{sc}").reset_index()
            piv.columns.name = None
            cols = ["domain","K","citation","simcse","supervised_thr","supervised_topk"]
            print(f"\n  -- MAE_avg ({em}, {sc}) --")
            print(piv[cols].to_string(index=False))

    print(f"\n💾 Saved → {OUT}")


if __name__ == "__main__":
    main()
