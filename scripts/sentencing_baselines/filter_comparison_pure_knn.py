#!/usr/bin/env python3
"""
PURE kNN filter comparison — does SimCSE-embedding filter beat citation filter
for sentencing-range prediction, WITHOUT using any LLM panel scores?

For every in-set verdict (drugs/weapon, has sentencing range, conf='גבוהה'):
  1. Pick top-K neighbors using filter F  (F ∈ {citation, simcse})
  2. Predict (low, high) = median of neighbors' (low, high)
  3. MAE vs the held-out true range

Two filters compared (no LLM in either):
  - citation: rank by citation_type strength  (1hop=3, 2hop=2, cocite=1)
              ties broken by absolute LLM score? NO — pure citation: random tie-break.
  - simcse:   rank by cosine similarity (already L2-normalized embeddings).

Restrictions:
  - same domain (matches existing pipeline)
  - exclude self (trivially)
  - candidate must also be in_set (only those have sentencing range to copy from)

Output: results/2_sentencing_range/predictions/filter_comparison_pure_knn.csv
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
EXP  = ROOT / "experiments"
EMB  = EXP / "simcse_outputs/verdict_embeddings.npy"
IDX  = EXP / "simcse_outputs/verdict_index.csv"
CIT  = EXP / "data_per_domain/network_analysis/citation_pair_types.csv"
MST  = EXP / "data_per_domain/master_inventory.csv"
OUT  = EXP / "results/2_sentencing_range/predictions/filter_comparison_pure_knn.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

K_VALUES   = [3, 5, 10, 20, 50]
CIT_WEIGHT = {"1hop": 3, "2hop": 2, "cocite": 1, "none": 0}


def cit_strength(citation_type: str) -> int:
    """Combined citation type → max strength. e.g. '1hop,cocite' → 3."""
    if not isinstance(citation_type, str): return 0
    return max((CIT_WEIGHT.get(p, 0) for p in citation_type.split(",")), default=0)


def load_data():
    # ---- in-set queries (have sentencing range) ----
    m = pd.read_csv(MST, usecols=["canonical_id","domain","sentencing_range_low",
                                  "sentencing_range_high","sentencing_confidence"])
    inset = m[m.domain.isin(["drugs","weapon"])
              & m.sentencing_range_low.notna()
              & (m.sentencing_confidence == "גבוהה")].copy()
    inset = inset.drop_duplicates("canonical_id").reset_index(drop=True)
    print(f"in_set: {len(inset):,}  ({inset.domain.value_counts().to_dict()})")

    # ---- SimCSE embeddings ----
    emb = np.load(EMB).astype(np.float32)
    idx = pd.read_csv(IDX)
    v2i = {v: i for i, v in enumerate(idx.verdict)}

    # restrict in_set to verdicts we have an embedding for (otherwise SimCSE filter unusable)
    inset = inset[inset.canonical_id.isin(v2i)].reset_index(drop=True)
    print(f"in_set ∩ embedding: {len(inset):,}")

    range_low  = dict(zip(inset.canonical_id, inset.sentencing_range_low))
    range_high = dict(zip(inset.canonical_id, inset.sentencing_range_high))
    domain_of  = dict(zip(inset.canonical_id, inset.domain))
    in_set_ids = set(inset.canonical_id)

    # ---- citation pairs ----
    cit = pd.read_csv(CIT, usecols=["verdict_1","verdict_2","domain","citation_type"])
    cit = cit[(cit.citation_type != "none")].copy()
    cit["strength"] = cit.citation_type.map(cit_strength)
    print(f"citation positive pairs: {len(cit):,}")

    # build per-verdict citation neighbors {qid: [(neighbor_id, strength), ...]}
    cit_neighbors = {q: [] for q in inset.canonical_id}
    for r in cit.itertuples(index=False):
        for src, tgt in [(r.verdict_1, r.verdict_2), (r.verdict_2, r.verdict_1)]:
            if src in cit_neighbors and tgt in in_set_ids and domain_of.get(src) == domain_of.get(tgt):
                cit_neighbors[src].append((tgt, r.strength))

    return inset, emb, idx, v2i, in_set_ids, range_low, range_high, domain_of, cit_neighbors


def predict_citation(qid, k, cit_neighbors, range_low, range_high, rng):
    """Pick top-K citation neighbors by strength (random tie-break)."""
    candidates = cit_neighbors.get(qid, [])
    if not candidates: return None, None, 0
    # sort by strength desc, random tie-break
    keys = [(-strength, rng.random()) for _, strength in candidates]
    order = sorted(range(len(candidates)), key=lambda i: keys[i])
    picked = [candidates[i][0] for i in order[:k]]
    if not picked: return None, None, 0
    lo = np.median([range_low[p]  for p in picked])
    hi = np.median([range_high[p] for p in picked])
    return lo, hi, len(picked)


def predict_simcse(qid, k, emb, v2i, in_set_ids, domain_of, range_low, range_high):
    """Pick top-K cosine neighbors among in-set, same-domain."""
    qi = v2i[qid]
    qd = domain_of[qid]
    # candidate indices = same-domain in_set verdicts that are not the query
    cand_ids = [v for v in in_set_ids if domain_of.get(v) == qd and v != qid and v in v2i]
    cand_idx = np.array([v2i[v] for v in cand_ids])
    sims = emb[qi] @ emb[cand_idx].T   # cosine since normalized
    order = np.argsort(-sims)[:k]
    picked = [cand_ids[i] for i in order]
    if not picked: return None, None, 0
    lo = np.median([range_low[p]  for p in picked])
    hi = np.median([range_high[p] for p in picked])
    return lo, hi, len(picked)


def main():
    inset, emb, idx, v2i, in_set_ids, range_low, range_high, domain_of, cit_neighbors = load_data()
    rng = np.random.default_rng(42)

    # ---- Global median baseline per domain (predict same value for everyone) ----
    print("\n=== GLOBAL-MEDIAN BASELINE (predict domain median for all) ===")
    base_rows = []
    for dom in ["drugs", "weapon"]:
        sub = inset[inset.domain == dom]
        med_lo = float(np.median(sub.sentencing_range_low))
        med_hi = float(np.median(sub.sentencing_range_high))
        lo_err = (sub.sentencing_range_low  - med_lo).abs()
        hi_err = (sub.sentencing_range_high - med_hi).abs()
        base_rows.append({"domain": dom, "median_low": med_lo, "median_high": med_hi,
                          "MAE_low": round(lo_err.mean(), 2),
                          "MAE_high": round(hi_err.mean(), 2),
                          "MAE_avg": round(np.mean(np.concatenate([lo_err, hi_err])), 2)})
    print(pd.DataFrame(base_rows).to_string(index=False))

    rows = []
    for k in K_VALUES:
        for dom in ["drugs", "weapon"]:
            queries = inset[inset.domain == dom].canonical_id.tolist()

            # store per-query results so we can do "intersection only" comparison
            per_q = {}   # qid → {"cit": (lo,hi), "sim": (lo,hi)}
            for q in queries:
                true_lo, true_hi = range_low[q], range_high[q]
                rec = {"true_lo": true_lo, "true_hi": true_hi}

                lo_c, hi_c, n_c = predict_citation(q, k, cit_neighbors, range_low, range_high, rng)
                if lo_c is not None: rec["cit"] = (lo_c, hi_c, n_c)

                lo_s, hi_s, n_s = predict_simcse(q, k, emb, v2i, in_set_ids, domain_of, range_low, range_high)
                if lo_s is not None: rec["sim"] = (lo_s, hi_s, n_s)

                per_q[q] = rec

            # ---- Eval mode 1: native subset (each filter on its own coverage) ----
            for fname, key in [("citation","cit"), ("simcse","sim")]:
                preds = [r for r in per_q.values() if key in r]
                lo_err = [abs(r[key][0] - r["true_lo"]) for r in preds]
                hi_err = [abs(r[key][1] - r["true_hi"]) for r in preds]
                n_used = [r[key][2] for r in preds]
                rows.append({
                    "mode": "native", "filter": fname, "domain": dom, "K": k,
                    "n_queries": len(queries), "n_predicted": len(preds),
                    "coverage": round(100 * len(preds) / len(queries), 1),
                    "MAE_low":  round(float(np.mean(lo_err)), 2) if lo_err else None,
                    "MAE_high": round(float(np.mean(hi_err)), 2) if hi_err else None,
                    "MAE_avg":  round(float(np.mean(lo_err + hi_err)), 2) if lo_err else None,
                    "median_n_neighbors_used": int(np.median(n_used)) if n_used else 0,
                })

            # ---- Eval mode 2: intersection (queries where BOTH filters predicted) ----
            inter_qs = [q for q, r in per_q.items() if "cit" in r and "sim" in r]
            for fname, key in [("citation","cit"), ("simcse","sim")]:
                lo_err = [abs(per_q[q][key][0] - per_q[q]["true_lo"]) for q in inter_qs]
                hi_err = [abs(per_q[q][key][1] - per_q[q]["true_hi"]) for q in inter_qs]
                rows.append({
                    "mode": "intersection", "filter": fname, "domain": dom, "K": k,
                    "n_queries": len(queries), "n_predicted": len(inter_qs),
                    "coverage": round(100 * len(inter_qs) / len(queries), 1),
                    "MAE_low":  round(float(np.mean(lo_err)), 2) if lo_err else None,
                    "MAE_high": round(float(np.mean(hi_err)), 2) if hi_err else None,
                    "MAE_avg":  round(float(np.mean(lo_err + hi_err)), 2) if lo_err else None,
                    "median_n_neighbors_used": None,
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("\n\n=== HEAD-TO-HEAD: NATIVE coverage (each filter on its own queries) ===")
    nat = df[df["mode"] =="native"].pivot_table(index=["domain","K"], columns="filter",
                                              values="MAE_avg").reset_index()
    nat["delta_simcse_vs_cit_pct"] = ((nat.simcse - nat.citation) / nat.citation * 100).round(1)
    print(nat.to_string(index=False))

    print("\n=== HEAD-TO-HEAD: INTERSECTION (same query subset for both) ===")
    inter = df[df["mode"] =="intersection"].pivot_table(index=["domain","K"], columns="filter",
                                                     values="MAE_avg").reset_index()
    inter["delta_simcse_vs_cit_pct"] = ((inter.simcse - inter.citation) / inter.citation * 100).round(1)
    print(inter.to_string(index=False))
    print("\n  delta_*_pct: positive = SimCSE worse than citation, negative = SimCSE better")

    # Lift over global median
    print("\n=== LIFT vs GLOBAL-MEDIAN BASELINE ===")
    base_dict = {r["domain"]: r["MAE_avg"] for r in base_rows}
    lift = df[df["mode"] =="native"].copy()
    lift["baseline_MAE"] = lift.domain.map(base_dict)
    lift["lift_pct"] = ((lift.baseline_MAE - lift.MAE_avg) / lift.baseline_MAE * 100).round(1)
    show = lift.pivot_table(index=["domain","K"], columns="filter", values="lift_pct").reset_index()
    print(show.to_string(index=False))
    print("\n  lift_pct: positive = filter beats global-median baseline (good)")

    print(f"\n💾 Saved → {OUT}")


if __name__ == "__main__":
    main()
