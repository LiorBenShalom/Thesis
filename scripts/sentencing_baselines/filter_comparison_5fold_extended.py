#!/usr/bin/env python3
"""
Extended version of filter_comparison_5fold.py.

Adds, per query (K=10 only):
  • σ_lo, σ_hi for each method (sup, sup_llm, cit, all)
  • Neighbor IDs picked by each method (for leakage / overlap analysis)
  • Per-source isolated predictions: simcse-only, supervised-only, 5fold-only
    (citation-only is identical to existing `cit`)
  • Leave-one-source-out predictions: all_no_cit, all_no_simcse, all_no_sup, all_no_5fold
  • σ over full LLM-all pool (not just top-K) — "natural variance" benchmark
  • Mean |Δyear| between target and neighbors per method — leakage proxy

Output: results/2_sentencing_range/predictions/cv_5fold_extended.csv
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
EXP  = ROOT / "experiments"
OUT  = EXP / "results/2_sentencing_range/predictions/cv_5fold_extended.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

K = 10
N_FOLDS = 5
TOP_POOL = 20

LLM_PATHS = {
    "cit":     EXP / "data_per_domain/similarity_scores_combined.csv",
    "simcse":  EXP / "data_per_domain/similarity_batch_simcse/results/similarity_scores_simcse.csv",
    "sup":     EXP / "data_per_domain/similarity_batch_supervised/results/similarity_scores_supervised.csv",
    "fold5":   EXP / "data_per_domain/similarity_batch_5fold/results/similarity_scores_5fold.csv",
}


def load_llm_sources() -> dict[str, dict[tuple[str, str], float]]:
    """Return {source_name → {(a,b): score}} with sorted (a,b) keys."""
    out = {}
    for name, path in LLM_PATHS.items():
        if not path.exists():
            print(f"  ⚠ missing: {path}")
            out[name] = {}
            continue
        d = pd.read_csv(path)
        m = {}
        for r in d.itertuples(index=False):
            if pd.notna(r.similarity_score):
                a, b = sorted([r.verdict_1, r.verdict_2])
                m[(a, b)] = float(r.similarity_score)
        print(f"  {name}: {len(m):,} pairs")
        out[name] = m
    return out


def merge_sources(srcs: dict, keep: list[str]) -> dict[tuple, float]:
    """Union of selected source dicts. Later sources overwrite earlier on conflict."""
    out = {}
    for k in keep:
        out.update(srcs[k])
    return out


def median_pred(picked, lo_map, hi_map):
    if not picked: return None, None, None, None
    los = np.array([lo_map[p] for p in picked], dtype=float)
    his = np.array([hi_map[p] for p in picked], dtype=float)
    return float(np.median(los)), float(np.median(his)), float(los.std()), float(his.std())


def top_k_by_llm(qid, train_set, llm_dict, k):
    """Pick top-K training cases by LLM score for this query."""
    scored = [(t, llm_dict[tuple(sorted([qid, t]))])
              for t in train_set
              if t != qid and tuple(sorted([qid, t])) in llm_dict]
    scored.sort(key=lambda x: -x[1])
    return [n for n, _ in scored[:k]], len(scored)


def full_pool_sigma(qid, train_set, llm_dict, lo_map, hi_map):
    """σ across ALL LLM-scored similars in train (for 'natural variance' baseline)."""
    pool = [t for t in train_set
            if t != qid and tuple(sorted([qid, t])) in llm_dict]
    if not pool:
        return None, None, 0
    los = np.array([lo_map[p] for p in pool], dtype=float)
    his = np.array([hi_map[p] for p in pool], dtype=float)
    return float(los.std()), float(his.std()), len(pool)


def main():
    print("=== Loading master inventory ===")
    m = pd.read_csv(EXP / "data_per_domain/master_inventory.csv",
                    usecols=["canonical_id","domain","sentencing_range_low",
                             "sentencing_range_high","sentencing_confidence","year"])
    inset = (m[m.domain.isin(["drugs","weapon"])
              & m.sentencing_range_low.notna()
              & (m.sentencing_confidence == "גבוהה")]
             .drop_duplicates("canonical_id"))
    range_low  = dict(zip(inset.canonical_id, inset.sentencing_range_low))
    range_high = dict(zip(inset.canonical_id, inset.sentencing_range_high))
    year_of    = dict(zip(inset.canonical_id, inset.year))

    print("\n=== Loading LLM similarity sources ===")
    srcs = load_llm_sources()
    llm_all = merge_sources(srcs, ["cit", "simcse", "sup", "fold5"])
    print(f"  ALL union: {len(llm_all):,}")
    llm_no = {
        "cit":    merge_sources(srcs, ["simcse", "sup", "fold5"]),
        "simcse": merge_sources(srcs, ["cit", "sup", "fold5"]),
        "sup":    merge_sources(srcs, ["cit", "simcse", "fold5"]),
        "fold5":  merge_sources(srcs, ["cit", "simcse", "sup"]),
    }

    print("\n=== Loading 5 folds × 2 domains ===")
    folds = {}
    for dom in ["drugs","weapon"]:
        for f in range(1, N_FOLDS+1):
            emb = np.load(EXP / f"simcse_outputs/supervised/verdict_embeddings_{dom}_topk_fold{f}.npy").astype(np.float32)
            idx = pd.read_csv(EXP / f"simcse_outputs/supervised/verdict_index_{dom}_topk_fold{f}.csv")
            v2i = {v: i for i, v in enumerate(idx.verdict)}
            train_ids = idx[idx.split == "train"].verdict.tolist()
            test_ids  = idx[idx.split == "test"].verdict.tolist()
            folds[(dom, f)] = {"emb": emb, "v2i": v2i,
                               "train_ids": train_ids, "test_ids": test_ids}
        print(f"  {dom}: loaded 5 folds")

    print("\n=== Predicting per fold ===")
    rows = []
    for dom in ["drugs", "weapon"]:
        for fold in range(1, N_FOLDS+1):
            ff = folds[(dom, fold)]
            emb = ff["emb"]; v2i = ff["v2i"]
            train_ids = ff["train_ids"]; test_ids = ff["test_ids"]
            train_set = set(train_ids)
            train_idx = np.array([v2i[v] for v in train_ids])

            for q in test_ids:
                if q not in v2i: continue
                qi = v2i[q]
                t_lo, t_hi = range_low[q], range_high[q]
                y_q = year_of.get(q)

                row = {"qid": q, "fold": fold, "domain": dom,
                       "true_lo": t_lo, "true_hi": t_hi, "year": y_q}

                # ===== supervised top-pool =====
                sims = emb[qi] @ emb[train_idx].T
                pool_order = np.argsort(-sims)[:TOP_POOL]
                sup_pool = [train_ids[i] for i in pool_order]

                # sup top-K (no LLM)
                sup_picked = sup_pool[:K]
                p_lo, p_hi, s_lo, s_hi = median_pred(sup_picked, range_low, range_high)
                row.update({
                    "sup_pred_lo": p_lo, "sup_pred_hi": p_hi,
                    "sup_lo_err": abs(p_lo - t_lo) if p_lo is not None else None,
                    "sup_hi_err": abs(p_hi - t_hi) if p_hi is not None else None,
                    "sup_sig_lo": s_lo, "sup_sig_hi": s_hi,
                    "sup_neighbors": ";".join(sup_picked),
                })

                # sup top-K from pool reranked by all-LLM
                rerank = [(n, llm_all.get(tuple(sorted([q, n])))) for n in sup_pool]
                rerank = [(n, s) for n, s in rerank if s is not None]
                rerank.sort(key=lambda x: -x[1])
                sup_llm_picked = [n for n, _ in rerank[:K]]
                if len(sup_llm_picked) >= K:
                    p_lo, p_hi, s_lo, s_hi = median_pred(sup_llm_picked, range_low, range_high)
                    row.update({
                        "sup_llm_pred_lo": p_lo, "sup_llm_pred_hi": p_hi,
                        "sup_llm_lo_err": abs(p_lo - t_lo), "sup_llm_hi_err": abs(p_hi - t_hi),
                        "sup_llm_sig_lo": s_lo, "sup_llm_sig_hi": s_hi,
                        "sup_llm_pool": len(rerank),
                        "sup_llm_neighbors": ";".join(sup_llm_picked),
                    })
                else:
                    row.update({k: None for k in ["sup_llm_pred_lo","sup_llm_pred_hi",
                        "sup_llm_lo_err","sup_llm_hi_err","sup_llm_sig_lo","sup_llm_sig_hi",
                        "sup_llm_neighbors"]})
                    row["sup_llm_pool"] = len(rerank)

                # citation-only (= "cit" in original)
                cit_picked, cit_pool_size = top_k_by_llm(q, train_set, srcs["cit"], K)
                if len(cit_picked) >= K:
                    p_lo, p_hi, s_lo, s_hi = median_pred(cit_picked, range_low, range_high)
                    row.update({
                        "cit_pred_lo": p_lo, "cit_pred_hi": p_hi,
                        "cit_lo_err": abs(p_lo - t_lo), "cit_hi_err": abs(p_hi - t_hi),
                        "cit_sig_lo": s_lo, "cit_sig_hi": s_hi,
                        "cit_neighbors": ";".join(cit_picked),
                    })
                else:
                    row.update({k: None for k in ["cit_pred_lo","cit_pred_hi",
                        "cit_lo_err","cit_hi_err","cit_sig_lo","cit_sig_hi","cit_neighbors"]})
                row["cit_pool"] = cit_pool_size

                # all-sources LLM
                all_picked, all_pool_size = top_k_by_llm(q, train_set, llm_all, K)
                if len(all_picked) >= K:
                    p_lo, p_hi, s_lo, s_hi = median_pred(all_picked, range_low, range_high)
                    row.update({
                        "all_pred_lo": p_lo, "all_pred_hi": p_hi,
                        "all_lo_err": abs(p_lo - t_lo), "all_hi_err": abs(p_hi - t_hi),
                        "all_sig_lo": s_lo, "all_sig_hi": s_hi,
                        "all_neighbors": ";".join(all_picked),
                    })
                else:
                    row.update({k: None for k in ["all_pred_lo","all_pred_hi",
                        "all_lo_err","all_hi_err","all_sig_lo","all_sig_hi","all_neighbors"]})
                row["all_pool"] = all_pool_size

                # ===== Per-source isolated predictions (besides cit which we already have) =====
                for sname in ["simcse", "sup", "fold5"]:
                    picked, pool_size = top_k_by_llm(q, train_set, srcs[sname], K)
                    prefix = f"{sname}_only"
                    if len(picked) >= K:
                        p_lo, p_hi, s_lo, s_hi = median_pred(picked, range_low, range_high)
                        row.update({
                            f"{prefix}_lo_err": abs(p_lo - t_lo),
                            f"{prefix}_hi_err": abs(p_hi - t_hi),
                            f"{prefix}_sig_lo": s_lo, f"{prefix}_sig_hi": s_hi,
                            f"{prefix}_neighbors": ";".join(picked),
                        })
                    else:
                        row.update({c: None for c in [
                            f"{prefix}_lo_err", f"{prefix}_hi_err",
                            f"{prefix}_sig_lo", f"{prefix}_sig_hi", f"{prefix}_neighbors"]})
                    row[f"{prefix}_pool"] = pool_size

                # ===== Leave-one-source-out predictions =====
                for sname in ["cit", "simcse", "sup", "fold5"]:
                    picked, pool_size = top_k_by_llm(q, train_set, llm_no[sname], K)
                    prefix = f"all_no_{sname}"
                    if len(picked) >= K:
                        p_lo, p_hi, s_lo, s_hi = median_pred(picked, range_low, range_high)
                        row.update({
                            f"{prefix}_lo_err": abs(p_lo - t_lo),
                            f"{prefix}_hi_err": abs(p_hi - t_hi),
                            f"{prefix}_sig_lo": s_lo, f"{prefix}_sig_hi": s_hi,
                            f"{prefix}_neighbors": ";".join(picked),
                        })
                    else:
                        row.update({c: None for c in [
                            f"{prefix}_lo_err", f"{prefix}_hi_err",
                            f"{prefix}_sig_lo", f"{prefix}_sig_hi", f"{prefix}_neighbors"]})
                    row[f"{prefix}_pool"] = pool_size

                # ===== Full-pool σ (natural variance baseline — variant C) =====
                fp_lo, fp_hi, fp_n = full_pool_sigma(q, train_set, llm_all, range_low, range_high)
                row["full_pool_sig_lo"] = fp_lo
                row["full_pool_sig_hi"] = fp_hi
                row["full_pool_n"] = fp_n

                rows.append(row)

            print(f"  {dom} fold {fold}: {len(test_ids)} queries done")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\n💾 Saved → {OUT}  ({len(df):,} rows, {df.shape[1]} cols)")
    print("\n=== Quick sanity: per-method MAE_avg at full coverage ===")
    for dom in ["drugs", "weapon"]:
        sub = df[df.domain == dom]
        print(f"\n{dom} (n={len(sub)}):")
        for m_ in ["sup","sup_llm","cit","all","simcse_only","sup_only","fold5_only",
                  "all_no_cit","all_no_simcse","all_no_sup","all_no_fold5"]:
            lo = sub[f"{m_}_lo_err"].dropna()
            hi = sub[f"{m_}_hi_err"].dropna()
            mae_avg = (lo.tolist() + hi.tolist())
            n_valid = len(lo)
            if mae_avg:
                print(f"  {m_:<15s}  n={n_valid:>5d}  MAE_lo={lo.mean():.2f}  MAE_hi={hi.mean():.2f}  MAE_avg={np.mean(mae_avg):.2f}")


if __name__ == "__main__":
    main()
