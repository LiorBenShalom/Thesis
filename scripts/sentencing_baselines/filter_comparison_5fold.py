#!/usr/bin/env python3
"""
5-fold CV evaluation of supervised_topk vs LLM-citation, per domain.

For each fold f (1..5):
  - test = the 1/5 of verdicts assigned to fold f
  - train = the other 4/5
  - supervised_topk model trained on this fold's train (already done on AWS)
  - For each test query: predict (low,high) via top-K cosine over fold-train

Aggregation:
  - Each verdict appears in test EXACTLY ONCE (across the 5 folds)
  - Compute MAE per verdict, then aggregate
  - Also report mean ± std across folds for stability check

Two filters compared:
  - supervised_topk (5-fold CV)
  - LLM-citation (uses existing 140K scores, restricted to fold-train per query)

Output: results/2_sentencing_range/predictions/cv_5fold_supervised_topk.csv
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
EXP  = ROOT / "experiments"
OUT  = EXP / "results/2_sentencing_range/predictions/cv_5fold_supervised_topk.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

K_VALUES = [3, 5, 10, 20]
N_FOLDS  = 5


def load_data():
    m = pd.read_csv(EXP / "data_per_domain/master_inventory.csv",
                    usecols=["canonical_id","domain","sentencing_range_low","sentencing_range_high",
                             "sentencing_confidence"])
    inset = m[m.domain.isin(["drugs","weapon"])
              & m.sentencing_range_low.notna()
              & (m.sentencing_confidence == "גבוהה")].drop_duplicates("canonical_id")
    range_low  = dict(zip(inset.canonical_id, inset.sentencing_range_low))
    range_high = dict(zip(inset.canonical_id, inset.sentencing_range_high))

    # Citation-LLM scores
    df = pd.read_csv(EXP / "data_per_domain/similarity_scores_combined.csv")
    llm_cit = {}
    for r in df.itertuples(index=False):
        a, b = sorted([r.verdict_1, r.verdict_2])
        if pd.notna(r.similarity_score):
            llm_cit[(a, b)] = float(r.similarity_score)
    print(f"  LLM citation scores loaded: {len(llm_cit):,}")

    # ALL LLM scores (citation + simcse + supervised single-split + 5-fold)
    llm_all = dict(llm_cit)  # start from citation
    for path in [EXP / "data_per_domain/similarity_batch_simcse/results/similarity_scores_simcse.csv",
                 EXP / "data_per_domain/similarity_batch_supervised/results/similarity_scores_supervised.csv",
                 EXP / "data_per_domain/similarity_batch_5fold/results/similarity_scores_5fold.csv"]:
        if not path.exists():
            print(f"  ⚠ missing: {path.name}")
            continue
        d = pd.read_csv(path)
        for r in d.itertuples(index=False):
            a, b = sorted([r.verdict_1, r.verdict_2])
            if pd.notna(r.similarity_score):
                llm_all[(a, b)] = float(r.similarity_score)
        print(f"  + {len(d):,} from {path.name}")
    print(f"  TOTAL LLM scores (all sources): {len(llm_all):,}")

    # Per fold, per domain — load embeddings + split index
    folds = {}  # (dom, fold) → {emb, idx, v2i, train_ids, test_ids}
    for dom in ["drugs","weapon"]:
        for f in range(1, N_FOLDS+1):
            emb = np.load(EXP / f"simcse_outputs/supervised/verdict_embeddings_{dom}_topk_fold{f}.npy").astype(np.float32)
            idx = pd.read_csv(EXP / f"simcse_outputs/supervised/verdict_index_{dom}_topk_fold{f}.csv")
            train_ids = idx[idx.split == "train"].verdict.tolist()
            test_ids  = idx[idx.split == "test"].verdict.tolist()
            v2i = {v: i for i, v in enumerate(idx.verdict)}
            folds[(dom, f)] = {"emb": emb, "v2i": v2i,
                               "train_ids": train_ids, "test_ids": test_ids}
        # Sanity: each verdict appears in test exactly once
        all_test = []
        for f in range(1, N_FOLDS+1):
            all_test.extend(folds[(dom, f)]["test_ids"])
        if len(set(all_test)) != len(all_test):
            print(f"  ⚠ {dom}: some verdicts appear in test of multiple folds!")
        if len(all_test) < len(folds[(dom,1)]["train_ids"] + folds[(dom,1)]["test_ids"]) * 0.95:
            print(f"  ⚠ {dom}: union of test sets is smaller than full set")
        print(f"  {dom}: union of all 5 test folds = {len(set(all_test))} verdicts")

    return inset, range_low, range_high, llm_cit, llm_all, folds


def predict_per_fold(folds, dom, fold, K, range_low, range_high, llm_cit, llm_all, top_pool=20):
    """Returns per-query records with multiple modes:
       - sup       : supervised top-K cosine
       - sup+LLM   : supervised top-pool, rerank by LLM, take top-K
       - LLM_cit   : top-K LLM-citation neighbors
       - LLM_all   : top-K LLM-from-all-sources neighbors
    """
    f = folds[(dom, fold)]
    emb = f["emb"]; v2i = f["v2i"]
    train_ids = f["train_ids"]; test_ids = f["test_ids"]
    train_idx = np.array([v2i[v] for v in train_ids])
    train_set = set(train_ids)

    def median_pred(picked):
        if not picked: return None, None
        return (float(np.median([range_low[p]  for p in picked])),
                float(np.median([range_high[p] for p in picked])))

    rows = []
    for q in test_ids:
        if q not in v2i: continue
        qi = v2i[q]
        true_lo, true_hi = range_low[q], range_high[q]

        # ===== supervised top-pool (for both sup-noLLM and sup-+LLM) =====
        sims = emb[qi] @ emb[train_idx].T
        pool_order = np.argsort(-sims)[:top_pool]
        sup_pool = [train_ids[i] for i in pool_order]

        # sup top-K (no LLM)
        sup_picked = sup_pool[:K]
        sup_pred_lo, sup_pred_hi = median_pred(sup_picked)

        # sup top-K from pool, reranked by LLM (take only those with LLM scores)
        sup_llm_scored = [(n, llm_all.get(tuple(sorted([q, n])))) for n in sup_pool]
        sup_llm_scored = [(n, s) for n, s in sup_llm_scored if s is not None]
        sup_llm_scored.sort(key=lambda x: -x[1])
        sup_llm_picked = [n for n, _ in sup_llm_scored[:K]]
        sup_llm_pred_lo, sup_llm_pred_hi = median_pred(sup_llm_picked) if len(sup_llm_picked) >= K else (None, None)

        # ===== LLM-citation only (no embedding model at all) =====
        cit_scored = sorted(
            [(t, llm_cit[tuple(sorted([q, t]))]) for t in train_set
             if t != q and tuple(sorted([q, t])) in llm_cit],
            key=lambda x: -x[1])
        cit_picked = [n for n, _ in cit_scored[:K]]
        cit_pred_lo, cit_pred_hi = median_pred(cit_picked) if len(cit_picked) >= K else (None, None)

        # ===== LLM-all sources (no embedding model) =====
        all_scored = sorted(
            [(t, llm_all[tuple(sorted([q, t]))]) for t in train_set
             if t != q and tuple(sorted([q, t])) in llm_all],
            key=lambda x: -x[1])
        all_picked = [n for n, _ in all_scored[:K]]
        all_pred_lo, all_pred_hi = median_pred(all_picked) if len(all_picked) >= K else (None, None)

        rows.append({
            "qid": q, "fold": fold, "domain": dom, "K": K,
            "true_lo": true_lo, "true_hi": true_hi,
            "sup_lo_err": abs(sup_pred_lo - true_lo) if sup_pred_lo is not None else None,
            "sup_hi_err": abs(sup_pred_hi - true_hi) if sup_pred_hi is not None else None,
            "sup_n":      len(sup_picked),
            "sup_llm_lo_err": abs(sup_llm_pred_lo - true_lo) if sup_llm_pred_lo is not None else None,
            "sup_llm_hi_err": abs(sup_llm_pred_hi - true_hi) if sup_llm_pred_hi is not None else None,
            "sup_llm_pool": len(sup_llm_scored),
            "cit_lo_err": abs(cit_pred_lo - true_lo) if cit_pred_lo is not None else None,
            "cit_hi_err": abs(cit_pred_hi - true_hi) if cit_pred_hi is not None else None,
            "cit_n_pool": len(cit_scored),
            "all_lo_err": abs(all_pred_lo - true_lo) if all_pred_lo is not None else None,
            "all_hi_err": abs(all_pred_hi - true_hi) if all_pred_hi is not None else None,
            "all_n_pool": len(all_scored),
        })
    return rows


def main():
    print("=== Loading data + 5 folds × 2 domains ===")
    inset, range_low, range_high, llm_cit, llm_all, folds = load_data()

    print("\n=== Computing predictions per fold per K ===")
    all_rows = []
    for dom in ["drugs","weapon"]:
        for fold in range(1, N_FOLDS+1):
            for K in K_VALUES:
                all_rows.extend(predict_per_fold(folds, dom, fold, K, range_low, range_high, llm_cit, llm_all))
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT, index=False)
    print(f"  saved per-query records → {OUT}  ({len(df):,} rows)")

    # ---- Aggregation: pooled across folds (each verdict in test exactly once per K) ----
    print("\n" + "=" * 110)
    print("POOLED ACROSS FOLDS (each verdict in test exactly once)")
    print("=" * 110)
    agg = []
    for dom in ["drugs","weapon"]:
        for K in K_VALUES:
            sub = df[(df.domain == dom) & (df.K == K)]
            row = {"domain": dom, "K": K, "n_total": len(sub)}
            for label, lo_col, hi_col in [
                ("sup",     "sup_lo_err",     "sup_hi_err"),
                ("sup_llm", "sup_llm_lo_err", "sup_llm_hi_err"),
                ("cit",     "cit_lo_err",     "cit_hi_err"),
                ("all",     "all_lo_err",     "all_hi_err"),
            ]:
                ssub = sub.dropna(subset=[lo_col, hi_col])
                lo = ssub[lo_col].tolist(); hi = ssub[hi_col].tolist()
                row[f"{label}_n"] = len(ssub)
                row[f"{label}_cov"] = round(100*len(ssub)/len(sub), 1) if sub.shape[0] else 0
                row[f"{label}_MAE_lo"]  = round(float(np.mean(lo)), 2) if lo else None
                row[f"{label}_MAE_hi"]  = round(float(np.mean(hi)), 2) if lo else None
                row[f"{label}_MAE_avg"] = round(float(np.mean(lo + hi)), 2) if lo else None
            agg.append(row)
    pdf = pd.DataFrame(agg)
    print(pdf.to_string(index=False))
    pdf.to_csv(OUT.with_name("cv_5fold_summary.csv"), index=False)

    # ---- Per-fold mean ± std ----
    print("\n" + "=" * 110)
    print("PER-FOLD MAE_avg (mean ± std across 5 folds)")
    print("=" * 110)
    fold_rows = []
    for dom in ["drugs","weapon"]:
        for K in K_VALUES:
            sub = df[(df.domain == dom) & (df.K == K)]
            row = {"domain": dom, "K": K}
            for label, lo_col, hi_col in [
                ("sup",     "sup_lo_err",     "sup_hi_err"),
                ("sup_llm", "sup_llm_lo_err", "sup_llm_hi_err"),
                ("cit",     "cit_lo_err",     "cit_hi_err"),
                ("all",     "all_lo_err",     "all_hi_err"),
            ]:
                ssub = sub.dropna(subset=[lo_col, hi_col])
                if len(ssub) == 0:
                    row[f"{label}_mean"] = None; row[f"{label}_std"] = None
                    continue
                per_fold = ssub.groupby("fold").apply(
                    lambda x: float(np.mean(list(x[lo_col]) + list(x[hi_col]))), include_groups=False).values
                row[f"{label}_mean"] = round(float(per_fold.mean()), 2)
                row[f"{label}_std"]  = round(float(per_fold.std()), 2)
            fold_rows.append(row)
    fdf = pd.DataFrame(fold_rows)
    print(fdf.to_string(index=False))

    print(f"\n💾 Saved → {OUT}")
    print(f"💾 Saved → {OUT.with_name('cv_5fold_summary.csv')}")


if __name__ == "__main__":
    main()
