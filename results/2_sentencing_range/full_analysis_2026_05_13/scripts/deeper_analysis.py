"""
EVEN DEEPER analyses to enrich the thesis:

H. Confidence-weighted prediction — use LLM scores as weights (not just median)
I. Per-quartile LLM contribution — does LLM help more on hard/easy cases?
J. Top-1 LLM score correlation with prediction error
K. Out-of-supervised-pool win analysis — when does LLM-from-all beat sup+LLM?
L. Pool overlap with citation 1hop — % of high-quality citation pairs captured by sup pool
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
EXP  = ROOT / "experiments"
FILTERED_DIR = ROOT / "simcse_cuda_bundle/outputs_supervised_filtered"

N_FOLDS = 5
K_FINAL = 10

m = pd.read_csv(EXP / "data_per_domain/master_inventory.csv",
                usecols=["canonical_id","domain","sentencing_range_low",
                         "sentencing_range_high","sentencing_confidence"])
m = m[m.domain.isin(["drugs","weapon"])
      & m.sentencing_range_low.notna()
      & (m.sentencing_confidence == "גבוהה")].drop_duplicates("canonical_id")
rng_lo = dict(zip(m.canonical_id, m.sentencing_range_low))
rng_hi = dict(zip(m.canonical_id, m.sentencing_range_high))

# LLM scores
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

cit_1hop = set()
cit_df = pd.read_csv(EXP / "data_per_domain/network_analysis/citation_pair_types.csv")
for r in cit_df.itertuples(index=False):
    if r.citation_type == "1hop":
        cit_1hop.add(tuple(sorted([r.verdict_1, r.verdict_2])))

folds = {}
for dom in ("drugs", "weapon"):
    for f in range(1, N_FOLDS + 1):
        ep = FILTERED_DIR / f"verdict_embeddings_{dom}_topk_fold{f}_offenseFiltered.npy"
        ip = FILTERED_DIR / f"verdict_index_{dom}_topk_fold{f}_offenseFiltered.csv"
        if not ep.exists(): continue
        emb = np.load(ep).astype(np.float32)
        idx = pd.read_csv(ip)
        train_ids = idx[idx.split == "train"].verdict.tolist()
        test_ids  = idx[idx.split == "test"].verdict.tolist()
        v2i = {v: i for i, v in enumerate(idx.verdict)}
        folds[(dom, f)] = {"emb": emb, "v2i": v2i,
                           "train_ids": train_ids, "test_ids": test_ids}


def supervised_pool_with_scores(q, ff, size):
    emb, v2i = ff["emb"], ff["v2i"]
    if q not in v2i: return [], []
    qi = v2i[q]
    train_ids = ff["train_ids"]
    train_idx = np.array([v2i[v] for v in train_ids])
    sims = emb[qi] @ emb[train_idx].T
    order = np.argsort(-sims)
    pool_ids = [train_ids[i] for i in order[:size]] if size != "all" else [train_ids[i] for i in order]
    pool_sims = [float(sims[i]) for i in order[:size]] if size != "all" else [float(sims[i]) for i in order]
    return pool_ids, pool_sims


# ============ H. WEIGHTED MEDIAN by LLM score ============
print("=== H. WEIGHTED MEDIAN by LLM score ===")
def weighted_median(values, weights):
    """Weighted median (interpolated)."""
    sv = sorted(zip(values, weights))
    cum = 0
    total = sum(weights)
    if total == 0: return float(np.median(values))
    for v, w in sv:
        cum += w
        if cum >= total / 2:
            return float(v)
    return float(sv[-1][0])

weighted_rows = []
for dom in ("drugs", "weapon"):
    for ps in [50, 100, 200, 500, 1000, "all"]:
        lo_errs_med = []; hi_errs_med = []
        lo_errs_wmed = []; hi_errs_wmed = []
        lo_errs_wmean = []; hi_errs_wmean = []
        for (d, fid), ff in folds.items():
            if d != dom: continue
            for q in ff["test_ids"]:
                if q not in rng_lo: continue
                pool, _ = supervised_pool_with_scores(q, ff, ps)
                scored = [(c, llm_scores.get(tuple(sorted([q, c])))) for c in pool]
                scored = [(c, s) for c, s in scored if s is not None]
                scored.sort(key=lambda x: -x[1])
                picked = [(c, s) for c, s in scored[:K_FINAL] if c in rng_lo]
                if not picked: continue
                lows  = [rng_lo[c] for c, _ in picked]
                highs = [rng_hi[c] for c, _ in picked]
                wts   = [s for _, s in picked]
                # Method 1: unweighted median
                plo_m  = float(np.median(lows))
                phi_m  = float(np.median(highs))
                # Method 2: LLM-weighted median
                plo_wm = weighted_median(lows, wts)
                phi_wm = weighted_median(highs, wts)
                # Method 3: LLM-weighted mean
                w_total = sum(wts)
                if w_total <= 0:
                    plo_wmean = float(np.mean(lows)); phi_wmean = float(np.mean(highs))
                else:
                    plo_wmean = float(sum(l*w for l,w in zip(lows, wts)) / w_total)
                    phi_wmean = float(sum(h*w for h,w in zip(highs, wts)) / w_total)
                lo_errs_med.append(abs(plo_m  - rng_lo[q]))
                hi_errs_med.append(abs(phi_m  - rng_hi[q]))
                lo_errs_wmed.append(abs(plo_wm - rng_lo[q]))
                hi_errs_wmed.append(abs(phi_wm - rng_hi[q]))
                lo_errs_wmean.append(abs(plo_wmean - rng_lo[q]))
                hi_errs_wmean.append(abs(phi_wmean - rng_hi[q]))
        print(f"  {dom:6s} pool={str(ps):>4s}: "
              f"median={np.mean(lo_errs_med):.2f}/{np.mean(hi_errs_med):.2f}  "
              f"w_median={np.mean(lo_errs_wmed):.2f}/{np.mean(hi_errs_wmed):.2f}  "
              f"w_mean={np.mean(lo_errs_wmean):.2f}/{np.mean(hi_errs_wmean):.2f}")
        weighted_rows.append({"domain": dom, "pool_size": str(ps),
                              "median_lo": np.mean(lo_errs_med), "median_hi": np.mean(hi_errs_med),
                              "wmedian_lo": np.mean(lo_errs_wmed), "wmedian_hi": np.mean(hi_errs_wmed),
                              "wmean_lo": np.mean(lo_errs_wmean), "wmean_hi": np.mean(hi_errs_wmean)})
pd.DataFrame(weighted_rows).to_csv("/tmp/deeper_weighted.csv", index=False)


# ============ J. TOP-1 LLM SCORE → PREDICTION ERROR ============
print("\n=== J. CONFIDENCE-AS-LLM-TOP1: Does high LLM-top1 score predict low error? ===")
conf_rows = []
for dom in ("drugs", "weapon"):
    # Use pool=500 as representative
    ps = 500
    top1_scores = []
    errors = []
    for (d, fid), ff in folds.items():
        if d != dom: continue
        for q in ff["test_ids"]:
            if q not in rng_lo: continue
            pool, _ = supervised_pool_with_scores(q, ff, ps)
            scored = [(c, llm_scores.get(tuple(sorted([q, c])))) for c in pool]
            scored = [(c, s) for c, s in scored if s is not None]
            scored.sort(key=lambda x: -x[1])
            if not scored: continue
            top1_score = scored[0][1]
            picked = [c for c, _ in scored[:K_FINAL] if c in rng_lo]
            if not picked: continue
            plo = float(np.median([rng_lo[p] for p in picked]))
            phi = float(np.median([rng_hi[p] for p in picked]))
            err = (abs(plo - rng_lo[q]) + abs(phi - rng_hi[q])) / 2
            top1_scores.append(top1_score)
            errors.append(err)
    # Bucket by top1 score
    arr1 = np.array(top1_scores)
    arr2 = np.array(errors)
    rho, p = spearmanr(arr1, arr2)
    print(f"  {dom}: n={len(arr1)}, Spearman(top1_score, error) = {rho:.3f}  (p={p:.3g})")
    print(f"    by top1 bucket:")
    for lo, hi in [(0, 50), (50, 70), (70, 85), (85, 100)]:
        mask = (arr1 >= lo) & (arr1 < hi)
        if mask.sum() == 0: continue
        print(f"      top1 ∈ [{lo:>3d}, {hi:>3d}): n={mask.sum():>4d}, mean_error={arr2[mask].mean():.2f}")
        conf_rows.append({"domain": dom, "top1_bucket": f"[{lo},{hi})",
                          "n": int(mask.sum()), "mean_error": float(arr2[mask].mean())})
pd.DataFrame(conf_rows).to_csv("/tmp/deeper_confidence.csv", index=False)


# ============ K. WHEN DOES LLM-from-all BEAT sup+LLM ============
print("\n=== K. Out-of-supervised-pool win — when LLM-from-all picks better ===")
win_rows = []
for dom in ("drugs", "weapon"):
    sup_wins = 0; llm_wins = 0; tied = 0
    sup_total_err = 0; llm_total_err = 0; n = 0
    for (d, fid), ff in folds.items():
        if d != dom: continue
        for q in ff["test_ids"]:
            if q not in rng_lo: continue
            # sup+LLM (pool=100)
            pool_sup, _ = supervised_pool_with_scores(q, ff, 100)
            scored_sup = [(c, llm_scores.get(tuple(sorted([q, c])))) for c in pool_sup]
            scored_sup = [(c, s) for c, s in scored_sup if s is not None]
            scored_sup.sort(key=lambda x: -x[1])
            picked_sup = [c for c, _ in scored_sup[:K_FINAL] if c in rng_lo]
            # LLM-from-all
            scored_all = [(t, llm_scores.get(tuple(sorted([q, t]))))
                          for t in ff["train_ids"] if t != q]
            scored_all = [(t, s) for t, s in scored_all if s is not None]
            scored_all.sort(key=lambda x: -x[1])
            picked_all = [t for t, _ in scored_all[:K_FINAL] if t in rng_lo]
            if not picked_sup or not picked_all: continue
            plo_sup = float(np.median([rng_lo[p] for p in picked_sup]))
            phi_sup = float(np.median([rng_hi[p] for p in picked_sup]))
            plo_all = float(np.median([rng_lo[p] for p in picked_all]))
            phi_all = float(np.median([rng_hi[p] for p in picked_all]))
            err_sup = (abs(plo_sup - rng_lo[q]) + abs(phi_sup - rng_hi[q])) / 2
            err_all = (abs(plo_all - rng_lo[q]) + abs(phi_all - rng_hi[q])) / 2
            sup_total_err += err_sup; llm_total_err += err_all; n += 1
            if err_sup < err_all - 0.5: sup_wins += 1
            elif err_all < err_sup - 0.5: llm_wins += 1
            else: tied += 1
            # How much overlap?
            overlap = len(set(picked_sup) & set(picked_all)) / K_FINAL
    print(f"  {dom}: sup+LLM wins {sup_wins:>4d}, LLM-all wins {llm_wins:>4d}, tied {tied:>4d} "
          f"(n={n}) | mean err sup={sup_total_err/n:.2f}, all={llm_total_err/n:.2f}")
    win_rows.append({"domain": dom, "sup_wins": sup_wins, "llm_wins": llm_wins, "tied": tied,
                     "mean_err_sup": sup_total_err/n, "mean_err_llm": llm_total_err/n})
pd.DataFrame(win_rows).to_csv("/tmp/deeper_win_analysis.csv", index=False)


# ============ L. Pool overlap with citation 1hop ============
print("\n=== L. Sup pool ⊃ Citation 1hop — what fraction of high-quality citations are inside sup pool? ===")
overlap_rows = []
for dom in ("drugs", "weapon"):
    for ps in [10, 50, 100, 500, 1000]:
        recalls = []
        for (d, fid), ff in folds.items():
            if d != dom: continue
            train_set = set(ff["train_ids"])
            for q in ff["test_ids"]:
                cit_nbrs = [t for t in train_set if t != q and tuple(sorted([q, t])) in cit_1hop]
                if not cit_nbrs: continue
                pool, _ = supervised_pool_with_scores(q, ff, ps)
                pool_set = set(pool)
                recalls.append(len(set(cit_nbrs) & pool_set) / len(cit_nbrs))
        r = float(np.mean(recalls)) if recalls else 0
        print(f"  {dom:6s} pool={ps:>4d}: recall of citation-1hop neighbors = {r:.3f}")
        overlap_rows.append({"domain": dom, "pool_size": ps,
                             "recall_citation_1hop": r, "n_test": len(recalls)})
pd.DataFrame(overlap_rows).to_csv("/tmp/deeper_overlap.csv", index=False)

print("\n✅ All deeper analyses saved.")
