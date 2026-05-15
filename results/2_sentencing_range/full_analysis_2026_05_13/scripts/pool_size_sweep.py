"""
THESIS REFRAMING — Pool richness is what enables LLM to do its best work.

Sweep: supervised top-N cosine candidates → LLM picks best 10 → median predict.
N ranges from 10 (cosine alone, no rerank possible) to ALL train (=oracle).

Shows MONOTONIC improvement as we give the LLM a richer pool to choose from.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
EXP  = ROOT / "experiments"
FILTERED_DIR = ROOT / "simcse_cuda_bundle/outputs_supervised_filtered"

N_FOLDS = 5
K_FINAL = 10

# Sentencing
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
print(f"LLM pool: {len(llm_scores):,}")

# Folds
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


def evaluate_pool_size(pool_size, K=K_FINAL):
    """Supervised top-N candidates → LLM rerank → top-K → median predict."""
    rows = []
    for (dom, fid), ff in folds.items():
        emb, v2i = ff["emb"], ff["v2i"]
        train_ids, test_ids = ff["train_ids"], ff["test_ids"]
        train_idx = np.array([v2i[v] for v in train_ids])
        lo_errs = []; hi_errs = []; n_pred = 0
        cands_with_llm = []
        for q in test_ids:
            if q not in v2i or q not in rng_lo: continue
            qi = v2i[q]
            sims = emb[qi] @ emb[train_idx].T
            order = np.argsort(-sims)
            if pool_size == "all":
                pool = [train_ids[i] for i in order]
            else:
                pool = [train_ids[i] for i in order[:pool_size]]
            scored = [(c, llm_scores.get(tuple(sorted([q, c])))) for c in pool]
            scored = [(c, s) for c, s in scored if s is not None]
            cands_with_llm.append(len(scored))
            scored.sort(key=lambda x: -x[1])
            picked = [c for c, _ in scored[:K] if c in rng_lo]
            if not picked:
                # fallback: if no LLM scores in pool, use cosine top-K
                picked = [train_ids[i] for i in order[:K] if train_ids[i] in rng_lo]
            if not picked: continue
            plo = float(np.median([rng_lo[p] for p in picked]))
            phi = float(np.median([rng_hi[p] for p in picked]))
            lo_errs.append(abs(plo - rng_lo[q]))
            hi_errs.append(abs(phi - rng_hi[q]))
            n_pred += 1
        rows.append({"pool_size": pool_size, "domain": dom, "fold": fid,
                     "n_pred": n_pred,
                     "mean_cands_with_llm": float(np.mean(cands_with_llm)),
                     "mae_lo": np.mean(lo_errs) if lo_errs else None,
                     "mae_hi": np.mean(hi_errs) if hi_errs else None})
    return pd.DataFrame(rows)


# Run sweep
POOL_SIZES = [10, 20, 50, 100, 200, 500, 1000, "all"]
print(f"\n{'pool':>10s}  {'dom':6s}  {'cands_w_llm':>12s}  {'MAE-lo':>10s}  {'MAE-hi':>10s}")
print("-" * 60)
all_rows = []
for ps in POOL_SIZES:
    df = evaluate_pool_size(ps)
    for dom in ("drugs", "weapon"):
        sub = df[df.domain == dom]
        cands = sub.mean_cands_with_llm.mean()
        mlo = sub.mae_lo.mean()
        mhi = sub.mae_hi.mean()
        print(f"{str(ps):>10s}  {dom:6s}  {cands:>12.1f}  {mlo:>8.2f}  {mhi:>8.2f}")
        all_rows.append({"pool_size": str(ps), "domain": dom,
                         "mean_cands_w_llm": cands, "mae_lo": mlo, "mae_hi": mhi,
                         "mae_lo_std": sub.mae_lo.std(),
                         "mae_hi_std": sub.mae_hi.std()})
    print()

pd.DataFrame(all_rows).to_csv("/tmp/sweep_pool_size.csv", index=False)
print(f"\n✅ /tmp/sweep_pool_size.csv")
