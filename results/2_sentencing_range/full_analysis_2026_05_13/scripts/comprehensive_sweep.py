"""
COMPREHENSIVE SWEEP — all key configurations for the thesis.

Sweeps:
  - K values: 1, 3, 5, 10, 15, 20, 30, 50
  - source-set sampling: 25%, 50%, 75%, 100% of train pool
  - min_k requirement: 1, K/2, K
  - filters: random, citation_all, supervised_filtered, llm_top
  - with vs without LLM rerank where applicable

Outputs:
  /tmp/sweep_K.csv
  /tmp/sweep_source.csv
  /tmp/sweep_min_k.csv
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
EXP  = ROOT / "experiments"
FILTERED_DIR = ROOT / "simcse_cuda_bundle/outputs_supervised_filtered"

N_FOLDS = 5

# Load sentencing
m = pd.read_csv(EXP / "data_per_domain/master_inventory.csv",
                usecols=["canonical_id","domain","sentencing_range_low",
                         "sentencing_range_high","sentencing_confidence"])
m = m[m.domain.isin(["drugs","weapon"])
      & m.sentencing_range_low.notna()
      & (m.sentencing_confidence == "גבוהה")].drop_duplicates("canonical_id")
rng_lo = dict(zip(m.canonical_id, m.sentencing_range_low))
rng_hi = dict(zip(m.canonical_id, m.sentencing_range_high))
glob_med = {dom: (m[m.domain == dom].sentencing_range_low.median(),
                  m[m.domain == dom].sentencing_range_high.median())
            for dom in ("drugs", "weapon")}

# Load LLM scores
print("Loading LLM scores...")
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
print(f"  {len(llm_scores):,} pairs")

# Citation pairs (all types combined)
cit_pairs = set()
cit_df = pd.read_csv(EXP / "data_per_domain/network_analysis/citation_pair_types.csv")
for r in cit_df.itertuples(index=False):
    if r.citation_type in ("1hop", "2hop", "cocite"):
        cit_pairs.add(tuple(sorted([r.verdict_1, r.verdict_2])))

# Load filtered folds
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


def get_candidates(filter_name, q, ff, use_llm=False, top_pool=100, source_frac=1.0):
    """Return ordered list of candidate verdicts for q.
       source_frac < 1.0: subsample fold-train uniformly at random (per query seeded)."""
    train_ids = ff["train_ids"]
    if source_frac < 1.0:
        rng = np.random.default_rng(hash(q) % 2**32)
        n_keep = max(1, int(len(train_ids) * source_frac))
        keep_idx = rng.choice(len(train_ids), size=n_keep, replace=False)
        train_ids = [train_ids[i] for i in sorted(keep_idx)]

    if filter_name == "random":
        rng = np.random.default_rng(hash(q + "rand") % 2**32)
        idx = rng.permutation(len(train_ids))
        cands = [train_ids[i] for i in idx[:50] if train_ids[i] != q]
    elif filter_name == "citation_all":
        cands = [t for t in train_ids if t != q and tuple(sorted([q, t])) in cit_pairs]
    elif filter_name == "supervised":
        emb, v2i = ff["emb"], ff["v2i"]
        if q not in v2i: return []
        qi = v2i[q]
        # need train_ids in v2i for cosine
        train_idx_arr = np.array([v2i[v] for v in train_ids if v in v2i])
        if len(train_idx_arr) == 0: return []
        sims = emb[qi] @ emb[train_idx_arr].T
        order = np.argsort(-sims)
        train_ordered = [v for v in train_ids if v in v2i]
        if use_llm:
            pool = [train_ordered[i] for i in order[:top_pool]]
            scored = [(c, llm_scores.get(tuple(sorted([q, c])))) for c in pool]
            scored = [(c, s) for c, s in scored if s is not None]
            scored.sort(key=lambda x: -x[1])
            return [c for c, _ in scored]
        return [train_ordered[i] for i in order]
    elif filter_name == "llm_top":
        cands = []
        for t in train_ids:
            if t == q: continue
            s = llm_scores.get(tuple(sorted([q, t])))
            if s is not None: cands.append((t, s))
        cands.sort(key=lambda x: -x[1])
        return [t for t, _ in cands]
    else:
        raise ValueError(filter_name)

    if use_llm and filter_name != "supervised":
        scored = [(c, llm_scores.get(tuple(sorted([q, c])))) for c in cands]
        scored = [(c, s) for c, s in scored if s is not None]
        scored.sort(key=lambda x: -x[1])
        return [c for c, _ in scored]
    return cands


def predict_mae(filter_name, K, use_llm, source_frac, min_k):
    """Run prediction for all test queries; return per-domain (mae_lo, mae_hi, coverage)."""
    results = {}
    for dom in ("drugs", "weapon"):
        lo_errs = []; hi_errs = []; n_pred = 0; n_total = 0
        for (d, fid), ff in folds.items():
            if d != dom: continue
            for q in ff["test_ids"]:
                if q not in rng_lo: continue
                n_total += 1
                if filter_name == "global_median":
                    plo, phi = glob_med[dom]
                    lo_errs.append(abs(plo - rng_lo[q]))
                    hi_errs.append(abs(phi - rng_hi[q]))
                    n_pred += 1
                    continue
                cands = get_candidates(filter_name, q, ff,
                                       use_llm=use_llm, source_frac=source_frac)
                picked = [c for c in cands[:K] if c in rng_lo]
                if len(picked) < min_k: continue
                plo = float(np.median([rng_lo[p] for p in picked]))
                phi = float(np.median([rng_hi[p] for p in picked]))
                lo_errs.append(abs(plo - rng_lo[q]))
                hi_errs.append(abs(phi - rng_hi[q]))
                n_pred += 1
        results[dom] = {
            "n_total": n_total, "n_pred": n_pred,
            "coverage": n_pred / n_total if n_total else 0,
            "mae_lo": np.mean(lo_errs) if lo_errs else None,
            "mae_hi": np.mean(hi_errs) if hi_errs else None,
        }
    return results


# ============ SWEEP 1: K values ============
print("\n=== SWEEP 1: K values (source=100%, min_k=1) ===")
K_VALUES = [1, 3, 5, 10, 15, 20, 30, 50]
configs_K = [
    ("global_median", False),
    ("random", False), ("random", True),
    ("citation_all", False), ("citation_all", True),
    ("supervised", False), ("supervised", True),
    ("llm_top", False),
]
rows = []
for filter_name, use_llm in configs_K:
    label = f"{filter_name}{'_llm' if use_llm else ''}"
    for K in K_VALUES:
        r = predict_mae(filter_name, K, use_llm, source_frac=1.0, min_k=1)
        for dom, vals in r.items():
            rows.append({
                "sweep": "K", "filter": label, "domain": dom, "K": K,
                **vals,
            })
        print(f"  {label:25s} K={K:>2d}  drugs: {r['drugs']['mae_lo']:.2f}/{r['drugs']['mae_hi']:.2f}  "
              f"weapon: {r['weapon']['mae_lo']:.2f}/{r['weapon']['mae_hi']:.2f}")
df_K = pd.DataFrame(rows)
df_K.to_csv("/tmp/sweep_K.csv", index=False)


# ============ SWEEP 2: source-set size ============
print("\n=== SWEEP 2: source-set sampling (K=10, min_k=1) ===")
SRC_FRACS = [0.25, 0.50, 0.75, 1.00]
rows = []
for filter_name, use_llm in configs_K:
    label = f"{filter_name}{'_llm' if use_llm else ''}"
    for sf in SRC_FRACS:
        r = predict_mae(filter_name, 10, use_llm, source_frac=sf, min_k=1)
        for dom, vals in r.items():
            rows.append({
                "sweep": "source", "filter": label, "domain": dom,
                "source_frac": sf, **vals,
            })
        print(f"  {label:25s} src={sf:.2f}  drugs: {r['drugs']['mae_lo']:.2f}/{r['drugs']['mae_hi']:.2f}  "
              f"weapon: {r['weapon']['mae_lo']:.2f}/{r['weapon']['mae_hi']:.2f}")
df_src = pd.DataFrame(rows)
df_src.to_csv("/tmp/sweep_source.csv", index=False)


# ============ SWEEP 3: min_k threshold ============
print("\n=== SWEEP 3: min_k requirement (K=10, source=100%) ===")
MIN_KS = [1, 3, 5, 10]
rows = []
for filter_name, use_llm in configs_K:
    label = f"{filter_name}{'_llm' if use_llm else ''}"
    for mk in MIN_KS:
        r = predict_mae(filter_name, 10, use_llm, source_frac=1.0, min_k=mk)
        for dom, vals in r.items():
            rows.append({
                "sweep": "min_k", "filter": label, "domain": dom,
                "min_k": mk, **vals,
            })
        print(f"  {label:25s} min_k={mk:>2d}  drugs cov={r['drugs']['coverage']*100:>5.1f}% "
              f"MAE-lo={r['drugs']['mae_lo']:.2f}  weapon cov={r['weapon']['coverage']*100:>5.1f}% "
              f"MAE-lo={r['weapon']['mae_lo']:.2f}")
df_mk = pd.DataFrame(rows)
df_mk.to_csv("/tmp/sweep_min_k.csv", index=False)

print("\n✅ Saved: /tmp/sweep_K.csv, /tmp/sweep_source.csv, /tmp/sweep_min_k.csv")
