"""
THESIS FINAL: For multiple filter strategies, show how LLM rerank improves MAE.
The headline: LLM consistently adds value regardless of filter choice.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
EXP  = ROOT / "experiments"
FILTERED_DIR = ROOT / "simcse_cuda_bundle/outputs_supervised_filtered"
BASELINE_DIR = EXP / "simcse_outputs/supervised"

N_FOLDS = 5
K = 10

# Sentencing
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

# LLM scores
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

# Citation pairs
cit_pairs = {"1hop": set(), "2hop": set(), "cocite": set()}
cit_df = pd.read_csv(EXP / "data_per_domain/network_analysis/citation_pair_types.csv")
for r in cit_df.itertuples(index=False):
    a, b = sorted([r.verdict_1, r.verdict_2])
    if r.citation_type in cit_pairs:
        cit_pairs[r.citation_type].add((a, b))
cit_all = cit_pairs["1hop"] | cit_pairs["2hop"] | cit_pairs["cocite"]


def load_folds(emb_dir, suffix):
    folds = {}
    for dom in ("drugs", "weapon"):
        for f in range(1, N_FOLDS + 1):
            ep = emb_dir / f"verdict_embeddings_{dom}_topk_fold{f}{suffix}.npy"
            ip = emb_dir / f"verdict_index_{dom}_topk_fold{f}{suffix}.csv"
            if not ep.exists(): continue
            emb = np.load(ep).astype(np.float32)
            idx = pd.read_csv(ip)
            train_ids = idx[idx.split == "train"].verdict.tolist()
            test_ids  = idx[idx.split == "test"].verdict.tolist()
            v2i = {v: i for i, v in enumerate(idx.verdict)}
            folds[(dom, f)] = {"emb": emb, "v2i": v2i,
                               "train_ids": train_ids, "test_ids": test_ids}
    return folds


folds_base = load_folds(BASELINE_DIR, suffix="")
folds_filt = load_folds(FILTERED_DIR, suffix="_offenseFiltered")


def predict(filter_name, q, ff, K=K, use_llm=False, top_pool=None):
    """Returns picked neighbors (verdict ids) for prediction."""
    train_ids = ff["train_ids"]
    if filter_name == "global_median":
        return None  # special

    elif filter_name == "random":
        rng = np.random.default_rng(hash(q) % 2**32)
        idx = rng.permutation(len(train_ids))
        cands = [train_ids[i] for i in idx[:50] if train_ids[i] != q]
        if use_llm:
            scored = [(c, llm_scores.get(tuple(sorted([q, c])))) for c in cands]
            scored = [(c, s) for c, s in scored if s is not None]
            scored.sort(key=lambda x: -x[1])
            return [c for c, _ in scored[:K]]
        return cands[:K]

    elif filter_name.startswith("citation_"):
        types = filter_name.split("_", 1)[1].split("+")
        type_sets = [cit_pairs[t] for t in types]
        cands = []
        for t in train_ids:
            if t == q: continue
            key = tuple(sorted([q, t]))
            if any(key in s for s in type_sets):
                cands.append(t)
        if use_llm:
            scored = [(c, llm_scores.get(tuple(sorted([q, c])))) for c in cands]
            scored = [(c, s) for c, s in scored if s is not None]
            scored.sort(key=lambda x: -x[1])
            return [c for c, _ in scored[:K]]
        return cands[:K]

    elif filter_name in ("baseline_sup", "filtered_sup"):
        emb, v2i = ff["emb"], ff["v2i"]
        if q not in v2i: return []
        qi = v2i[q]
        train_idx_arr = np.array([v2i[v] for v in train_ids])
        sims = emb[qi] @ emb[train_idx_arr].T
        if use_llm:
            pool_size = top_pool or 100
            pool_idx = np.argsort(-sims)[:pool_size]
            pool = [train_ids[i] for i in pool_idx]
            scored = [(c, llm_scores.get(tuple(sorted([q, c])))) for c in pool]
            scored = [(c, s) for c, s in scored if s is not None]
            scored.sort(key=lambda x: -x[1])
            return [c for c, _ in scored[:K]]
        else:
            top_idx = np.argsort(-sims)[:K]
            return [train_ids[i] for i in top_idx]

    elif filter_name == "llm_oracle":
        cand = [(t, llm_scores.get(tuple(sorted([q, t])))) for t in train_ids if t != q]
        cand = [(t, s) for t, s in cand if s is not None]
        cand.sort(key=lambda x: -x[1])
        return [t for t, _ in cand[:K]]

    raise ValueError(filter_name)


def evaluate(filter_name, folds, use_llm=False, top_pool=None):
    rows = []
    for (dom, fid), ff in folds.items():
        lo_errs = []; hi_errs = []; n_pred = 0; n_total = 0
        for q in ff["test_ids"]:
            if q not in rng_lo: continue
            n_total += 1
            if filter_name == "global_median":
                pred_lo, pred_hi = glob_med[dom]
                lo_errs.append(abs(pred_lo - rng_lo[q]))
                hi_errs.append(abs(pred_hi - rng_hi[q]))
                n_pred += 1
                continue
            picked = predict(filter_name, q, ff, use_llm=use_llm, top_pool=top_pool)
            picked = [p for p in picked if p in rng_lo] if picked else []
            if not picked: continue
            pred_lo = float(np.median([rng_lo[p] for p in picked]))
            pred_hi = float(np.median([rng_hi[p] for p in picked]))
            lo_errs.append(abs(pred_lo - rng_lo[q]))
            hi_errs.append(abs(pred_hi - rng_hi[q]))
            n_pred += 1
        rows.append({"domain": dom, "fold": fid, "n_total": n_total, "n_pred": n_pred,
                     "coverage": n_pred/n_total if n_total else 0,
                     "mae_lo": np.mean(lo_errs) if lo_errs else None,
                     "mae_hi": np.mean(hi_errs) if hi_errs else None})
    return pd.DataFrame(rows)


def summarize(df, label):
    out = []
    for dom in ("drugs", "weapon"):
        sub = df[df.domain == dom]
        out.append({"config": label, "domain": dom,
                    "coverage_mean": sub.coverage.mean(),
                    "mae_lo": sub.mae_lo.mean(), "mae_lo_std": sub.mae_lo.std(),
                    "mae_hi": sub.mae_hi.mean(), "mae_hi_std": sub.mae_hi.std()})
    return pd.DataFrame(out)


# ===== Run all configurations =====
print(f"\n{'='*100}")
print(f" LLM value across multiple filters")
print(f"{'='*100}\n")
print(f"{'Strategy':45s} {'dom':6s} {'cov':>5s} {'MAE-lo':>14s} {'MAE-hi':>14s}")
print("-" * 90)

configs = [
    ("Global median (no similarity)",      "global_median",          False, None,  folds_filt),
    ("Random + median",                    "random",                 False, None,  folds_filt),
    ("Random + LLM rerank",                "random",                 True,  None,  folds_filt),
    ("Citation 1hop + median",             "citation_1hop",          False, None,  folds_filt),
    ("Citation 1hop + LLM rerank",         "citation_1hop",          True,  None,  folds_filt),
    ("Citation ALL + median",              "citation_1hop+2hop+cocite", False, None, folds_filt),
    ("Citation ALL + LLM rerank",          "citation_1hop+2hop+cocite", True,  None, folds_filt),
    ("Baseline supervised + median",       "baseline_sup",           False, None,  folds_base),
    ("Baseline supervised + LLM rerank",   "baseline_sup",           True,  100,   folds_base),
    ("Filtered supervised + median",       "filtered_sup",           False, None,  folds_filt),
    ("Filtered supervised + LLM rerank",   "filtered_sup",           True,  100,   folds_filt),
    ("LLM oracle (no filter, UB)",         "llm_oracle",             False, None,  folds_filt),
]

all_summaries = []
for label, fname, use_llm, top_pool, folds in configs:
    df = evaluate(fname, folds, use_llm=use_llm, top_pool=top_pool)
    s = summarize(df, label)
    all_summaries.append(s)
    for _, r in s.iterrows():
        if r.mae_lo is None:
            print(f"{label:45s} {r.domain:6s} {r.coverage_mean*100:>4.0f}% {'—':>14s} {'—':>14s}")
        else:
            print(f"{label:45s} {r.domain:6s} {r.coverage_mean*100:>4.0f}% "
                  f"{r.mae_lo:>5.2f} ± {r.mae_lo_std:>4.2f}  {r.mae_hi:>5.2f} ± {r.mae_hi_std:>4.2f}")
    print()

result = pd.concat(all_summaries, ignore_index=True)
result.to_csv("/tmp/llm_value_across_filters.csv", index=False)

# ===== LLM contribution per filter =====
print(f"\n{'='*100}")
print(f" LLM CONTRIBUTION — improvement when adding LLM rerank to each filter")
print(f"{'='*100}\n")
print(f"{'Filter':30s} {'dom':6s} {'no-LLM':>16s} {'+LLM':>16s} {'ΔMAE-lo':>10s} {'ΔMAE-hi':>10s} {'%lo':>7s} {'%hi':>7s}")
print("-" * 110)

pairs = [
    ("Random",               "Random + median",                "Random + LLM rerank"),
    ("Citation 1hop",        "Citation 1hop + median",         "Citation 1hop + LLM rerank"),
    ("Citation ALL",         "Citation ALL + median",          "Citation ALL + LLM rerank"),
    ("Baseline supervised",  "Baseline supervised + median",   "Baseline supervised + LLM rerank"),
    ("Filtered supervised",  "Filtered supervised + median",   "Filtered supervised + LLM rerank"),
]
for name, before_label, after_label in pairs:
    bef = result[result.config == before_label]
    aft = result[result.config == after_label]
    for dom in ("drugs", "weapon"):
        b = bef[bef.domain == dom].iloc[0]
        a = aft[aft.domain == dom].iloc[0]
        if b.mae_lo is None or a.mae_lo is None: continue
        dlo = a.mae_lo - b.mae_lo
        dhi = a.mae_hi - b.mae_hi
        plo = dlo / b.mae_lo * 100
        phi = dhi / b.mae_hi * 100
        print(f"{name:30s} {dom:6s} "
              f"{b.mae_lo:>5.2f}/{b.mae_hi:>5.2f}  "
              f"{a.mae_lo:>5.2f}/{a.mae_hi:>5.2f}  "
              f"{dlo:>+8.2f}  {dhi:>+8.2f}  {plo:>+6.1f}% {phi:>+6.1f}%")

print(f"\n✅ /tmp/llm_value_across_filters.csv")
