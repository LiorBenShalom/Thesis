"""
Phase A v2 — RICH per-query raw data.

For each (test query × method × K), saves:
  - predicted (low, high)
  - error (low, high)
  - picked neighbors (JSON list of verdict IDs)
  - mean neighbor LLM score
  - mean neighbor cosine sim (where applicable)

Methods (retrieval-based, supports K-sweep):
  - random_llm
  - citation_llm
  - sup_only (cosine top-K)
  - sup_llm (cosine top-100 → LLM rerank → top-K)
  - llm_best (LLM-only top-K)
  - bm25 (top-K)
  - offense_matched_random (top-K random from offense-matched)

Methods (direct, no K):
  - global_median
  - tfidf_ridge

K values: 1, 3, 5, 7, 10, 15, 20, 30, 50

Output: /tmp/rigor_raw_per_query_K.csv  (~200K rows)
"""
from pathlib import Path
import json, re, time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from rank_bm25 import BM25Okapi

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
EXP  = ROOT / "experiments"
FILTERED_DIR = ROOT / "simcse_cuda_bundle/outputs_supervised_filtered"

N_FOLDS = 5
K_VALUES = [1, 3, 5, 7, 10, 15, 20, 30, 50]
TOP_POOL = 100  # for sup+LLM (max top-pool)
RANDOM_POOL = 50  # for random+LLM

# Load sentencing
sup_csv = ROOT / "simcse_cuda_bundle/data/supervised_data.csv"
sup = pd.read_csv(sup_csv)
v_to_text = dict(zip(sup.verdict, sup.indictment_facts))
v_to_dom = dict(zip(sup.verdict, sup.domain))

m = pd.read_csv(EXP / "data_per_domain/master_inventory.csv",
                usecols=["canonical_id","domain","sentencing_range_low",
                         "sentencing_range_high","sentencing_confidence","year"])
m = m[m.domain.isin(["drugs","weapon"])
      & m.sentencing_range_low.notna()
      & (m.sentencing_confidence == "גבוהה")].drop_duplicates("canonical_id")
rng_lo = dict(zip(m.canonical_id, m.sentencing_range_low))
rng_hi = dict(zip(m.canonical_id, m.sentencing_range_high))
year_of = dict(zip(m.canonical_id, m.year))
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
print(f"  {len(llm_scores):,}")

# Citation pairs
cit_pairs = set()
cit_df = pd.read_csv(EXP / "data_per_domain/network_analysis/citation_pair_types.csv")
for r in cit_df.itertuples(index=False):
    if r.citation_type in ("1hop", "2hop", "cocite"):
        cit_pairs.add(tuple(sorted([r.verdict_1, r.verdict_2])))

# H-Full offense sets
print("Loading H-Full features...")
with open(EXP / "data/sentencing_range-old/hfull_features/hybrid_full_cache.json") as f:
    hf = json.load(f)
def yesno(v):
    if v is None: return False
    s = str(v).strip()
    return s not in ("", "לא", "nan", "None", "0", "0.0")
def drugs_offense_set(feats):
    if not feats: return set()
    s = set()
    for sec in ("6","7","13","14","19"):
        if yesno(feats.get(f"section_{sec}")): s.add(f"sec_{sec}")
    if yesno(feats.get("other_drug_offense")): s.add("other")
    return s
WPAT = [(r"144\s*\(\s*א\s*\)","144a"),(r"144\s*\(\s*ב\s*2\s*\)","144b2"),
        (r"144\s*\(\s*ב\s*\)","144b"),(r"144\s*\(\s*ג\s*\)","144c"),
        (r"144\s*\(\s*ז\s*\)","144g"),(r"\b145\b","145"),(r"\b146\b","146")]
def weapon_offense_set(feats):
    if not feats: return set()
    blob = " ".join(str(feats.get(k, "")) for k in ("offense_number","offense_type","additional_offenses"))
    return {label for pat, label in WPAT if re.search(pat, blob)}
verdict_offenses = {v: (drugs_offense_set(hf.get(v, {})) if v_to_dom[v] == "drugs"
                       else weapon_offense_set(hf.get(v, {})))
                    for v in v_to_dom}

# Folds
print("Loading filtered embedding folds...")
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


def make_record(query, dom, fold, method, K, picked, true_lo, true_hi, year):
    """Build a single record from a list of picked verdict IDs."""
    valid = [p for p in picked if p in rng_lo]
    valid_K = valid[:K]
    if not valid_K:
        return None
    plo = float(np.median([rng_lo[p] for p in valid_K]))
    phi = float(np.median([rng_hi[p] for p in valid_K]))
    # mean LLM score
    llms = [llm_scores.get(tuple(sorted([query, p]))) for p in valid_K]
    llms = [s for s in llms if s is not None]
    mean_llm = float(np.mean(llms)) if llms else None
    return {
        "query": query, "domain": dom, "fold": fold, "year": year,
        "method": method, "K": K,
        "true_lo": true_lo, "true_hi": true_hi,
        "pred_lo": plo, "pred_hi": phi,
        "err_lo": abs(plo - true_lo), "err_hi": abs(phi - true_hi),
        "n_actual": len(valid_K),
        "neighbors": json.dumps(valid_K, ensure_ascii=False),
        "mean_llm_in_picked": mean_llm,
    }


records = []
print(f"\nComputing per-query × method × K (K_VALUES={K_VALUES})...")
t0 = time.time()

for (dom, fid), ff in folds.items():
    print(f"\n[{dom} fold {fid}] {time.time()-t0:.0f}s elapsed")
    emb = ff["emb"]; v2i = ff["v2i"]
    train_ids = ff["train_ids"]; test_ids = ff["test_ids"]
    train_idx_arr = np.array([v2i[v] for v in train_ids])

    # Pre-fit fold-level baselines
    valid_train = [v for v in train_ids if v in v_to_text and v in rng_lo]
    train_texts = [v_to_text[v] for v in valid_train]
    train_lows  = np.array([rng_lo[v] for v in valid_train])
    train_highs = np.array([rng_hi[v] for v in valid_train])

    print(f"  Fitting TF-IDF + Ridge on {len(valid_train)} train texts...")
    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2, max_features=50_000)
    X_train = tfidf.fit_transform(train_texts)
    ridge_lo = Ridge(alpha=10.0).fit(X_train, train_lows)
    ridge_hi = Ridge(alpha=10.0).fit(X_train, train_highs)

    print(f"  Fitting BM25 on {len(train_texts)} train texts...")
    train_tokens = [t.split() for t in train_texts]
    bm25 = BM25Okapi(train_tokens)

    for q in test_ids:
        if q not in rng_lo or q not in v_to_text: continue
        if q not in v2i: continue
        true_lo, true_hi = rng_lo[q], rng_hi[q]
        year = year_of.get(q)
        qi = v2i[q]
        q_text = v_to_text[q]

        # === Method: global_median (no K) ===
        plo, phi = glob_med[dom]
        records.append({
            "query": q, "domain": dom, "fold": fid, "year": year,
            "method": "global_median", "K": None,
            "true_lo": true_lo, "true_hi": true_hi,
            "pred_lo": plo, "pred_hi": phi,
            "err_lo": abs(plo - true_lo), "err_hi": abs(phi - true_hi),
            "n_actual": 0, "neighbors": "[]", "mean_llm_in_picked": None,
        })

        # === Method: TF-IDF + Ridge (no K) ===
        X_q = tfidf.transform([q_text])
        plo = float(ridge_lo.predict(X_q)[0])
        phi = float(ridge_hi.predict(X_q)[0])
        records.append({
            "query": q, "domain": dom, "fold": fid, "year": year,
            "method": "tfidf_ridge", "K": None,
            "true_lo": true_lo, "true_hi": true_hi,
            "pred_lo": plo, "pred_hi": phi,
            "err_lo": abs(plo - true_lo), "err_hi": abs(phi - true_hi),
            "n_actual": 0, "neighbors": "[]", "mean_llm_in_picked": None,
        })

        # === Pre-compute candidate orders for K-NN methods ===
        # Supervised cosine ranking
        sims = emb[qi] @ emb[train_idx_arr].T
        sup_order = [train_ids[i] for i in np.argsort(-sims)]

        # Sup+LLM: top-100 cosine, rerank by LLM
        sup_pool = sup_order[:TOP_POOL]
        sup_llm_scored = [(c, llm_scores.get(tuple(sorted([q, c])))) for c in sup_pool]
        sup_llm_scored = [(c, s) for c, s in sup_llm_scored if s is not None]
        sup_llm_scored.sort(key=lambda x: -x[1])
        sup_llm_order = [c for c, _ in sup_llm_scored]

        # LLM-best: all train candidates with LLM scores
        llm_best_scored = [(t, llm_scores.get(tuple(sorted([q, t])))) for t in train_ids if t != q]
        llm_best_scored = [(t, s) for t, s in llm_best_scored if s is not None]
        llm_best_scored.sort(key=lambda x: -x[1])
        llm_best_order = [t for t, _ in llm_best_scored]

        # Random + LLM
        rng = np.random.default_rng(hash(q + "rand") % 2**32)
        rand_idx = rng.permutation(len(train_ids))[:RANDOM_POOL]
        rand_cands = [train_ids[i] for i in rand_idx if train_ids[i] != q]
        rand_scored = [(c, llm_scores.get(tuple(sorted([q, c])))) for c in rand_cands]
        rand_scored = [(c, s) for c, s in rand_scored if s is not None]
        rand_scored.sort(key=lambda x: -x[1])
        random_llm_order = [c for c, _ in rand_scored]

        # Citation + LLM
        train_set = set(train_ids)
        cit_cands = [t for t in train_set if t != q and tuple(sorted([q, t])) in cit_pairs]
        cit_scored = [(c, llm_scores.get(tuple(sorted([q, c])))) for c in cit_cands]
        cit_scored = [(c, s) for c, s in cit_scored if s is not None]
        cit_scored.sort(key=lambda x: -x[1])
        citation_llm_order = [c for c, _ in cit_scored]

        # BM25
        q_tokens = q_text.split()
        bm25_scores = bm25.get_scores(q_tokens)
        bm25_order = [valid_train[i] for i in np.argsort(-bm25_scores)]

        # Offense-matched random
        q_off = verdict_offenses.get(q, set())
        if q_off:
            off_cands = [t for t in train_ids if t != q and (verdict_offenses.get(t, set()) & q_off)]
            rng2 = np.random.default_rng((hash(q) + 1) % 2**32)
            shuffled_idx = rng2.permutation(len(off_cands))
            offense_matched_order = [off_cands[i] for i in shuffled_idx]
        else:
            offense_matched_order = []

        # === For each K-NN method × K, record ===
        for K in K_VALUES:
            for method_name, order in [
                ("sup_only", sup_order),
                ("sup_llm", sup_llm_order),
                ("llm_best", llm_best_order),
                ("random_llm", random_llm_order),
                ("citation_llm", citation_llm_order),
                ("bm25", bm25_order),
                ("offense_matched_random", offense_matched_order),
            ]:
                rec = make_record(q, dom, fid, method_name, K, order, true_lo, true_hi, year)
                if rec is not None:
                    records.append(rec)

print(f"\nTotal: {len(records):,} records in {time.time()-t0:.0f}s")

df_out = pd.DataFrame(records)
df_out.to_csv("/tmp/rigor_raw_per_query_K.csv", index=False)
print(f"✅ /tmp/rigor_raw_per_query_K.csv")
print(f"   Size: {df_out.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB in memory")

# Quick MAE summary by K
print(f"\n=== MAE by K (sanity check) ===")
summary = df_out[df_out.method.isin(["sup_llm", "sup_only", "llm_best", "bm25"])].groupby(["domain","method","K"]).agg(
    n=("err_lo","size"), mae_lo=("err_lo","mean"), mae_hi=("err_hi","mean")
).reset_index()
print(summary.pivot_table(index=["domain","method"], columns="K", values="mae_lo").round(2))
