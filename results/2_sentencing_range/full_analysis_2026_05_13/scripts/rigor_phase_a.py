"""
Phase A — Compute per-query (test) errors for ALL methods, save to a single CSV.
Methods:
  - global_median
  - random + LLM rerank
  - citation + LLM rerank (all types)
  - supervised (cosine top-10)
  - supervised + LLM rerank (top-100 → top-10)
  - LLM-best (top-10 from existing LLM-scored pool)
  - TF-IDF + Ridge regression (direct prediction)
  - BM25 + median-of-top-K
  - Offense-matched random + median

For each method × test query: error_low, error_high (absolute).
Output: /tmp/rigor_per_query_errors.csv
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
SIMCSE_DIR   = ROOT / "simcse_cuda_bundle/outputs_simcse_5fold"

N_FOLDS = 5
K = 10
TOP_POOL = 100

# ============ Load sentencing ============
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

# ============ LLM scores ============
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

# ============ Citation pairs ============
cit_pairs = set()
cit_df = pd.read_csv(EXP / "data_per_domain/network_analysis/citation_pair_types.csv")
for r in cit_df.itertuples(index=False):
    if r.citation_type in ("1hop", "2hop", "cocite"):
        cit_pairs.add(tuple(sorted([r.verdict_1, r.verdict_2])))

# ============ H-Full features (for offense-matched) ============
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

WPAT = [
    (r"144\s*\(\s*א\s*\)",  "144a"),
    (r"144\s*\(\s*ב\s*2\s*\)","144b2"),
    (r"144\s*\(\s*ב\s*\)","144b"),
    (r"144\s*\(\s*ג\s*\)","144c"),
    (r"144\s*\(\s*ז\s*\)","144g"),
    (r"\b145\b","145"),
    (r"\b146\b","146"),
]
def weapon_offense_set(feats):
    if not feats: return set()
    blob = " ".join(str(feats.get(k, "")) for k in ("offense_number","offense_type","additional_offenses"))
    return {label for pat, label in WPAT if re.search(pat, blob)}

verdict_offenses = {}
for v in v_to_dom:
    feats = hf.get(v, {})
    s = drugs_offense_set(feats) if v_to_dom[v] == "drugs" else weapon_offense_set(feats)
    verdict_offenses[v] = s

# ============ Load folds ============
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
        rec = {"emb": emb, "v2i": v2i,
               "train_ids": train_ids, "test_ids": test_ids}
        # SimCSE fold embeddings (same split, own verdict order) — Methods 10/11
        sep = SIMCSE_DIR / f"verdict_embeddings_simcse_{dom}_fold{f}.npy"
        sip = SIMCSE_DIR / f"verdict_index_simcse_{dom}_fold{f}.csv"
        if sep.exists() and sip.exists():
            semb = np.load(sep).astype(np.float32)
            sidx = pd.read_csv(sip); sidx["verdict"] = sidx.verdict.astype(str)
            sv2i = {v: i for i, v in enumerate(sidx.verdict)}
            s_train = sidx[sidx.split == "train"].verdict.tolist()
            rec.update(sim_emb=semb, sim_v2i=sv2i, sim_train_ids=s_train,
                       sim_tarr=np.array([sv2i[v] for v in s_train]))
        folds[(dom, f)] = rec


# ============ Helper: median prediction ============
def median_pred(picked):
    """Return (pred_low, pred_high) as median of picked verdicts' ranges."""
    valid = [p for p in picked if p in rng_lo]
    if not valid: return None, None
    return (float(np.median([rng_lo[p] for p in valid])),
            float(np.median([rng_hi[p] for p in valid])))


# ============ Per-query computation for each method ============
print("\nComputing per-query errors for all methods (this takes ~10 min)...")
records = []
t0 = time.time()

for (dom, fid), ff in folds.items():
    print(f"\n[{dom} fold {fid}] {time.time()-t0:.0f}s elapsed")
    emb = ff["emb"]; v2i = ff["v2i"]
    train_ids = ff["train_ids"]; test_ids = ff["test_ids"]
    train_idx_arr = np.array([v2i[v] for v in train_ids])

    # --- Pre-fit TF-IDF + Ridge for this fold ---
    train_texts = [v_to_text[v] for v in train_ids if v in v_to_text]
    train_lows  = [rng_lo[v] for v in train_ids if v in v_to_text and v in rng_lo]
    train_highs = [rng_hi[v] for v in train_ids if v in v_to_text and v in rng_hi]
    # only use train_ids that have all 3
    valid_train = [v for v in train_ids if v in v_to_text and v in rng_lo]
    train_texts = [v_to_text[v] for v in valid_train]
    train_lows  = np.array([rng_lo[v] for v in valid_train])
    train_highs = np.array([rng_hi[v] for v in valid_train])

    print(f"  Fitting TF-IDF (3-5 char ngrams) + Ridge on {len(valid_train)} train texts...")
    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2, max_features=50_000)
    X_train = tfidf.fit_transform(train_texts)
    ridge_lo = Ridge(alpha=10.0).fit(X_train, train_lows)
    ridge_hi = Ridge(alpha=10.0).fit(X_train, train_highs)

    print(f"  Fitting BM25 on {len(train_texts)} train texts...")
    # Tokenize with whitespace (naive for Hebrew, but standard)
    train_tokens = [t.split() for t in train_texts]
    bm25 = BM25Okapi(train_tokens)

    # --- For each test query, compute all methods ---
    for q_i, q in enumerate(test_ids):
        if q not in rng_lo or q not in v_to_text: continue
        if q not in v2i: continue
        true_lo, true_hi = rng_lo[q], rng_hi[q]
        qi = v2i[q]
        q_text = v_to_text[q]

        # === Method 1: global_median ===
        plo, phi = glob_med[dom]
        records.append({
            "query": q, "domain": dom, "fold": fid, "year": year_of.get(q),
            "method": "global_median",
            "err_lo": abs(plo - true_lo), "err_hi": abs(phi - true_hi)
        })

        # === Method 2: random + LLM ===
        rng = np.random.default_rng(hash(q + "rand") % 2**32)
        rand_idx = rng.permutation(len(train_ids))[:50]
        rand_cands = [train_ids[i] for i in rand_idx if train_ids[i] != q]
        scored = [(c, llm_scores.get(tuple(sorted([q, c])))) for c in rand_cands]
        scored = [(c, s) for c, s in scored if s is not None]
        scored.sort(key=lambda x: -x[1])
        picked = [c for c, _ in scored[:K]]
        plo, phi = median_pred(picked)
        if plo is not None:
            records.append({
                "query": q, "domain": dom, "fold": fid, "year": year_of.get(q),
                "method": "random_llm",
                "err_lo": abs(plo - true_lo), "err_hi": abs(phi - true_hi)
            })

        # === Method 3: citation + LLM ===
        train_set = set(train_ids)
        cit_cands = [t for t in train_set if t != q and tuple(sorted([q, t])) in cit_pairs]
        scored = [(c, llm_scores.get(tuple(sorted([q, c])))) for c in cit_cands]
        scored = [(c, s) for c, s in scored if s is not None]
        scored.sort(key=lambda x: -x[1])
        picked = [c for c, _ in scored[:K]]
        plo, phi = median_pred(picked)
        if plo is not None:
            records.append({
                "query": q, "domain": dom, "fold": fid, "year": year_of.get(q),
                "method": "citation_llm",
                "err_lo": abs(plo - true_lo), "err_hi": abs(phi - true_hi)
            })

        # === Method 4: supervised cosine top-K (no LLM) ===
        sims = emb[qi] @ emb[train_idx_arr].T
        order = np.argsort(-sims)
        sup_top10 = [train_ids[i] for i in order[:K]]
        plo, phi = median_pred(sup_top10)
        if plo is not None:
            records.append({
                "query": q, "domain": dom, "fold": fid, "year": year_of.get(q),
                "method": "sup_only",
                "err_lo": abs(plo - true_lo), "err_hi": abs(phi - true_hi)
            })

        # === Method 5: supervised + LLM rerank (top-100 → top-10) ===
        sup_pool = [train_ids[i] for i in order[:TOP_POOL]]
        scored = [(c, llm_scores.get(tuple(sorted([q, c])))) for c in sup_pool]
        scored = [(c, s) for c, s in scored if s is not None]
        scored.sort(key=lambda x: -x[1])
        picked = [c for c, _ in scored[:K]]
        plo, phi = median_pred(picked)
        if plo is not None:
            records.append({
                "query": q, "domain": dom, "fold": fid, "year": year_of.get(q),
                "method": "sup_llm",
                "err_lo": abs(plo - true_lo), "err_hi": abs(phi - true_hi)
            })

        # === Method 6: LLM-best (over all fold-train) ===
        scored_all = [(t, llm_scores.get(tuple(sorted([q, t]))))
                      for t in train_ids if t != q]
        scored_all = [(t, s) for t, s in scored_all if s is not None]
        scored_all.sort(key=lambda x: -x[1])
        picked = [t for t, _ in scored_all[:K]]
        plo, phi = median_pred(picked)
        if plo is not None:
            records.append({
                "query": q, "domain": dom, "fold": fid, "year": year_of.get(q),
                "method": "llm_best",
                "err_lo": abs(plo - true_lo), "err_hi": abs(phi - true_hi)
            })

        # === Method 7 (NEW): TF-IDF + Ridge (direct prediction) ===
        X_q = tfidf.transform([q_text])
        plo = float(ridge_lo.predict(X_q)[0])
        phi = float(ridge_hi.predict(X_q)[0])
        records.append({
            "query": q, "domain": dom, "fold": fid, "year": year_of.get(q),
            "method": "tfidf_ridge",
            "err_lo": abs(plo - true_lo), "err_hi": abs(phi - true_hi)
        })

        # === Method 8 (NEW): BM25 + median-of-top-K ===
        q_tokens = q_text.split()
        bm25_scores = bm25.get_scores(q_tokens)
        order_bm25 = np.argsort(-bm25_scores)[:K]
        picked = [valid_train[i] for i in order_bm25]
        plo, phi = median_pred(picked)
        if plo is not None:
            records.append({
                "query": q, "domain": dom, "fold": fid, "year": year_of.get(q),
                "method": "bm25",
                "err_lo": abs(plo - true_lo), "err_hi": abs(phi - true_hi)
            })

        # === Method 9 (NEW): Offense-matched random ===
        q_off = verdict_offenses.get(q, set())
        if q_off:
            off_match_cands = [t for t in train_ids
                               if t != q and (verdict_offenses.get(t, set()) & q_off)]
            if len(off_match_cands) >= K:
                rng2 = np.random.default_rng((hash(q) + 1) % 2**32)
                sampled_idx = rng2.permutation(len(off_match_cands))[:K]
                picked = [off_match_cands[i] for i in sampled_idx]
            else:
                picked = off_match_cands
            plo, phi = median_pred(picked) if picked else (None, None)
            if plo is not None:
                records.append({
                    "query": q, "domain": dom, "fold": fid, "year": year_of.get(q),
                    "method": "offense_matched_random",
                    "err_lo": abs(plo - true_lo), "err_hi": abs(phi - true_hi)
                })

        # === Method 10: SimCSE cosine top-K (no LLM) ===
        if "sim_emb" in ff and q in ff["sim_v2i"]:
            s_emb = ff["sim_emb"]; s_v2i = ff["sim_v2i"]
            s_train_ids = ff["sim_train_ids"]; s_tarr = ff["sim_tarr"]
            s_sims = s_emb[s_v2i[q]] @ s_emb[s_tarr].T
            s_order = np.argsort(-s_sims)
            picked = [s_train_ids[i] for i in s_order[:K]]
            plo, phi = median_pred(picked)
            if plo is not None:
                records.append({
                    "query": q, "domain": dom, "fold": fid, "year": year_of.get(q),
                    "method": "simcse_only",
                    "err_lo": abs(plo - true_lo), "err_hi": abs(phi - true_hi)
                })

            # === Method 11: SimCSE + LLM rerank (top-100 → top-10) ===
            s_pool = [s_train_ids[i] for i in s_order[:TOP_POOL]]
            scored = [(c, llm_scores.get(tuple(sorted([q, c])))) for c in s_pool]
            scored = [(c, s) for c, s in scored if s is not None]
            scored.sort(key=lambda x: -x[1])
            picked = [c for c, _ in scored[:K]]
            plo, phi = median_pred(picked)
            if plo is not None:
                records.append({
                    "query": q, "domain": dom, "fold": fid, "year": year_of.get(q),
                    "method": "simcse_llm",
                    "err_lo": abs(plo - true_lo), "err_hi": abs(phi - true_hi)
                })

df_out = pd.DataFrame(records)
df_out.to_csv("/tmp/rigor_per_query_errors.csv", index=False)
print(f"\n✅ /tmp/rigor_per_query_errors.csv  ({len(df_out):,} rows)")
print(f"Total elapsed: {time.time()-t0:.0f}s")

# Quick sanity check — MAE per method per domain
print("\n=== Sanity check: MAE per method per domain ===")
g = df_out.groupby(["domain", "method"]).agg(
    n=("err_lo", "size"),
    mae_lo=("err_lo", "mean"),
    mae_hi=("err_hi", "mean"),
).reset_index()
print(g.to_string(index=False))
