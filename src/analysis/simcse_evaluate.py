#!/usr/bin/env python3
"""
SimCSE filter — first-look evaluation.

Compares the unsupervised SimCSE embeddings against the LLM-panel similarity
scores and the citation-network filter.

Three checks:
  1. SANITY            — Spearman ρ between cosine and LLM-panel score
                         (over the 140K pairs we have LLM scores for)
  2. PER-CITATION-TYPE — cosine distribution stratified by citation_type
                         (1hop / 2hop / cocite / none) — does cosine see
                         what the citation graph sees?
  3. COVERAGE @ K      — for each in-set verdict, how many of its top-K cosine
                         neighbors are also in the citation-filter set?
                         Also: distribution of LLM scores over those neighbours.

NOTE: We only have LLM scores for citation-filtered pairs (~140K). We CANNOT
directly evaluate "did the embedding find new high-quality pairs the citation
filter missed" from this data alone — that would require re-scoring with LLMs.
What we CAN do: show that the embedding learned something sensible.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
EMB  = ROOT / "experiments/simcse_outputs/verdict_embeddings.npy"
IDX  = ROOT / "experiments/simcse_outputs/verdict_index.csv"
LLM  = ROOT / "experiments/data_per_domain/similarity_scores_combined.csv"
CIT  = ROOT / "experiments/data_per_domain/network_analysis/citation_pair_types.csv"
OUT  = ROOT / "experiments/results/0_preprocessing/embedding_filter"
OUT.mkdir(parents=True, exist_ok=True)


def load_all():
    emb = np.load(EMB).astype(np.float32)
    idx = pd.read_csv(IDX)
    v2i = {v: i for i, v in enumerate(idx.verdict)}
    llm = pd.read_csv(LLM)
    cit = pd.read_csv(CIT)
    print(f"Loaded:")
    print(f"  embeddings : {emb.shape}, mean row-norm={np.linalg.norm(emb, axis=1).mean():.3f}")
    print(f"  index      : {len(idx):,} verdicts ({idx.domain.value_counts().to_dict()})")
    print(f"  LLM pairs  : {len(llm):,}")
    print(f"  citation   : {len(cit):,} (types: {cit.citation_type.value_counts().head().to_dict()})")
    return emb, idx, v2i, llm, cit


def cosine_for_pairs(emb, v2i, df):
    """Lookup embeddings by id and compute cosine for each pair. Drop pairs
    where either id is missing from the embedding index."""
    a_idx = df.verdict_1.map(v2i)
    b_idx = df.verdict_2.map(v2i)
    keep  = a_idx.notna() & b_idx.notna()
    n_drop = (~keep).sum()
    if n_drop:
        print(f"    (dropped {n_drop:,} pairs missing from embedding index)")
    df = df[keep].copy()
    a = emb[a_idx[keep].astype(int).values]
    b = emb[b_idx[keep].astype(int).values]
    df["cosine"] = (a * b).sum(axis=1)   # already L2-normalized
    return df


# ---------- Check 1: sanity ----------
def check1_sanity(emb, v2i, llm):
    print("\n=== 1. SANITY: cosine vs LLM-panel score ===")
    df = cosine_for_pairs(emb, v2i, llm[["verdict_1","verdict_2","domain","similarity_score"]])

    for dom in ["drugs", "weapon"]:
        sub = df[df.domain == dom]
        rho, p = spearmanr(sub.cosine, sub.similarity_score)
        print(f"  {dom:7s}: n={len(sub):>6,}  Spearman ρ = {rho:+.3f}  (p={p:.1e})")

    rho, p = spearmanr(df.cosine, df.similarity_score)
    print(f"  ALL    : n={len(df):>6,}  Spearman ρ = {rho:+.3f}  (p={p:.1e})")

    bins = pd.cut(df.similarity_score, [-0.01, 0.5, 1.5, 2.5, 3.01],
                  labels=["~0", "1", "2", "3"])
    g = df.groupby(bins, observed=True).cosine.agg(["mean","std","count"])
    print(f"\n  Cosine by LLM-score bucket:")
    print(g.round(3).to_string())
    return df


# ---------- Check 2: per-citation-type ----------
def check2_per_citation(emb, v2i, cit):
    print("\n=== 2. COSINE BY CITATION TYPE ===")
    df = cosine_for_pairs(emb, v2i, cit[["verdict_1","verdict_2","domain","citation_type","similarity_score"]])
    df["primary"] = df.citation_type.str.split(",").str[0]   # use first label

    g = df.groupby("primary", observed=True).agg(
        n=("cosine","count"),
        cosine_mean=("cosine","mean"),
        cosine_std =("cosine","std"),
        llm_mean   =("similarity_score","mean"),
    ).sort_values("cosine_mean", ascending=False)
    print(g.round(3).to_string())
    print("\n  Interpretation: if SimCSE learned something legal-relevant,")
    print("  citation-connected pairs (1hop/2hop/cocite) should have higher")
    print("  cosine than 'none' pairs.")
    return df


# ---------- Check 3: coverage @ K ----------
def check3_coverage(emb, idx, v2i, cit, llm, k_values=(5, 10, 20, 50)):
    print("\n=== 3. TOP-K COVERAGE ===")

    cit_pairs = set()
    for r in cit.itertuples(index=False):
        if r.citation_type != "none":
            a, b = sorted([r.verdict_1, r.verdict_2])
            cit_pairs.add((a, b))
    print(f"  citation-filter positive pairs: {len(cit_pairs):,}")

    llm_score = {}
    for r in llm.itertuples(index=False):
        a, b = sorted([r.verdict_1, r.verdict_2])
        llm_score[(a, b)] = r.similarity_score
    overall_llm_mean = llm.similarity_score.mean()
    high_thr = llm.similarity_score.quantile(0.75)
    print(f"  LLM score scale: 0-100, overall mean={overall_llm_mean:.1f}, "
          f"P75={high_thr:.0f}  (using P75 as 'high' threshold)")

    rows = []
    for dom in ["drugs", "weapon"]:
        dom_mask = (idx.domain == dom).values
        dom_emb  = emb[dom_mask]
        dom_ids  = idx.verdict.values[dom_mask]
        sims     = dom_emb @ dom_emb.T
        np.fill_diagonal(sims, -1)

        n = len(dom_ids)
        K = max(k_values)
        topK = np.argpartition(-sims, K, axis=1)[:, :K]
        for i in range(n):
            order = topK[i][np.argsort(-sims[i, topK[i]])]
            topK[i] = order

        for k in k_values:
            overlap_pairs   = set()
            llm_picked      = []
            for i in range(n):
                src = dom_ids[i]
                for j in topK[i, :k]:
                    tgt = dom_ids[j]
                    a, b = sorted([src, tgt])
                    if (a, b) in cit_pairs:
                        overlap_pairs.add((a, b))
                    if (a, b) in llm_score:
                        llm_picked.append(llm_score[(a, b)])
            n_pairs_emb = (n * k) // 2   # each pair counted from both sides
            llm_mean    = np.mean(llm_picked) if llm_picked else float("nan")
            llm_p_high  = (np.array(llm_picked) >= high_thr).mean() if llm_picked else float("nan")

            rows.append({
                "domain": dom, "K": k,
                "n_emb_pairs": n_pairs_emb,
                "overlap_with_citation": len(overlap_pairs),
                "pct_overlap": round(100 * len(overlap_pairs) / n_pairs_emb, 2) if n_pairs_emb else 0,
                "in_LLM_set": len(llm_picked) // 2,
                "LLM_mean_picked":    round(llm_mean, 1),
                "LLM_pct_above_P75":  round(100 * llm_p_high, 1),
            })
    cov = pd.DataFrame(rows)
    print(cov.to_string(index=False))
    print(f"\n  Reading the table:")
    print(f"  - LLM_mean_picked: of SimCSE's top-K pairs that have an LLM score,")
    print(f"    what's their mean LLM score? Compare to overall mean = {overall_llm_mean:.1f}.")
    print(f"  - LLM_pct_above_P75: % of those scored pairs above the global P75={high_thr:.0f}.")
    print(f"    A random sample would be ~25%; higher = SimCSE picks better-than-random.")
    return cov


def main():
    emb, idx, v2i, llm, cit = load_all()
    df1 = check1_sanity(emb, v2i, llm)
    df2 = check2_per_citation(emb, v2i, cit)
    cov = check3_coverage(emb, idx, v2i, cit, llm)

    df1.to_csv(OUT / "eval_1_sanity_pairs.csv", index=False)
    df2.to_csv(OUT / "eval_2_by_citation_type.csv", index=False)
    cov.to_csv(OUT / "eval_3_coverage_topK.csv", index=False)
    print(f"\n💾 Saved → {OUT}/eval_*.csv")


if __name__ == "__main__":
    main()
