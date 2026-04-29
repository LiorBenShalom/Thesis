#!/usr/bin/env python3
"""
TF-IDF cosine similarity for the 85K pairs in similarity_scores.csv,
on indictment_facts text from verdicts_clean.csv.

Output: similarity_scores_tfidf.csv (verdict_1, verdict_2, domain, similarity_score in [0, 100]).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!")
EXP = ROOT / "new_try/experiments"
SIM_CSV = EXP / "data_per_domain/similarity_scores_combined.csv"
VERDICTS = ROOT / "new_try/innovation_submission/data_master_final/verdicts_clean.csv"
OUT_CSV = EXP / "data_per_domain/similarity_scores_tfidf_combined.csv"


def main():
    pairs = pd.read_csv(SIM_CSV, usecols=["verdict_1", "verdict_2", "domain"])
    clean = pd.read_csv(VERDICTS)

    txt_map = {}
    for _, r in clean.iterrows():
        cid = r["canonical_id"]
        if cid in txt_map:
            continue
        t = r.get("indictment_facts") or r.get("indictment_facts_raw") or ""
        if isinstance(t, str) and t.strip():
            txt_map[cid] = t

    unique_ids = sorted(set(pairs["verdict_1"]).union(pairs["verdict_2"]))
    have_text = [v for v in unique_ids if v in txt_map]
    print(f"unique verdicts: {len(unique_ids):,}, with text: {len(have_text):,}")

    texts = [txt_map[v] for v in have_text]
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    print(f"TF-IDF matrix: {X.shape}")

    idx = {v: i for i, v in enumerate(have_text)}
    scores = np.full(len(pairs), np.nan, dtype=np.float32)
    for i, (v1, v2) in enumerate(zip(pairs["verdict_1"], pairs["verdict_2"])):
        if v1 in idx and v2 in idx:
            r1 = X[idx[v1]]
            r2 = X[idx[v2]]
            cos = float((r1 @ r2.T).toarray()[0, 0])
            scores[i] = max(0.0, min(1.0, cos)) * 100.0

    out = pairs.copy()
    out["similarity_score"] = scores
    n_valid = int((~out["similarity_score"].isna()).sum())
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV}  ({n_valid:,}/{len(out):,} pairs scored)")


if __name__ == "__main__":
    main()
