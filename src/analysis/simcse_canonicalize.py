#!/usr/bin/env python3
"""
Re-label the SimCSE embedding index with canonical Hebrew verdict IDs and
deduplicate. Saves new files alongside the originals (originals kept as .raw).

Why: training data (verdicts_master.csv) mixed court-system IDs (SH-/ME-/ST-)
with canonical Hebrew IDs (תפ_/ת"פ/תפח_). Same verdict appears under multiple
keys → ~1,680 duplicates. Re-labeling collapses them to 6,766 unique verdicts
in canonical form, matching the IDs used in similarity_scores_combined.csv.
"""
from __future__ import annotations
import re, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
OUT  = ROOT / "experiments/simcse_outputs"
ALIAS = ROOT / "innovation_submission/data_master_final/verdict_alias.csv"


def canonical(s):
    if not s or pd.isna(s): return ""
    s = unicodedata.normalize("NFKC", str(s).strip())
    s = re.sub(r'["\'״׳`]', '', s)
    s = re.sub(r'\(.*?\)', '', s)
    s = re.sub(r'[\s/∕\\.]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_- ')
    return s


def main():
    alias = pd.read_csv(ALIAS)
    orig_to_canon = dict(zip(alias.original_id.astype(str), alias.canonical_id.astype(str)))
    canon_set = set(alias.canonical_id.astype(str))

    def best_lookup(t):
        if not t: return None
        if t in canon_set: return t
        c = canonical(t)
        if c in canon_set: return c
        a = orig_to_canon.get(t)
        if a:
            if a in canon_set: return a
            ac = canonical(a)
            if ac in canon_set: return ac
        return None

    emb = np.load(OUT / "verdict_embeddings.npy")
    idx = pd.read_csv(OUT / "verdict_index.csv")
    print(f"Loaded {len(idx):,} rows, embeddings shape {emb.shape}")

    idx["canonical_id"] = idx.verdict.map(best_lookup)
    n_failed = idx.canonical_id.isna().sum()
    print(f"  canonicalized: {(~idx.canonical_id.isna()).sum():,}  failed: {n_failed:,}")
    if n_failed:
        print(f"  sample failed IDs: {idx[idx.canonical_id.isna()].verdict.head().to_list()}")

    # Drop rows that failed to canonicalize
    keep = idx.canonical_id.notna()
    idx2 = idx[keep].reset_index(drop=True)
    emb2 = emb[keep.values]

    # Dedupe — keep first occurrence per canonical_id
    before = len(idx2)
    first  = ~idx2.canonical_id.duplicated(keep="first")
    idx2   = idx2[first].reset_index(drop=True)
    emb2   = emb2[first.values]
    n_dup  = before - len(idx2)
    print(f"  dedupe: removed {n_dup:,} duplicate canonical IDs → {len(idx2):,} unique verdicts")

    # Backup originals once, then overwrite
    raw_emb = OUT / "verdict_embeddings.raw.npy"
    raw_idx = OUT / "verdict_index.raw.csv"
    if not raw_emb.exists():
        np.save(raw_emb, emb)
        idx.to_csv(raw_idx, index=False)
        print(f"  backed up originals → *.raw.{{npy,csv}}")

    # Save canonical version (overwrite the originals so downstream picks them up)
    out_idx = pd.DataFrame({
        "verdict": idx2.canonical_id.values,    # now canonical Hebrew
        "domain":  idx2.domain.values,
    })
    out_idx.to_csv(OUT / "verdict_index.csv", index=False)
    np.save(OUT / "verdict_embeddings.npy", emb2.astype(np.float32))
    print(f"\n💾 saved canonical:")
    print(f"  {OUT/'verdict_index.csv'}        ({len(out_idx):,} rows, all-Hebrew IDs)")
    print(f"  {OUT/'verdict_embeddings.npy'}   shape {emb2.shape}")
    print(f"\n  domain distribution:")
    print(f"  {out_idx.domain.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
