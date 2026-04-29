#!/usr/bin/env python3
"""
Random-K null baseline: shuffle the similarity_scores.csv `similarity_score`
column within each domain. Predictions made from these random scores are the
chance floor for the kNN+selective-prediction pipeline.

Output: similarity_scores_random.csv (same schema as input).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!")
EXP = ROOT / "new_try/experiments"
SIM_CSV = EXP / "data_per_domain/similarity_scores_combined.csv"
OUT_CSV = EXP / "data_per_domain/similarity_scores_random_combined.csv"
SEED = 42


def main():
    df = pd.read_csv(SIM_CSV)
    rng = np.random.default_rng(SEED)
    out_parts = []
    for dom, sub in df.groupby("domain"):
        sub = sub.copy()
        idx = sub.index.to_numpy().copy()
        rng.shuffle(idx)
        sub["similarity_score"] = df.loc[idx, "similarity_score"].to_numpy()
        out_parts.append(sub)
    out = pd.concat(out_parts).sort_index()
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV}  ({len(out):,} pairs, shuffled within domain)")


if __name__ == "__main__":
    main()
