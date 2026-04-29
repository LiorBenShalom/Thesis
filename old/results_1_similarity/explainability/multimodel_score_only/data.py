"""
Load (rep, domain) → list of pair tuples (verdict_1, verdict_2, GT, fv1, fv2).

For 'facts' kind: fv1/fv2 = indicment_facts text.
For 'features' kind: fv1/fv2 = JSON feature_vector text (kept as-is).
"""
from __future__ import annotations
import pandas as pd
from typing import Iterator
from .config import REPS, domain_dir


def iter_rep_pairs(rep_id: str, kind: str, domain: str) -> Iterator[dict]:
    fname = next((f for r, f, k in REPS if r == rep_id), None)
    if fname is None:
        raise ValueError(f"unknown rep {rep_id}")
    df = pd.read_csv(domain_dir(domain) / fname)
    if kind == "facts":
        # facts CSV has 'indicment_facts_1' / 'indicment_facts_2' columns
        if "indicment_facts_1" in df.columns:
            fv1_col, fv2_col = "indicment_facts_1", "indicment_facts_2"
        elif "feature_vector_1" in df.columns:
            fv1_col, fv2_col = "feature_vector_1", "feature_vector_2"
        else:
            raise RuntimeError(f"facts rep missing facts/feature_vector columns: {df.columns.tolist()}")
    else:
        fv1_col, fv2_col = "feature_vector_1", "feature_vector_2"

    for idx, row in df.iterrows():
        yield {
            "pair_id": int(idx),
            "verdict_1": row["verdict_1"],
            "verdict_2": row["verdict_2"],
            "gt": int(row["similarity_scale"]),
            "fv1": row[fv1_col],
            "fv2": row[fv2_col],
        }


def load_all_pairs() -> dict:
    """Returns: {(rep_id, domain): [pair_dicts]}"""
    out = {}
    for rep_id, _, kind in REPS:
        for domain in ["drugs", "weapon"]:
            out[(rep_id, domain)] = list(iter_rep_pairs(rep_id, kind, domain))
    return out
