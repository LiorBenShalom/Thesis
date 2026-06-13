"""Shared loader for representation feature-vector analysis.

Each similarity_database_*.csv stores VERDICT PAIRS. We dedupe per verdict id
across both columns/all rows, and return verdict_id -> feature dict per (domain, rep).
"""
from __future__ import annotations
import csv, json, os, re
from pathlib import Path

csv.field_size_limit(10**9)

ROOT = Path(__file__).resolve().parents[2]  # .../new_try

# rep_key -> filename (same filename used in both drugs/ and weapon/ dirs)
REPS = {
    "Hybrid-Manual": "similarity_database_hybrid.csv",
    "Hybrid-GPT":    "similarity_database_hybrid_gpt.csv",   # intermediate, kept for reference
    "Hybrid-Full":   "similarity_database_hybrid_full_gpt.csv",
    "GPT-Law":       "similarity_database_with_gpt_law_features.csv",
    "GPT-Free":      "similarity_database_with_gpt_features.csv",
}
# the 4 the user asked about (Hybrid-GPT included only as context)
FOCUS = ["Hybrid-Manual", "Hybrid-Full", "GPT-Law", "GPT-Free"]
DOMAINS = ["drugs", "weapon"]


def _clean_key(s: str) -> str:
    if s is None:
        return s
    return s.strip().lstrip("﻿").strip('"')


def _parse_json(cell: str):
    if not cell:
        return None
    t = cell.strip()
    if not t.startswith("{"):
        # find first {...}
        i, j = t.find("{"), t.rfind("}")
        if i == -1 or j == -1:
            return None
        t = t[i:j + 1]
    for attempt in (t, re.sub(r",\s*}", "}", t)):
        try:
            d = json.loads(attempt, strict=False)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return None


def load_rep(domain: str, rep: str) -> dict[str, dict]:
    """Return {verdict_id: feature_dict} deduped across the pair file."""
    path = ROOT / domain / REPS[rep]
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        header = [_clean_key(h) for h in next(r)]
        idx = {h: i for i, h in enumerate(header)}
        for row in r:
            if not row:
                continue
            for vcol, fcol in (("verdict_1", "feature_vector_1"),
                               ("verdict_2", "feature_vector_2")):
                vid = row[idx[vcol]].strip()
                d = _parse_json(row[idx[fcol]])
                if vid and d is not None and vid not in out:
                    # normalise keys (strip whitespace)
                    out[vid] = {str(k).strip(): v for k, v in d.items()}
    return out


if __name__ == "__main__":
    for dom in DOMAINS:
        for rep in FOCUS:
            data = load_rep(dom, rep)
            keys = set()
            for d in data.values():
                keys.update(d.keys())
            print(f"{dom:7s} {rep:13s} verdicts={len(data):4d}  distinct_keys={len(keys)}")
