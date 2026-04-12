#!/usr/bin/env python3
"""
Aggregate binary_0 F1 from existing stats JSONs (standard similarity_experiment + v6 multimodel).

Writes:
  new_try/code/leaderboard_similarity_binary0.csv
  new_try/code/leaderboard_v6score_binary0.csv (if v6 stats exist)

Does not call APIs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent


def load_standard_stats(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None
    m = d.get("metrics_binary_0") or {}
    info = d.get("experiment_info") or {}
    return {
        "file": str(path.name),
        "domain": info.get("domain"),
        "model": info.get("model"),
        "representation": info.get("representation"),
        "f1": m.get("f1"),
        "precision": m.get("precision"),
        "recall": m.get("recall"),
        "accuracy": m.get("accuracy"),
    }


def load_v6_stats(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None
    if d.get("method") != "v6_score_multimodel":
        return None
    g = d.get("global_best") or {}
    loo = d.get("loo") or {}
    return {
        "file": str(path.name),
        "domain": d.get("domain"),
        "model": d.get("model"),
        "representation_id": d.get("representation_id"),
        "csv": d.get("csv"),
        "f1_global": g.get("f1"),
        "f1_loo": loo.get("f1"),
        "precision": g.get("precision"),
        "recall": g.get("recall"),
        "threshold": d.get("best_threshold"),
    }


def main():
    rows_std = []
    rows_v6 = []

    for domain_dir in (BASE / "drugs", BASE / "weapon"):
        if not domain_dir.is_dir():
            continue
        sub = "results_drugs" if domain_dir.name == "drugs" else "results_weapon"
        rd = domain_dir / sub
        if not rd.is_dir():
            continue
        for p in rd.glob("*_stats_summary.json"):
            r = load_standard_stats(p)
            if r and r.get("f1") is not None:
                rows_std.append(r)
        for p in rd.glob("*_v6score_*_stats.json"):
            r = load_v6_stats(p)
            if r:
                rows_v6.append(r)

    out_std = BASE / "code" / "leaderboard_similarity_binary0.csv"
    out_v6 = BASE / "code" / "leaderboard_v6score_binary0.csv"

    pd.DataFrame(rows_std).to_csv(out_std, index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_v6).to_csv(out_v6, index=False, encoding="utf-8-sig")

    print(f"Wrote {len(rows_std)} rows -> {out_std}")
    print(f"Wrote {len(rows_v6)} rows -> {out_v6}")

    if rows_std:
        df = pd.DataFrame(rows_std)
        # Best F1 per domain for gpt4 + features (hybrid-style filenames)
        g4 = df[df["model"] == "gpt4"].copy()
        if len(g4):
            print("\nGPT-4.1 standard runs (sample, by F1):")
            print(
                g4.sort_values("f1", ascending=False)
                .head(15)[["domain", "representation", "f1", "file"]]
                .to_string(index=False)
            )


if __name__ == "__main__":
    main()
