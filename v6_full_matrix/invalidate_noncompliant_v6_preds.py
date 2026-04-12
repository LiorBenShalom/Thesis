#!/usr/bin/env python3
"""
Mark rows as failed (parse_error) when status=ok but response lacks a matching
SIMILARITY_SCORE: line — so the next v6 run will refetch those rows (resume).

Usage:
  python invalidate_noncompliant_v6_preds.py
  python invalidate_noncompliant_v6_preds.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V6_ROOT = Path(__file__).resolve().parent
NEW_TRY = V6_ROOT.parents[1]

MATRIX_MODELS = [
    "gpt4",
    "gpt5mini",
    "qwen3_235b",
    "mistral",
    "llama3_70b",
    "gpt52",
    "gpt51_thinking",
    "qwen_hf",
    "gemini_25_pro",
    "gemini_3_flash",
    "gemma3_27b",
]

_SIM = re.compile(r"SIMILARITY_SCORE\s*:\s*(\d+)", re.IGNORECASE)


def row_fails_strict(row: pd.Series) -> bool:
    if str(row.get("status", "")) != "ok":
        return False
    resp = row.get("response")
    if pd.isna(resp):
        resp = ""
    else:
        resp = str(resp)
    m = _SIM.search(resp)
    if not m:
        return True
    try:
        declared = float(m.group(1))
        stored = float(row["score"])
    except (TypeError, ValueError):
        return True
    return abs(declared - stored) > 0.501


def save_preds_like_checkpoint(out_csv: Path, df: pd.DataFrame) -> None:
    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
    df.to_csv(
        tmp,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_NONNUMERIC,
        na_rep="",
    )
    tmp.replace(out_csv)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total_invalidated = 0
    files_touched = 0

    for domain in ("drugs", "weapon"):
        rdir = V6_ROOT / domain / f"results_{domain}"
        if not rdir.is_dir():
            continue
        for preds_path in sorted(rdir.glob("*_v6score_*_binary_0_preds.csv")):
            m = re.match(r"^(.+)_v6score_(.+)_binary_0_preds\.csv$", preds_path.name)
            if not m:
                continue
            model = m.group(2)
            if model not in MATRIX_MODELS:
                continue

            pr = pd.read_csv(preds_path)
            if "status" not in pr.columns or "response" not in pr.columns:
                continue
            for col in ("status", "last_error", "response"):
                if col in pr.columns:
                    pr[col] = pr[col].astype(object)

            changed = False
            for i in pr.index:
                if not row_fails_strict(pr.loc[i]):
                    continue
                if str(pr.loc[i, "status"]) != "ok":
                    continue
                pr.loc[i, "score"] = np.nan
                pr.loc[i, "status"] = "parse_error"
                pr.loc[i, "last_error"] = "missing_or_mismatched_SIMILARITY_SCORE"
                changed = True
                total_invalidated += 1

            if changed:
                files_touched += 1
                if not args.dry_run:
                    save_preds_like_checkpoint(preds_path, pr)
                print(f"{'[dry-run] ' if args.dry_run else ''}{preds_path.name}: invalidated rows")

    print(f"Done. Files touched: {files_touched}, rows invalidated: {total_invalidated}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
