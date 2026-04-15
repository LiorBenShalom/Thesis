#!/usr/bin/env python3
"""
אחרי שינוי similarity_scale / similarity_binary_* ב-similarity_database_fe.csv:

1. מעדכן את עמודות התיוג בכל קובץ *_v6score_*_binary_0_preds.csv תחת v6_final/drugs/results_drugs
   (אותם זוגות — הציונים נשארים; רק ה-GT משתנה).

2. מחשב מחדש *_binary_0_stats.json ו-*_binary_1_stats.json מול הציונים השמורים — בלי קריאות API.

שימוש:
  python relabel_drugs_v6_preds_from_gt.py
  python relabel_drugs_v6_preds_from_gt.py --gt ../drugs/similarity_database_fe.csv --results-dir ../experiments/v6_final/drugs/results_drugs
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from v6_score_multimodel_experiment import _v6_build_stats  # noqa: E402


def _parse_preds_meta(path: Path) -> tuple[str, str, str] | None:
    """
    similarity_database_fe_v6score_gpt4_binary_0_preds.csv
    -> stem_csv=similarity_database_fe, model=gpt4
    """
    m = re.match(r"^(.+)_v6score_(.+)_binary_0_preds\.csv$", path.name)
    if not m:
        return None
    return m.group(1), m.group(2), path.name


def load_gt_index(gt_path: Path) -> pd.DataFrame:
    gt = pd.read_csv(gt_path, encoding="utf-8-sig")
    keys = ["verdict_1", "verdict_2"]
    cols = ["similarity_scale", "similarity_binary_0", "similarity_binary_1"]
    return gt.drop_duplicates(subset=keys).set_index(keys)[cols]


def apply_gt_to_preds(df: pd.DataFrame, gt_idx: pd.DataFrame) -> int:
    """מחזיר כמה תאי תיוג עודכנו."""
    n = 0
    for i, row in df.iterrows():
        k = (row["verdict_1"], row["verdict_2"])
        if k not in gt_idx.index:
            continue
        g = gt_idx.loc[k]
        if isinstance(g, pd.DataFrame):
            g = g.iloc[0]
        for c in gt_idx.columns:
            nv = int(pd.to_numeric(g[c], errors="coerce"))
            ov = int(pd.to_numeric(row.get(c, np.nan), errors="coerce"))
            if ov != nv:
                n += 1
            df.at[i, c] = nv
    return n


def recompute_stats_for_preds(
    preds_path: Path,
    df: pd.DataFrame,
    csv_name: str,
    model: str,
    meta_from_json: dict | None,
) -> None:
    sc = pd.to_numeric(df["score"], errors="coerce")
    if "status" in df.columns:
        valid = sc.notna() & (df["status"].astype(str) == "ok")
    else:
        valid = sc.notna()
    n_failed = int((~valid).sum())
    mask = valid.to_numpy()
    scv = sc[valid].values.astype(float)

    rep_id = "unknown"
    if meta_from_json:
        rep_id = meta_from_json.get("representation_id", rep_id)

    for task in ("binary_0", "binary_1"):
        y = df[f"similarity_{task}"].values.astype(int)
        yv = y[mask]
        stats = _v6_build_stats(
            scv,
            yv,
            "drugs",
            rep_id,
            csv_name,
            model,
            task,
            len(df),
            n_failed,
            str(preds_path.resolve()),
            from_same_scores=True,
        )
        if meta_from_json:
            stats["representation_id"] = meta_from_json.get("representation_id", stats["representation_id"])
            stats["csv"] = meta_from_json.get("csv", stats["csv"])

        out_stats = preds_path.parent / preds_path.name.replace(
            "_binary_0_preds.csv", f"_{task}_stats.json"
        )
        with open(out_stats, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"  stats -> {out_stats.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gt",
        type=Path,
        default=None,
        help="similarity_database_fe.csv (drugs)",
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="תיקיית results_drugs של v6",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent.parent
    gt_path = (args.gt or (base / "drugs" / "similarity_database_fe.csv")).resolve()
    results_dir = (args.results_dir or (base / "experiments" / "v6_final" / "drugs" / "results_drugs")).resolve()

    if not gt_path.is_file():
        print(f"חסר GT: {gt_path}", file=sys.stderr)
        sys.exit(1)
    if not results_dir.is_dir():
        print(f"חסרה תיקייה: {results_dir}", file=sys.stderr)
        sys.exit(1)

    gt_idx = load_gt_index(gt_path)
    preds_files = sorted(results_dir.glob("*_v6score_*_binary_0_preds.csv"))
    if not preds_files:
        print(f"לא נמצאו קבצי preds תואמים ב-{results_dir}")
        return

    print(f"GT: {gt_path}")
    print(f"תיקייה: {results_dir}  ({len(preds_files)} קבצי preds)\n")

    for preds_path in preds_files:
        meta = _parse_preds_meta(preds_path)
        if not meta:
            continue
        stem_csv, model, _ = meta
        csv_name = f"{stem_csv}.csv"

        df = pd.read_csv(preds_path, encoding="utf-8-sig")
        changed = apply_gt_to_preds(df, gt_idx)

        stats_json_path = preds_path.parent / preds_path.name.replace(
            "_binary_0_preds.csv", "_binary_0_stats.json"
        )
        meta_from = None
        if stats_json_path.is_file():
            try:
                meta_from = json.loads(stats_json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

        print(f"{preds_path.name}: עודכנו {changed} תאי תיוג (השוואה ל-GT)")

        if args.dry_run:
            continue

        df.to_csv(
            preds_path,
            index=False,
            encoding="utf-8-sig",
            quoting=csv.QUOTE_NONNUMERIC,
            na_rep="",
        )

        recompute_stats_for_preds(preds_path, df, csv_name, model, meta_from)

    if args.dry_run:
        print("\n(dry-run — לא נשמר)")
    else:
        print("\nסיום. להריץ regenerate_v6_tables.py כדי לעדכן טבלאות Excel-style.")


if __name__ == "__main__":
    main()
