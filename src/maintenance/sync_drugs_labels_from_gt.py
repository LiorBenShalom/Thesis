#!/usr/bin/env python3
"""
מעדכן similarity_scale / similarity_binary_* בכל קבצי הייצוג של סמים מול
similarity_database_fe.csv (מקור אמת לתיוגים).

כולל את שבעת הקבצים שממופים ל-experiments/data/drugs/*.csv וגם (אופציונלי)
manual_format + legacy_from_structured.

  python sync_drugs_labels_from_gt.py
  python sync_drugs_labels_from_gt.py --gt /path/to/similarity_database_fe.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_BASE = Path(__file__).resolve().parent.parent / "drugs"

REPRESENTATION_CSVS = [
    "similarity_database_with_indicment_facts.csv",
    "similarity_database_fe.csv",
    "similarity_database_fe_gpt_schema.csv",
    "similarity_database_with_gpt_law_features.csv",
    "similarity_database_with_gpt_features.csv",
    "similarity_database_hybrid.csv",
    "similarity_database_hybrid_full_gpt.csv",
]

EXTRA = [
    "similarity_database_fe_manual_format.csv",
    "similarity_database_fe_legacy_from_structured.csv",
]

KEYS = ("verdict_1", "verdict_2")
LABELS = ("similarity_scale", "similarity_binary_0", "similarity_binary_1")


def sync_file(gt_idx: pd.DataFrame, path: Path) -> tuple[int, int]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = 0
    changed = 0
    for i, row in df.iterrows():
        k = (row["verdict_1"], row["verdict_2"])
        if k not in gt_idx.index:
            missing += 1
            continue
        g = gt_idx.loc[k]
        if isinstance(g, pd.DataFrame):
            g = g.iloc[0]
        for c in LABELS:
            nv = int(pd.to_numeric(g[c], errors="coerce"))
            ov = int(pd.to_numeric(row[c], errors="coerce"))
            if nv != ov:
                changed += 1
            df.at[i, c] = nv
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return missing, changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gt",
        type=Path,
        default=None,
        help="ברירת מחדל: new_try/drugs/similarity_database_fe.csv",
    )
    ap.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
        help="תיקיית drugs (קבצי CSV)",
    )
    ap.add_argument(
        "--no-extra",
        action="store_true",
        help="לא לעדכן manual_format ו-legacy_from_structured",
    )
    args = ap.parse_args()

    base = args.base.resolve()
    gt_path = (args.gt or (base / "similarity_database_fe.csv")).resolve()
    gt = pd.read_csv(gt_path, encoding="utf-8-sig")
    gt_idx = gt.drop_duplicates(subset=list(KEYS)).set_index(list(KEYS))[
        list(LABELS)
    ]

    files = list(REPRESENTATION_CSVS)
    if not args.no_extra:
        files.extend(EXTRA)

    print(f"GT: {gt_path} ({len(gt)} שורות)\n")
    for name in files:
        p = base / name
        if not p.is_file():
            print(f"SKIP (חסר): {name}")
            continue
        miss, ch = sync_file(gt_idx, p)
        print(f"{name}: missing_pairs={miss}, label_cells_updated={ch}")


if __name__ == "__main__":
    main()
