#!/usr/bin/env python3
"""
אחרי עדכון GT ב־similarity_database_fe.csv (סמים) — מסנכרן תוויות, ממיר ל־legacy מובנה,
מריץ דיפולטים על הידני, ומדפיס מה להריץ הלאה (v6, טבלאות).

שלבים (מקומיים, בלי API):
  1. מעתיק similarity_scale / similarity_binary_* מה-GT הידני ל־similarity_database_fe_manual_format.csv
     (אותם זוגות — רק עמודות התיוג מתעדכנות; וקטורי הפיצ'רים במבנה מסודר נשארים).
  2. apply_manual_drugs_defaults.py על similarity_database_fe.csv (אלא אם --no-defaults)
  3. manual_format_to_legacy_fe — יוצר מחדש similarity_database_fe_legacy_from_structured.csv

אחר כך (מודפס): ריצת v6 מחדש על ייצוגים רלוונטיים + regenerate_v6_tables + חילוץ/היבריד אם צריך.

שימוש:
  python refresh_drugs_gt_downstream.py
  python refresh_drugs_gt_downstream.py --no-defaults
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
CODE = Path(__file__).resolve().parent
DRUGS = BASE / "drugs"

GT = DRUGS / "similarity_database_fe.csv"
MANUAL_FMT = DRUGS / "similarity_database_fe_manual_format.csv"
LEGACY_STRUCT = DRUGS / "similarity_database_fe_legacy_from_structured.csv"
V6_ROOT = BASE / "experiments" / "v6_final"
RESULTS_DRUGS = V6_ROOT / "drugs" / "results_drugs"


def sync_manual_format_labels_from_gt() -> int:
    """מעתיק עמודות תיוג מ-GT הידני לקובץ המבנה המסודר. מחזיר מספר תאים ששונו."""
    gt = pd.read_csv(GT, encoding="utf-8-sig")
    mf = pd.read_csv(MANUAL_FMT, encoding="utf-8-sig")
    keys = ["verdict_1", "verdict_2"]
    label_cols = ["similarity_scale", "similarity_binary_0", "similarity_binary_1"]
    gt_idx = gt.drop_duplicates(subset=keys).set_index(keys)[label_cols]
    n_changed = 0
    for i, row in mf.iterrows():
        k = (row["verdict_1"], row["verdict_2"])
        if k not in gt_idx.index:
            continue
        g = gt_idx.loc[k]
        if isinstance(g, pd.DataFrame):
            g = g.iloc[0]
        for c in label_cols:
            new_v = int(pd.to_numeric(g[c], errors="coerce"))
            old_v = int(pd.to_numeric(row[c], errors="coerce"))
            if old_v != new_v:
                n_changed += 1
            mf.at[i, c] = new_v
    mf.to_csv(MANUAL_FMT, index=False, encoding="utf-8-sig")
    return n_changed


def main() -> None:
    ap = argparse.ArgumentParser(description="רענון נגזרים אחרי תיקון GT סמים")
    ap.add_argument("--no-defaults", action="store_true", help="לא להריץ apply_manual_drugs_defaults")
    ap.add_argument(
        "--no-sync-manual-format",
        action="store_true",
        help="לא לעדכן תוויות ב-similarity_database_fe_manual_format.csv",
    )
    args = ap.parse_args()

    if not GT.is_file():
        print(f"חסר קובץ GT: {GT}", file=sys.stderr)
        sys.exit(1)
    if not MANUAL_FMT.is_file():
        print(f"חסר: {MANUAL_FMT}", file=sys.stderr)
        sys.exit(1)

    print("=== 1. סנכרון תוויות GT → similarity_database_fe_manual_format.csv ===")
    if args.no_sync_manual_format:
        print("  (דולג)")
    else:
        n = sync_manual_format_labels_from_gt()
        print(f"  עודכן קובץ: {MANUAL_FMT}")
        print(f"  שינויי ערך בעמודות תיוג (סה\"כ השוואות תאים): {n}")

    if not args.no_defaults:
        print("\n=== 2. ברירות מחדל על similarity_database_fe.csv (legacy) ===")
        subprocess.run(
            [sys.executable, str(CODE / "apply_manual_drugs_defaults.py"), "--csv", str(GT), "--in-place"],
            check=True,
        )
    else:
        print("\n=== 2. (דולג) apply_manual_drugs_defaults ===")

    print("\n=== 3. המרה מובנית → similarity_database_fe_legacy_from_structured.csv ===")
    sys.path.insert(0, str(CODE))
    from manual_format_to_legacy_fe import convert_csv

    convert_csv(MANUAL_FMT, LEGACY_STRUCT, "drugs")
    print(f"  נכתב: {LEGACY_STRUCT}")

    print("\n=== מה להריץ הלאה (API / זמן) ===")
    print(
        """
A) שינוי similarity / תיוגי GT בלבד (בלי שינוי טקסט פיצ'רים ל-API):
   — כל חישובי המטריקות מול GT (PR-AUC, דיוק, וכו') תלויים ב-similarity_binary_0/1 הנכונים.
   — ייצוגים שמבוססים רק על GPT (למשל fe_gpt_schema): אין צורך בקריאות API מחדש — הציונים נשארים;
     לעדכן את עמודות התיוג בקבצי preds ואת ה-stats בלי API:

   cd new_try/code
   python relabel_drugs_v6_preds_from_gt.py

   ואז טבלאות:

   cd ../experiments/v6_final
   python regenerate_v6_tables.py

B) ייצוגי hybrid (משלבים פיצ'רים ידניים): אם הידני/legacy_from_structured השתנה — צריך להריץ מחדש את ציוני ה-dim
   (הפיצ'רים הידניים בתוך הייצוג השתנו). ייצוגי GPT טהורים לא חייבים ריצה מחודשת רק בגלל שינוי ידני.

C) דמיון v6 מלא עם --fresh (כשהטקסט/הפיצ'רים שהמודל רואה השתנו, או כשמוסיפים מודל/ייצוג):

   cd new_try/code
   python v6_score_multimodel_experiment.py --domain drugs --reps manual_fe fe_manual_format fe_legacy_from_structured \\
     --models gpt4 --task binary_0 --fresh --output-root ../experiments/v6_final

D) אם עדכנת טקסט/כמויות ב-GT הידני ורוצים שגם החילוץ האוטומטי (מסודר) יתאים — להריץ מחדש:

   python extract_features_manual_format.py --domain drugs --refetch
   (או בלי refetch רק לזוגות ששונו — לפי הצורך)

E) היבריד / gpt_schema — אם נבנים מחדש מקובצי ביניים, להריץ את הסקריפטים הרלוונטיים (run_gpt_schema_extraction וכו').

F) ייצוא לפני/אחרי פורמט (אופציונלי):

   python export_features_before_after_legacy.py --domain drugs
"""
    )

    print("\n=== גיבוי preds ישן (אופציונלי) ===")
    print(
        f"  לפני --fresh, אפשר להעתיק תיקייה:\n  {RESULTS_DRUGS}\n"
        f"  למשל: cp -a {RESULTS_DRUGS} {RESULTS_DRUGS}_backup_$(date +%Y%m%d)"
    )


if __name__ == "__main__":
    main()
