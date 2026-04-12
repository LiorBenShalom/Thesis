#!/usr/bin/env python3
"""
חילוץ מחדש (העשרת GPT) לפיצ'רים היברידיים — רק למזהי פסק דין נתונים.

הלוגיקה של enrich_single_case זהה ל־gpt_feature_database.ipynb (מיובא מ־gpt_feature_database_hybrid_enrich).
  - קלט ידני: new_try/drugs/similarity_database_fe.csv (זהה ל־experiments/data/drugs/manual_fe.csv)
  - עובדות: similarity_database_with_indicment_facts.csv
  - פלט מעודכן: similarity_database_hybrid.csv
  אחרי העשרה: ממזגים שוב את שני צידי הפיצ'ר מה־GT הידני כדי שהערכים העדכניים ב־CSV ינצחו.

דוגמה:
  export OPENAI_API_KEY=sk-your-key-here
  python hybrid_manual_reextract_verdicts.py --verdicts ME-16-06-6788-21 ME-23-12-14553-571

  # או במקום סביבה:
  python hybrid_manual_reextract_verdicts.py --verdicts ... --api-key sk-your-key-here
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI

from gpt_feature_database_hybrid_enrich import enrich_single_case

BASE = Path(__file__).resolve().parent.parent / "drugs"

# פלייסהולדר לדוגמה בלבד — חייב להישאר טקסט שלא שווה למפתח אמיתי (אל תדביקי כאן מפתח)
OPENAI_API_KEY_PLACEHOLDER = "sk-your-key-here"


def facts_columns(df: pd.DataFrame) -> tuple[str, str]:
    c1 = "indicment_facts_1" if "indicment_facts_1" in df.columns else "indictment_facts_1"
    c2 = "indicment_facts_2" if "indicment_facts_2" in df.columns else "indictment_facts_2"
    return c1, c2


def overlay_manual_gt_on_rows(
    hybrid: pd.DataFrame,
    manual: pd.DataFrame,
    target: set[str],
) -> int:
    """
    אחרי העשרת GPT: דורס מפתחות שקיימים ב־similarity_database_fe.csv לפי ה־CSV שנטען כאן
    (הידני תמיד מנצח — מונע מצב שבו נשאר FE ישן או ערכי GPT בשדות ידניים).
    """
    manual_idx = manual.set_index(["verdict_1", "verdict_2"])
    n_changed = 0
    for i, row in hybrid.iterrows():
        v1, v2 = str(row["verdict_1"]), str(row["verdict_2"])
        if v1 not in target and v2 not in target:
            continue
        key = (v1, v2)
        if key not in manual_idx.index:
            continue
        mr = manual_idx.loc[key]
        if isinstance(mr, pd.DataFrame):
            mr = mr.iloc[0]
        for col_h, col_m in (
            ("feature_vector_1", "feature_vector_1"),
            ("feature_vector_2", "feature_vector_2"),
        ):
            try:
                hyb = json.loads(hybrid.at[i, col_h])
                man = json.loads(mr[col_m])
            except (json.JSONDecodeError, TypeError):
                continue
            merged = {**hyb, **man}
            if merged != hyb:
                n_changed += 1
            hybrid.at[i, col_h] = json.dumps(merged, ensure_ascii=False)
    return n_changed


def _canonical_side_text(
    df: pd.DataFrame,
    vid: str,
    col_if_v1: str,
    col_if_v2: str,
    label: str,
) -> str:
    """אותו פסק חייב להופיע עם אותו טקסט בכל הזוגות — אחרת ה-GT לא עקבי."""
    texts: list[str] = []
    for _, r in df.iterrows():
        if r["verdict_1"] == vid:
            texts.append(str(r[col_if_v1]))
        elif r["verdict_2"] == vid:
            texts.append(str(r[col_if_v2]))
    if not texts:
        raise ValueError(f"לא נמצא הפסק {vid} ב־{label}")
    unique = set(texts)
    if len(unique) > 1:
        print(
            f"שגיאה: {vid} — {len(unique)} גרסאות שונות של {label} בין זוגות (ה-GT לא אחיד).",
            file=sys.stderr,
        )
        for i, t in enumerate(list(unique)[:3]):
            print(f"  גרסה {i+1} (התחלה): {t[:200]}...", file=sys.stderr)
        sys.exit(1)
    return texts[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--verdicts",
        nargs="+",
        required=True,
        help="מזהי פסק דין לחילוץ מחדש (מופיעים כ-verdict_1 או verdict_2)",
    )
    ap.add_argument(
        "--manual-csv",
        type=Path,
        default=BASE / "similarity_database_fe.csv",
    )
    ap.add_argument(
        "--facts-csv",
        type=Path,
        default=BASE / "similarity_database_with_indicment_facts.csv",
    )
    ap.add_argument(
        "--hybrid-out",
        type=Path,
        default=BASE / "similarity_database_hybrid.csv",
    )
    ap.add_argument(
        "--model",
        default=os.getenv("GPT_FEATURE_MODEL", "gpt-4-turbo"),
        help="ברירת מחדל כמו במחברת (enrich_single_case)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="רק מדפיס אילו שורות יעודכנו — בלי API",
    )
    ap.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        help=(
            f"מפתח OpenAI (חלופה ל־OPENAI_API_KEY). לא להדביק מפתח בקוד — "
            f"פורמט לדוגמה: {OPENAI_API_KEY_PLACEHOLDER}"
        ),
    )
    args = ap.parse_args()

    api_key = (args.api_key or "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not args.dry_run:
        if not api_key or api_key == OPENAI_API_KEY_PLACEHOLDER:
            print(
                "חסר מפתח API: הגדרי export OPENAI_API_KEY=... או העברי --api-key "
                f"(לא להשאיר את הפלייסהולדר {OPENAI_API_KEY_PLACEHOLDER!r})",
                file=sys.stderr,
            )
            sys.exit(1)

    target = set(str(v).strip() for v in args.verdicts)
    manual_path = args.manual_csv.resolve()
    facts_path = args.facts_csv.resolve()
    hybrid_path = args.hybrid_out.resolve()
    manual = pd.read_csv(manual_path, encoding="utf-8-sig")
    facts = pd.read_csv(facts_path, encoding="utf-8-sig")
    fc1, fc2 = facts_columns(facts)
    merged = pd.merge(
        manual,
        facts[["verdict_1", "verdict_2", fc1, fc2]],
        on=["verdict_1", "verdict_2"],
        how="inner",
    )

    print(f"קובץ GT ידני: {manual_path}")
    try:
        print(f"  (שינוי אחרון בקובץ: {datetime.fromtimestamp(manual_path.stat().st_mtime)})")
    except OSError:
        pass
    print(f"עובדות: {facts_path}")
    print(f"פלט hybrid: {hybrid_path}")

    client: OpenAI | None = None
    if not args.dry_run:
        client = OpenAI(api_key=api_key)

    cache: dict[str, dict] = {}
    for vid in target:
        hit = merged[
            (merged["verdict_1"] == vid) | (merged["verdict_2"] == vid)
        ]
        if hit.empty:
            print(f"WARNING: לא נמצא זוג עם הפסק {vid} ב-merge manual+facts", file=sys.stderr)
            continue
        try:
            mtxt = _canonical_side_text(
                manual, vid, "feature_vector_1", "feature_vector_2", "feature_vector (GT)"
            )
            ftxt = _canonical_side_text(merged, vid, fc1, fc2, "indictment facts")
        except ValueError as e:
            print(f"WARNING: {e}", file=sys.stderr)
            continue
        q_preview = mtxt.replace("\n", " ")[:120]
        print(f"מכין חילוץ ל-{vid} (facts ~{len(str(ftxt))} תווים, GT ידני: {q_preview}...)")
        if args.dry_run:
            cache[vid] = {}
            continue
        assert client is not None
        cache[vid] = enrich_single_case(client, str(ftxt), str(mtxt), args.model)
        time.sleep(0.3)

    if args.dry_run:
        print("dry-run: לא נכתב פלט")
        return

    hybrid = pd.read_csv(args.hybrid_out, encoding="utf-8-sig")
    n_up = 0
    for i, row in hybrid.iterrows():
        v1, v2 = str(row["verdict_1"]), str(row["verdict_2"])
        d1 = cache.get(v1)
        d2 = cache.get(v2)
        if d1 is None and d2 is None:
            continue
        if d1 is not None:
            hybrid.at[i, "feature_vector_1"] = json.dumps(d1, ensure_ascii=False)
            n_up += 1
        if d2 is not None:
            hybrid.at[i, "feature_vector_2"] = json.dumps(d2, ensure_ascii=False)
            n_up += 1

    n_overlay = overlay_manual_gt_on_rows(hybrid, manual, target)
    hybrid.to_csv(args.hybrid_out, index=False, encoding="utf-8-sig")
    print(
        f"נכתב {args.hybrid_out} — עודכנו {n_up} צדי פיצ'ר מהעשרה; "
        f"סינכרון מחדש מול GT ידני: {n_overlay} תאים שוּנוּ."
    )


if __name__ == "__main__":
    main()
