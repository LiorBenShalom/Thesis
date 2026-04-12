#!/usr/bin/env python3
"""
Convert structured drug/weapon feature dicts (manual-format / GT-aligned schema)
to the legacy `similarity_database_fe.csv` JSON shape — **deterministic**, no LLM.

Input: objects like those in `similarity_database_fe_manual_format.csv`.
Output: objects like those in `similarity_database_fe.csv` — **שמות מפתחות, סדר מפתחות,
ופורמט JSON** (מפרידים עם רווח אחרי `:` ו-`,`) כמו בקבצי ה-GT הידניים:

- סמים: תמיד בדיוק 6 מפתחות בסדר `DRUG_LEGACY_KEY_ORDER`.
- נשק: סדר `WEAPON_LEGACY_KEY_ORDER`; רשת `סוג הנשק [...]` כמו בטופס (אותו סדר כמו
  `_WEAPON_TYPE_GRID` ב-extract_features_manual_format); ערכי כמות ברשת כ-float.

Usage:
  python manual_format_to_legacy_fe.py --domain both \\
    --input-dir ../drugs ../weapon \\
    --output ../drugs/similarity_database_fe_legacy_from_structured.csv ...

Or use defaults pointing at new_try/drugs and new_try/weapon with -o per domain.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping

# סדר מפתחות כמו ברוב שורות `similarity_database_fe.csv` (ידני)
DRUG_LEGACY_KEY_ORDER: tuple[str, ...] = (
    "מכירה לסוכן",
    "מעבדה",
    "סוג הסם, כמות",
    "עבירה",
    "עבירות נלוות כן/לא",
    "תפקיד",
)

# רשת סוגי נשק — אותו סדר כמו ב-extract_features_manual_format._WEAPON_TYPE_GRID
_WEAPON_GRID_KEYS_IN_FORM_ORDER: tuple[str, ...] = (
    "סוג הנשק [אקדח]",
    "סוג הנשק [תת מקלע]",
    "סוג הנשק [תת מקלע מאולתר]",
    "סוג הנשק [בקבוק תבערה]",
    "סוג הנשק [מטען חבלה]",
    "סוג הנשק [רימון רסס]",
    "סוג הנשק [רובה סער ]",
    "סוג הנשק [רימון הלם/גז]",
    "סוג הנשק [טיל לאו]",
    "סוג הנשק [טיל מטאדור]",
    "סוג הנשק [רובה צייד]",
    "סוג הנשק [רובה צלפים]",
    "סוג הנשק [מטען חבלה מאולתר]",
    "סוג הנשק [רובה סער מאולתר]",
)

# סדר שדות טקסט/קטגוריה אחרי מספר עבירה ולפני/אחרי הרשת — תואם טופס ידני טיפוסי
WEAPON_LEGACY_KEY_ORDER: tuple[str, ...] = (
    "אופן החזקת הנשק",
    "אופן קבלת הנשק",
    "כמות תחמושת",
    "מטרה-סיבת העבירה",
    "מספר עבירה",
) + _WEAPON_GRID_KEYS_IN_FORM_ORDER + (
    "סוג עבירה",
    "סטטוס הנשק",
    "עבירות נוספות",
    "שימוש",
    "תכנון",
    "סוג הנשק - אם לא נמצא בטבלה",
)

# Reuse GT parsing to validate drug עבירה round-trip (same rules as compare_fe_gt_soft)
_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from compare_fe_gt_soft import (  # noqa: E402
    DRUG_OFFENSE_KEYS,
    WEAPON_OFFENSE_PHRASES,
    WEAPON_STORAGE_PHRASES,
    parse_gt_drug_offense_flags,
)

# --- drugs -----------------------------------------------------------------


def _legacy_drug_qty_string(s: str) -> str:
    """Match legacy CSV style: MDMA/LSD often suffixed with מ (no double-מ)."""
    if not (s or "").strip():
        return s or ""
    s = re.sub(r"-MDMA(?!מ)(?=[,\s]|$)", "-MDMAמ", s)
    s = re.sub(r"-LSD(?!מ)(?=[,\s]|$)", "-LSDמ", s)
    return s


def _build_drug_offense_blob(flags: Dict[str, str]) -> str:
    """Inverse of parse_gt_drug_offense_flags — fixed order, comma-separated."""
    parts: List[str] = []
    if flags.get("עבירה_החזקה") == "כן":
        parts.append("החזקה שלא לצריכה עצמית")
    if flags.get("עבירה_יבוא_סחר") == "כן":
        parts.append("יבוא/סחר")
    if flags.get("עבירה_ייצור") == "כן":
        parts.append("ייצור")
    if flags.get("עבירה_כלים") == "כן":
        parts.append("כלים")
    if flags.get("עבירה_19") == "כן":
        parts.append("19")
    if flags.get("עבירה_ניסיון_סחר") == "כן":
        parts.append("ניסיון לסחר")
    if flags.get("עבירה_ניסיון_ייצור") == "כן":
        parts.append("ניסיון לייצור/גידול")
    return ", ".join(parts)


def drug_structured_to_legacy(d: Dict[str, Any]) -> OrderedDict[str, Any]:
    if not any(k in d for k in DRUG_OFFENSE_KEYS):
        raise ValueError("Not a structured drug dict (missing offense flags)")
    flags = {k: ("כן" if str(d.get(k, "לא")).strip() == "כן" else "לא") for k in DRUG_OFFENSE_KEYS}
    עבירה = _build_drug_offense_blob(flags)
    back = parse_gt_drug_offense_flags(עבירה)
    for k in DRUG_OFFENSE_KEYS:
        if back.get(k) != flags.get(k):
            raise RuntimeError(f"עבירה round-trip mismatch on {k}: {flags.get(k)} vs {back.get(k)}")

    role_s = str(d.get("תפקיד_בעלות_סמים", "")).strip()
    lab = str(d.get("תפקיד_בעלות_מעבדה", "")).strip()
    if lab and lab != "לא רלוונטי":
        תפקיד = f"{role_s}, {lab}"
    else:
        תפקיד = role_s

    qty = str(d.get("סוג הסם, כמות", "")).strip()
    qty = _legacy_drug_qty_string(qty)

    inner = {
        "מכירה לסוכן": str(d.get("מכירה לסוכן", "לא")).strip(),
        "מעבדה": str(d.get("מעבדה", "לא")).strip(),
        "סוג הסם, כמות": qty,
        "עבירה": עבירה,
        "עבירות נלוות כן/לא": str(d.get("עבירות נלוות כן/לא", "לא")).strip(),
        "תפקיד": תפקיד,
    }
    return OrderedDict((k, inner[k]) for k in DRUG_LEGACY_KEY_ORDER)


# --- weapon ----------------------------------------------------------------


def _as_legacy_weapon_quantity(v: Any) -> float:
    """כמו ב-CSV הידני: מספרים ברשת כ-float (למשל 2.0, 1.0)."""
    return float(v)


def weapon_structured_to_legacy(d: Dict[str, Any]) -> OrderedDict[str, Any]:
    parts_off: List[str] = []
    for key, phrase in WEAPON_OFFENSE_PHRASES:
        if str(d.get(key, "לא")).strip() == "כן":
            parts_off.append(phrase)
    סוג_עבירה = ", ".join(parts_off)

    parts_stor: List[str] = []
    for key, phrase in WEAPON_STORAGE_PHRASES:
        if str(d.get(key, "לא")).strip() == "כן":
            parts_stor.append(phrase)
    אופן_החזקה = ", ".join(parts_stor)

    raw: Dict[str, Any] = {}
    if אופן_החזקה:
        raw["אופן החזקת הנשק"] = אופן_החזקה
    for k in (
        "אופן קבלת הנשק",
        "כמות תחמושת",
        "מטרה-סיבת העבירה",
        "מספר עבירה",
        "סטטוס הנשק",
        "עבירות נוספות",
        "שימוש",
        "תכנון",
        "סוג הנשק - אם לא נמצא בטבלה",
    ):
        if k in d and d[k] not in (None, ""):
            raw[k] = d[k]
    for k in _WEAPON_GRID_KEYS_IN_FORM_ORDER:
        if k not in d:
            continue
        v = d[k]
        if isinstance(v, (int, float)):
            if float(v) == 0.0:
                continue
            raw[k] = _as_legacy_weapon_quantity(v)
        else:
            raw[k] = v
    if סוג_עבירה:
        raw["סוג עבירה"] = סוג_עבירה
    return _order_weapon_legacy(raw)


def _order_weapon_legacy(raw: Dict[str, Any]) -> OrderedDict[str, Any]:
    """רק מפתחות שקיימים ב-raw, בסדר WEAPON_LEGACY_KEY_ORDER; שאר מפתחות (אם יש) בסוף."""
    od: OrderedDict[str, Any] = OrderedDict()
    seen: set[str] = set()
    for k in WEAPON_LEGACY_KEY_ORDER:
        if k not in raw:
            continue
        v = raw[k]
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        od[k] = v
        seen.add(k)
    for k, v in raw.items():
        if k in seen:
            continue
        od[k] = v
    return od


def _dump_legacy_json(obj: Mapping[str, Any]) -> str:
    """מפרידים כמו ברירת המחדל של json.dumps: פסיק ורווח אחרי ':' ו-',' — כמו בקבצי ה-GT."""
    return json.dumps(obj, ensure_ascii=False)


def convert_feature_vector(raw: str, domain: str) -> str:
    d = json.loads(raw)
    if domain == "drugs":
        new_d = drug_structured_to_legacy(d)
    else:
        new_d = weapon_structured_to_legacy(d)
    return _dump_legacy_json(new_d)


def convert_csv(
    input_path: Path,
    output_path: Path,
    domain: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(newline="", encoding="utf-8-sig") as fin:
        r = csv.DictReader(fin)
        fieldnames = list(r.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"Empty or invalid CSV: {input_path}")
        rows_out: List[Dict[str, str]] = []
        for row in r:
            out = dict(row)
            for side in ("feature_vector_1", "feature_vector_2"):
                if side in out and out[side]:
                    out[side] = convert_feature_vector(out[side], domain)
            rows_out.append(out)
    with output_path.open("w", newline="", encoding="utf-8-sig") as fout:
        w = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Structured manual-format FE → legacy FE JSON shape")
    ap.add_argument(
        "--drugs-in",
        type=Path,
        default=None,
        help="Default: new_try/drugs/similarity_database_fe_manual_format.csv",
    )
    ap.add_argument(
        "--weapon-in",
        type=Path,
        default=None,
        help="Default: new_try/weapon/similarity_database_fe_manual_format.csv",
    )
    ap.add_argument(
        "--drugs-out",
        type=Path,
        default=None,
        help="Default: new_try/drugs/similarity_database_fe_legacy_from_structured.csv",
    )
    ap.add_argument(
        "--weapon-out",
        type=Path,
        default=None,
        help="Default: new_try/weapon/similarity_database_fe_legacy_from_structured.csv",
    )
    ap.add_argument(
        "--domain",
        choices=("drugs", "weapon", "both"),
        default="both",
    )
    args = ap.parse_args()

    base = Path(__file__).resolve().parent.parent
    drugs_in = args.drugs_in or (base / "drugs" / "similarity_database_fe_manual_format.csv")
    weapon_in = args.weapon_in or (base / "weapon" / "similarity_database_fe_manual_format.csv")
    drugs_out = args.drugs_out or (base / "drugs" / "similarity_database_fe_legacy_from_structured.csv")
    weapon_out = args.weapon_out or (base / "weapon" / "similarity_database_fe_legacy_from_structured.csv")

    if args.domain in ("drugs", "both"):
        convert_csv(drugs_in, drugs_out, "drugs")
        print(f"Wrote {drugs_out}")
    if args.domain in ("weapon", "both"):
        convert_csv(weapon_in, weapon_out, "weapon")
        print(f"Wrote {weapon_out}")


if __name__ == "__main__":
    main()
