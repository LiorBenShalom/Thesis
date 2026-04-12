"""
convert_gt_to_fe.py  (drugs)
Converts per-verdict GT CSV → pair-based feature-vector CSV
compatible with experiments/data/drugs/manual_fe.csv format.

Usage:
    python convert_gt_to_fe.py \
        --gt    ../data/gt_manual_drugs.csv \
        --pairs ../../data/drugs/facts.csv \
        --out   ../results/fe_gt.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import pathlib
import re

BASE = pathlib.Path(__file__).resolve().parent.parent

DRUG_COLS = [
    "LSD", "METHAMPHETAMINE", "האיוואסקה", "קתינון", "קטמין",
    "חשיש", "מתילמקאתינון", "קנבוס בשתילים", "קנבוס", "MDMA", "קוקאין",
]

UNIT_NORMALIZE = {
    "נוזל":    "מיליליטר",
    "יחידות":  "שתילים",
}


def _parse_drug_field(raw: str) -> str:
    """'[1301.44-גרם, 13189-טבליות]' → '1301.44 גרם, 13189 טבליות'"""
    if not raw or not raw.strip():
        return ""
    raw = raw.strip()
    SEP = "\u2063"
    raw = re.sub(r'(\d),(\d)', lambda m: m.group(1) + SEP + m.group(2), raw)
    raw = re.sub(r'^[^\[\d]*\[', '', raw)
    raw = raw.rstrip(']')
    parts = []
    for token in raw.split(','):
        token = token.replace('\u2063', ',').strip()
        if not token:
            continue
        m = re.match(r'^([\d\.,]+)-(.+)$', token)
        if m:
            amt = m.group(1).replace(',', '')
            unit = UNIT_NORMALIZE.get(m.group(2).strip(), m.group(2).strip())
            try:
                parts.append(f"{float(amt):g} {unit}")
            except ValueError:
                parts.append(f"{amt} {unit}")
            continue
        m = re.match(r'^([^\d]+)-([\d\.,]+)$', token)
        if m:
            unit = UNIT_NORMALIZE.get(m.group(1).strip(), m.group(1).strip())
            amt = m.group(2).replace(',', '')
            try:
                parts.append(f"{float(amt):g} {unit}")
            except ValueError:
                parts.append(f"{amt} {unit}")
            continue
        m = re.match(r'^(\S+)[-–]([\d\.,]+)\s+(\S+)$', token)
        if m:
            unit = UNIT_NORMALIZE.get(m.group(3).strip(), m.group(3).strip())
            amt = m.group(2).replace(',', '')
            try:
                parts.append(f"{float(amt):g} {unit}")
            except ValueError:
                parts.append(f"{amt} {unit}")
            continue
        parts.append(token)
    return ", ".join(parts)


def _build_drug_quantity(gt_row: dict) -> str:
    parts = []
    for drug in DRUG_COLS:
        raw = gt_row.get(f"סוג הסם [{drug}]", "").strip()
        if not raw:
            continue
        parsed = _parse_drug_field(raw)
        if parsed:
            parts.append(f"{drug}: {parsed}")
    return ", ".join(parts)


def _build_offense(gt_row: dict) -> str:
    parts = []
    if gt_row.get("עבירות [סעיף 6]", "").strip():
        parts.append("ייצור")
    if gt_row.get("עבירות [סעיף 7]", "").strip():
        parts.append("החזקה שלא לצריכה עצמית")
    if gt_row.get("עבירות [סעיף 13]", "").strip():
        parts.append("יבוא/סחר")
    if gt_row.get("עבירות [סעיף 14]", "").strip() and "יבוא/סחר" not in parts:
        parts.append("יבוא/סחר")
    if gt_row.get("עבירות [סעיף 19]", "").strip():
        parts.append("19")
    if gt_row.get("עבירות [סעיף 21]", "").strip():
        parts.append("כלים")
    if gt_row.get("עבירות [סעיף 22]", "").strip():
        parts.append("ניסיון")
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p); out.append(p)
    return ", ".join(out)


def build_feature_vector(gt_row: dict) -> dict:
    role = gt_row.get("תפקיד", "").strip() or "בעל הסמים"
    lab  = gt_row.get("מעבדה", "").strip() or "לא"
    sale = gt_row.get("מכירה לסוכן", "").strip() or "לא"
    side = gt_row.get("עבירות נלוות כן/לא", "").strip() or "לא"
    return {
        "מכירה לסוכן":        sale,
        "מעבדה":               lab,
        "סוג הסם, כמות":      _build_drug_quantity(gt_row),
        "עבירה":               _build_offense(gt_row),
        "עבירות נלוות כן/לא":  side,
        "תפקיד":               role,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt",    type=pathlib.Path,
                    default=BASE / "data/gt_manual_drugs.csv")
    ap.add_argument("--pairs", type=pathlib.Path,
                    default=BASE.parent / "data/drugs/facts.csv")
    ap.add_argument("--out",   type=pathlib.Path,
                    default=BASE / "results/fe_gt.csv")
    args = ap.parse_args()

    # load GT per verdict
    gt: dict[str, dict] = {}
    with open(args.gt, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("data_source", "").strip() != "GT":
                continue
            vid = row.get("שם קובץ התיק", "").strip()
            if vid:
                gt[vid] = row
    print(f"Loaded GT: {len(gt)} verdicts")

    # load pairs
    with open(args.pairs, newline="", encoding="utf-8-sig") as f:
        pairs = list(csv.DictReader(f))
    print(f"Loaded pairs: {len(pairs)}")

    missing: set[str] = set()
    rows_out = []
    for row in pairs:
        v1, v2 = row["verdict_1"], row["verdict_2"]
        g1, g2 = gt.get(v1), gt.get(v2)
        if g1 is None: missing.add(v1)
        if g2 is None: missing.add(v2)
        fv1 = build_feature_vector(g1) if g1 else {}
        fv2 = build_feature_vector(g2) if g2 else {}
        rows_out.append({
            "verdict_1":           v1,
            "verdict_2":           v2,
            "similarity_scale":    row["similarity_scale"],
            "similarity_binary_0": row["similarity_binary_0"],
            "similarity_binary_1": row["similarity_binary_1"],
            "feature_vector_1":    json.dumps(fv1, ensure_ascii=False),
            "feature_vector_2":    json.dumps(fv2, ensure_ascii=False),
        })

    if missing:
        print(f"⚠  Missing in GT ({len(missing)}): {sorted(missing)[:10]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["verdict_1","verdict_2","similarity_scale",
                                           "similarity_binary_0","similarity_binary_1",
                                           "feature_vector_1","feature_vector_2"])
        w.writeheader()
        w.writerows(rows_out)

    print(f"✅ Wrote {len(rows_out)} pairs → {args.out}")


if __name__ == "__main__":
    main()
