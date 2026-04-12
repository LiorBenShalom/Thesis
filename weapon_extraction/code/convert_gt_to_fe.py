"""
convert_gt_to_fe.py  (weapon)
Converts per-verdict GT CSV → pair-based feature-vector CSV
compatible with experiments/data/wep/manual_fe.csv format.

Usage:
    python convert_gt_to_fe.py \
        --gt    ../data/manual-fe-gt.csv \
        --pairs ../../data/wep/facts.csv \
        --out   ../results/fe_gt.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import pathlib

BASE = pathlib.Path(__file__).resolve().parent.parent

WEAPON_GT_COLS = [
    "אקדח", "תת מקלע", "תת מקלע מאולתר", "בקבוק תבערה",
    "מטען חבלה", "רימון רסס", "רימון הלם/גז", "טיל לאו ",
    "טיל מטאדור", "רובה צייד", "רובה צלפים", "מטען חבלה מאולתר",
    "רובה סער ", "רובה סער מאולתר",
]


def build_feature_vector(gt_row: dict) -> dict:
    fv: dict = {}

    # מספר עבירה
    off_num = gt_row.get("מספר עבירה", "").strip()
    if off_num:
        fv["מספר עבירה"] = off_num

    # סוג עבירה
    off_type = gt_row.get("סוג עבירה", "").strip()
    if off_type:
        fv["סוג עבירה"] = off_type

    # עבירות נוספות
    side = gt_row.get("עבירות נוספות", "").strip()
    if side:
        fv["עבירות נוספות"] = side

    # סוג הנשק — include all weapons present (value '1' or '2')
    for weapon in WEAPON_GT_COLS:
        val = gt_row.get(f"סוג הנשק [{weapon}]", "").strip()
        if val and val not in ("0", ""):
            try:
                fv[f"סוג הנשק [{weapon.strip()}]"] = float(val)
            except ValueError:
                fv[f"סוג הנשק [{weapon.strip()}]"] = 1.0

    other_weapon = gt_row.get("סוג הנשק - אם לא נמצא בטבלה", "").strip()
    if other_weapon:
        fv["סוג הנשק - אם לא נמצא בטבלה"] = other_weapon

    # סטטוס הנשק
    status = gt_row.get("סטטוס הנשק", "").strip()
    if status:
        fv["סטטוס הנשק"] = status

    # תכנון
    planning = gt_row.get("תכנון", "").strip()
    if planning:
        fv["תכנון"] = planning

    # אופן החזקת הנשק
    storage = gt_row.get("אופן החזקת הנשק", "").strip()
    if storage:
        fv["אופן החזקת הנשק"] = storage

    # אופן קבלת הנשק
    obtained = gt_row.get("אופן קבלת הנשק", "").strip()
    if obtained:
        fv["אופן קבלת הנשק"] = obtained

    # כמות תחמושת
    ammo = gt_row.get("כמות תחמושת", "").strip()
    if ammo:
        fv["כמות תחמושת"] = ammo

    # מטרה-סיבת העבירה
    purpose = gt_row.get("מטרה-סיבת העבירה", "").strip()
    if purpose:
        fv["מטרה-סיבת העבירה"] = purpose

    # שימוש
    usage = gt_row.get("שימוש", "").strip()
    if usage:
        fv["שימוש"] = usage

    return fv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt",    type=pathlib.Path,
                    default=BASE / "data/manual-fe-gt.csv")
    ap.add_argument("--pairs", type=pathlib.Path,
                    default=BASE.parent / "data/wep/facts.csv")
    ap.add_argument("--out",   type=pathlib.Path,
                    default=BASE / "results/fe_gt.csv")
    args = ap.parse_args()

    # load GT per verdict (GT rows only, deduplicated)
    gt: dict[str, dict] = {}
    with open(args.gt, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("data_source", "").strip() != "GT":
                continue
            vid = row.get("case", "").strip()
            if vid and vid not in gt:
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
