"""
convert_gpt_cache_to_fe.py
Converts GPT extraction cache → feature-vector CSV compatible with
experiments/data/drugs/fe_gpt_schema.csv / manual_fe.csv format.

Each row in the output is a verdict-pair with:
  verdict_1, verdict_2, similarity_scale, similarity_binary_0, similarity_binary_1,
  feature_vector_1, feature_vector_2

Usage:
    python convert_gpt_cache_to_fe.py \
        --cache  ../cache/eval_drugs_results_gpt_cache.json \
        --pairs  ../../data/drugs/facts.csv \
        --out    ../results/fe_gpt_extracted.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path

# ── Section → offense-type label (same mapping as rebuild_manual_fe_from_gt.py) ──
SECTION_TO_OFFENSE: list[tuple[str, str]] = [
    ("סעיף_6",  "ייצור"),
    ("סעיף_7",  "החזקה שלא לצריכה עצמית"),
    ("סעיף_13", "יבוא/סחר"),
    ("סעיף_14", "יבוא/סחר"),   # same label – deduped below
    ("סעיף_19", "19"),
    ("סעיף_21", "כלים"),
    ("סעיף_22", "ניסיון"),
]

# ── Drug key in cache → display name in feature vector ──
DRUG_ORDER: list[tuple[str, str]] = [
    ("LSD",              "LSD"),
    ("METHAMPHETAMINE",  "METHAMPHETAMINE"),
    ("האיוואסקה",        "האיוואסקה"),
    ("קתינון",           "קתינון"),
    ("קטמין",            "קטמין"),
    ("חשיש",             "חשיש"),
    ("מתילמקאתינון",     "מתילמקאתינון"),
    ("קנבוס_בשתילים",    "קנבוס בשתילים"),
    ("קנבוס",            "קנבוס"),
    ("MDMA",             "MDMA"),
    ("קוקאין",           "קוקאין"),
]


def _format_amount(s: str) -> str:
    """'1255.48-גרם' → '1255.48 גרם',  'גרם-52.6' → '52.6 גרם'"""
    s = s.strip()
    parts = s.split("-", 1)
    if len(parts) == 2:
        a, b = parts
        try:
            return f"{float(a):g} {b}"
        except ValueError:
            pass
        try:
            return f"{float(b):g} {a}"
        except ValueError:
            pass
    return s


def _build_offense(offense_dict: dict) -> str:
    seen, out = set(), []
    for key, label in SECTION_TO_OFFENSE:
        if offense_dict.get(key, ""):
            if label not in seen:
                seen.add(label)
                out.append(label)
    return ", ".join(out)


def _build_drugs(drugs_dict: dict) -> str:
    parts = []
    for cache_key, display in DRUG_ORDER:
        amounts = drugs_dict.get(cache_key, [])
        if amounts:
            formatted = ", ".join(_format_amount(a) for a in amounts)
            parts.append(f"{display}: {formatted}")
    return ", ".join(parts)


def cache_to_feature_vector(pred: dict) -> dict:
    """Convert one cached GPT prediction to the standard feature-vector dict."""
    sale = "כן" if pred.get("מכירה_לסוכן", {}).get("מכירה_לסוכן") == 1 else "לא"
    lab  = "כן" if pred.get("מעבדה",        {}).get("מעבדה")        == 1 else "לא"
    side = "כן" if pred.get("עבירות_נלוות", {}).get("עבירות_נלוות") == 1 else "לא"

    tafkid = pred.get("תפקיד", {})
    role_parts = ["בעל הסמים" if tafkid.get("בעל_הסמים") == 1 else "לא בעל הסמים"]
    if "בעל_המעבדה" in tafkid:
        role_parts.append("בעל המעבדה" if tafkid.get("בעל_המעבדה") == 1 else "לא בעל המעבדה")
    role = ", ".join(role_parts)

    return {
        "מכירה לסוכן":        sale,
        "מעבדה":               lab,
        "סוג הסם, כמות":      _build_drugs(pred.get("סוג_הסם", {})),
        "עבירה":               _build_offense(pred.get("מספר_עבירה", {})),
        "עבירות נלוות כן/לא":  side,
        "תפקיד":               role,
    }


def main():
    BASE = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path,
                    default=BASE / "cache/eval_drugs_results_gpt_cache.json",
                    help="GPT extraction cache JSON")
    ap.add_argument("--pairs", type=Path,
                    default=BASE.parent / "data/drugs/facts.csv",
                    help="CSV with verdict pairs + similarity labels")
    ap.add_argument("--out",   type=Path,
                    default=BASE / "results/fe_gpt_extracted.csv",
                    help="Output feature-vector CSV")
    args = ap.parse_args()

    # Load cache
    with open(args.cache, encoding="utf-8") as f:
        cache: dict = json.load(f)
    print(f"Loaded {len(cache)} cached predictions")

    # Load pairs
    with open(args.pairs, newline="", encoding="utf-8-sig") as f:
        pairs = list(csv.DictReader(f))
    print(f"Loaded {len(pairs)} pairs")

    missing: set[str] = set()
    rows_out = []
    for row in pairs:
        v1, v2 = row["verdict_1"], row["verdict_2"]
        p1, p2 = cache.get(v1), cache.get(v2)
        if p1 is None: missing.add(v1)
        if p2 is None: missing.add(v2)
        fv1 = cache_to_feature_vector(p1) if p1 else {}
        fv2 = cache_to_feature_vector(p2) if p2 else {}
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
        print(f"⚠  Missing in cache ({len(missing)}): {sorted(missing)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=[
            "verdict_1", "verdict_2", "similarity_scale",
            "similarity_binary_0", "similarity_binary_1",
            "feature_vector_1", "feature_vector_2",
        ])
        w.writeheader()
        w.writerows(rows_out)

    print(f"✅ Wrote {len(rows_out)} pairs → {args.out}")


if __name__ == "__main__":
    main()
