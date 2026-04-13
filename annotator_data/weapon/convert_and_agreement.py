"""
convert_and_agreement.py  (weapon)
End-to-end: normalize annotator form → compute Cohen's kappa per feature.

Normalization:
- strip whitespace + .doc/.docx extensions
- strip leading zeros in tik numbers (e.g., "0189-01-16" → "189-01-16")
- map old tik IDs to canonical file-name IDs via Manual Feature Extraction - mapping.csv

Only cases appearing in experiments/data/wep/facts.csv are considered.

Usage:
    python convert_and_agreement.py \
        --responses "../Manual Feature Extraction Form 2017_verdicts to FE - V2 (תגובות) - תגובות לטופס 1.csv" \
        --mapping   "../Manual Feature Extraction - mapping.csv" \
        --facts     "../../data/wep/facts.csv" \
        --out       "weapon_v2_11_features_kappa.csv"
"""
from __future__ import annotations
import argparse
import pathlib
import re
from itertools import combinations

import numpy as np
import pandas as pd


def strip_zeros(s: str) -> str:
    return re.sub(r"\b0+(\d)", r"\1", re.sub(r"\.docx?$", "", str(s).strip()))


def normalize_tik(x, mapping: dict[str, str]) -> str:
    key = strip_zeros(x)
    return mapping.get(key, key)


def manual_kappa(a, b, weights=None):
    a, b = list(a), list(b)
    cats = sorted(set(a) | set(b))
    k = len(cats)
    if k <= 1:
        return 1.0
    idx = {c: i for i, c in enumerate(cats)}
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    n = O.sum()
    O /= n
    r = O.sum(axis=1)
    c = O.sum(axis=0)
    E = np.outer(r, c)
    if weights is None:
        W = 1 - np.eye(k)
    elif weights == "linear":
        W = np.array([[abs(i - j) for j in range(k)] for i in range(k)], float) / (k - 1)
    elif weights == "quadratic":
        W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], float) / (k - 1) ** 2
    num = (W * O).sum()
    den = (W * E).sum()
    return 1 - num / den if den else 1.0


FEATURES = [
    "מספר עבירה", "סוג העבירה", "שימוש", "עבירות נוספות",
    "כמות תחמושת", "מטרה-סיבת העבירה", "סטטוס הנשק",
    "אופן קבלת הנשק", "אופן החזקת הנשק", "תכנון",
]
FREE_TEXT = {
    "עונש", "עבירות נוספות", "כסף ששולם", "כמות תחמושת",
    "מתחם ענישה - מאשימה", "מתחם ענישה - שופט", "מטרה-סיבת העבירה",
}


def norm(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def to_cat(val, col):
    t = norm(val)
    if col in FREE_TEXT or col.startswith("סוג הנשק ["):
        return "ne" if t else "e"
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True, type=pathlib.Path)
    ap.add_argument("--mapping", required=True, type=pathlib.Path)
    ap.add_argument("--facts", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--only-in-facts", action="store_true", default=True)
    args = ap.parse_args()

    mapping = pd.read_csv(args.mapping)
    map_dict = {
        strip_zeros(k): strip_zeros(str(v).strip())
        for k, v in zip(mapping["מספר תיק_ישן"], mapping["case"])
    }

    df = pd.read_csv(args.responses)
    df.columns = [c.strip() for c in df.columns]
    df["fname"] = df["מספר תיק"].apply(lambda x: normalize_tik(x, map_dict))
    df = df[df["fname"].notna()]

    if args.only_in_facts:
        facts = pd.read_csv(args.facts)
        verdicts = {strip_zeros(x) for x in facts["verdict_1"]} | {strip_zeros(x) for x in facts["verdict_2"]}
        df = df[df["fname"].isin(verdicts)]

    dup = df.groupby("fname").filter(lambda g: len(g) > 1)
    print(f"multi-tagged cases: {dup['fname'].nunique()} | rows: {len(dup)}")

    weapon_cols = [c for c in df.columns if c.startswith("סוג הנשק [")]

    rows = []
    for col in FEATURES:
        if col not in df.columns:
            print(f"! missing column: {col}")
            continue
        A, B, agree, tot = [], [], 0, 0
        for _, g in dup.groupby("fname"):
            vals = [to_cat(v, col) for v in g[col]]
            tot += 1
            if len(set(vals)) == 1:
                agree += 1
            for i, j in combinations(range(len(vals)), 2):
                A.append(vals[i]); B.append(vals[j])
        rows.append({
            "feature": col,
            "agreement": f"{agree/tot:.1%}",
            "cohen_kappa": round(manual_kappa(A, B), 3),
            "weighted_kappa_linear": round(manual_kappa(A, B, "linear"), 3),
            "weighted_kappa_quadratic": round(manual_kappa(A, B, "quadratic"), 3),
            "n_cases": tot,
            "n_pairs": len(A),
        })

    # weapon set (sorted tuple of present weapon types)
    A, B, agree, tot = [], [], 0, 0
    for _, g in dup.groupby("fname"):
        sets = [tuple(sorted(c for c in weapon_cols if norm(r[c]))) for _, r in g.iterrows()]
        tot += 1
        if len(set(sets)) == 1:
            agree += 1
        for i, j in combinations(range(len(sets)), 2):
            A.append(sets[i]); B.append(sets[j])
    rows.append({
        "feature": "סוג הנשק (סט)",
        "agreement": f"{agree/tot:.1%}",
        "cohen_kappa": round(manual_kappa(A, B), 3),
        "weighted_kappa_linear": np.nan,  # not meaningful for set-valued feature
        "weighted_kappa_quadratic": np.nan,
        "n_cases": tot,
        "n_pairs": len(A),
    })

    res = pd.DataFrame(rows).sort_values("cohen_kappa", ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(res.to_string(index=False))
    print(f"\n✅ wrote {args.out}")


if __name__ == "__main__":
    main()
