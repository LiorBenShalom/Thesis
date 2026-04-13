"""
compute_agreement_kappa.py
Computes pairwise Cohen's kappa + exact agreement rate per feature
over multi-annotator cases in a gt-format annotator CSV.

Usage:
    python compute_agreement_kappa.py \
        --in  annotator_as_gt.csv \
        --out annotator_agreement_kappa.csv
"""
from __future__ import annotations
import argparse
import pathlib
import re
from itertools import combinations

import numpy as np
import pandas as pd


def norm(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def cohen_kappa(a, b):
    a, b = list(a), list(b)
    if len(a) < 2:
        return np.nan
    cats = sorted(set(a) | set(b))
    if len(cats) == 1:
        return 1.0
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def parse_role(val):
    """'תפקיד' encodes two binary sub-features: בעל הסמים and בעל המעבדה.
    Returns (owner: bool, lab: bool|None). Empty cell defaults owner=True.
    """
    if pd.isna(val) or not str(val).strip():
        return (True, None)
    s = str(val)
    owner = False if "לא בעל הסמים" in s else ("בעל הסמים" in s or True)
    if "לא בעל המעבדה" in s:
        lab = False
    elif "בעל המעבדה" in s:
        lab = True
    else:
        lab = None
    return (bool(owner), lab)


def to_category(val, col):
    """Map a cell value to a category for kappa.
    Free-text / drug-type columns: binary non_empty vs empty.
    Categorical fields (מעבדה, מכירה, נלוות, sections): use normalized text.
    `תפקיד` is handled separately (split into 2 sub-features).
    """
    v = norm(val)
    free_text = {
        "עונש עיקרי", "ענש נלווה", "הערות מחשבות",
        "מתחם ענישה מאשימה (פרקליט)", "מתחם ענישה בא כוח (סנגור)",
        "מתחם ענישה שופט",
        "עבירת סמים שלא הייתה ברשימה",
    }
    if col in free_text or col.startswith("סוג הסם ["):
        return "non_empty" if v else "empty"
    if col.startswith("עבירות ["):
        return "non_empty" if v else "empty"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    print(f"rows: {len(df)} | annotators: {df['שם המתייג'].value_counts().to_dict()}")

    # multi-annotator cases
    dup = df.groupby("שם קובץ התיק").filter(lambda g: len(g) > 1)
    print(f"cases w/ ≥2 annotators: {dup['שם קובץ התיק'].nunique()} | rows: {len(dup)}")

    sec_cols = [c for c in df.columns if c.startswith("עבירות [") or c in
                {"עבירת סמים שלא הייתה ברשימה", "עבירות נלוות כן/לא"}]
    drug_cols = [c for c in df.columns if c.startswith("סוג הסם [")]
    other_cols = [
        "מעבדה", "מכירה לסוכן", "עונש עיקרי", "ענש נלווה",
        "מתחם ענישה מאשימה (פרקליט)", "מתחם ענישה בא כוח (סנגור)",
        "מתחם ענישה שופט",
    ]
    all_cols = [c for c in sec_cols + drug_cols + other_cols if c in df.columns]

    results = []
    for col in all_cols:
        agree_cases = 0
        total_cases = 0
        pair_a, pair_b = [], []
        for _, grp in dup.groupby("שם קובץ התיק"):
            vals = [to_category(v, col) for v in grp[col]]
            total_cases += 1
            if len(set(vals)) == 1:
                agree_cases += 1
            for i, j in combinations(range(len(vals)), 2):
                pair_a.append(vals[i])
                pair_b.append(vals[j])
        k = cohen_kappa(pair_a, pair_b) if pair_a else np.nan
        results.append({
            "feature": col,
            "agreement_rate": agree_cases / total_cases if total_cases else 0,
            "cohen_kappa": k,
            "n_cases": total_cases,
            "n_pairs": len(pair_a),
        })

    # תפקיד — split into 2 binary sub-features
    if "תפקיד" in df.columns:
        # בעל הסמים (always present; empty defaults to True)
        pa, pb, ag, tot = [], [], 0, 0
        for _, grp in dup.groupby("שם קובץ התיק"):
            vals = [parse_role(v)[0] for v in grp["תפקיד"]]
            tot += 1
            if len(set(vals)) == 1:
                ag += 1
            for i, j in combinations(range(len(vals)), 2):
                pa.append(vals[i]); pb.append(vals[j])
        results.append({"feature": "תפקיד - בעל הסמים",
                        "agreement_rate": ag/tot if tot else 0,
                        "cohen_kappa": cohen_kappa(pa, pb) if pa else np.nan,
                        "n_cases": tot, "n_pairs": len(pa)})
        # בעל המעבדה — only cases where ≥2 annotators reported it
        pa, pb, ag, tot = [], [], 0, 0
        for _, grp in dup.groupby("שם קובץ התיק"):
            vals = [parse_role(v)[1] for v in grp["תפקיד"]]
            reported = [v for v in vals if v is not None]
            if len(reported) >= 2:
                tot += 1
                if len(set(reported)) == 1:
                    ag += 1
            for i, j in combinations(range(len(vals)), 2):
                if vals[i] is None or vals[j] is None:
                    continue
                pa.append(vals[i]); pb.append(vals[j])
        results.append({"feature": "תפקיד - בעל המעבדה",
                        "agreement_rate": ag/tot if tot else 0,
                        "cohen_kappa": cohen_kappa(pa, pb) if pa else np.nan,
                        "n_cases": tot, "n_pairs": len(pa)})

    res = pd.DataFrame(results).sort_values("cohen_kappa", ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(f"\n✅ wrote {args.out}")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
