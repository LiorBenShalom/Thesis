"""
compute_similarity_agreement.py
Computes inter-annotator agreement for the PAIR-SIMILARITY tagging (scale 1–3 or 1–4).
Handles both iterations (itr1 / itr2) for drugs and weapon.

Metrics:
- exact agreement
- Cohen's kappa (unweighted)
- weighted kappa (linear)
- weighted kappa (quadratic) — penalizes |1-3| distance more than |1-2|
- Pearson correlation

Usage:
    python compute_similarity_agreement.py \
        --weapon   weapon_similarity_gt.csv \
        --drugs    drugs_similarity_gt.csv \
        --out      similarity_agreement.csv
"""
from __future__ import annotations
import argparse
import pathlib
import numpy as np
import pandas as pd


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
    n = O.sum(); O /= n
    r = O.sum(axis=1); c = O.sum(axis=0)
    E = np.outer(r, c)
    if weights is None:
        W = 1 - np.eye(k)
    elif weights == "linear":
        W = np.array([[abs(i-j) for j in range(k)] for i in range(k)], float) / (k-1)
    elif weights == "quadratic":
        W = np.array([[(i-j)**2 for j in range(k)] for i in range(k)], float) / (k-1)**2
    num = (W * O).sum(); den = (W * E).sum()
    return 1 - num/den if den else 1.0


def metrics(df, label):
    """df has columns 'tagger_a' and 'tagger_b' (numeric)."""
    a = df["tagger_a"].astype(int).tolist()
    b = df["tagger_b"].astype(int).tolist()
    return {
        "dataset": label,
        "n_pairs": len(a),
        "exact_agreement": f"{sum(x==y for x,y in zip(a,b))/len(a):.1%}",
        "cohen_kappa": round(manual_kappa(a, b), 3),
        "weighted_kappa_linear": round(manual_kappa(a, b, "linear"), 3),
        "weighted_kappa_quadratic": round(manual_kappa(a, b, "quadratic"), 3),
        "pearson": round(pd.Series(a).corr(pd.Series(b)), 3),
    }


def parse_weapon(path: pathlib.Path):
    """Weapon similarity GT has two side-by-side blocks (itr1 cols 0-4, itr2 cols 8-12)."""
    raw = pd.read_csv(path, header=None, skiprows=4)
    def block(cols):
        b = raw.iloc[:, cols].copy()
        b.columns = ["v1","v2","tagger_a","tagger_b","final"]
        for c in ["tagger_a","tagger_b","final"]:
            b[c] = pd.to_numeric(b[c], errors="coerce")
        return b.dropna(subset=["tagger_a","tagger_b"])
    return block([0,1,2,3,4]), block([8,9,10,11,12])


def parse_drugs(path: pathlib.Path):
    """Drugs similarity GT has 'itr 2:' marker row between iterations."""
    d = pd.read_csv(path)
    idx = d[d["verdict_1"].astype(str).str.contains("itr 2", na=False)].index
    split = idx[0] if len(idx) else len(d)
    def block(df):
        b = df.rename(columns={"similarity_1":"tagger_a","similarity_2":"tagger_b"}).copy()
        b["tagger_a"] = pd.to_numeric(b["tagger_a"], errors="coerce")
        b["tagger_b"] = pd.to_numeric(b["tagger_b"], errors="coerce")
        return b.dropna(subset=["tagger_a","tagger_b"])
    itr1 = block(d.iloc[:split])
    itr2 = block(d.iloc[split+2:]) if len(idx) else pd.DataFrame()
    return itr1, itr2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weapon", type=pathlib.Path, required=True)
    ap.add_argument("--drugs", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    w1, w2 = parse_weapon(args.weapon)
    d1, d2 = parse_drugs(args.drugs)

    rows = [
        metrics(w1, "weapon itr1"),
        metrics(w2, "weapon itr2"),
        metrics(d1, "drugs itr1"),
        metrics(d2, "drugs itr2"),
    ]
    res = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200):
        print(res.to_string(index=False))
    print(f"\n✅ wrote {args.out}")


if __name__ == "__main__":
    main()
