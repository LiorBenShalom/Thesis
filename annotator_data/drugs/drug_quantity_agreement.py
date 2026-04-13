"""
drug_quantity_agreement.py

Per-pair agreement on drug annotations with partial credit for drug-set overlap
AND quantity tolerance.

Steps:
1. For each annotator row, parse the free-text "סוג הסם, כמות" cell via LLM into
   a structured list [{"drug": str, "amount": float, "unit": str}, ...].
   Results are cached locally (JSON) so the script can be re-run cheaply.
2. For each pair of annotators on the same case, compute a Jaccard-style score:
       score = |{d : d ∈ S_A ∩ S_B ∧ quantity_within_tolerance(A[d], B[d])}|
               / |S_A ∪ S_B|
   — "agreement" only counts a drug if both name AND quantity match.
3. Aggregate: mean Jaccard, and binary Cohen's κ (threshold = 0.5).

Tolerance: quantities are within tolerance if |a − b| ≤ 1 unit OR 0.9 ≤ a/b ≤ 1.1
(both required to pass is OR, so small absolute differences are forgiven even
at large ratios, and large nominal quantities tolerate proportional drift).

Usage:
    export OPENAI_API_KEY=...
    python drug_quantity_agreement.py \
        --responses annotator_responses.csv \
        --cache     drug_parse_cache.json \
        --out       drug_quantity_agreement.csv
"""
from __future__ import annotations
import argparse
import json
import os
import pathlib
import re
from itertools import combinations

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).resolve().parents[2] / ".env")
except Exception:
    pass

from openai import OpenAI

PROMPT = """
You are parsing an Israeli annotator's free-text drug-quantity entry from a
Google-Forms legal-case annotation. Extract every drug mentioned with its
quantity and unit.

Return STRICT JSON: {"drugs": [{"drug": "<hebrew name>", "amount": <number>, "unit": "<hebrew unit>"}, ...]}

Rules:
- Canonicalize drug names: קנבוס/קנאביס/קנביס → "קנבוס"; MDMAמ/אקסטזי → "MDMA";
  KETAMINE/קטמין → "קטמין"; קוקאין stays; LSD stays; חשיש stays;
  מתילמקאתינון/מתילמטאקתינון → "מתילמקאתינון"; "קנבוס בשתילים" if plants.
- Units: גרם / ק"ג / מ"ג / טבליות / שתילים / מ"ל / יחידות.
- If multiple entries of same drug appear, return one record per amount.
- If the text is empty or unparseable, return {"drugs": []}.
- Output ONLY the JSON, no prose, no code fences.

Input text:
{text}
"""


def call_llm(client: OpenAI, text: str, model: str) -> list[dict]:
    if not text or not text.strip():
        return []
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": PROMPT.replace("{text}", text)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    try:
        return json.loads(content).get("drugs", [])
    except json.JSONDecodeError:
        return []


def to_grams(amount: float, unit: str) -> tuple[float, str]:
    """Normalize weight-based units to grams. Non-weight units stay as-is."""
    u = unit.strip().replace('"', "").replace("'", "")
    if u in {"ק\"ג", "קג", "קילו", "קילוגרם"} or u == "קג":
        return amount * 1000, "גרם"
    if u in {"מ\"ג", "מג", "מיליגרם"}:
        return amount / 1000, "גרם"
    if u in {"גרם", "גר"}:
        return amount, "גרם"
    # non-weight units unchanged
    return amount, u


def within_tolerance(a: float, ua: str, b: float, ub: str) -> bool:
    an, un = to_grams(a, ua)
    bn, ub_n = to_grams(b, ub)
    if un != ub_n:
        return False  # can't compare incompatible units (e.g., גרם vs טבליות)
    if an == 0 or bn == 0:
        return an == bn
    if abs(an - bn) <= 1.0:
        return True
    return 0.9 <= min(an, bn) / max(an, bn) <= 1.0  # i.e. >=0.9 ratio


def pair_jaccard(a: list[dict], b: list[dict]) -> float:
    """Weighted Jaccard: a drug counts as agreed only if quantity within tolerance.
    |{d in A∩B with qty-match}| / |A∪B|.  If both empty → 1.0.
    """
    by_a = {d["drug"]: d for d in a if d.get("drug")}
    by_b = {d["drug"]: d for d in b if d.get("drug")}
    union = set(by_a) | set(by_b)
    if not union:
        return 1.0
    agreed = 0
    for drug in set(by_a) & set(by_b):
        da, db = by_a[drug], by_b[drug]
        if within_tolerance(
            float(da.get("amount", 0)), str(da.get("unit", "")),
            float(db.get("amount", 0)), str(db.get("unit", "")),
        ):
            agreed += 1
    return agreed / len(union)


def kappa_binary(a: list[int], b: list[int]) -> float:
    cats = sorted(set(a) | set(b))
    if len(cats) <= 1:
        return 1.0
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True, type=pathlib.Path)
    ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("drug_parse_cache.json"))
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="pair Jaccard ≥ threshold counts as agreement for binary κ")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set — check your .env")
    client = OpenAI()

    df = pd.read_csv(args.responses)
    df.columns = [c.strip() for c in df.columns]
    fname_col = "שם קובץ התיק"
    drug_col = "סוג הסם, כמות"
    df = df[df[fname_col].notna()].copy()
    df[fname_col] = df[fname_col].astype(str).str.strip().str.replace(r"\.docx?$", "", regex=True)

    # Load cache
    cache: dict[str, list[dict]] = {}
    if args.cache.exists():
        cache = json.loads(args.cache.read_text())

    # Parse each row's drug text via LLM (with caching)
    parsed = []
    for _, r in df.iterrows():
        text = str(r.get(drug_col, "") or "").strip()
        key = text
        if key not in cache:
            cache[key] = call_llm(client, text, args.model)
            args.cache.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
        parsed.append(cache[key])
    df["_drugs"] = parsed

    # Pairs per case
    dup = df.groupby(fname_col).filter(lambda g: len(g) > 1)
    rows = []
    jacs = []
    for fn, g in dup.groupby(fname_col):
        ds = g["_drugs"].tolist()
        taggers = g.get("שם המתייג", pd.Series(["?"] * len(g))).tolist()
        for i, j in combinations(range(len(ds)), 2):
            score = pair_jaccard(ds[i], ds[j])
            jacs.append(score)
            rows.append({
                "fname": fn,
                "tagger_a": taggers[i],
                "tagger_b": taggers[j],
                "drugs_a": json.dumps(ds[i], ensure_ascii=False),
                "drugs_b": json.dumps(ds[j], ensure_ascii=False),
                "jaccard": round(score, 3),
                "agreement": int(score >= args.threshold),
            })

    pair_df = pd.DataFrame(rows)
    pair_df.to_csv(args.out, index=False)

    mean_j = float(np.mean(jacs)) if jacs else float("nan")
    # binary κ on pair-level agreement vs chance (self-pair expected agreement ≈ mean)
    # We can compute Cohen's κ across all (tagger_a, tagger_b) pairs as a set,
    # but simplest: mean Jaccard + fraction above threshold.
    frac_agree = (pair_df["agreement"].sum() / len(pair_df)) if len(pair_df) else float("nan")

    print(f"pairs: {len(pair_df)}")
    print(f"mean Jaccard:             {mean_j:.3f}")
    print(f"fraction ≥ {args.threshold}:         {frac_agree:.3f}")
    print(f"fraction full agreement:  {(pair_df['jaccard']==1.0).mean():.3f}")
    print(f"fraction no overlap:      {(pair_df['jaccard']==0.0).mean():.3f}")
    print(f"\nsaved per-pair results → {args.out}")


if __name__ == "__main__":
    main()
