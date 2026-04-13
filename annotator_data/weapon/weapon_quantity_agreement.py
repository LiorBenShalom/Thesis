"""
weapon_quantity_agreement.py

Per-pair inter-annotator agreement on weapon types + ammo quantities.

The form encodes weapon types as separate columns (`סוג הנשק [אקדח]`, `סוג הנשק [תת מקלע]`, …)
plus a free-text `כמות תחמושת` field. We treat each annotator response as a structured
record: the set of weapon types selected + parsed ammo quantity.

Score per pair = Jaccard on weapon-type set, multiplied by ammo-quantity-match
(1.0 if within tolerance, else 0.5 for partial credit).

Usage:
    export OPENAI_API_KEY=...
    python weapon_quantity_agreement.py \
        --responses weapon_v2_responses.csv \
        --mapping   mapping.csv \
        --facts     ../../data/wep/facts.csv \
        --cache     ammo_parse_cache.json \
        --out       weapon_quantity_agreement.csv
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


def strip_zeros(s: str) -> str:
    return re.sub(r"\b0+(\d)", r"\1", re.sub(r"\.docx?$", "", str(s).strip()))


PROMPT = """
Parse this Israeli annotator's free-text ammunition-quantity entry.
Return STRICT JSON: {"items": [{"kind": "<bullets|magazines|other>", "amount": <number>, "raw_unit": "<hebrew>"}, ...]}

Rules:
- "כדורים", "תחמושת", "קליעים", "כדור", "טען" → kind="bullets"
- "מחסניות", "מחסנית" → kind="magazines"
- Otherwise → kind="other"
- Empty / unparseable → {"items": []}.
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
    try:
        return json.loads(resp.choices[0].message.content).get("items", [])
    except json.JSONDecodeError:
        return []


def ammo_within_tolerance(a: list[dict], b: list[dict]) -> float:
    """Returns 1.0 if quantity-match per kind, 0.5 partial, 0.0 mismatch."""
    by_a = {x["kind"]: float(x.get("amount", 0) or 0) for x in a}
    by_b = {x["kind"]: float(x.get("amount", 0) or 0) for x in b}
    if not by_a and not by_b:
        return 1.0
    if not by_a or not by_b:
        return 0.5
    common = set(by_a) & set(by_b)
    if not common:
        return 0.0
    matches = 0
    for k in common:
        ax, bx = by_a[k], by_b[k]
        if ax == 0 and bx == 0:
            matches += 1
        elif abs(ax - bx) <= 1 or 0.9 <= min(ax, bx) / max(ax, bx, 1e-9) <= 1.0:
            matches += 1
    return matches / max(len(by_a), len(by_b))


def jaccard_set(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True, type=pathlib.Path)
    ap.add_argument("--mapping", required=True, type=pathlib.Path)
    ap.add_argument("--facts", required=True, type=pathlib.Path)
    ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path("ammo_parse_cache.json"))
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set — check your .env")
    client = OpenAI()

    mapping = pd.read_csv(args.mapping)
    map_dict = {strip_zeros(k): strip_zeros(str(v).strip())
                for k, v in zip(mapping["מספר תיק_ישן"], mapping["case"])}

    df = pd.read_csv(args.responses)
    df.columns = [c.strip() for c in df.columns]
    df["fname"] = df["מספר תיק"].apply(lambda x: map_dict.get(strip_zeros(x), strip_zeros(x)))
    df = df[df["fname"].notna()].copy()

    facts = pd.read_csv(args.facts)
    verdicts = {strip_zeros(x) for x in facts["verdict_1"]} | {strip_zeros(x) for x in facts["verdict_2"]}
    df = df[df["fname"].isin(verdicts)]

    weapon_cols = [c for c in df.columns if c.startswith("סוג הנשק [")]
    ammo_col = "כמות תחמושת"

    # cache LLM ammo parses
    cache: dict[str, list[dict]] = {}
    if args.cache.exists():
        cache = json.loads(args.cache.read_text())

    parsed_ammo = []
    for _, r in df.iterrows():
        text = str(r.get(ammo_col, "") or "").strip()
        if text not in cache:
            cache[text] = call_llm(client, text, args.model)
            args.cache.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
        parsed_ammo.append(cache[text])
    df["_ammo"] = parsed_ammo

    def weapon_set(row):
        return frozenset(
            c[len("סוג הנשק ["):-1]
            for c in weapon_cols
            if pd.notna(row[c]) and str(row[c]).strip() not in ("", "0", "0.0")
        )
    df["_weapons"] = df.apply(weapon_set, axis=1)

    dup = df.groupby("fname").filter(lambda g: len(g) > 1)
    rows, scores = [], []
    for fn, g in dup.groupby("fname"):
        ws = g["_weapons"].tolist()
        ams = g["_ammo"].tolist()
        taggers = g.get("מתייג", pd.Series(["?"] * len(g))).tolist()
        for i, j in combinations(range(len(ws)), 2):
            j_w = jaccard_set(ws[i], ws[j])
            a_match = ammo_within_tolerance(ams[i], ams[j])
            score = j_w * (0.5 + 0.5 * a_match)  # weight ammo as half
            scores.append(score)
            rows.append({
                "fname": fn,
                "tagger_a": taggers[i],
                "tagger_b": taggers[j],
                "weapons_a": ", ".join(sorted(ws[i])),
                "weapons_b": ", ".join(sorted(ws[j])),
                "ammo_a": json.dumps(ams[i], ensure_ascii=False),
                "ammo_b": json.dumps(ams[j], ensure_ascii=False),
                "weapon_jaccard": round(j_w, 3),
                "ammo_match": round(a_match, 3),
                "combined_score": round(score, 3),
            })

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"pairs: {len(out)}")
    print(f"mean weapon-set Jaccard: {np.mean([r['weapon_jaccard'] for r in rows]):.3f}")
    print(f"mean ammo match:         {np.mean([r['ammo_match'] for r in rows]):.3f}")
    print(f"mean combined score:     {np.mean(scores):.3f}")
    print(f"pairs with full match:   {(out['combined_score']==1.0).mean():.1%}")
    print(f"\nsaved per-pair → {args.out}")


if __name__ == "__main__":
    main()
