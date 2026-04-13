"""
consolidate_per_case.py  (drugs)

Per-case aggregation of drug-quantity annotations across annotators.
Uses the LLM parses cached in drug_parse_cache.json (created by drug_quantity_agreement.py).

Output: per_case_drug_aggregate.csv with one row per case:
    fname, n_annotators, consolidated_drugs (median per drug if quantities agree),
    annotator_<name>_drugs (per-annotator parsed JSON), quantity_conflicts.

Usage:
    python consolidate_per_case.py \
        --responses annotator_responses.csv \
        --cache     drug_parse_cache.json \
        --out       per_case_drug_aggregate.csv
"""
from __future__ import annotations
import argparse
import json
import pathlib
import re
from collections import defaultdict

import pandas as pd

from drug_quantity_agreement import to_grams, within_tolerance


def consolidate(quantities: list[tuple[float, str]]) -> dict:
    if not quantities:
        return {"agrees": True, "median": None}
    normalized = [to_grams(a, u) for a, u in quantities]
    units = {u for _, u in normalized}
    if len(units) > 1:
        return {"agrees": False, "median": None}
    only_unit = list(units)[0]
    amounts = sorted(a for a, _ in normalized)
    median = amounts[len(amounts) // 2]
    agrees = all(within_tolerance(a, only_unit, median, only_unit) for a in amounts)
    return {"agrees": agrees, "median": (median, only_unit)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True, type=pathlib.Path)
    ap.add_argument("--cache", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()

    responses = pd.read_csv(args.responses)
    responses.columns = [c.strip() for c in responses.columns]
    cache = json.loads(args.cache.read_text())

    fname_col = "שם קובץ התיק"
    drug_col = "סוג הסם, כמות"
    tagger_col = "שם המתייג"

    responses = responses[responses[fname_col].notna()].copy()
    responses[fname_col] = responses[fname_col].astype(str).str.strip().str.replace(r"\.docx?$", "", regex=True)

    per_case: dict[str, dict] = {}
    for _, r in responses.iterrows():
        fn = str(r[fname_col]).strip()
        text = str(r.get(drug_col, "") or "").strip()
        tagger = str(r.get(tagger_col, "?") or "?").strip()
        parsed = cache.get(text, [])
        per_case.setdefault(fn, {"per_annotator": {}, "drugs": defaultdict(list)})
        per_case[fn]["per_annotator"][tagger] = parsed
        for d in parsed:
            per_case[fn]["drugs"][d["drug"]].append(
                (float(d.get("amount", 0) or 0), str(d.get("unit", "")))
            )

    rows = []
    for fn, info in per_case.items():
        consolidated, conflicts = [], []
        for drug, quants in info["drugs"].items():
            s = consolidate(quants)
            if s["median"]:
                consolidated.append({
                    "drug": drug,
                    "amount": s["median"][0],
                    "unit": s["median"][1],
                })
            if not s["agrees"]:
                conflicts.append(f'{drug}: {quants}')
        row = {
            "fname": fn,
            "n_annotators": len(info["per_annotator"]),
            "consolidated_drugs": json.dumps(consolidated, ensure_ascii=False),
            "quantity_conflicts": "; ".join(conflicts),
        }
        for tagger, ds in info["per_annotator"].items():
            row[f"annotator_{tagger}"] = json.dumps(ds, ensure_ascii=False)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("fname")
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(out)} cases, {(out['quantity_conflicts']!='').sum()} with quantity conflicts)")


if __name__ == "__main__":
    main()
