"""
consolidate_per_case.py  (weapon)

Per-case aggregation of weapon-types + ammo annotations across annotators.
Uses ammo_parse_cache.json (created by weapon_quantity_agreement.py).

Output: per_case_weapon_aggregate.csv with one row per case:
    fname, n_annotators,
    consolidated_weapons (union of weapon types across annotators),
    consolidated_ammo (median per kind),
    annotator_<name>_weapons, annotator_<name>_ammo,
    ammo_conflicts (drugs where quantities diverge beyond tolerance).

Usage:
    python consolidate_per_case.py \
        --responses weapon_v2_responses.csv \
        --mapping   mapping.csv \
        --facts     ../../data/wep/facts.csv \
        --cache     ammo_parse_cache.json \
        --out       per_case_weapon_aggregate.csv
"""
from __future__ import annotations
import argparse
import json
import pathlib
import re
from collections import defaultdict

import pandas as pd

from weapon_quantity_agreement import strip_zeros, ammo_within_tolerance


def median(xs: list[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True, type=pathlib.Path)
    ap.add_argument("--mapping", required=True, type=pathlib.Path)
    ap.add_argument("--facts", required=True, type=pathlib.Path)
    ap.add_argument("--cache", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()

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

    cache = json.loads(args.cache.read_text())
    weapon_cols = [c for c in df.columns if c.startswith("סוג הנשק [")]
    ammo_col = "כמות תחמושת"
    tagger_col = "מתייג"

    per_case: dict[str, dict] = {}
    for _, r in df.iterrows():
        fn = r["fname"]
        weapons = sorted(
            c[len("סוג הנשק ["):-1]
            for c in weapon_cols
            if pd.notna(r[c]) and str(r[c]).strip() not in ("", "0", "0.0")
        )
        text = str(r.get(ammo_col, "") or "").strip()
        ammo = cache.get(text, [])
        tagger = str(r.get(tagger_col, "?") or "?").strip()
        per_case.setdefault(fn, {"per_annotator": []})
        per_case[fn]["per_annotator"].append({
            "tagger": tagger, "weapons": weapons, "ammo": ammo,
        })

    rows = []
    for fn, info in per_case.items():
        # union of weapon types
        all_weapons = sorted({w for ann in info["per_annotator"] for w in ann["weapons"]})
        # consolidate ammo per kind
        by_kind: dict[str, list[float]] = defaultdict(list)
        for ann in info["per_annotator"]:
            for item in ann["ammo"]:
                k = item.get("kind", "other")
                a = float(item.get("amount", 0) or 0)
                by_kind[k].append(a)
        consolidated_ammo, conflicts = [], []
        for kind, amounts in by_kind.items():
            med = median(amounts)
            agrees = all(abs(x - med) <= 1 or 0.9 <= min(x, med) / max(x, med, 1e-9) <= 1.0
                         for x in amounts)
            consolidated_ammo.append({"kind": kind, "amount": med})
            if not agrees:
                conflicts.append(f"{kind}: {amounts}")
        row = {
            "fname": fn,
            "n_annotators": len(info["per_annotator"]),
            "consolidated_weapons": ", ".join(all_weapons),
            "consolidated_ammo": json.dumps(consolidated_ammo, ensure_ascii=False),
            "ammo_conflicts": "; ".join(conflicts),
        }
        for ann in info["per_annotator"]:
            row[f"annotator_{ann['tagger']}_weapons"] = ", ".join(ann["weapons"])
            row[f"annotator_{ann['tagger']}_ammo"] = json.dumps(ann["ammo"], ensure_ascii=False)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("fname")
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(out)} cases, {(out['ammo_conflicts']!='').sum()} with ammo conflicts)")


if __name__ == "__main__":
    main()
