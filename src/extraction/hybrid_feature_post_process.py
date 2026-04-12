#!/usr/bin/env python3
"""
Post-processing pipeline for hybrid features — Enrich existing concept mapping.

Pipeline:
  1. Load existing semantic_analysis_results.json (concept mapping)
  2. Collect all unique feature keys from the 241 pairs (hybrid_full_gpt.csv)
  3. Identify keys NOT covered by the existing mapping
  4. Send ONLY missing keys to GPT with the list of existing concepts
  5. GPT assigns each missing key to an existing concept or creates a new one
  6. Merge into a unified mapping, convert feature vectors, save
"""

import csv
import json
import os
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception: pass
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI

csv.field_size_limit(500_000)

SCRIPT_DIR = Path(__file__).resolve().parent
NEW_TRY = SCRIPT_DIR.parent

HYBRID_DRUGS = NEW_TRY / "drugs" / "similarity_database_hybrid_full_gpt.csv"
HYBRID_WEAPON = NEW_TRY / "weapon" / "similarity_database_hybrid_full_gpt.csv"
EXISTING_MAPPING = SCRIPT_DIR / "semantic_analysis_results.json"
OUTPUT_DIR = SCRIPT_DIR / "post_process_output"

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    os.environ.get("OPENAI_API_KEY", ""),
)
GPT_MODEL = "o3"


# ─── Load existing mapping ───────────────────────────────────────────────────

def load_existing_mapping(path: Path) -> Dict[str, Dict[str, str]]:
    """Load semantic_analysis_results.json → {domain: {phrase: concept}}"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    result = {}
    for domain, domain_data in data.items():
        phrase_to_concept = {}
        for concept, phrases in domain_data.get("concept_features", {}).items():
            for p in phrases:
                phrase_to_concept[p] = concept
                phrase_to_concept[p.replace("_", " ")] = concept
                phrase_to_concept[p.replace(" ", "_")] = concept
        result[domain] = phrase_to_concept
    return result


def get_existing_concepts(path: Path, domain: str) -> Dict[str, List[str]]:
    """Get concept_name → [phrases] for a domain."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get(domain, {}).get("concept_features", {})


# ─── CSV / feature helpers ────────────────────────────────────────────────────

def load_pairs(csv_path):
    with open(csv_path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_feature_json(raw):
    if not raw or not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def collect_all_keys(pairs):
    counter = Counter()
    verdicts = set()
    for row in pairs:
        for col in ("feature_vector_1", "feature_vector_2"):
            d = parse_feature_json(row.get(col, ""))
            for k in d.keys():
                counter[k] += 1
        verdicts.add(row.get("verdict_1", ""))
        verdicts.add(row.get("verdict_2", ""))
    return counter, verdicts


# ─── GPT: assign missing keys using existing concepts ─────────────────────────

SYSTEM_ENRICH = """אתה מומחה ל-NLP ומשפט פלילי ישראלי.

## הקשר
יש לנו מיפוי קיים של פיצ'רים ל-concepts סמנטיים.
תקבל:
1. רשימת ה-concepts הקיימים עם דוגמאות לפיצ'רים שכבר ממופים אליהם
2. רשימת פיצ'רים חדשים שעדיין לא ממופים

## המשימה
לכל פיצ'ר חדש:
- אם הוא מתאר **אותו סוג מידע** כמו concept קיים — שייך אותו לאותו concept.
  (למשל "כמות הסם הכוללת" שייך ל-drug_quantity הקיים, "שיטת ההסתרה" שייך ל-concealment_method הקיים)
- אם אין concept מתאים — צור concept חדש (snake_case באנגלית).

## הנחיה חשובה
פיצ'ר שמשלב שני סוגי מידע יחד (למשל "סוג וכמות הסם") — שייך אותו ל-concept נפרד (כמו type_and_quantity),
כי פירוק מאבד את הקשר בין חלקי המידע.

## פורמט
ענה **רק** JSON: {"feature_name": "concept_name", ...}
כל פיצ'ר חדש שקיבלת חייב להופיע בתשובה."""


def gpt_enrich(missing_keys: List[str], existing_concepts: Dict[str, List[str]],
               domain: str, client) -> Dict[str, str]:
    """Send missing keys + existing concept list to GPT for assignment."""
    concept_summary = []
    for concept, phrases in sorted(existing_concepts.items()):
        examples = phrases[:5]
        concept_summary.append(f"  - {concept}: {examples}")
    concepts_text = "\n".join(concept_summary)

    missing_text = "\n".join(f'  {i+1}. "{k}"' for i, k in enumerate(missing_keys))

    user_msg = (
        f"דומיין: {domain}\n\n"
        f"## Concepts קיימים ({len(existing_concepts)}):\n{concepts_text}\n\n"
        f"## פיצ'רים חדשים לשיוך ({len(missing_keys)}):\n{missing_text}"
    )

    for attempt in range(3):
        try:
            print(f"    Calling {GPT_MODEL} with {len(missing_keys)} missing keys...")
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                max_completion_tokens=16384,
                messages=[
                    {"role": "developer", "content": SYSTEM_ENRICH},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content.strip())

            missing = [k for k in missing_keys if k not in result]
            if missing:
                print(f"    ⚠️  {len(missing)} keys missing from GPT response")
                if missing and attempt < 2:
                    print(f"    Retrying... (missing: {missing[:5]}...)")
                    time.sleep(3)
                    continue
                for k in missing:
                    result[k] = k

            print(f"    ✅ Got {len(result)} mappings")
            return result
        except Exception as e:
            print(f"    [attempt {attempt+1}] {e}")
            time.sleep(2 ** attempt)

    return {k: k for k in missing_keys}


# ─── Convert feature vectors ─────────────────────────────────────────────────

def _value_to_str(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def convert_features_to_concepts(feature_json, key_to_concept):
    raw = parse_feature_json(feature_json)
    if not raw:
        return feature_json

    concept_vals = defaultdict(list)
    for key, val in raw.items():
        concept = key_to_concept.get(key, key)
        concept_vals[concept].append(val)

    result = {}
    for concept, vals in sorted(concept_vals.items()):
        if len(vals) == 1:
            result[concept] = vals[0]
        else:
            numerics = [float(v) for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if len(numerics) == len(vals):
                result[concept] = round(sum(numerics) / len(numerics), 4)
            else:
                str_vals = []
                seen = set()
                for v in vals:
                    s = _value_to_str(v)
                    if s and s not in seen:
                        seen.add(s)
                        str_vals.append(s)
                result[concept] = "; ".join(str_vals) if len(str_vals) > 1 else (str_vals[0] if str_vals else vals[0])

    return json.dumps(result, ensure_ascii=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def process_domain(csv_path, domain, client, existing_phrase_map, existing_concepts):
    print(f"\n{'=' * 60}")
    print(f"  Domain: {domain.upper()}")
    print(f"{'=' * 60}")

    pairs = load_pairs(csv_path)
    key_counter, verdicts = collect_all_keys(pairs)
    all_keys = sorted(key_counter.keys())
    print(f"  {len(pairs)} pairs, {len(verdicts)} verdicts, {len(all_keys)} unique keys")

    # Split into already-mapped vs missing
    key_to_concept = {}
    missing_keys = []
    for k in all_keys:
        if k in existing_phrase_map:
            key_to_concept[k] = existing_phrase_map[k]
        elif k.replace("_", " ") in existing_phrase_map:
            key_to_concept[k] = existing_phrase_map[k.replace("_", " ")]
        elif k.replace(" ", "_") in existing_phrase_map:
            key_to_concept[k] = existing_phrase_map[k.replace(" ", "_")]
        else:
            missing_keys.append(k)

    print(f"\n  Already mapped: {len(key_to_concept)} keys")
    print(f"  Missing:        {len(missing_keys)} keys")

    if missing_keys:
        print(f"\n  Step 1: GPT enrichment ({len(missing_keys)} missing keys, {len(existing_concepts)} existing concepts)...")
        new_mappings = gpt_enrich(missing_keys, existing_concepts, domain, client)

        new_concepts = set()
        assigned_existing = 0
        for k, c in new_mappings.items():
            key_to_concept[k] = c
            if c in existing_concepts:
                assigned_existing += 1
            else:
                new_concepts.add(c)
        print(f"    → {assigned_existing} assigned to existing concepts")
        print(f"    → {len(new_concepts)} new concepts created")
        if new_concepts:
            print(f"    New concepts: {sorted(new_concepts)}")
    else:
        print(f"\n  All keys already mapped!")

    # Build concept → keys
    concept_to_keys: Dict[str, List[str]] = defaultdict(list)
    for key, concept in key_to_concept.items():
        concept_to_keys[concept].append(key)
    concept_to_keys = dict(concept_to_keys)
    n_concepts = len(concept_to_keys)
    print(f"\n  Total: {len(all_keys)} keys → {n_concepts} concepts ({(1 - n_concepts / len(all_keys)) * 100:.1f}% reduction)")

    # Convert feature vectors
    print(f"\n  Step 2: Converting feature vectors...")
    new_rows = []
    concepts_in_output = Counter()
    for row in pairs:
        fv1 = convert_features_to_concepts(row.get("feature_vector_1", ""), key_to_concept)
        fv2 = convert_features_to_concepts(row.get("feature_vector_2", ""), key_to_concept)
        for fv in (fv1, fv2):
            for c in parse_feature_json(fv):
                concepts_in_output[c] += 1
        new_rows.append({
            "verdict_1": row["verdict_1"],
            "verdict_2": row["verdict_2"],
            "similarity_scale": row.get("similarity_scale", ""),
            "similarity_binary_0": row.get("similarity_binary_0", ""),
            "similarity_binary_1": row.get("similarity_binary_1", ""),
            "feature_vector_1": fv1,
            "feature_vector_2": fv2,
        })

    actual = len(concepts_in_output)
    print(f"\n  FINAL: {len(all_keys)} raw keys → {actual} concepts")
    print(f"  Reduction: {len(all_keys) - actual} keys merged ({(1 - actual / len(all_keys)) * 100:.1f}%)")
    print(f"\n  Top 20 concepts:")
    for c, cnt in concepts_in_output.most_common(20):
        print(f"    {c}: {cnt}")

    multi_key = {c: ks for c, ks in concept_to_keys.items() if len(ks) > 1}
    print(f"\n  Concepts with multiple merged keys ({len(multi_key)}):")
    for c, ks in sorted(multi_key.items()):
        print(f"    {c}: {ks}")

    stats = {
        "domain": domain,
        "n_pairs": len(pairs),
        "raw_key_count": len(all_keys),
        "concept_count": actual,
        "concept_features": {c: ks for c, ks in concept_to_keys.items()},
        "feature_to_concept": key_to_concept,
    }
    return stats, new_rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=OPENAI_API_KEY)

    print("Loading existing mapping from semantic_analysis_results.json...")
    existing_maps = load_existing_mapping(EXISTING_MAPPING)

    all_stats = {}
    for csv_path, domain in [(HYBRID_DRUGS, "drugs"), (HYBRID_WEAPON, "weapon")]:
        if not csv_path.exists():
            print(f"  SKIP: {csv_path}")
            continue

        existing_phrase_map = existing_maps.get(domain, {})
        existing_concepts = get_existing_concepts(EXISTING_MAPPING, domain)
        print(f"\n  Existing mapping for {domain}: {len(existing_phrase_map)} phrases → {len(existing_concepts)} concepts")

        stats, new_rows = process_domain(csv_path, domain, client, existing_phrase_map, existing_concepts)
        all_stats[domain] = stats

        out_csv = OUTPUT_DIR / f"similarity_database_hybrid_concepts_{domain}.csv"
        with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "verdict_1", "verdict_2", "similarity_scale",
                "similarity_binary_0", "similarity_binary_1",
                "feature_vector_1", "feature_vector_2",
            ])
            w.writeheader()
            w.writerows(new_rows)
        print(f"\n  Saved: {out_csv}")

    # Save enriched mapping
    sem = {}
    for domain, stats in all_stats.items():
        sem[domain] = {
            "domain": domain,
            "raw_key_count": stats["raw_key_count"],
            "concept_count": stats["concept_count"],
            "method": "enrich_existing_mapping",
            "gpt_model": GPT_MODEL,
            "concept_features": stats["concept_features"],
            "feature_to_concept": stats["feature_to_concept"],
        }
    out_json = OUTPUT_DIR / "semantic_analysis_results_v2.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(sem, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_json}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — Enriched Concept Mapping")
    print("=" * 70)
    print(f"{'Domain':<12} {'Raw Keys':>10} {'Mapped':>10} {'Missing':>10} {'Concepts':>10} {'Reduction':>8}")
    print("-" * 70)
    for domain, s in all_stats.items():
        r, c = s["raw_key_count"], s["concept_count"]
        mapped = r - len([k for k in s["feature_to_concept"] if s["feature_to_concept"][k] == k])
        print(f"{domain.upper():<12} {r:>10} {mapped:>10} {r - mapped:>10} {c:>10} {(1 - c / r) * 100:>6.1f}%")
    print("-" * 70)
    tr = sum(s["raw_key_count"] for s in all_stats.values())
    tc = sum(s["concept_count"] for s in all_stats.values())
    print(f"{'TOTAL':<12} {tr:>10} {'':>10} {'':>10} {tc:>10} {(1 - tc / tr) * 100:>6.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
