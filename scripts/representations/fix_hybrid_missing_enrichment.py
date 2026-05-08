#!/usr/bin/env python3
"""
Fix hybrid_full_gpt: re-enrich the ~35 drug verdicts that were left with base-only (6 keys).
Saves comparison and updated CSV.

Output:
  data/final/drugs/hybrid_full_gpt_enrichment_fix/
    - comparison.csv           (before vs after, per verdict)
    - hybrid_full_gpt.csv      (updated full CSV)
    - enrichment_cache.json    (raw GPT responses)

Usage:
  cd new_try/experiments/scripts/representations
  python fix_hybrid_missing_enrichment.py
"""
import os, json, time
from pathlib import Path
import pandas as pd
from openai import OpenAI

try:
    from dotenv import load_dotenv
    for p in [Path(__file__).resolve().parents[2] / ".env",
              Path(__file__).resolve().parents[3] / ".env"]:
        if p.exists():
            load_dotenv(p)
            break
except Exception:
    pass

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    # Try reading directly
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                break

client = OpenAI(api_key=api_key)

EXP = Path(__file__).resolve().parents[2]
HYBRID_CSV = EXP / "data" / "final" / "drugs" / "hybrid_full_gpt.csv"
FACTS_CSV = EXP / "data" / "final" / "drugs" / "facts.csv"
OUT_DIR = EXP / "data" / "final" / "drugs" / "hybrid_full_gpt_enrichment_fix"
OUT_DIR.mkdir(exist_ok=True)

ENRICH_PROMPT = """אתה עוזר משפטי מומחה בנושאי סמים ונשק בישראל. תפקידך לחלץ מידע נוסף מפסק דין בעברית.

קיבלת:
1. **עובדות בתיק** - טקסט עובדות מכתב האישום או פסק הדין
2. **features שכבר חולצו** - מידע מובנה שחולץ מהפסק דין (JSON)

המשימה שלך:
- קרא את **עובדות בתיק**
- השתמש בעובדות כדי **להשלים ולהעשיר** את ה-features הקיימים
- **אל תמציא מידע!** רק אם מופיע במפורש בעובדות
- **אל תשנה fields קיימים** - רק תוסיף fields חדשים

הוסף **רק fields רלוונטיים** כגון:
- פרטי נסיבות (מקום, זמן, אופן ביצוע)
- שיתופי פעולה עם גורמים נוספים
- תוצאות חקירה (מעצר, חיפוש, עדויות)
- נסיבות אישיות (משפחה, עבודה, מצב כלכלי)
- כל מידע אחר שיכול להועיל לחיזוי דמיון בין תיקים

**החזר JSON** עם:
1. כל ה-fields מה-features המקוריים (ללא שינוי)
2. fields חדשים שהוספת מתוך העובדות

עובדות בתיק:
{facts}

Features שכבר חולצו:
{manual_features}

החזר JSON בלבד (ללא markdown, ללא ```json):
"""


def enrich(facts: str, features_json: str) -> dict:
    features = json.loads(features_json)
    prompt = ENRICH_PROMPT.format(
        facts=facts,
        manual_features=json.dumps(features, ensure_ascii=False, indent=2),
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "אתה עוזר משפטי מומחה. החזר JSON בלבד."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=2500,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content.strip()
            enriched = json.loads(content)
            result = features.copy()
            result.update(enriched)
            return result
        except Exception as e:
            print(f"    attempt {attempt+1} error: {e}")
            time.sleep(2 ** attempt)
    return features  # fallback


if __name__ == "__main__":
    # Load data
    hdf = pd.read_csv(HYBRID_CSV)
    fdf = pd.read_csv(FACTS_CSV)

    # Build facts lookup
    facts_lookup = {}
    for _, r in fdf.iterrows():
        v1, v2 = str(r["verdict_1"]), str(r["verdict_2"])
        f1 = str(r.get("indicment_facts_1", ""))
        f2 = str(r.get("indicment_facts_2", ""))
        if f1 and f1 != "nan" and len(f1) > 50:
            facts_lookup[v1] = f1
        if f2 and f2 != "nan" and len(f2) > 50:
            facts_lookup[v2] = f2

    # Find non-enriched verdicts
    non_enriched = set()
    for _, r in hdf.iterrows():
        for col in ["feature_vector_1", "feature_vector_2"]:
            vid = str(r["verdict_1"] if col == "feature_vector_1" else r["verdict_2"])
            fv = json.loads(r[col])
            if len(fv) <= 6:
                non_enriched.add(vid)

    print(f"Non-enriched verdicts to fix: {len(non_enriched)}")
    print(f"Facts available: {sum(1 for v in non_enriched if v in facts_lookup)}/{len(non_enriched)}")

    # Build features lookup (base features)
    features_lookup = {}
    for _, r in hdf.iterrows():
        v1, v2 = str(r["verdict_1"]), str(r["verdict_2"])
        if v1 not in features_lookup:
            features_lookup[v1] = r["feature_vector_1"]
        if v2 not in features_lookup:
            features_lookup[v2] = r["feature_vector_2"]

    # Enrich
    cache_path = OUT_DIR / "enrichment_cache.json"
    cache = {}
    if cache_path.exists():
        cache = json.load(open(cache_path, encoding="utf-8"))
        print(f"Loaded {len(cache)} from cache")

    comparison_rows = []
    to_process = [v for v in sorted(non_enriched) if v in facts_lookup and v not in cache]
    print(f"API calls needed: {len(to_process)}")

    for i, vid in enumerate(to_process):
        print(f"  [{i+1}/{len(to_process)}] {vid} ...", end=" ", flush=True)
        facts = facts_lookup[vid]
        base_features = features_lookup.get(vid, "{}")
        enriched = enrich(facts, base_features)
        cache[vid] = enriched
        new_keys = len(enriched) - len(json.loads(base_features))
        print(f"+{new_keys} keys")
        time.sleep(0.5)

        # Checkpoint
        if (i + 1) % 5 == 0:
            json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # Final save cache
    json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # Build comparison
    for vid in sorted(non_enriched):
        base = json.loads(features_lookup.get(vid, "{}"))
        enriched = cache.get(vid, base)
        comparison_rows.append({
            "verdict": vid,
            "base_keys": len(base),
            "enriched_keys": len(enriched),
            "added_keys": len(enriched) - len(base),
            "new_fields": ", ".join(sorted(set(enriched.keys()) - set(base.keys()))),
            "base_json": json.dumps(base, ensure_ascii=False),
            "enriched_json": json.dumps(enriched, ensure_ascii=False),
            "facts_length": len(facts_lookup.get(vid, "")),
        })

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(OUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
    print(f"\nComparison saved: {OUT_DIR / 'comparison.csv'}")
    print(f"  Average new keys: {comp_df['added_keys'].mean():.1f}")
    print(f"  Min: {comp_df['added_keys'].min()}, Max: {comp_df['added_keys'].max()}")

    # Build updated hybrid CSV
    updated_rows = []
    for _, r in hdf.iterrows():
        v1, v2 = str(r["verdict_1"]), str(r["verdict_2"])
        fv1 = cache.get(v1, json.loads(r["feature_vector_1"]))
        fv2 = cache.get(v2, json.loads(r["feature_vector_2"]))
        updated_rows.append({
            "verdict_1": r["verdict_1"],
            "verdict_2": r["verdict_2"],
            "similarity_scale": r["similarity_scale"],
            "similarity_binary_0": r["similarity_binary_0"],
            "similarity_binary_1": r["similarity_binary_1"],
            "feature_vector_1": json.dumps(fv1, ensure_ascii=False),
            "feature_vector_2": json.dumps(fv2, ensure_ascii=False),
        })

    udf = pd.DataFrame(updated_rows)
    udf.to_csv(OUT_DIR / "hybrid_full_gpt_fixed.csv", index=False, encoding="utf-8")

    # Stats
    enriched_count = sum(1 for _, r in udf.iterrows() if len(json.loads(r["feature_vector_1"])) > 6 or len(json.loads(r["feature_vector_2"])) > 6)
    print(f"\nUpdated CSV: {OUT_DIR / 'hybrid_full_gpt_fixed.csv'}")
    print(f"  Enriched rows: {enriched_count}/{len(udf)}")
    print(f"\nDone!")
