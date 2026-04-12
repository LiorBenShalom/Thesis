#!/usr/bin/env python3
"""
Run hybrid enrichment on MY schema features (from features_extract_drugs/wep.py)
Creates similarity_database_hybrid_my_schema.csv
"""

import os
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception: pass
import json
import pandas as pd
from openai import OpenAI
from typing import Dict, Any
from pathlib import Path
import time
from tqdm import tqdm

# Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)

# Same prompt as used in gpt_feature_database.ipynb
user_prompt = """
אתה עוזר משפטי מומחה בנושאי סמים ונשק בישראל. תפקידך לחלץ מידע נוסף מפסק דין בעברית.

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


def parse_feature_json(json_str: str) -> Dict[str, Any]:
    """Parse JSON from string, handling various formats"""
    if pd.isna(json_str) or json_str == '' or json_str is None:
        return {}
    try:
        if isinstance(json_str, dict):
            return json_str
        # Clean up common issues
        json_str = json_str.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        return json.loads(json_str.strip())
    except:
        return {}


def enrich_single_case(facts: str, features_str: str) -> Dict[str, Any]:
    """Enrich features with facts using GPT"""
    # Parse features
    features_dict = parse_feature_json(features_str)
    if not features_dict:
        return {}
    
    # Prepare prompt
    features_json_display = json.dumps(features_dict, ensure_ascii=False, indent=2)
    prompt = user_prompt.format(facts=facts, manual_features=features_json_display)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "אתה עוזר משפטי מומחה. החזר JSON בלבד."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=2500,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        enriched = json.loads(content)
        
        # Merge: keep original features, add new ones
        result = features_dict.copy()
        result.update(enriched)
        return result
    
    except Exception as e:
        print(f"Error in enrichment: {e}")
        return features_dict  # Return original on error


def run_hybrid_on_my_schema(domain: str, base_path: str, checkpoint_every: int = 5):
    """Run hybrid enrichment on my_schema features"""
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING HYBRID ENRICHMENT ON MY_SCHEMA - {domain.upper()}")
    print(f"{'='*60}\n")
    
    # Paths
    my_schema_csv = Path(base_path) / "similarity_database_fe_my_schema.csv"
    facts_csv = Path(base_path) / "similarity_database_with_indicment_facts.csv"
    output_csv = Path(base_path) / "similarity_database_hybrid_my_schema.csv"
    cache_json = Path(base_path) / "hybrid_my_schema_cache.json"
    
    # Check inputs
    if not my_schema_csv.exists():
        print(f"❌ My schema features not found: {my_schema_csv}")
        print("   Run run_my_feature_extraction.py first!")
        return
    
    if not facts_csv.exists():
        print(f"❌ Facts file not found: {facts_csv}")
        return
    
    # Load data
    print(f"📂 Loading my_schema features: {my_schema_csv}")
    df_features = pd.read_csv(my_schema_csv)
    
    print(f"📂 Loading facts: {facts_csv}")
    df_facts = pd.read_csv(facts_csv)
    
    # Get facts column names
    facts_col_1 = 'indicment_facts_1' if 'indicment_facts_1' in df_facts.columns else 'indictment_facts_1'
    facts_col_2 = 'indicment_facts_2' if 'indicment_facts_2' in df_facts.columns else 'indictment_facts_2'
    
    # Build facts lookup
    facts_lookup = {}
    for _, row in df_facts.iterrows():
        v1 = row['verdict_1']
        v2 = row['verdict_2']
        if v1 not in facts_lookup and pd.notna(row.get(facts_col_1)):
            facts_lookup[v1] = row[facts_col_1]
        if v2 not in facts_lookup and pd.notna(row.get(facts_col_2)):
            facts_lookup[v2] = row[facts_col_2]
    
    print(f"📊 Facts available for {len(facts_lookup)} verdicts")
    
    # Load cache
    enrichment_cache = {}
    if cache_json.exists():
        with open(cache_json, 'r', encoding='utf-8') as f:
            enrichment_cache = json.load(f)
        print(f"📥 Loaded {len(enrichment_cache)} cached enrichments")
    
    # Get unique verdicts to process
    verdicts_in_features = set()
    for _, row in df_features.iterrows():
        verdicts_in_features.add(row['verdict_1'])
        verdicts_in_features.add(row['verdict_2'])
    
    # Build features lookup from df
    features_lookup = {}
    for _, row in df_features.iterrows():
        v1 = row['verdict_1']
        v2 = row['verdict_2']
        if v1 not in features_lookup:
            features_lookup[v1] = row['feature_vector_1']
        if v2 not in features_lookup:
            features_lookup[v2] = row['feature_vector_2']
    
    verdicts_to_process = [v for v in verdicts_in_features 
                          if v not in enrichment_cache and v in facts_lookup]
    
    print(f"📊 Verdicts to process: {len(verdicts_to_process)}")
    print(f"📊 Already cached: {len([v for v in verdicts_in_features if v in enrichment_cache])}")
    
    if verdicts_to_process:
        print(f"\n⚠️  This will make ~{len(verdicts_to_process)} API calls!")
        print("   Press Ctrl+C to cancel, or wait 5 seconds to continue...")
        time.sleep(5)
    
    # Process each verdict
    for i, verdict_id in enumerate(tqdm(verdicts_to_process, desc="Enriching")):
        facts = facts_lookup.get(verdict_id, "")
        features = features_lookup.get(verdict_id, "{}")
        
        if not facts:
            continue
        
        try:
            enriched = enrich_single_case(facts, features)
            enrichment_cache[verdict_id] = enriched
            
            # Checkpoint
            if (i + 1) % checkpoint_every == 0:
                with open(cache_json, 'w', encoding='utf-8') as f:
                    json.dump(enrichment_cache, f, ensure_ascii=False, indent=2)
                print(f"\n💾 Checkpoint: {len(enrichment_cache)} enrichments")
        
        except Exception as e:
            print(f"\n❌ Error processing {verdict_id}: {e}")
        
        time.sleep(0.3)  # Rate limiting
    
    # Final save
    with open(cache_json, 'w', encoding='utf-8') as f:
        json.dump(enrichment_cache, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Final cache: {len(enrichment_cache)} enrichments")
    
    # Build output CSV
    print("\n📝 Building output CSV...")
    output_records = []
    for _, row in df_features.iterrows():
        v1 = row['verdict_1']
        v2 = row['verdict_2']
        
        feat1 = enrichment_cache.get(v1, parse_feature_json(row['feature_vector_1']))
        feat2 = enrichment_cache.get(v2, parse_feature_json(row['feature_vector_2']))
        
        output_records.append({
            'verdict_1': v1,
            'verdict_2': v2,
            'similarity_scale': row['similarity_scale'],
            'similarity_binary_0': row['similarity_binary_0'],
            'similarity_binary_1': row['similarity_binary_1'],
            'feature_vector_1': json.dumps(feat1, ensure_ascii=False),
            'feature_vector_2': json.dumps(feat2, ensure_ascii=False)
        })
    
    df_output = pd.DataFrame(output_records)
    df_output.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\n✅ Output saved to: {output_csv}")
    
    return str(output_csv)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run hybrid enrichment on my_schema features")
    parser.add_argument("--domain", choices=["weapon", "drugs", "both"], default="both",
                       help="Domain to process")
    parser.add_argument("--checkpoint", type=int, default=5,
                       help="Save checkpoint every N verdicts")
    args = parser.parse_args()
    
    BASE_PATHS = {
        "weapon": "/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/weapon/",
        "drugs": "/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/drugs/"
    }
    
    domains = ["weapon", "drugs"] if args.domain == "both" else [args.domain]
    
    for domain in domains:
        run_hybrid_on_my_schema(domain, BASE_PATHS[domain], args.checkpoint)
    
    print("\n" + "="*60)
    print("🎉 HYBRID ENRICHMENT COMPLETE!")
    print("="*60)
