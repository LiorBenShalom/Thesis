#!/usr/bin/env python3
"""
Extract H-Full features for the 5,191 sentencing-range verdicts.

Pipeline:
  Step 1 (schema, per-domain):
    - Use DrugFeatureExtractor / FeatureExtractor from new_try/code/features_extract*{,_drugs} 2.py
    - ~5 GPT calls per drugs verdict, ~11 per weapon verdict
    - Model: gpt-4.1-mini
    - Output: schema_cache_{domain}.json + features_schema.csv

  Step 2 (enrichment):
    - Take step-1 schema + full verdict text
    - One GPT call per verdict to add case-specific "free" fields
    - Model: gpt-4.1
    - Output: hybrid_full_cache.json + features_hybrid_full.csv

Usage:
  # Step 1 pilot (10 verdicts per domain):
  python3 extract_hfull_sentencing.py --step schema --limit 10

  # Step 1 full run:
  python3 extract_hfull_sentencing.py --step schema

  # Step 2 full run (after step 1 complete):
  python3 extract_hfull_sentencing.py --step enrich
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd
from tqdm import tqdm

# ---------- paths ----------
ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!")
CODE_DIR = ROOT / "new_try" / "code"
DATA_DIR = ROOT / "new_try" / "experiments" / "data" / "sentencing_range"
OUT_DIR = DATA_DIR / "hfull_features"
CLEAN_CSV = DATA_DIR / "verdicts_clean.csv"
VERDICT_CSV_DIR = DATA_DIR / "verdict_csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- models ----------
SCHEMA_MODEL = "gpt-4.1-mini"
ENRICH_MODEL = "gpt-4.1"


def _load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def get_extractor(domain: str) -> Callable[[str, str], Dict[str, Any]]:
    """Returns: (verdict_text, model) -> feature dict."""
    if domain == "drugs":
        path = CODE_DIR / "features_extract_drugs 2.py"
        mod = _load_module_from_path("features_extract_drugs_2", path)
        cls = mod.DrugFeatureExtractor
    elif domain == "weapon":
        path = CODE_DIR / "features_extract 2.py"
        mod = _load_module_from_path("features_extract_2", path)
        cls = mod.FeatureExtractor
    else:
        raise ValueError(f"Unknown domain: {domain}")

    def call(text: str, model: str) -> Dict[str, Any]:
        ex = cls(model=model, debug=False)
        ex.extract_all_features(text)
        return ex.to_dict()

    return call


def read_verdict_text(normalized_id: str) -> Optional[str]:
    """Read full verdict text from data_master/csv/<normalized_id>.csv."""
    fpath = VERDICT_CSV_DIR / f"{normalized_id}.csv"
    if not fpath.exists():
        return None
    try:
        df = pd.read_csv(fpath)
        if "text" not in df.columns:
            return None
        return "\n".join(df["text"].dropna().astype(str).tolist())
    except Exception:
        return None


def step1_schema_extraction(
    domain: str,
    limit: Optional[int] = None,
    checkpoint_every: int = 10,
    sleep_between: float = 0.3,
    refetch: bool = False,
) -> Path:
    """Run step 1 (GPT schema) for a single domain."""
    print(f"\n{'='*70}")
    print(f"STEP 1: GPT schema extraction — domain={domain}")
    print(f"  model={SCHEMA_MODEL}  limit={limit}  refetch={refetch}")
    print(f"{'='*70}\n")

    df = pd.read_csv(CLEAN_CSV)
    df = df[df["domain"] == domain].copy()
    print(f"📊 Verdicts in domain {domain}: {len(df):,}")

    if limit is not None:
        df = df.head(limit)
        print(f"📊 Limited to first {limit}")

    cache_path = OUT_DIR / f"schema_cache_{domain}.json"
    cache: Dict[str, Dict[str, Any]] = {}
    if cache_path.exists() and not refetch:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"📥 Loaded {len(cache):,} cached entries from {cache_path.name}")
    elif refetch:
        print("🔄 Refetch mode: ignoring existing cache")

    to_process = [
        (str(row["verdict"]), str(row["normalized_id"]))
        for _, row in df.iterrows()
        if str(row["verdict"]) not in cache
    ]
    print(f"📊 To process (not cached): {len(to_process):,}")

    if not to_process:
        print("✅ Nothing to do — all verdicts cached.")
        return cache_path

    extract_fn = get_extractor(domain)
    new_count = 0
    for verdict_id, norm_id in tqdm(to_process, desc=f"Extract {domain}"):
        text = read_verdict_text(norm_id)
        if not text:
            cache[verdict_id] = {"__error": f"no_text_for_{norm_id}"}
            continue

        try:
            cache[verdict_id] = extract_fn(text, SCHEMA_MODEL)
        except Exception as e:
            tqdm.write(f"  ❌ {verdict_id}: {type(e).__name__}: {e}")
            cache[verdict_id] = {"__error": str(e)}

        new_count += 1
        if new_count % checkpoint_every == 0:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            tqdm.write(f"  💾 checkpoint @ {len(cache):,}")

        if sleep_between > 0:
            time.sleep(sleep_between)

    # Final save
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: {cache_path}")
    print(f"   total entries: {len(cache):,}")
    print(f"   error entries: {sum(1 for v in cache.values() if '__error' in v):,}")
    return cache_path


def combine_schema_csv() -> Path:
    """Combine per-domain schema caches into a single per-verdict CSV."""
    rows = []
    for domain in ["drugs", "weapon"]:
        cache_path = OUT_DIR / f"schema_cache_{domain}.json"
        if not cache_path.exists():
            print(f"⚠️  Missing {cache_path.name}")
            continue
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        for vid, features in cache.items():
            if "__error" in features:
                continue
            rows.append({"verdict": vid, "domain": domain, "features_schema": json.dumps(features, ensure_ascii=False)})

    out = OUT_DIR / "features_schema.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"✅ Combined schema CSV: {out}  ({len(rows):,} rows)")
    return out


# ------------------ STEP 2 ------------------

ENRICHMENT_SYSTEM = "You are an expert legal analyst extracting structured facts from Israeli criminal verdicts."

def build_enrichment_prompt(indictment_facts: str, schema_features: dict, domain: str) -> str:
    schema_str = json.dumps(schema_features, ensure_ascii=False, indent=2)
    domain_he = "סמים" if domain == "drugs" else "נשק"
    return f"""להלן עובדות כתב אישום של תיק {domain_he}, יחד עם פיצ'רים שכבר חולצו.
המטרה: להעשיר את הפיצ'רים בשדות נוספים רלוונטיים לתיק הספציפי הזה (שאינם בסכימה הקיימת).

דוגמאות לשדות העשרה אפשריים: שיטת ביצוע, מסלול, שעת העבירה, סכום כספי, אמצעי תקשורת, אופן התפיסה, מניע, נסיבות מיוחדות, וכו'.

הנחיות:
- החזר אך ורק JSON תקין (ללא הסברים).
- כלול את *כל* השדות הקיימים בסכימה + שדות חדשים.
- שדה חדש צריך להיות פרטני, קצר, בעברית, ומבוסס על עובדות הכתב אישום בלבד.

## הסכימה הקיימת:
{schema_str}

## עובדות כתב האישום:
{indictment_facts}

החזר JSON אחד עם כל השדות:"""


def step2_enrichment(
    limit: Optional[int] = None,
    checkpoint_every: int = 10,
    sleep_between: float = 0.3,
    refetch: bool = False,
) -> Path:
    """Run step 2 (hybrid enrichment) on step-1 schema output."""
    print(f"\n{'='*70}")
    print(f"STEP 2: Hybrid enrichment")
    print(f"  model={ENRICH_MODEL}  limit={limit}  refetch={refetch}")
    print(f"{'='*70}\n")

    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        sys.exit(1)
    oai = OpenAI(api_key=api_key)

    # Load clean CSV + schema caches
    df = pd.read_csv(CLEAN_CSV)

    schema_by_verdict: Dict[str, dict] = {}
    for domain in ["drugs", "weapon"]:
        cache_path = OUT_DIR / f"schema_cache_{domain}.json"
        if not cache_path.exists():
            continue
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        for vid, feats in cache.items():
            if "__error" not in feats:
                schema_by_verdict[vid] = feats

    print(f"📥 Schema features available for {len(schema_by_verdict):,} verdicts")

    # Load enrichment cache
    enrich_path = OUT_DIR / "hybrid_full_cache.json"
    enrich_cache: Dict[str, dict] = {}
    if enrich_path.exists() and not refetch:
        with open(enrich_path, "r", encoding="utf-8") as f:
            enrich_cache = json.load(f)
        print(f"📥 Loaded {len(enrich_cache):,} cached enrichments")

    # Target set: verdicts that have schema + no enrichment yet
    target = df[df["verdict"].astype(str).isin(schema_by_verdict)].copy()
    target = target[~target["verdict"].astype(str).isin(enrich_cache)]
    if limit is not None:
        target = target.head(limit)
    print(f"📊 To enrich: {len(target):,}")

    if target.empty:
        print("✅ Nothing to do.")
        return enrich_path

    new_count = 0
    for _, row in tqdm(list(target.iterrows()), desc="Enrich"):
        vid = str(row["verdict"])
        facts = row.get("indictment_facts") or ""
        schema = schema_by_verdict.get(vid, {})
        prompt = build_enrichment_prompt(facts, schema, row["domain"])
        try:
            resp = oai.chat.completions.create(
                model=ENRICH_MODEL,
                messages=[
                    {"role": "system", "content": ENRICHMENT_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            raw = resp.choices[0].message.content or "{}"
            enriched = json.loads(raw)
            enrich_cache[vid] = enriched
        except Exception as e:
            tqdm.write(f"  ❌ {vid}: {type(e).__name__}: {e}")
            enrich_cache[vid] = {"__error": str(e)}

        new_count += 1
        if new_count % checkpoint_every == 0:
            with open(enrich_path, "w", encoding="utf-8") as f:
                json.dump(enrich_cache, f, ensure_ascii=False, indent=2)

        if sleep_between > 0:
            time.sleep(sleep_between)

    with open(enrich_path, "w", encoding="utf-8") as f:
        json.dump(enrich_cache, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: {enrich_path}  ({len(enrich_cache):,} entries)")
    return enrich_path


def combine_hfull_csv() -> Path:
    """Combine enrichment cache → per-verdict CSV."""
    enrich_path = OUT_DIR / "hybrid_full_cache.json"
    if not enrich_path.exists():
        print("⚠️  No enrichment cache found")
        return None
    with open(enrich_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    df = pd.read_csv(CLEAN_CSV)
    rows = []
    for _, row in df.iterrows():
        vid = str(row["verdict"])
        if vid in cache and "__error" not in cache[vid]:
            rows.append({
                "verdict": vid,
                "domain": row["domain"],
                "features_hybrid_full": json.dumps(cache[vid], ensure_ascii=False),
            })
    out = OUT_DIR / "features_hybrid_full.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"✅ Combined hybrid_full CSV: {out}  ({len(rows):,} rows)")
    return out


# ------------------ CLI ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["schema", "enrich", "combine"], required=True)
    ap.add_argument("--domain", choices=["drugs", "weapon", "both"], default="both",
                    help="Only used for step=schema")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N verdicts per domain")
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--refetch", action="store_true")
    args = ap.parse_args()

    if args.step == "schema":
        domains = ["drugs", "weapon"] if args.domain == "both" else [args.domain]
        for d in domains:
            step1_schema_extraction(
                domain=d,
                limit=args.limit,
                checkpoint_every=args.checkpoint_every,
                sleep_between=args.sleep,
                refetch=args.refetch,
            )
        combine_schema_csv()

    elif args.step == "enrich":
        step2_enrichment(
            limit=args.limit,
            checkpoint_every=args.checkpoint_every,
            sleep_between=args.sleep,
            refetch=args.refetch,
        )
        combine_hfull_csv()

    elif args.step == "combine":
        combine_schema_csv()
        combine_hfull_csv()


if __name__ == "__main__":
    main()
