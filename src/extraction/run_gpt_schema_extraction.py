"""
GPT Schema Extraction Pipeline
==============================
Builds `similarity_database_fe_gpt_schema.csv` using the **same granular extraction logic**
as `features_extract_drugs 2.py` (drugs) and `features_extract 2.py` (weapon): same classes,
prompts, grouping, and `to_dict()` output — not a single-shot JSON schema prompt.

Verdict scope (GT only):
- Unique IDs are taken **only** from `similarity_database_fe.csv` (columns verdict_1, verdict_2).
- Files under `verdict_csv/` are **not** scanned wholesale; only those GT IDs are read.

Workflow:
1. Collect unique verdict IDs from the GT CSV (same as manual / similarity labels)
2. Read full verdict text from verdict_csv/<ID>.csv (all parts concatenated)
3. Run DrugFeatureExtractor / FeatureExtractor on the full text (many GPT calls per verdict)
4. Save feature JSON into the same pair-rows format as the manual CSV (by default aligned to
   manual keys via `manual_schema_align.py` for apples-to-apples comparison; cache stores raw granular).
   שדה «עבירה» נכתב בפורמט קטגוריות קנוני (רשימת התוויות ב-`drug_offense_categories.py`) — מיד אחרי המיפוי מהחילוץ הגרנולרי, לא בסקריפט נפרד.
5. לחילוץ עבירה מעובדות כתב אישום בלבד (regex): `drug_offense_categories.canonical_offense_label_from_indictment_facts(text)`.
6. Optional: use output as MANUAL_CSV for hybrid enrichment in gpt_feature_database.ipynb

Cache file is separate from the legacy single-prompt pipeline so old caches are not reused.

Set ``OPENAI_API_KEY`` in the environment before running (the extractors use ``os.getenv``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import pandas as pd
from tqdm import tqdm

from manual_schema_align import align_granular_to_manual

# Same default model as DrugFeatureExtractor / FeatureExtractor in the feature modules
DEFAULT_MODEL = "gpt-4.1-mini"

# New cache name — incompatible with the old single-JSON prompt cache
CACHE_FILENAME = "gpt_schema_feature_cache_granular_fullverdict.json"


def _load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _get_granular_extractor_factory(code_dir: Path, domain: str) -> Callable[[str, str], Dict[str, Any]]:
    """Return a callable: full_verdict_text -> dict (same as extractor.to_dict())."""
    if domain == "drugs":
        path = code_dir / "features_extract_drugs 2.py"
        mod = _load_module_from_path("features_extract_drugs_2", path)
        cls = mod.DrugFeatureExtractor

        def factory(text: str, model: str) -> Dict[str, Any]:
            ex = cls(model=model, debug=False)
            ex.extract_all_features(text)
            return ex.to_dict()

        return factory

    if domain == "weapon":
        path = code_dir / "features_extract 2.py"
        mod = _load_module_from_path("features_extract_2", path)
        cls = mod.FeatureExtractor

        def factory(text: str, model: str) -> Dict[str, Any]:
            ex = cls(model=model, debug=False)
            ex.extract_all_features(text)
            return ex.to_dict()

        return factory

    raise ValueError(f"Unknown domain: {domain}")


def _feature_dict_for_csv(domain: str, granular: Dict[str, Any], align_manual_schema: bool) -> Dict[str, Any]:
    """Cache is always granular; align to manual FE keys when writing CSV (unless disabled)."""
    if not align_manual_schema:
        return granular
    if domain == "drugs":
        if "מכירה לסוכן" in granular and "section_13" not in granular:
            return granular
        return align_granular_to_manual("drugs", granular)
    if domain == "weapon":
        if "סוג הנשק [אקדח]" in granular and "pistol" not in granular:
            return granular
        return align_granular_to_manual("weapon", granular)
    return granular


def _json_default(o: Any):
    if hasattr(o, "item") and callable(o.item):
        try:
            return o.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def get_unique_verdicts(manual_csv_path: str) -> set:
    """Get unique verdict IDs from manual CSV"""
    df = pd.read_csv(manual_csv_path)
    verdicts = set(df["verdict_1"].unique()) | set(df["verdict_2"].unique())
    return verdicts


def _pick_verdict_ids_limited(
    df_manual: pd.DataFrame, unique_verdicts: Set[str], max_verdicts: int
) -> Set[str]:
    """
    Pick up to max_verdicts IDs in GT row order (not plain sort) so at least one pair
    is often fully inside the set — smoke tests get non-empty pair CSVs.
    """
    picked: List[str] = []
    seen: Set[str] = set()
    for _, row in df_manual.iterrows():
        for col in ("verdict_1", "verdict_2"):
            vid = row[col]
            if vid not in seen:
                seen.add(vid)
                picked.append(vid)
                if len(picked) >= max_verdicts:
                    return set(picked)
    for vid in sorted(unique_verdicts - seen):
        picked.append(vid)
        if len(picked) >= max_verdicts:
            break
    return set(picked[:max_verdicts])


def build_verdict_lookup(verdict_csv_dir: str, verdict_ids: set) -> Dict[str, str]:
    """Build lookup table: verdict_id -> full verdict text (all parts concatenated)."""
    lookup = {}
    vdir = Path(verdict_csv_dir)
    for vid in verdict_ids:
        fpath = vdir / f"{vid}.csv"
        if not fpath.exists():
            continue
        try:
            df = pd.read_csv(fpath)
            full_text = "\n".join(df["text"].dropna().astype(str).tolist())
            lookup[vid] = full_text
        except Exception as e:
            print(f"  Warning: could not read {fpath}: {e}")
    return lookup


def run_gpt_schema_extraction(
    domain: str,
    base_path: str,
    checkpoint_every: int = 10,
    model: str = DEFAULT_MODEL,
    sleep_between_verdicts: float = 0.5,
    max_verdicts: Optional[int] = None,
    artifact_tag: Optional[str] = None,
    align_manual_schema: bool = True,
    refetch: bool = False,
) -> str:
    """
    Run granular extraction (same logic as features_extract_* 2.py) and write fe_gpt_schema CSV.

    Args:
        domain: "weapon" or "drugs"
        base_path: Path to the domain folder (contains similarity_database_fe.csv, verdict_csv/)
        checkpoint_every: Save JSON cache every N newly processed verdicts
        model: OpenAI model name passed to DrugFeatureExtractor / FeatureExtractor
        sleep_between_verdicts: Pause after each verdict (many API calls per verdict)
        max_verdicts: If set, only process this many verdict IDs (sorted order, for smoke tests).
        artifact_tag: Suffix for cache/output files; if None and max_verdicts is set, defaults to smoke{N}.
        align_manual_schema: If True, CSV `feature_vector_*` use manual-schema keys (see manual_schema_align).
        refetch: If True, ignore cached entries for verdicts in this run and re-call the API for them.
    """
    print(f"\n{'='*60}")
    print(f"GPT Schema Extraction (granular, same as features_extract *2.py) - {domain.upper()}")
    print(f"  model={model}")
    print(f"  align_manual_schema={align_manual_schema} (CSV; cache stays granular)")
    print(f"  refetch={refetch} (re-extract from API, ignoring cache for selected verdicts)")
    if max_verdicts is not None:
        print(f"  max_verdicts={max_verdicts} (smoke / partial run)")
    print(f"{'='*60}\n")

    code_dir = Path(__file__).resolve().parent
    extract_fn = _get_granular_extractor_factory(code_dir, domain)

    tag = artifact_tag
    if tag is None and max_verdicts is not None:
        tag = f"smoke{max_verdicts}"
    tag_suffix = f"_{tag}" if tag else ""

    manual_csv = Path(base_path) / "similarity_database_fe.csv"
    verdict_csv_dir = Path(base_path) / "verdict_csv"
    output_csv = Path(base_path) / f"similarity_database_fe_gpt_schema{tag_suffix}.csv"
    feature_cache_json = Path(base_path) / CACHE_FILENAME.replace(".json", f"{tag_suffix}.json")

    print(f"📂 Manual CSV (GT): {manual_csv}")
    print("   ← מזהי תיקים נלקחים רק מכאן (לא כל קבצי verdict_csv).")
    print(f"📂 Verdict CSV dir: {verdict_csv_dir}")
    print(f"📂 Output CSV:      {output_csv}")
    print(f"📂 Cache:           {feature_cache_json}")

    feature_cache: Dict[str, Any] = {}
    if feature_cache_json.exists():
        with open(feature_cache_json, "r", encoding="utf-8") as f:
            feature_cache = json.load(f)
        print(f"📥 Loaded {len(feature_cache)} cached features")

    df_manual = pd.read_csv(manual_csv)
    unique_verdicts = set(df_manual["verdict_1"].unique()) | set(df_manual["verdict_2"].unique())
    print(f"\n📊 Found {len(unique_verdicts)} unique verdicts (GT: similarity_database_fe.csv)")

    extraction_set: Set[str] = set(unique_verdicts)
    if max_verdicts is not None:
        extraction_set = _pick_verdict_ids_limited(df_manual, unique_verdicts, max_verdicts)
        print(
            f"📊 Limited to {len(extraction_set)} verdict IDs (GT row order): {sorted(extraction_set)}"
        )

    verdict_lookup = build_verdict_lookup(str(verdict_csv_dir), extraction_set)
    print(f"📊 Full verdict text available: {len(verdict_lookup)} / {len(extraction_set)}")
    missing_verdicts = extraction_set - set(verdict_lookup.keys())
    if missing_verdicts:
        print(f"⚠️  Missing verdict files: {missing_verdicts}")

    if refetch:
        n_drop = 0
        for vid in extraction_set:
            if vid in feature_cache:
                del feature_cache[vid]
                n_drop += 1
        print(f"🔄 Refetch: removed {n_drop} verdicts from cache (will re-extract from API)")

    verdicts_to_process = [v for v in sorted(extraction_set) if v not in feature_cache and v in verdict_lookup]
    print(f"📊 Verdicts to process: {len(verdicts_to_process)}")
    print(
        f"📊 Already cached (in this artifact): {len([v for v in extraction_set if v in feature_cache])}"
    )

    new_count = 0
    for i, verdict_id in enumerate(tqdm(verdicts_to_process, desc="Extracting features")):
        facts = verdict_lookup.get(verdict_id, "")
        if not facts:
            continue

        try:
            feature_cache[verdict_id] = extract_fn(facts, model)
        except Exception as e:
            print(f"\n❌ Failed {verdict_id}: {e}")
            feature_cache[verdict_id] = {}

        new_count += 1
        if new_count % checkpoint_every == 0:
            with open(feature_cache_json, "w", encoding="utf-8") as f:
                json.dump(feature_cache, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Checkpoint saved: {len(feature_cache)} verdicts in cache")

        if sleep_between_verdicts > 0:
            time.sleep(sleep_between_verdicts)

    with open(feature_cache_json, "w", encoding="utf-8") as f:
        json.dump(feature_cache, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Final cache saved: {len(feature_cache)} verdicts")

    print("\n📝 Building output CSV (manual row layout)...")

    if max_verdicts is not None:
        mask = df_manual["verdict_1"].isin(extraction_set) & df_manual["verdict_2"].isin(extraction_set)
        df_pair = df_manual.loc[mask].copy()
        print(
            f"   Smoke mode: {len(df_pair)} pair rows where both verdicts are in the "
            f"{len(extraction_set)}-verdict extraction set (of {len(df_manual)} GT rows)."
        )
    else:
        df_pair = df_manual

    output_records = []
    for _, row in df_pair.iterrows():
        v1 = row["verdict_1"]
        v2 = row["verdict_2"]

        raw1 = feature_cache.get(v1, {})
        raw2 = feature_cache.get(v2, {})
        feat1 = _feature_dict_for_csv(domain, raw1, align_manual_schema)
        feat2 = _feature_dict_for_csv(domain, raw2, align_manual_schema)

        output_records.append(
            {
                "verdict_1": v1,
                "verdict_2": v2,
                "similarity_scale": row["similarity_scale"],
                "similarity_binary_0": row["similarity_binary_0"],
                "similarity_binary_1": row["similarity_binary_1"],
                "feature_vector_1": json.dumps(feat1, ensure_ascii=False, default=_json_default),
                "feature_vector_2": json.dumps(feat2, ensure_ascii=False, default=_json_default),
            }
        )

    df_output = pd.DataFrame(output_records)
    df_output.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n✅ Output saved to: {output_csv}")
    print(f"   Total pairs: {len(df_output)}")

    return str(output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build fe_gpt_schema CSV using granular extractors (features_extract *2.py)"
    )
    parser.add_argument(
        "--domain",
        choices=["weapon", "drugs", "both"],
        default="both",
        help="Domain to process",
    )
    parser.add_argument("--checkpoint", type=int, default=10, help="Save cache every N new verdicts")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model (default: {DEFAULT_MODEL}, same as feature extractors)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to sleep after each verdict (default 0.5; set 0 to disable)",
    )
    parser.add_argument(
        "--max-verdicts",
        type=int,
        default=None,
        metavar="N",
        help="Only process N GT verdict IDs (sorted). Writes *_smokeN.csv and separate cache unless --artifact-tag is set.",
    )
    parser.add_argument(
        "--artifact-tag",
        type=str,
        default=None,
        help="Suffix for output CSV and cache filenames (e.g. myrun). Default when using --max-verdicts: smoke{N}.",
    )
    parser.add_argument(
        "--no-manual-align",
        action="store_true",
        help="Write raw granular JSON to feature_vector_* (English keys). Default: align to manual keys via manual_schema_align.py.",
    )
    parser.add_argument(
        "--refetch",
        action="store_true",
        help="Ignore cache for GT verdicts in this run and re-call the API (still merges into same cache file).",
    )
    args = parser.parse_args()

    BASE_PATHS = {
        "weapon": "/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/weapon/",
        "drugs": "/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/drugs/",
    }

    domains = ["weapon", "drugs"] if args.domain == "both" else [args.domain]

    output_paths = {}
    for dom in domains:
        output_paths[dom] = run_gpt_schema_extraction(
            dom,
            BASE_PATHS[dom],
            checkpoint_every=args.checkpoint,
            model=args.model,
            sleep_between_verdicts=args.sleep,
            max_verdicts=args.max_verdicts,
            artifact_tag=args.artifact_tag,
            align_manual_schema=not args.no_manual_align,
            refetch=args.refetch,
        )

    print("\n" + "=" * 60)
    print("🎉 EXTRACTION COMPLETE!")
    print("=" * 60)
    print("\nOutput files (use as MANUAL_CSV for hybrid enrichment):")
    for dom, path in output_paths.items():
        print(f"  {dom}: {path}")
    print(f"\nCache (granular): see per-domain path printed above (e.g. *{CACHE_FILENAME} or *_smoke*.json)")
    print("\n📋 Next steps:")
    print("1. Point gpt_feature_database.ipynb MANUAL_CSV to similarity_database_fe_gpt_schema.csv")
    print("2. Run hybrid enrichment → similarity_database_hybrid_full_gpt.csv")
    print("3. Run downstream similarity experiments")
