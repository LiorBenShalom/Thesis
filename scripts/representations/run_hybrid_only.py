#!/usr/bin/env python3
"""
Run ONLY hybrid experiments and merge results into the combined CSV.
This is a simple standalone script that won't affect other representations.

Uses the SAME logic as run_all_domains_unified_binary_no_fewshot_combined.py
"""

import csv
import os
import subprocess
import sys
from datetime import datetime
from typing import List, Tuple

DOMAINS = ["drugs", "wep"]
MODELS = ["gpt4", "gpt5mini", "mistral", "dicta"]

# Same build_suffix function from the original script
def build_suffix(*, use_cot: bool, use_few_shot: bool, use_unified_binary_prompt: bool, request_explanation: bool, request_confidence: bool) -> str:
    """Must match run_all_expirments.build_suffix()"""
    suffix_parts = []
    if use_cot:
        suffix_parts.append("cot")
    if not use_few_shot:
        suffix_parts.append("no_fewshot")
    if use_unified_binary_prompt:
        suffix_parts.append("unified_binary")
    if not request_explanation and not request_confidence:
        suffix_parts.append("no_extras")
    elif not request_explanation:
        suffix_parts.append("no_explanation")
    elif not request_confidence:
        suffix_parts.append("no_confidence")
    return "_" + "_".join(suffix_parts) if suffix_parts else ""


# Same PROMPT_VARIANTS from the original script
PROMPT_VARIANTS = [
    {
        "name": "labeling_only",
        "use_cot": False,
        "request_explanation": False,
        "request_confidence": False,
        "extra_args": ["--no-cot-only", "--no-explanation-confidence"],
    },
    {
        "name": "cot_labeling",
        "use_cot": True,
        "request_explanation": False,
        "request_confidence": False,
        "extra_args": ["--cot-only", "--no-explanation-confidence"],
    },
    {
        "name": "explainability",
        "use_cot": False,
        "request_explanation": True,
        "request_confidence": True,
        "extra_args": ["--no-cot-only"],
    },
]

COMBINED_CSV = "results_combined/unified_binary_no_fewshot_all_domains_all_prompt_variants.csv"


def find_summary_csv(domain: str, variant: dict) -> str:
    """Same logic as original script"""
    suffix = build_suffix(
        use_cot=variant["use_cot"],
        use_few_shot=False,
        use_unified_binary_prompt=True,
        request_explanation=variant["request_explanation"],
        request_confidence=variant["request_confidence"],
    )
    path = os.path.join(f"results_{domain}", f"final_results_summary_{domain}{suffix}.csv")
    return path


def read_csv(path: str, domain: str, prompt_variant: str) -> Tuple[List[str], List[dict]]:
    """Same logic as original script"""
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = []
        for row in reader:
            row_out = {"Domain": domain, "PromptVariant": prompt_variant}
            row_out.update(row)
            rows.append(row_out)
        return header, rows


def run_hybrid_experiments():
    """Run all hybrid experiments for both domains - same logic as run_one()"""
    print("\n" + "=" * 80)
    print("🔬 RUNNING HYBRID EXPERIMENTS ONLY")
    print("=" * 80)
    
    for domain in DOMAINS:
        for variant in PROMPT_VARIANTS:
            cmd = [
                sys.executable,
                "run_all_expirments.py",
                "--domain", domain,
                "--models", *MODELS,
                "--tasks", "binary_0",
                "--unified-binary-prompt",
                "--no-few-shot-only",
                "--representations", "hybrid",
                "--clear-checkpoint",  # Force fresh run for hybrid
                *variant["extra_args"],
            ]
            
            print(f"\n{'='*90}")
            print(f"🚀 Domain={domain} | PromptVariant={variant['name']} | Reps=hybrid | Models={MODELS}")
            print("Command:")
            print(f"  {' '.join(cmd)}")
            print(f"{'='*90}\n")
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ Error: {e}")


def merge_into_combined():
    """Merge hybrid results into the combined CSV - same logic as original"""
    print("\n" + "=" * 80)
    print("📦 MERGING HYBRID RESULTS INTO COMBINED CSV")
    print("=" * 80)
    
    if not os.path.exists(COMBINED_CSV):
        print(f"❌ Combined CSV not found: {COMBINED_CSV}")
        print("   Run full experiments first to create the combined file.")
        return
    
    # Load existing combined CSV
    with open(COMBINED_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        existing_rows = list(reader)
    
    print(f"   Loaded {len(existing_rows)} rows from combined CSV")
    
    # Remove old hybrid rows
    non_hybrid_rows = [r for r in existing_rows if r.get("Representation") != "Hybrid Manual+GPT"]
    print(f"   Keeping {len(non_hybrid_rows)} non-hybrid rows")
    
    # Read new hybrid results from per-domain summary CSVs (same logic as original)
    new_hybrid_rows = []
    for domain in DOMAINS:
        for variant in PROMPT_VARIANTS:
            summary_path = find_summary_csv(domain, variant)
            
            if not os.path.exists(summary_path):
                print(f"   ⚠️  Summary not found: {summary_path}")
                continue
            
            header, rows = read_csv(summary_path, domain=domain, prompt_variant=variant["name"])
            
            # Only keep hybrid rows
            for row in rows:
                if row.get("Representation") == "Hybrid Manual+GPT":
                    new_hybrid_rows.append(row)
    
    print(f"   Found {len(new_hybrid_rows)} new hybrid rows")
    
    # Combine and save
    all_rows = non_hybrid_rows + new_hybrid_rows
    
    with open(COMBINED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)
    
    print(f"   ✅ Saved {len(all_rows)} total rows to {COMBINED_CSV}")


def main():
    start = datetime.now()
    
    # Step 1: Run experiments
    run_hybrid_experiments()
    
    # Step 2: Merge results
    merge_into_combined()
    
    dur = datetime.now() - start
    print("\n" + "=" * 80)
    print(f"✅ ALL DONE in {dur}")
    print("=" * 80)


if __name__ == "__main__":
    main()
