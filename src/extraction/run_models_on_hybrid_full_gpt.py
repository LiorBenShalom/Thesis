"""
Run models on hybrid_full_gpt representation and compare with original hybrid.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

BASE_PATH = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
CODE_PATH = BASE_PATH / "code"

# Models to test (3 representative models)
MODELS = ["gpt4", "gpt5mini", "dicta"]
DOMAINS = ["drugs", "weapon"]


def run_single_model(domain: str, model: str, csv_path: str, output_suffix: str = "hybrid_full_gpt"):
    """Run a single model on a CSV file."""
    print(f"\n{'='*60}")
    print(f"🚀 Running {model} on {domain} ({output_suffix})")
    print(f"   CSV: {csv_path}")
    print(f"{'='*60}\n")
    
    # Build command
    cmd = [
        "python3", str(CODE_PATH / "similarity_experiment.py"),
        "--csv", csv_path,
        "--representation", "features",
        "--model", model,
        "--task", "binary",
        "--binary_label", "binary_0",
        "--domain", domain,
        "--no_fewshot"  # Use clean predictions without few-shot
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=str(CODE_PATH), capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr[:500] if result.stderr else 'No error output'}")
        return None
    
    print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["drugs", "weapon", "both"], default="both")
    parser.add_argument("--model", choices=MODELS + ["all"], default="all")
    parser.add_argument("--dry-run", action="store_true", help="Just show what would be run")
    args = parser.parse_args()
    
    domains = DOMAINS if args.domain == "both" else [args.domain]
    models = MODELS if args.model == "all" else [args.model]
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     Running Models on hybrid_full_gpt Representation                 ║
║     Comparing: GPT Schema → GPT Free vs Human → GPT Free             ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    print(f"Domains: {domains}")
    print(f"Models: {models}")
    print(f"Total runs: {len(domains) * len(models)}")
    
    if args.dry_run:
        print("\n[DRY RUN - not actually running]")
        return
    
    for domain in domains:
        domain_dir = domain
        csv_path = str(BASE_PATH / domain_dir / "similarity_database_hybrid_full_gpt.csv")
        
        if not Path(csv_path).exists():
            print(f"❌ CSV not found: {csv_path}")
            continue
        
        for model in models:
            run_single_model(domain, model, csv_path)
    
    print("\n" + "="*60)
    print("✅ All model runs complete!")
    print("="*60)
    print("\nNext: Run comparison script to analyze results")


if __name__ == "__main__":
    main()
