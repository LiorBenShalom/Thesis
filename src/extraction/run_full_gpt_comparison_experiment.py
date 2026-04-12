"""
Full GPT Comparison Experiment
==============================
This script runs the complete experiment comparing:
- Human Manual → Free GPT enrichment (current approach)  
- GPT Schema → Free GPT enrichment (alternative approach)

Steps:
1. Run GPT schema extraction on all verdicts
2. Run free GPT enrichment on the GPT schema features
3. Run 3 models on both hybrid representations
4. Generate comparison report
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
from datetime import datetime

# Import from existing modules
sys.path.append(str(Path(__file__).parent))

BASE_PATH = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
CODE_PATH = BASE_PATH / "code"

# Configuration
MODELS_TO_TEST = ["gpt4", "gpt5mini", "dicta"]  # 3 representative models
DOMAINS = ["drugs", "weapon"]


def step1_run_gpt_schema_extraction():
    """Step 1: Extract features using GPT with predefined schema"""
    print("\n" + "="*70)
    print("🔧 STEP 1: GPT Schema Feature Extraction")
    print("="*70)
    
    script_path = CODE_PATH / "run_gpt_schema_extraction.py"
    
    # Run for both domains
    cmd = ["python3", str(script_path), "--domain", "both", "--checkpoint", "10"]
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=str(CODE_PATH), capture_output=False)
    
    if result.returncode != 0:
        print("❌ GPT schema extraction failed!")
        return False
    
    # Verify output files exist
    for domain in DOMAINS:
        domain_path = BASE_PATH / domain
        output_file = domain_path / "similarity_database_fe_gpt_schema.csv"
        if not output_file.exists():
            print(f"❌ Output file not found: {output_file}")
            return False
        print(f"✅ {domain}: {output_file} created")
    
    return True


def step2_run_free_gpt_enrichment():
    """Step 2: Run free GPT enrichment on the GPT schema features"""
    print("\n" + "="*70)
    print("🔧 STEP 2: Free GPT Enrichment (on GPT Schema Features)")
    print("="*70)
    print("""
⚠️  MANUAL STEP REQUIRED:
    
1. Open gpt_feature_database.ipynb
2. For each domain, update MANUAL_CSV to point to the new GPT schema file:
   
   For DRUGS:
   MANUAL_CSV = base_path + "similarity_database_fe_gpt_schema.csv"
   OUTPUT_CSV = base_path + "similarity_database_hybrid_full_gpt.csv"
   
   For WEAPON:
   MANUAL_CSV = base_path + "similarity_database_fe_gpt_schema.csv"
   OUTPUT_CSV = base_path + "similarity_database_hybrid_full_gpt.csv"

3. Run the notebook for both domains
4. Verify output files:
   - /Users/liorb/.../new_try/drugs/similarity_database_hybrid_full_gpt.csv
   - /Users/liorb/.../new_try/weapon/similarity_database_hybrid_full_gpt.csv

Press ENTER when ready to continue...
""")
    input()
    
    # Verify output files
    for domain in DOMAINS:
        output_file = BASE_PATH / domain / "similarity_database_hybrid_full_gpt.csv"
        if not output_file.exists():
            print(f"❌ Output file not found: {output_file}")
            print("   Please run the notebook first!")
            return False
        print(f"✅ {domain}: {output_file} found")
    
    return True


def step3_run_model_predictions():
    """Step 3: Run models on the new hybrid_full_gpt representation"""
    print("\n" + "="*70)
    print("🔧 STEP 3: Running Model Predictions")
    print("="*70)
    
    # Add hybrid_full_gpt to representation options
    script_path = CODE_PATH / "similarity_experiment.py"
    
    results = {}
    
    for domain in DOMAINS:
        domain_dir = "weapon" if domain == "wep" else domain
        csv_path = BASE_PATH / domain_dir / "similarity_database_hybrid_full_gpt.csv"
        
        if not csv_path.exists():
            print(f"❌ CSV not found: {csv_path}")
            continue
        
        for model in MODELS_TO_TEST:
            print(f"\n📊 Running {model} on {domain} (hybrid_full_gpt)...")
            
            cmd = [
                "python3", str(script_path),
                "--csv", str(csv_path),
                "--representation", "features",
                "--model", model,
                "--task", "binary",
                "--binary_label", "binary_0",
                "--domain", domain
            ]
            
            print(f"   Command: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=str(CODE_PATH), capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"   ❌ Failed: {result.stderr}")
            else:
                print(f"   ✅ Completed")
                
            # Parse output for F1 score if available
            if "F1" in result.stdout:
                # Try to extract F1 score
                for line in result.stdout.split('\n'):
                    if 'F1' in line:
                        print(f"   {line.strip()}")
    
    return True


def step4_collect_and_compare():
    """Step 4: Collect results and generate comparison report"""
    print("\n" + "="*70)
    print("🔧 STEP 4: Collecting Results and Generating Comparison")
    print("="*70)
    
    results = []
    
    representations = {
        "hybrid": "similarity_database_hybrid.csv",  # Human manual + GPT free
        "hybrid_full_gpt": "similarity_database_hybrid_full_gpt.csv"  # GPT schema + GPT free
    }
    
    for domain in DOMAINS:
        domain_dir = "weapon" if domain == "wep" else domain
        domain_path = BASE_PATH / domain_dir
        
        for rep_name, csv_name in representations.items():
            for model in MODELS_TO_TEST:
                # Look for prediction files
                pred_pattern = f"*{rep_name}*{model}*binary_binary_0_preds.csv"
                pred_files = list(domain_path.glob(pred_pattern))
                
                if not pred_files:
                    # Try alternate pattern
                    if rep_name == "hybrid":
                        pred_pattern = f"similarity_database_hybrid_features_{model}_binary_binary_0_preds.csv"
                    else:
                        pred_pattern = f"similarity_database_hybrid_full_gpt_features_{model}_binary_binary_0_preds.csv"
                    
                    pred_file = domain_path / pred_pattern
                    if pred_file.exists():
                        pred_files = [pred_file]
                
                if pred_files:
                    df_pred = pd.read_csv(pred_files[0])
                    
                    # Calculate F1
                    if 'pred_binary' in df_pred.columns and 'similarity_binary_0' in df_pred.columns:
                        y_true = df_pred['similarity_binary_0'].values
                        y_pred = df_pred['pred_binary'].values
                        
                        # Calculate metrics
                        tp = np.sum((y_true == 1) & (y_pred == 1))
                        fp = np.sum((y_true == 0) & (y_pred == 1))
                        fn = np.sum((y_true == 1) & (y_pred == 0))
                        tn = np.sum((y_true == 0) & (y_pred == 0))
                        
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        
                        results.append({
                            'domain': domain.upper(),
                            'representation': rep_name,
                            'model': model,
                            'f1_score': f1,
                            'precision': precision,
                            'recall': recall,
                            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                            'n_samples': len(df_pred)
                        })
                        print(f"✅ {domain}/{rep_name}/{model}: F1={f1:.3f}")
                    else:
                        print(f"⚠️  Missing columns in {pred_files[0]}")
                else:
                    print(f"❓ No predictions found for {domain}/{rep_name}/{model}")
    
    # Create comparison DataFrame
    if results:
        df_results = pd.DataFrame(results)
        
        # Save results
        output_path = CODE_PATH / "hybrid_comparison_results.csv"
        df_results.to_csv(output_path, index=False)
        print(f"\n📁 Results saved to: {output_path}")
        
        # Generate comparison report
        generate_comparison_report(df_results)
    
    return True


def generate_comparison_report(df: pd.DataFrame):
    """Generate a detailed comparison report"""
    print("\n" + "="*70)
    print("📊 COMPARISON REPORT: Human vs GPT Schema")
    print("="*70)
    
    # Pivot for comparison
    pivot = df.pivot_table(
        index=['domain', 'model'], 
        columns='representation', 
        values='f1_score'
    ).reset_index()
    
    print("\n📈 F1 Scores by Domain and Model:\n")
    print(pivot.to_string(index=False))
    
    # Calculate improvement/degradation
    if 'hybrid' in pivot.columns and 'hybrid_full_gpt' in pivot.columns:
        pivot['diff'] = pivot['hybrid_full_gpt'] - pivot['hybrid']
        pivot['diff_pct'] = (pivot['diff'] / pivot['hybrid'] * 100).round(1)
        
        print("\n📊 Difference Analysis (Full GPT vs Human+GPT):\n")
        for _, row in pivot.iterrows():
            direction = "↑" if row['diff'] >= 0 else "↓"
            print(f"  {row['domain']}/{row['model']}: {direction} {abs(row['diff_pct']):.1f}%")
    
    # Overall summary
    print("\n" + "-"*70)
    print("📋 SUMMARY:\n")
    
    if 'hybrid' in df['representation'].values and 'hybrid_full_gpt' in df['representation'].values:
        human_avg = df[df['representation'] == 'hybrid']['f1_score'].mean()
        gpt_avg = df[df['representation'] == 'hybrid_full_gpt']['f1_score'].mean()
        
        print(f"  Human Manual + GPT Free (current):  Average F1 = {human_avg:.3f}")
        print(f"  GPT Schema + GPT Free (new):        Average F1 = {gpt_avg:.3f}")
        print(f"  Difference:                         {(gpt_avg - human_avg) * 100:+.1f}%")
        
        if gpt_avg > human_avg:
            print("\n  ✅ GPT Schema performs BETTER than Human annotation")
        elif gpt_avg < human_avg:
            print("\n  ⚠️ Human annotation still performs BETTER")
        else:
            print("\n  ➡️ Results are equivalent")
    
    # Save report
    report_path = CODE_PATH / "hybrid_comparison_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Hybrid Comparison Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"="*70 + "\n\n")
        f.write(df.to_string(index=False))
    
    print(f"\n📁 Full report saved to: {report_path}")


def main():
    """Main entry point"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     Full GPT Hybrid Comparison Experiment                            ║
║     Comparing: Human Manual → GPT Free                               ║
║            vs: GPT Schema → GPT Free                                 ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4], 
                       help="Run specific step only (1-4)")
    parser.add_argument("--skip-extraction", action="store_true",
                       help="Skip step 1 (schema extraction)")
    parser.add_argument("--skip-enrichment", action="store_true", 
                       help="Skip step 2 (free GPT enrichment)")
    parser.add_argument("--compare-only", action="store_true",
                       help="Only run step 4 (comparison)")
    args = parser.parse_args()
    
    if args.compare_only or args.step == 4:
        step4_collect_and_compare()
        return
    
    if args.step == 1:
        step1_run_gpt_schema_extraction()
        return
    
    if args.step == 2:
        step2_run_free_gpt_enrichment()
        return
    
    if args.step == 3:
        step3_run_model_predictions()
        return
    
    # Full pipeline
    print("Running full pipeline...\n")
    
    if not args.skip_extraction:
        if not step1_run_gpt_schema_extraction():
            print("❌ Step 1 failed. Aborting.")
            return
    
    if not args.skip_enrichment:
        if not step2_run_free_gpt_enrichment():
            print("❌ Step 2 failed. Aborting.")
            return
    
    if not step3_run_model_predictions():
        print("❌ Step 3 failed. Aborting.")
        return
    
    step4_collect_and_compare()
    
    print("\n" + "="*70)
    print("🎉 EXPERIMENT COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
