#!/usr/bin/env python3
"""
Create a summary CSV with F1 and F2 metrics (mean, CI lower, CI upper) for each experiment.
Each row represents one experiment: Domain, Model, Representation, Few-Shot, Binary Label
"""

import pandas as pd
import os

def create_experiment_summary(input_csv, output_csv):
    """
    Create summary CSV with F1 and F2 metrics for each experiment.
    
    Args:
        input_csv: Path to bootstrap_full_results.csv
        output_csv: Path to output summary CSV
    """
    # Load full results
    df = pd.read_csv(input_csv)
    
    # Filter for F1 and F2 metrics only
    df_metrics = df[df['Metric'].isin(['F1', 'F2'])].copy()
    
    # Create summary rows
    summary_rows = []
    
    # Group by experiment configuration
    for (domain, representation, model, few_shot, binary_label), group in df_metrics.groupby(
        ['Domain', 'Representation', 'Model', 'Few-Shot', 'Binary Label']
    ):
        # Get F1 and F2 rows
        f1_row = group[group['Metric'] == 'F1']
        f2_row = group[group['Metric'] == 'F2']
        
        # Create summary row
        summary_row = {
            'Domain': domain,
            'Model': model,
            'Representation': representation,
            'Few-Shot': few_shot,
            'Binary Label': binary_label,
            'Task': f"{domain}_{few_shot}_{binary_label}",  # Combined task identifier
        }
        
        # Add F1 metrics
        if not f1_row.empty:
            summary_row['F1_Mean'] = f1_row['Mean'].values[0]
            summary_row['F1_CI_Lower'] = f1_row['CI Lower'].values[0]
            summary_row['F1_CI_Upper'] = f1_row['CI Upper'].values[0]
        else:
            summary_row['F1_Mean'] = None
            summary_row['F1_CI_Lower'] = None
            summary_row['F1_CI_Upper'] = None
        
        # Add F2 metrics
        if not f2_row.empty:
            summary_row['F2_Mean'] = f2_row['Mean'].values[0]
            summary_row['F2_CI_Lower'] = f2_row['CI Lower'].values[0]
            summary_row['F2_CI_Upper'] = f2_row['CI Upper'].values[0]
        else:
            summary_row['F2_Mean'] = None
            summary_row['F2_CI_Lower'] = None
            summary_row['F2_CI_Upper'] = None
        
        summary_rows.append(summary_row)
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_rows)
    
    # Sort by Domain, Model, Representation, Few-Shot, Binary Label
    summary_df = summary_df.sort_values(
        by=['Domain', 'Model', 'Representation', 'Few-Shot', 'Binary Label']
    )
    
    # Round numeric columns to 3 decimal places
    numeric_cols = ['F1_Mean', 'F1_CI_Lower', 'F1_CI_Upper', 
                    'F2_Mean', 'F2_CI_Lower', 'F2_CI_Upper']
    for col in numeric_cols:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].round(3)
    
    # Reorder columns
    column_order = [
        'Domain', 'Model', 'Representation', 'Few-Shot', 'Binary Label', 'Task',
        'F1_Mean', 'F1_CI_Lower', 'F1_CI_Upper',
        'F2_Mean', 'F2_CI_Lower', 'F2_CI_Upper'
    ]
    summary_df = summary_df[column_order]
    
    # Save to CSV
    summary_df.to_csv(output_csv, index=False)
    
    print(f"✅ Summary CSV created: {output_csv}")
    print(f"   Total experiments: {len(summary_df)}")
    print(f"\nFirst few rows:")
    print(summary_df.head(10).to_string())
    
    return summary_df


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(script_dir, "bootstrap_analysis_results", "bootstrap_full_results.csv")
    output_csv = os.path.join(script_dir, "bootstrap_analysis_results", "experiment_summary_f1_f2.csv")
    
    if not os.path.exists(input_csv):
        print(f"❌ Input file not found: {input_csv}")
        print("   Please run bootstrap_analysis.py first to generate the full results.")
        return
    
    print("📊 Creating experiment summary CSV...")
    print(f"   Input: {input_csv}")
    print(f"   Output: {output_csv}\n")
    
    summary_df = create_experiment_summary(input_csv, output_csv)
    
    print(f"\n✅ Done! Summary saved to: {output_csv}")


if __name__ == "__main__":
    main()








