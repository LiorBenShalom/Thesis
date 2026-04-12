#!/usr/bin/env python3
"""
Evaluates pipeline components against ground truth data.

Computes metrics for:
1. Sentencing Range Extraction - exact match, MAE, within tolerance
2. Citation Extraction - coverage, average count
3. Indictment Facts Extraction - coverage, length analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error


def evaluate_sentencing_range(gt_df: pd.DataFrame) -> dict:
    """Evaluate sentencing range extraction accuracy."""
    mask = gt_df['gt_range_low'].notna() & gt_df['model_range_low'].notna()
    valid = gt_df[mask]
    
    if len(valid) == 0:
        return {"error": "No valid samples for comparison"}
    
    # Exact match
    exact_low = (valid['gt_range_low'] == valid['model_range_low']).mean()
    exact_high = (valid['gt_range_high'] == valid['model_range_high']).mean()
    exact_both = ((valid['gt_range_low'] == valid['model_range_low']) & 
                  (valid['gt_range_high'] == valid['model_range_high'])).mean()
    
    # MAE
    mae_low = mean_absolute_error(valid['gt_range_low'], valid['model_range_low'])
    mae_high = mean_absolute_error(valid['gt_range_high'], valid['model_range_high'])
    
    # Within tolerance (±2 months)
    within_2_low = (abs(valid['gt_range_low'] - valid['model_range_low']) <= 2).mean()
    within_2_high = (abs(valid['gt_range_high'] - valid['model_range_high']) <= 2).mean()
    within_2_both = ((abs(valid['gt_range_low'] - valid['model_range_low']) <= 2) & 
                     (abs(valid['gt_range_high'] - valid['model_range_high']) <= 2)).mean()
    
    # Classification accuracy
    detected_positive = (gt_df['model_classification'] == 'POSITIVE').sum()
    
    return {
        "total_samples": len(gt_df),
        "valid_comparisons": len(valid),
        "exact_match_low": exact_low,
        "exact_match_high": exact_high,
        "exact_match_both": exact_both,
        "mae_low_months": mae_low,
        "mae_high_months": mae_high,
        "within_2_months_low": within_2_low,
        "within_2_months_high": within_2_high,
        "within_2_months_both": within_2_both,
        "classification_accuracy": detected_positive / len(gt_df)
    }


def evaluate_citations(gt_df: pd.DataFrame) -> dict:
    """Evaluate citation extraction coverage."""
    has_citations = gt_df['citations_count'] > 0
    
    return {
        "total_samples": len(gt_df),
        "verdicts_with_citations": has_citations.sum(),
        "coverage": has_citations.mean(),
        "avg_citations_per_verdict": gt_df['citations_count'].mean(),
        "max_citations": gt_df['citations_count'].max(),
        "min_citations": gt_df['citations_count'].min()
    }


def evaluate_indictment_facts(gt_df: pd.DataFrame) -> dict:
    """Evaluate indictment facts extraction coverage."""
    has_raw = gt_df['indictment_facts_raw'].notna() & (gt_df['indictment_facts_raw'] != '')
    has_gpt = gt_df['indictment_facts_gpt'].notna() & (gt_df['indictment_facts_gpt'] != '')
    
    avg_len_raw = gt_df.loc[has_raw, 'indictment_facts_raw'].str.len().mean()
    avg_len_gpt = gt_df.loc[has_gpt, 'indictment_facts_gpt'].str.len().mean()
    
    return {
        "total_samples": len(gt_df),
        "raw_extraction_coverage": has_raw.mean(),
        "gpt_extraction_coverage": has_gpt.mean(),
        "avg_length_raw_chars": avg_len_raw,
        "avg_length_gpt_chars": avg_len_gpt,
        "compression_ratio": avg_len_gpt / avg_len_raw if avg_len_raw > 0 else 0
    }


def print_metrics(metrics: dict, title: str):
    """Pretty print metrics."""
    print(f"\n{'=' * 50}")
    print(f"📊 {title}")
    print('=' * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'coverage' in key or 'match' in key or 'accuracy' in key or 'within' in key:
                print(f"  {key}: {value:.1%}")
            else:
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


def main():
    # Load ground truth
    BASE_DIR = Path(__file__).parent.parent
    gt_path = BASE_DIR / "evaluation" / "ground_truth_weapon.csv"
    
    if not gt_path.exists():
        print(f"❌ Ground truth file not found: {gt_path}")
        return
    
    gt_df = pd.read_csv(gt_path)
    print(f"Loaded ground truth: {len(gt_df)} samples")
    
    # Evaluate each component
    sentencing_metrics = evaluate_sentencing_range(gt_df)
    print_metrics(sentencing_metrics, "SENTENCING RANGE EXTRACTION")
    
    citation_metrics = evaluate_citations(gt_df)
    print_metrics(citation_metrics, "CITATION EXTRACTION")
    
    facts_metrics = evaluate_indictment_facts(gt_df)
    print_metrics(facts_metrics, "INDICTMENT FACTS EXTRACTION")
    
    # Summary
    print(f"\n{'=' * 50}")
    print("📈 SUMMARY")
    print('=' * 50)
    print(f"  Sentencing Range: {sentencing_metrics['exact_match_both']:.1%} exact match")
    print(f"  Citations: {citation_metrics['coverage']:.1%} coverage")
    print(f"  Indictment Facts: {facts_metrics['gpt_extraction_coverage']:.1%} coverage")


if __name__ == "__main__":
    main()



