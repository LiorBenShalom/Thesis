#!/usr/bin/env python3
"""
Statistical Significance Analysis: REPRESENTATIONS Comparison
Tests if Hybrid Manual+GPT or Manual Feature Schema consistently and
statistically significantly outperform GPT Free, GPT Law, Raw Indictment Facts.

Uses McNemar's test for paired comparisons on the same samples.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """
    McNemar's test for paired nominal data.
    Tests if two representations have significantly different error rates.
    """
    # Build contingency table
    n12 = np.sum((y_pred_a == y_true) & (y_pred_b != y_true))  # A correct, B incorrect
    n21 = np.sum((y_pred_a != y_true) & (y_pred_b == y_true))  # A incorrect, B correct
    
    b, c = n12, n21
    
    if b + c == 0:
        return 0, 1.0, b, c
    
    # McNemar's test with continuity correction
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    # Use exact binomial test for small samples
    if b + c < 25:
        result = stats.binomtest(b, b + c, 0.5)
        p_value = result.pvalue
    
    return statistic, p_value, b, c


def benjamini_hochberg(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction for multiple comparisons."""
    n = len(p_values)
    if n == 0:
        return np.array([]), np.array([])
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    
    adjusted = np.zeros(n)
    for i, p in enumerate(sorted_p):
        adjusted[sorted_idx[i]] = p * n / (i + 1)
    
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    
    return adjusted, adjusted < alpha


def cohens_h(p1, p2):
    """Cohen's h effect size for proportions."""
    phi1 = 2 * np.arcsin(np.sqrt(max(0, min(1, p1))))
    phi2 = 2 * np.arcsin(np.sqrt(max(0, min(1, p2))))
    return phi1 - phi2


# ============================================================================
# DATA LOADING AND ALIGNMENT
# ============================================================================

def load_predictions(base_dir):
    """Load prediction files for all domains (labeling_only only)."""
    pred_files = {
        'drugs': os.path.join(base_dir, 'results_drugs', 'preds_csv', 'detailed_predictions_drugs.csv'),
        'wep': os.path.join(base_dir, 'results_wep', 'preds_csv', 'detailed_predictions_wep.csv'),
    }
    
    all_data = {}
    for domain, path in pred_files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['prediction'] = pd.to_numeric(df['prediction'], errors='coerce')
            df['true_label'] = pd.to_numeric(df['true_label'], errors='coerce')
            all_data[domain] = df
    
    return all_data


def get_aligned_predictions_by_representation(df, rep_a, rep_b, model, task):
    """
    Get aligned predictions for two REPRESENTATIONS on the same samples (same model).
    Returns matching y_true, y_pred_a, y_pred_b arrays.
    """
    df_a = df[(df['model'] == model) & (df['representation'] == rep_a) & (df['task'] == task)].copy()
    df_b = df[(df['model'] == model) & (df['representation'] == rep_b) & (df['task'] == task)].copy()
    
    if len(df_a) == 0 or len(df_b) == 0:
        return None, None, None
    
    # Create unique sample identifier
    df_a['sample_id'] = df_a['verdict_1'].astype(str) + '_' + df_a['verdict_2'].astype(str)
    df_b['sample_id'] = df_b['verdict_1'].astype(str) + '_' + df_b['verdict_2'].astype(str)
    
    # Find common samples
    common_samples = set(df_a['sample_id']) & set(df_b['sample_id'])
    
    if len(common_samples) == 0:
        return None, None, None
    
    # Align predictions
    df_a = df_a[df_a['sample_id'].isin(common_samples)].drop_duplicates('sample_id').set_index('sample_id')
    df_b = df_b[df_b['sample_id'].isin(common_samples)].drop_duplicates('sample_id').set_index('sample_id')
    
    df_a = df_a.sort_index()
    df_b = df_b.loc[df_a.index]
    
    # Filter valid predictions
    valid_mask = (
        df_a['prediction'].notna() & df_a['prediction'].isin([0, 1]) &
        df_b['prediction'].notna() & df_b['prediction'].isin([0, 1]) &
        df_a['true_label'].notna() & df_a['true_label'].isin([0, 1])
    )
    
    y_true = df_a.loc[valid_mask, 'true_label'].values.astype(int)
    y_pred_a = df_a.loc[valid_mask, 'prediction'].values.astype(int)
    y_pred_b = df_b.loc[valid_mask, 'prediction'].values.astype(int)
    
    return y_true, y_pred_a, y_pred_b


# ============================================================================
# REPRESENTATION NAME MAPPING
# ============================================================================

REP_NAME_MAP = {
    'facts': 'Raw Indictment Facts',
    'gpt_free': 'GPT Free Extraction',
    'gpt_law': 'GPT Law Extraction',
    'hybrid': 'Hybrid Manual+GPT',
    'manual': 'Manual Feature Schema',
}

REP_SHORT_MAP = {v: k for k, v in REP_NAME_MAP.items()}

# Structured vs Unstructured grouping
STRUCTURED_REPS = ['hybrid', 'manual']
UNSTRUCTURED_REPS = ['gpt_free', 'gpt_law', 'facts']


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 100)
    print("🔬 STATISTICAL SIGNIFICANCE: REPRESENTATION COMPARISON")
    print("   Testing if Hybrid/Manual consistently beat GPT Free/GPT Law/Raw Facts")
    print("=" * 100)
    
    # Load data
    all_data = load_predictions(base_dir)
    if not all_data:
        print("❌ No data found!")
        return
    
    # Combine for unified analysis
    df_combined = pd.concat(all_data.values(), ignore_index=True)
    all_data['unified'] = df_combined
    
    models = ['dicta', 'gpt4', 'gpt5mini', 'mistral']
    representations = ['facts', 'gpt_free', 'gpt_law', 'hybrid', 'manual']
    tasks = ['binary_0', 'binary_1']
    
    # =========================================================================
    # PAIRWISE REPRESENTATION COMPARISONS
    # =========================================================================
    print("\n" + "=" * 100)
    print("📊 PAIRWISE REPRESENTATION COMPARISONS (McNemar's Test)")
    print("=" * 100)
    
    all_comparisons = []
    
    for domain_name, df in all_data.items():
        for task in tasks:
            for model in models:
                rep_pairs = list(combinations(representations, 2))
                
                for rep_a, rep_b in rep_pairs:
                    y_true, y_pred_a, y_pred_b = get_aligned_predictions_by_representation(
                        df, rep_a, rep_b, model, task
                    )
                    
                    if y_true is None or len(y_true) < 10:
                        continue
                    
                    # Calculate metrics
                    f1_a = f1_score(y_true, y_pred_a, pos_label=1, zero_division=0)
                    f1_b = f1_score(y_true, y_pred_b, pos_label=1, zero_division=0)
                    acc_a = accuracy_score(y_true, y_pred_a)
                    acc_b = accuracy_score(y_true, y_pred_b)
                    
                    # McNemar's test
                    stat, p_value, b, c = mcnemar_test(y_true, y_pred_a, y_pred_b)
                    
                    # Effect size
                    effect_size = cohens_h(acc_a, acc_b)
                    
                    all_comparisons.append({
                        'Domain': domain_name.upper(),
                        'Task': task,
                        'Model': model,
                        'Rep_A': REP_NAME_MAP.get(rep_a, rep_a),
                        'Rep_B': REP_NAME_MAP.get(rep_b, rep_b),
                        'Rep_A_Code': rep_a,
                        'Rep_B_Code': rep_b,
                        'N_Samples': len(y_true),
                        'F1_A': f1_a,
                        'F1_B': f1_b,
                        'F1_Diff': f1_a - f1_b,
                        'Acc_A': acc_a,
                        'Acc_B': acc_b,
                        'McNemar_Stat': stat,
                        'P_Value': p_value,
                        'Cohens_h': effect_size,
                        'A_correct_B_wrong': b,
                        'B_correct_A_wrong': c,
                    })
    
    df_comparisons = pd.DataFrame(all_comparisons)
    
    if len(df_comparisons) == 0:
        print("❌ No comparisons found!")
        return
    
    # Apply FDR correction
    p_values = df_comparisons['P_Value'].values
    fdr_adjusted, fdr_sig = benjamini_hochberg(p_values)
    df_comparisons['P_FDR'] = fdr_adjusted
    df_comparisons['Sig_FDR'] = fdr_sig
    
    # Effect size interpretation
    df_comparisons['Effect_Size'] = df_comparisons['Cohens_h'].apply(
        lambda h: 'Large' if abs(h) > 0.8 else ('Medium' if abs(h) > 0.5 else ('Small' if abs(h) > 0.2 else 'Negligible'))
    )
    
    # =========================================================================
    # KEY ANALYSIS: Structured vs Unstructured
    # =========================================================================
    print("\n" + "=" * 100)
    print("🎯 KEY QUESTION: Do Structured Reps (Hybrid, Manual) beat Unstructured (GPT Free, GPT Law, Raw)?")
    print("=" * 100)
    
    # Filter to Structured vs Unstructured comparisons only
    structured_vs_unstructured = df_comparisons[
        ((df_comparisons['Rep_A_Code'].isin(STRUCTURED_REPS)) & (df_comparisons['Rep_B_Code'].isin(UNSTRUCTURED_REPS))) |
        ((df_comparisons['Rep_A_Code'].isin(UNSTRUCTURED_REPS)) & (df_comparisons['Rep_B_Code'].isin(STRUCTURED_REPS)))
    ].copy()
    
    # Determine winner in each comparison
    def get_winner_info(row):
        if row['F1_A'] > row['F1_B']:
            winner = row['Rep_A_Code']
            winner_name = row['Rep_A']
        else:
            winner = row['Rep_B_Code']
            winner_name = row['Rep_B']
        
        winner_type = 'Structured' if winner in STRUCTURED_REPS else 'Unstructured'
        return pd.Series({'Winner': winner_name, 'Winner_Type': winner_type})
    
    winner_info = structured_vs_unstructured.apply(get_winner_info, axis=1)
    structured_vs_unstructured = pd.concat([structured_vs_unstructured, winner_info], axis=1)
    
    # Summary statistics
    print("\n📈 OVERALL WIN RATE (Structured vs Unstructured):")
    print("-" * 60)
    
    total_comparisons = len(structured_vs_unstructured)
    structured_wins = (structured_vs_unstructured['Winner_Type'] == 'Structured').sum()
    structured_sig_wins = ((structured_vs_unstructured['Winner_Type'] == 'Structured') & 
                           (structured_vs_unstructured['Sig_FDR'])).sum()
    unstructured_wins = total_comparisons - structured_wins
    unstructured_sig_wins = ((structured_vs_unstructured['Winner_Type'] == 'Unstructured') & 
                              (structured_vs_unstructured['Sig_FDR'])).sum()
    
    print(f"  Total comparisons: {total_comparisons}")
    print(f"  Structured wins:   {structured_wins} ({structured_wins/total_comparisons*100:.1f}%) - {structured_sig_wins} statistically significant")
    print(f"  Unstructured wins: {unstructured_wins} ({unstructured_wins/total_comparisons*100:.1f}%) - {unstructured_sig_wins} statistically significant")
    
    # Binomial test: Is Structured winning significantly more than chance (50%)?
    binom_result = stats.binomtest(structured_wins, total_comparisons, 0.5, alternative='greater')
    print(f"\n  📊 Binomial test (H0: 50% win rate):")
    print(f"     Structured win rate: {structured_wins/total_comparisons*100:.1f}%")
    print(f"     p-value (one-sided): {binom_result.pvalue:.4f}")
    print(f"     Conclusion: {'✅ Structured SIGNIFICANTLY better' if binom_result.pvalue < 0.05 else '❌ NOT significantly better'}")
    
    # =========================================================================
    # DETAILED BREAKDOWN BY REPRESENTATION
    # =========================================================================
    print("\n" + "=" * 100)
    print("📊 DETAILED: Each Representation Pair")
    print("=" * 100)
    
    # Group by representation pair
    rep_pairs = [
        ('hybrid', 'gpt_free', 'Hybrid Manual+GPT', 'GPT Free Extraction'),
        ('hybrid', 'gpt_law', 'Hybrid Manual+GPT', 'GPT Law Extraction'),
        ('hybrid', 'facts', 'Hybrid Manual+GPT', 'Raw Indictment Facts'),
        ('manual', 'gpt_free', 'Manual Feature Schema', 'GPT Free Extraction'),
        ('manual', 'gpt_law', 'Manual Feature Schema', 'GPT Law Extraction'),
        ('manual', 'facts', 'Manual Feature Schema', 'Raw Indictment Facts'),
    ]
    
    summary_results = []
    
    for rep_a_code, rep_b_code, rep_a_name, rep_b_name in rep_pairs:
        # Get all comparisons for this pair
        pair_comps = df_comparisons[
            ((df_comparisons['Rep_A_Code'] == rep_a_code) & (df_comparisons['Rep_B_Code'] == rep_b_code)) |
            ((df_comparisons['Rep_A_Code'] == rep_b_code) & (df_comparisons['Rep_B_Code'] == rep_a_code))
        ].copy()
        
        if len(pair_comps) == 0:
            continue
        
        # Normalize so rep_a is always the structured one
        def normalize_row(row):
            if row['Rep_A_Code'] == rep_a_code:
                return row['F1_A'], row['F1_B'], row['Sig_FDR'], row['P_FDR']
            else:
                return row['F1_B'], row['F1_A'], row['Sig_FDR'], row['P_FDR']
        
        normalized = pair_comps.apply(normalize_row, axis=1, result_type='expand')
        normalized.columns = ['F1_Structured', 'F1_Unstructured', 'Sig_FDR', 'P_FDR']
        
        total = len(normalized)
        structured_wins = (normalized['F1_Structured'] > normalized['F1_Unstructured']).sum()
        sig_wins = ((normalized['F1_Structured'] > normalized['F1_Unstructured']) & normalized['Sig_FDR']).sum()
        sig_losses = ((normalized['F1_Structured'] < normalized['F1_Unstructured']) & normalized['Sig_FDR']).sum()
        
        avg_f1_diff = (normalized['F1_Structured'] - normalized['F1_Unstructured']).mean()
        
        # Sign test
        sign_test_result = stats.binomtest(structured_wins, total, 0.5)
        
        print(f"\n🔹 {rep_a_name} vs {rep_b_name}:")
        print(f"   Comparisons: {total}")
        print(f"   {rep_a_name} wins: {structured_wins}/{total} ({structured_wins/total*100:.1f}%)")
        print(f"   Avg F1 difference: {avg_f1_diff:+.3f}")
        print(f"   Significant wins: {sig_wins}, Significant losses: {sig_losses}")
        print(f"   Sign test p-value: {sign_test_result.pvalue:.4f}")
        print(f"   Conclusion: {'✅ Significantly better' if sign_test_result.pvalue < 0.05 and structured_wins > total/2 else ('⚠️ Significantly worse' if sign_test_result.pvalue < 0.05 else '❌ No significant difference')}")
        
        summary_results.append({
            'Structured_Rep': rep_a_name,
            'Unstructured_Rep': rep_b_name,
            'Total_Comparisons': total,
            'Structured_Wins': structured_wins,
            'Win_Rate': structured_wins/total*100,
            'Avg_F1_Diff': avg_f1_diff,
            'Sig_Wins': sig_wins,
            'Sig_Losses': sig_losses,
            'Sign_Test_P': sign_test_result.pvalue,
            'Conclusion': 'Significantly better' if sign_test_result.pvalue < 0.05 and structured_wins > total/2 else ('Significantly worse' if sign_test_result.pvalue < 0.05 else 'No significant difference')
        })
    
    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print("\n" + "=" * 100)
    print("📋 SUMMARY TABLE: Structured vs Unstructured Representations")
    print("=" * 100)
    
    df_summary = pd.DataFrame(summary_results)
    print("\n" + df_summary.to_string(index=False))
    
    # =========================================================================
    # FINAL ANSWER
    # =========================================================================
    print("\n" + "=" * 100)
    print("🏆 FINAL ANSWER: Is there a winner representation?")
    print("=" * 100)
    
    # Count how many times each structured rep beats all unstructured
    hybrid_wins_all = all([r['Conclusion'] == 'Significantly better' 
                           for r in summary_results if r['Structured_Rep'] == 'Hybrid Manual+GPT'])
    manual_wins_all = all([r['Conclusion'] == 'Significantly better' 
                           for r in summary_results if r['Structured_Rep'] == 'Manual Feature Schema'])
    
    hybrid_summary = [r for r in summary_results if r['Structured_Rep'] == 'Hybrid Manual+GPT']
    manual_summary = [r for r in summary_results if r['Structured_Rep'] == 'Manual Feature Schema']
    
    print("\n📌 Hybrid Manual+GPT:")
    for r in hybrid_summary:
        status = "✅" if r['Conclusion'] == 'Significantly better' else ("⚠️" if 'worse' in r['Conclusion'] else "❌")
        print(f"   {status} vs {r['Unstructured_Rep']}: {r['Win_Rate']:.1f}% wins, p={r['Sign_Test_P']:.4f}")
    
    print("\n📌 Manual Feature Schema:")
    for r in manual_summary:
        status = "✅" if r['Conclusion'] == 'Significantly better' else ("⚠️" if 'worse' in r['Conclusion'] else "❌")
        print(f"   {status} vs {r['Unstructured_Rep']}: {r['Win_Rate']:.1f}% wins, p={r['Sign_Test_P']:.4f}")
    
    print("\n" + "-" * 60)
    if hybrid_wins_all:
        print("✅ HYBRID MANUAL+GPT significantly beats ALL unstructured representations!")
    elif any(r['Conclusion'] == 'Significantly better' for r in hybrid_summary):
        print("⚠️ HYBRID MANUAL+GPT beats SOME but not ALL unstructured representations")
    else:
        print("❌ HYBRID MANUAL+GPT does NOT significantly beat unstructured representations")
    
    if manual_wins_all:
        print("✅ MANUAL FEATURE SCHEMA significantly beats ALL unstructured representations!")
    elif any(r['Conclusion'] == 'Significantly better' for r in manual_summary):
        print("⚠️ MANUAL FEATURE SCHEMA beats SOME but not ALL unstructured representations")
    else:
        print("❌ MANUAL FEATURE SCHEMA does NOT significantly beat unstructured representations")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    output_dir = os.path.join(base_dir, 'representation_significance_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all comparisons
    df_comparisons.round(4).to_csv(os.path.join(output_dir, 'all_representation_comparisons.csv'), index=False)
    
    # Save summary
    df_summary.round(4).to_csv(os.path.join(output_dir, 'representation_summary.csv'), index=False)
    
    print(f"\n✅ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()



