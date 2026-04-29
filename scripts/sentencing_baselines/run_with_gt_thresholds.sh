#!/bin/bash
# Run all 4 reps × 2 binarizations (binary_0 strict / binary_1 lenient) × no/with σ
# Uses F1-Oracle threshold from task-1 GT for each (rep, domain, binary).
set -e
cd "/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/experiments"

OUT=data_per_domain/prediction_results/gt_thresholds
mkdir -p $OUT

run() {
    local sim_csv="$1"; local rep="$2"; local thr_d="$3"; local thr_w="$4"; local tag="$5"
    echo ""
    echo "=== $rep ($tag) — drugs THR=$thr_d, weapon THR=$thr_w ==="
    python3 scripts/sentencing_baselines/predict_paper_style.py \
        --sim-csv "$sim_csv" --rep "${rep}_${tag}" \
        --thr-drugs "$thr_d" --thr-weapon "$thr_w" \
        --no-citation-filter --use-corrected-canonical \
        --out-dir "$OUT" 2>&1 | tail -7
}

# binary_0 (strict)
run "data_per_domain/similarity_scores_combined.csv"        Hybrid-Full  48.24  55.28  bin0
run "data_per_domain/similarity_scores_gemini_combined.csv" Gemini       94.51  94.68  bin0
run "data_per_domain/similarity_scores_tfidf_combined.csv"  TF-IDF       15.51   7.69  bin0

# binary_1 (lenient)
run "data_per_domain/similarity_scores_combined.csv"        Hybrid-Full  45.23  45.23  bin1
run "data_per_domain/similarity_scores_gemini_combined.csv" Gemini       93.69  94.11  bin1
run "data_per_domain/similarity_scores_tfidf_combined.csv"  TF-IDF       12.09   7.66  bin1

# Random — no GT, use binary_0-equivalent percentile of Hybrid-Full as anchor
# Hybrid-Full bin_0 thr=48.24 corresponds to which percentile in Random distribution?
# (Random within domain ~ uniform — so percentile ≈ same)
run "data_per_domain/similarity_scores_random_combined.csv" Random-K     48.24  55.28  bin0
run "data_per_domain/similarity_scores_random_combined.csv" Random-K     45.23  45.23  bin1
