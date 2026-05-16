#!/usr/bin/env bash
# 5-fold CV with offense-overlap filter + backfill (cap=12 months).
# Total expected runtime: ~3-5 hours on a single A10.
# Outputs: outputs_supervised_filtered/model_{domain}_topk_fold{1..5}_offenseFiltered/
#          outputs_supervised_filtered/verdict_embeddings_{domain}_topk_fold{N}_offenseFiltered.npy
#          outputs_supervised_filtered/verdict_index_{domain}_topk_fold{N}_offenseFiltered.csv
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== 5-fold CV with offense-filter: drugs + weapon × 5 folds = 10 trainings ====="
date

for domain in drugs weapon; do
  for fold in 1 2 3 4 5; do
    echo
    echo "----- ${domain} fold ${fold}/5 (offense-filtered, cap=12mo) -----"
    date
    python train_supervised_filtered.py \
      --domain "${domain}" \
      --mode topk \
      --topk-per-anchor 20 \
      --max-distance 12 \
      --fold "${fold}" --n-folds 5 \
      2>&1 | tee "train_${domain}_topk_filtered_fold${fold}.log"
  done
done

echo
echo "===== ALL DONE ====="
date
ls -lh outputs_supervised_filtered/verdict_embeddings_*_topk_fold*_offenseFiltered.npy
