#!/usr/bin/env bash
# 5-fold CV for both domains, top-K supervised mode.
# Total expected runtime: ~3-5 hours on a single A10.
# Outputs: outputs_supervised/model_{domain}_topk_fold{1..5}/
#          outputs_supervised/verdict_embeddings_{domain}_topk_fold{N}.npy
#          outputs_supervised/verdict_index_{domain}_topk_fold{N}.csv
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== 5-fold CV: drugs + weapon × 5 folds = 10 trainings ====="
date

for domain in drugs weapon; do
  for fold in 1 2 3 4 5; do
    echo
    echo "----- ${domain} fold ${fold}/5 -----"
    date
    python train_supervised.py \
      --domain "${domain}" \
      --mode topk \
      --topk-per-anchor 20 \
      --fold "${fold}" --n-folds 5 \
      2>&1 | tee "train_${domain}_topk_fold${fold}.log"
  done
done

echo
echo "===== ALL DONE ====="
date
ls -lh outputs_supervised/verdict_embeddings_*_topk_fold*.npy
