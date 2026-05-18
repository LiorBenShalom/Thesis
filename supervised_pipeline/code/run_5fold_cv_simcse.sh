#!/usr/bin/env bash
# SimCSE 5-fold CV on the 4,432 corpus (holdout-correct; splits read from
# outputs_supervised_filtered). drugs + weapon × 5 = 10 trainings.
# Output: outputs_simcse_5fold/verdict_embeddings_simcse_{dom}_fold{N}.npy + index
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== SimCSE 5-fold CV (4,432, holdout-correct) ====="
date
for domain in drugs weapon; do
  for fold in 1 2 3 4 5; do
    echo; echo "----- ${domain} fold ${fold}/5 (SimCSE, unsupervised) -----"; date
    python train_simcse_5fold.py --domain "${domain}" --fold "${fold}" --n-folds 5 \
      2>&1 | tee "train_simcse_${domain}_fold${fold}.log"
  done
done
echo; echo "===== ALL DONE ====="; date
ls -lh outputs_simcse_5fold/verdict_embeddings_simcse_*_fold*.npy
