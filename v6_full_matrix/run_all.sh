#!/usr/bin/env bash
# Full v6 matrix: 10 representations × all models × drugs + weapon (one --task; binary_1 from same scores).
# Reps: facts, manual FE, manual-format FE, legacy-from-structured FE, GPT-derived FE schema, gpt_law, gpt_free, hybrid (manual+GPT), hybrid_gpt, hybrid_full_gpt.
# Outputs under THIS_DIR (v6_full_matrix/). Seeds hybrid_full_gpt binary_0 cache from prior experiment folder if present.
set -euo pipefail

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$(cd "${THIS_DIR}/../../code" && pwd)"
OUT_ROOT="${THIS_DIR}"
PRIOR="${THIS_DIR}/../v6_hybrid_full_gpt_score_multimodel"

REPS=(
  facts
  manual_fe
  fe_manual_format
  fe_legacy_from_structured
  fe_gpt_schema
  gpt_law
  gpt_free
  hybrid_manual_gpt
  hybrid_gpt
  hybrid_full_gpt
)
MODELS=(
  gpt4 gpt5mini qwen3_235b mistral llama3_70b gpt52
  gpt51_thinking qwen_hf gemini_25_pro gemini_3_flash gemma3_27b
)

mkdir -p "${OUT_ROOT}/drugs/results_drugs" "${OUT_ROOT}/weapon/results_weapon"

if [[ -d "${PRIOR}/drugs/results_drugs" ]]; then
  echo "Seeding hybrid_full_gpt binary_0 preds/stats from prior experiment (cache reuse)..."
  shopt -s nullglob
  for f in "${PRIOR}/drugs/results_drugs/similarity_database_hybrid_full_gpt_v6score_"*_binary_0_*; do
    cp -f "$f" "${OUT_ROOT}/drugs/results_drugs/" || true
  done
  for f in "${PRIOR}/weapon/results_weapon/similarity_database_hybrid_full_gpt_v6score_"*_binary_0_*; do
    cp -f "$f" "${OUT_ROOT}/weapon/results_weapon/" || true
  done
  shopt -u nullglob
fi

# Parallel across models (threads); per-provider caps inside Python (OpenAI/HF/NIM/Gemini).
# Override: export V6_PARALLEL=6
PARALLEL="${V6_PARALLEL:-4}"

cd "${CODE_DIR}"
# Use e.g. export PYTHON=/path/to/venv/bin/python if `python` on PATH lacks deps.
PYTHON_BIN="${PYTHON:-python}"
TASK=binary_0
echo ""
echo "========== TASK=${TASK} (binary_1 stats from same API scores) =========="
echo "Using: ${PYTHON_BIN}  (set PYTHON=... to override)"
"${PYTHON_BIN}" -u v6_score_multimodel_experiment.py \
  --domain both \
  --reps "${REPS[@]}" \
  --models "${MODELS[@]}" \
  --task "${TASK}" \
  --sleep 0.2 \
  --parallel "${PARALLEL}" \
  --output-root "${OUT_ROOT}"

echo "All finished. Summary includes both binary tasks: ${OUT_ROOT}/v6_multimodel_summary_${TASK}.json"
