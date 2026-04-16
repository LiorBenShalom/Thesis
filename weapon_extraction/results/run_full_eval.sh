#!/bin/bash
# =====================================================================
# Full weapon extraction evaluation — reproducible pipeline
# =====================================================================
# This script reproduces the canonical accuracy numbers for weapon
# feature extraction (GPT vs manual GT), using the involved-only
# metric as the default (ignores trivial negatives).
#
# Partial-match rules applied:
#   - אופן החזקה: 0.8 when only 'מוסלק - מוסתר' differs
#   - סטטוס הנשק: 0.8 for 'תקין ↔ נשק מופרד מתחמושת' confusion
#   - סוג הנשק:   0.8 for X ↔ X_מאולתר of same base weapon
#
# Expected overall result: ~82% (involved) / ~96% (all-binaries)
#
# Usage:
#   cd experiments/weapon_extraction
#   bash results/run_full_eval.sh
# =====================================================================
set -e

BASE="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$BASE"

echo "▸ Base directory: $BASE"

# Ensure environment has OPENAI_API_KEY
if [ -z "$OPENAI_API_KEY" ]; then
    ENV_FILE="$BASE/../.env"
    if [ -f "$ENV_FILE" ]; then
        export $(grep -v '^#' "$ENV_FILE" | xargs)
    fi
fi

# Step 1: (Re-)run extraction if cache is missing. Otherwise use cache.
if [ ! -f "cache/eval_weapon_gpt_cache.json" ]; then
    echo "▸ No cache found — running full extraction (takes ~20 min)..."
    python3 code/extract_weapon_features_simple.py \
        --dir "../../weapon/weapon_docx/" \
        --out "results/fe_gpt_extracted.csv"
else
    echo "▸ Using existing extraction cache"
fi

# Step 2: Run eval (GT with defaults → per-feature accuracy, dual metric)
echo "▸ Running eval with defaults..."
python3 code/eval_with_defaults.py

# Step 3: Rebuild similarity_database_fe_gpt_schema_v2.csv from cache
echo "▸ Rebuilding similarity_database_fe_gpt_schema_v2.csv..."
python3 code/convert_gpt_cache_to_fe.py \
    --cache cache/eval_weapon_gpt_cache.json \
    --pairs "../../weapon/similarity_database_fe.csv" \
    --out "../../weapon/similarity_database_fe_gpt_schema_v2.csv"

# Step 4: Print summary
echo ""
echo "======================================================================"
echo "  RESULTS (involved-only, default metric)"
echo "======================================================================"
python3 << 'PYEOF'
import csv
SUMMARY = "results/eval_weapon_results_WITH_DEFAULTS_summary.csv"
rows = list(csv.DictReader(open(SUMMARY, encoding='utf-8-sig')))
print(f"{'פיצ\"ר':<28}{'involved':>14}{'all bins':>14}")
print("-" * 60)
for r in rows:
    if r['level'] == 'main':
        ai = float(r['accuracy_involved'])*100
        aa = float(r['accuracy_all'])*100
        print(f"{r['feature']:<28}{ai:>11.1f}% {aa:>11.1f}%")
    elif r['level'] == 'overall':
        ai = float(r['accuracy_involved'])*100
        aa = float(r['accuracy_all'])*100
        print("-" * 60)
        print(f"{'OVERALL':<28}{ai:>11.1f}% {aa:>11.1f}%")
PYEOF

echo ""
echo "✅ Done. Outputs in results/:"
echo "   - eval_weapon_results_WITH_DEFAULTS.csv (per-case, per-feature)"
echo "   - eval_weapon_results_WITH_DEFAULTS_summary.csv (aggregate)"
echo "   - fe_gpt_extracted.csv (per-case features)"
echo ""
echo "   Downstream feed: ../../weapon/similarity_database_fe_gpt_schema_v2.csv"
