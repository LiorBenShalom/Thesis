#!/usr/bin/env bash
# After v6_score_multimodel_experiment.py finishes: refresh tables + full statistical analysis.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-/opt/anaconda3/bin/python}"

echo "Waiting until no v6_score_multimodel_experiment.py process..."
while pgrep -f "v6_score_multimodel_experiment.py" >/dev/null 2>&1; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) still running..."
  sleep 60
done
echo "v6 processes ended."

cd "$ROOT"
"$PY" regenerate_v6_tables.py
"$PY" v6_full_matrix_statistical_analysis.py "$ROOT" --regen-tables

echo "Done. Tables: $ROOT/excel_tables/  Analysis: $ROOT/analysis_full/"
