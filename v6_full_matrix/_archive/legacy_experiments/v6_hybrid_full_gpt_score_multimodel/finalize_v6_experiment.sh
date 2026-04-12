#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT}/../../code"
python v6_experiment_report.py "${ROOT}"
echo "Analysis written under ${ROOT}/analysis/"
