#!/bin/bash
# Live progress monitor for the 4 new-model experiment.
# Usage:  bash monitor_4new_models.sh
# Prints a snapshot table; pair with `watch` or Monitor tool to refresh.

set -e
ROOT="/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/experiments/v6_pilot_5models"

# Load OPENROUTER_API_KEY for credits lookup
ENV_FILE="/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/experiments/.env"
[ -f "$ENV_FILE" ] && set -a && source "$ENV_FILE" && set +a

python3 - <<'PY'
import os, sys, json, glob, urllib.request, urllib.error, datetime
from pathlib import Path

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/experiments/v6_pilot_5models")

REPS = [
    ("Raw-Facts",     "similarity_database_with_indicment_facts"),
    ("Manual",        "similarity_database_fe"),
    ("GPT-Schema",    "similarity_database_fe_gpt_schema_v2"),
    ("GPT-Free",      "similarity_database_with_gpt_features"),
    ("GPT-Law",       "similarity_database_with_gpt_law_features"),
    ("Hybrid-Manual", "similarity_database_hybrid"),
    ("Hybrid-Full",   "similarity_database_hybrid_full_gpt"),
]
MODELS = ["claude_haiku_4_5", "llama4_maverick_or", "kimi_k26_or", "qwen35_plus_or"]
EXPECTED = {"drugs": 100, "weapon": 141}

print("=" * 96)
print(f"PROGRESS @ {datetime.datetime.now().strftime('%H:%M:%S')}   "
      f"(4 new models × 7 reps × 2 domains = 56 cells; expected 100+141=241 pairs/cell)")
print("=" * 96)

import pandas as pd

totals = {m: {"done": 0, "ok": 0, "fail": 0} for m in MODELS}
cell_total = sum(EXPECTED.values())  # 241

# Per-model rows
for m in MODELS:
    parts = []
    for rep_name, prefix in REPS:
        cell_done = 0
        cell_ok = 0
        for dom, exp in EXPECTED.items():
            p = ROOT / dom / f"results_{dom}" / f"{prefix}_v6score_{m}_binary_0_preds.csv"
            if p.exists():
                try:
                    df = pd.read_csv(p)
                    cell_done += len(df)
                    cell_ok += (df.get("status", "ok") == "ok").sum() if "status" in df.columns else len(df)
                except Exception:
                    pass
        totals[m]["done"] += cell_done
        totals[m]["ok"] += cell_ok
        pct = 100.0 * cell_done / cell_total if cell_total else 0
        parts.append(f"{rep_name[:11]:<11s} {cell_done:>3d}/{cell_total} ({pct:>3.0f}%)")
    print(f"{m:<22s}")
    for i in range(0, len(parts), 4):
        print("    " + "  ".join(parts[i:i+4]))

print("-" * 96)

# Totals + completion %
target_per_model = 7 * cell_total  # 7 reps × 241 pairs
print(f"{'MODEL':<22s} {'DONE':>10s} {'OK':>8s} {'%':>5s}")
for m, t in totals.items():
    pct = 100.0 * t["done"] / target_per_model
    print(f"{m:<22s} {t['done']:>5d}/{target_per_model:<5d} {t['ok']:>5d}    {pct:>3.0f}%")

# Process status
import subprocess
try:
    out = subprocess.check_output(["pgrep", "-f", "v6_score_multimodel_experiment.py"], text=True).strip()
    pids = out.split("\n") if out else []
    print(f"\nv6 process: {'RUNNING (pid=' + ', '.join(pids) + ')' if pids else 'NOT RUNNING'}")
except subprocess.CalledProcessError:
    print("\nv6 process: NOT RUNNING")

# OR credits
or_key = os.environ.get("OPENROUTER_API_KEY")
if or_key:
    try:
        req = urllib.request.Request("https://openrouter.ai/api/v1/credits",
                                     headers={"Authorization": f"Bearer {or_key}"})
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.load(r).get("data", {})
            used = d.get("total_usage", 0)
            balance = (d.get("total_credits") or 0) - used
            print(f"OpenRouter:  spent ${used:.2f}   balance ${balance:.2f}")
    except Exception as e:
        print(f"OpenRouter credits: error {e}")
PY
