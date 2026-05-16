#!/usr/bin/env bash
# Re-run all remaining analyses + plots on the 4,432 data. Continue-on-error.
cd "$(dirname "$0")"
log() { echo "==== $1 ===="; }
run() { log "START $1"; python3 "$1" > "_4432_$1.log" 2>&1 && echo "OK $1" || echo "FAIL $1 (see _4432_$1.log)"; }

# data/analysis producers
for s in comprehensive_sweep.py pool_size_sweep.py deep_analysis.py deeper_analysis.py \
         compare_filtered_vs_baseline.py llm_value_across_filters.py \
         offense_matched_overlap_sweep.py rigor_phase_a_v2.py; do run "$s"; done

# plots (after data exists)
for s in rigor_plots.py plot_pool_size.py plot_deep.py plot_story_part1.py generate_plots.py; do run "$s"; done

echo "==== ALL ANALYSIS DONE ===="
