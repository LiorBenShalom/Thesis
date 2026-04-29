# Sentencing-range prediction results

## Layout

### Primary results — for the paper

| File | What |
|---|---|
| `topk10_clean/` | Per-verdict predictions (top-10 kNN, no threshold) for all 4 reps |
| `comparison_topk10_clean.csv` | Aggregated MAE/IoU per (rep, domain, σ) |
| `wilcoxon_topk10_clean.csv` | Paired Wilcoxon FDR — Hybrid-Full vs each baseline |
| `aurc_topk10_clean.csv` | Area Under Risk-Coverage curve per (rep, domain, target) |
| `k_sweep_clean.csv` | K-sweep results: MAE/IoU at K ∈ {1,3,5,10,20,50,100} |
| `k_sweep_clean.png` | **Main figure**: 4 metrics × 2 domains × 4 reps × 7 K values |

### Secondary — threshold-based ablations

| File | What |
|---|---|
| `qwk_thresholds/` | Per-rep predictions using QWK-Oracle t2 thresholds (from task-1 GT) |
| `qwk_t1_thresholds/` | Per-rep predictions using QWK-Oracle t1 thresholds (lenient) |
| `comparison_qwk_self.csv` | Aggregated metrics per rep with QWK t2 |
| `wilcoxon_qwk.csv` | Wilcoxon FDR with QWK t2 |
| `comparison_t1_vs_t2.csv` | Side-by-side: t1 vs t2 thresholds |

### `old/` — superseded results

Earlier analyses kept for audit trail. Not for the paper.
- `comparison_3way*.csv`, `fair_3way*.csv`, `comparison_schema_vs_hfull.csv` —
  Schema vs Hybrid-Full vs Raw-Facts (replaced by 4-way with Gemini/TF-IDF/Random-K)
- `results_drugs.csv`, `results_weapon.csv`, `results_all.csv` —
  Output of the original `predict_sentencing_range.py` (citation-only filter)
- `comparison_baselines_topk*.csv`, `comparison_baselines_percentile.csv` —
  Original 4-way runs on the 85K dataset (before extension to 144K + dedup)
- `comparison_paper_style.csv`, `comparison_nofilter.csv`, `comparison_f1_self.csv` —
  Threshold-strategy variants (citation-only filter, F1-Oracle threshold)
- `aurc_qwk.csv`, `aurc_f1.csv`, `aurc_topk_combined.csv` —
  AURC variants superseded by `aurc_topk10_clean.csv`
- `FINAL_SUMMARY.{csv,md}` — interim summary
- Subdirectories `baselines/`, `paper_style/`, `paper_style_nofilter/`,
  `gt_thresholds/`, `f1_thresholds/`, `topk_combined/` — predictions from
  earlier pipeline configurations

## Pipeline used to generate the primary results

1. Combined dataset: 144,246 pairs (after removing 9 cross-ID duplicates +
   1 underscore-prefix quirk + 37 co-defendants from same criminal cases)
2. Top-K kNN with K=10 nearest neighbors per query (no similarity threshold)
3. Aggregation: median (drugs) / softmax (weapon)
4. σ-filter (with_sigma): keep verdicts where σ_combined ≤ Q50 of distribution

Scripts:
- `experiments/scripts/sentencing_baselines/predict_knn_pairwise.py`
- `experiments/scripts/sentencing_baselines/k_sweep.py`
- `experiments/scripts/sentencing_baselines/compute_aurc.py`
