# v6 full matrix — statistical analysis

Generated from `v6_full_matrix`.

## Files
- `master_runs.csv` — one row per complete (domain, task, model, representation).
- `summary_representation_domain_task.csv` — mean/std F1 per representation slice.
- `wilcoxon_representation_pairs.csv` — paired Wilcoxon on F1 differences (same model/domain/task).
- `mcnemar_representation_pairs_by_cell.csv` — McNemar discordant pairs per cell.
- `shuffled_baseline_all_runs.csv` — F1 vs label-shuffled predictions (p_vs_shuffled).
- `friedman_by_domain_task.csv` — Friedman test across representations (complete grid only).
- `vs_baseline_hybrid_full_gpt.csv` — ΔF1 vs baseline folder for `hybrid_full_gpt`.

Interpret FDR columns as exploratory when many tests are run.
