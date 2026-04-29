# LEGACY reports (N=11 panel + binary similarity task)

⚠ **DO NOT CITE THESE** — they are from an earlier configuration.

The current paper uses:
- **N=13** LLM panel (not N=11)
- **Ordinal 3-point similarity scale only** (not binary 0/1)

Why these are kept:
- Audit trail for the project's evolution
- Per-cell scores in the underlying CSVs (`../random_full.csv`,
  `../emb_full.csv`, `../emb_reps_full.csv`) ARE still valid because they
  are computed per (rep, domain, embedding-model, metric), not per LLM.
- Only the *aggregated summaries* ("mean across 11 models") and *binary
  task metrics* (F1, AP-binary) in these MDs are outdated.

## Where to find the current paper-ready numbers

| What you want | Look at |
|---|---|
| N=13 ordinal results (canonical) | `../../qwk/summary_qwk_n13.csv` + figs |
| Random null vs LLM-rep | `../random_per_task.csv` + `../final_significance_2_baselines.csv` |
| Embedding-Text vs LLM-rep | `../emb_full.csv` (use QWK columns only) |
| Embedding-on-rep vs LLM-rep | `../emb_reps_full.csv` (use QWK columns only) |

## Files in this directory

| File | Why legacy |
|---|---|
| `BASELINES_REPORT.md` | Header says "mean across 11 models"; F1/AP columns are no longer in paper. |
| `EMB_REPORT.md` | F1_Oracle_b0/b1, F1_CV_b0/b1, AP_b0/b1 columns are binary. QWK columns are still valid. |
| `EMB_REPS_REPORT.md` | Same — mix of binary + ordinal columns. |
| `RANDOM_REPORT.md` | Random null is task-agnostic (per-cell shuffle), but report headers reference binary tasks. |
| `FINAL_SIGNIFICANCE.md` | "fraction of the 11 LLM models that individually pass p<0.05" — N=11 panel. |
| `significance_summary.md` | Same. |
| `significance_text_summary.md` | Same. |
