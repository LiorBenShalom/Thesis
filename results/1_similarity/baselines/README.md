# Baselines for the Paper

Three complementary baselines that contextualise the main results in `../qwk/`:

1. **Random (permutation) baseline** — chance floor (advisor's GT-shuffle method)
2. **Embedding-Text** — cosine of full verdict text via OpenAI / Gemini / mE5 / BGE-M3
3. **Embedding-on-rep ablation** — same 4 embedding models on each rep's structured
   features (isolates rep-structure from LLM reasoning)

> **Important — paper scope is ORDINAL similarity (3-point scale):**
> Use only the **QWK-Oracle, QWK-CV, Spearman ρ, C-index** columns from the
> CSVs. The legacy `*_REPORT.md` files (in `legacy_n11_with_binary/`) also
> contain F1 / AP columns from a binary task that is no longer in the paper.
>
> Also, those legacy reports were generated with an **earlier 11-LLM panel**.
> The current paper uses **N=13** (see `../qwk/summary_qwk_n13.csv`). The
> embedding/random baseline scores per cell are unchanged (they're per-cell,
> not per-LLM), but mean LLM-rep numbers in the legacy reports refer to the
> N=11 panel.

## Files (current — for the paper)

| File | What |
|---|---|
| `random_full.csv` | Per-cell random null: observed, baseline_mean, CI, p_value |
| `random_per_task.csv` | Shared baseline per (domain, metric) |
| `random_summary.csv` | Aggregated per (domain, rep, metric) |
| `emb_full.csv` | Embedding-Text: per (domain, emb_model, metric) |
| `emb_reps_full.csv` | Emb-on-rep: per (domain, rep, emb_model, metric) — 56 cells |
| `comparison_table.csv` | Combined long-format comparison |
| `final_significance_2_baselines.csv` | LLM-rep vs Random + vs best embedding (Wilcoxon, BH-FDR) |
| `significance_vs_embedding.csv` | LLM-rep vs Emb-on-Manual best |
| `significance_vs_text_embedding.csv` | LLM-rep vs raw-text embedding best |
| `headline_plot.png` | 4-level comparison plot (Random → Emb-Text → Emb-on-rep → LLM-rep) |
| `ablation_rep_vs_emb.png` | Per-rep: Rep+Emb vs Rep+LLM |
| `emb_preds/`, `emb_reps_preds/` | Per-pair embedding scores |
| `emb_cache/` | Cached `.npy` embedding vectors (gitignored) |

## Files in `legacy_n11_with_binary/`

Older Markdown reports from when the panel was N=11 LLMs and the paper still
included a binary task. Kept for reference and reproducibility audit. **Not
to be cited** — use the per-cell CSVs above and the canonical summary in
`../qwk/summary_qwk_n13.csv` instead.

| File | Why legacy |
|---|---|
| `BASELINES_REPORT.md` | "mean across 11 models" + binary metrics |
| `EMB_REPORT.md` | Includes F1 / AP columns |
| `EMB_REPS_REPORT.md` | Includes F1 / AP columns |
| `RANDOM_REPORT.md` | (random null is task-agnostic, but mixed in) |
| `FINAL_SIGNIFICANCE.md` | "11 LLM models" + binary metrics |
| `significance_summary.md` | Same |
| `significance_text_summary.md` | Same |

## Reproducing (with current N=13 panel + ordinal-only)

```bash
cd new_try/experiments

# Random permutation baseline (~2-3 min, 12 workers)
python3 src/analysis/random_baseline.py --n-shuffles 1000 --workers 12

# Embedding-Text baseline (~2-5 min; OpenAI API + local MPS)
python3 src/analysis/embedding_baseline.py

# Embedding-on-rep ablation (~10-15 min)
python3 src/analysis/embedding_all_reps.py
```

To regenerate with N=13 + ordinal-only output, the report scripts
(`baseline_report.py`) need to be updated to filter to QWK/Spearman/C-index
columns only. The CSVs above (`random_full.csv`, `emb_full.csv`,
`emb_reps_full.csv`) already contain the correct per-cell numbers; only the
aggregated summary tables in the legacy MD files reflect the older panel.
