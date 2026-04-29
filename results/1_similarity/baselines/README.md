# Baselines for the Paper

This folder contains three complementary baselines that contextualise the main
results in `../results_paper/` and `../results_paper_qwk/`:

## 1. Random (permutation) baseline — advisor's method

Script: `src/analysis/random_baseline.py`

Follows the pattern from `new_try/code/calculate_baseline_CORRECT.py`:

- **Shuffle the ground truth** (not predictions) 1000 times per cell with a fixed seed.
- Shuffling a vector preserves its marginal — so class proportions are preserved exactly.
- Predictions are held fixed; we evaluate each metric on the 1000 shuffled GTs to build a null distribution.
- Report: `baseline_mean`, `baseline_ci_lo/hi`, `improvement = observed - baseline_mean`,
  `p_value = P(null ≥ observed)`, `significantly_better = p_value < 0.05`.

The *shared* baseline per (domain, metric) — what the paper typically cites —
is aggregated in `random_per_task.csv`.

## 2. Embedding-Text baseline (raw verdict only)

Script: `src/analysis/embedding_baseline.py`

Cosine similarity between embeddings of **full verdict facts** (`indicment_facts_1/2`),
using three top embedding models:

- `text-embedding-3-large` (OpenAI, paid)
- `intfloat/multilingual-e5-large-instruct` (HuggingFace, free, local)
- `BAAI/bge-m3` (HuggingFace, free, local, 8K context)

Answers: "how far can you go using only the raw text of the verdict?"

## 3. Embedding-on-rep ablation (all 7 reps × 3 emb models)

Script: `src/analysis/embedding_all_reps.py`

Same 3 embedding models, but applied to the **structured feature vector** of
every representation (Manual, GPT-Schema, GPT-Free, GPT-Law, Raw-Facts,
Hybrid-Manual, Hybrid-Full). Structured features are serialized to a
deterministic `key: value | key: value | ...` string before embedding.

Answers: "how much does the LLM scoring step add on top of the structured rep?"
Compare `Rep+Embedding` (this script) against `Rep+LLM` (main experiment).

## Metrics covered (all baselines)

`F1-Oracle`, `F1-CV` (b0 strict, b1 lenient), `AP-PR` (b0, b1),
`QWK-Oracle`, `QWK-CV (10-fold)` — identical to the main experiment.

## Reproducing

```bash
cd new_try/experiments

# 1. Random permutation baseline (~2-3 min on 12 workers)
python3 src/analysis/random_baseline.py --n-shuffles 1000 --workers 12

# 2. Embedding-Text baseline (~2-5 min; OpenAI API + local MPS)
python3 src/analysis/embedding_baseline.py

# 3. Embedding-on-rep ablation (~10-15 min; 7 reps x 3 embedding models)
python3 src/analysis/embedding_all_reps.py

# 4. Combined report
python3 src/analysis/baseline_report.py
```

## Key output files

| File | Script | Description |
|---|---|---|
| `random_full.csv` | random_baseline | Per-cell: observed, baseline_mean, CI, p_value, significantly_better |
| `random_per_task.csv` | random_baseline | **Shared baseline** per (domain, metric) — paper-ready |
| `random_summary.csv` | random_baseline | Aggregated per (domain, rep, metric) |
| `RANDOM_REPORT.md` | random_baseline | Per-domain markdown tables |
| `emb_full.csv` | embedding_baseline | Per (domain, emb_model, metric) — raw-text only |
| `EMB_REPORT.md` | embedding_baseline | Per-domain tables |
| `emb_reps_full.csv` | embedding_all_reps | Per (domain, rep, emb_model, metric) — all 42 cells |
| `EMB_REPS_REPORT.md` | embedding_all_reps | Per (domain, metric): rep × emb_model pivot |
| `BASELINES_REPORT.md` | baseline_report | **Combined paper-ready report** |
| `comparison_table.csv` | baseline_report | Long-format comparison |
| `headline_plot.png` | baseline_report | QWK-CV bar chart: Random → Emb-Text → Emb-on-rep → LLM-rep |
| `ablation_rep_vs_emb.png` | baseline_report | Per-rep grouped bar: Rep+Emb vs Rep+LLM |
| `emb_preds/` | embedding_baseline | Per-pair scores (raw-text embeddings) |
| `emb_reps_preds/` | embedding_all_reps | Per-pair scores (rep embeddings) |
| `emb_cache/` | (all emb scripts) | Cached .npy embeddings — gitignored, regen on demand |

## Interpretation for the paper

- **Random null** (chance floor) — any system that doesn't beat the 97.5% null
  CI-hi is indistinguishable from guessing with correct proportions.
- **Embedding-Text** — principled non-structured baseline; gap to Manual+LLM
  quantifies the combined value of structure + LLM reasoning.
- **Emb-on-rep** — holds structure fixed and replaces LLM with simple cosine;
  gap between Rep+LLM and Rep+Emb quantifies the contribution of LLM reasoning
  **on top of** the structured representation.
