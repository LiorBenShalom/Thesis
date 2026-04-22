# Baselines for the Paper

This folder contains two complementary baselines that contextualise the main
results in `../results_paper/` and `../results_paper_qwk/`:

1. **Random (permutation) baseline** — per-cell null distribution by shuffling
   each model's predicted scores 1000 times (100 times for CV/Oracle-QWK) and
   recomputing every metric against the fixed ground truth.
2. **Embedding baselines** — cosine similarity between full verdict facts
   (`indicment_facts_1/2`) under three embedding models:
   - `text-embedding-3-large` (OpenAI, paid, top MTEB ranking)
   - `intfloat/multilingual-e5-large-instruct` (open-source, top MMTEB)
   - `BAAI/bge-m3` (open-source, 8K context — useful for long verdicts)

Both baselines share the eight metrics used in the main paper:
`F1-Oracle`, `F1-CV` (b0 strict, b1 lenient), `AP-PR` (b0, b1),
`QWK-Oracle`, `QWK-CV (10-fold)`.

## Reproducing

```bash
cd new_try/experiments

# 1. Random permutation baseline (~3-5 min on 12 workers)
python3 src/analysis/random_baseline.py --n-perm 1000 --n-perm-cv 100 --workers 12

# 2. Embedding baseline (~2-5 min; OpenAI API + local MPS inference)
python3 src/analysis/embedding_baseline.py

# 3. Combined report
python3 src/analysis/baseline_report.py
```

## Files

| File | Produced by | Description |
|---|---|---|
| `random_full.csv` | `random_baseline.py` | Per-cell observed + null mean/std/CI/p_emp |
| `random_summary.csv` | `random_baseline.py` | Aggregated per (domain, rep, metric) |
| `RANDOM_REPORT.md` | `random_baseline.py` | Per-domain markdown tables |
| `emb_full.csv` | `embedding_baseline.py` | Per (domain, embedding_model, metric) |
| `emb_preds/*.csv` | `embedding_baseline.py` | Per-pair cosine-similarity preds |
| `emb_cache/*.npy` | `embedding_baseline.py` | Cached embeddings per verdict (resume-safe) |
| `EMB_REPORT.md` | `embedding_baseline.py` | Per-domain metric tables |
| `BASELINES_REPORT.md` | `baseline_report.py` | Combined Random / Embedding / LLM-reps comparison |
| `comparison_table.csv` | `baseline_report.py` | Long-format comparison table |
| `headline_plot.png` | `baseline_report.py` | Bar chart: baselines vs LLM reps on QWK-CV |

## Interpretation for the paper

- **Random null** is the floor: any system that doesn't beat the 97.5% null
  CI-hi is indistinguishable from guessing with the right marginal proportions.
- **Embedding baseline** is a *principled non-structured* alternative that
  only uses raw text. The gap between Embedding and Manual quantifies how much
  structured representations contribute beyond what surface-level text
  embedding already captures.
- **LLM reps** (Manual, Hybrid-Manual, …) should clearly dominate both on
  every metric; if a rep falls below the Embedding baseline, it is contributing
  nothing over reading the verdict text directly.
