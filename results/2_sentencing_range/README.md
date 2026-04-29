# Task 2 — Sentencing-range prediction

Predicting (low, high) sentence range for a criminal verdict via kNN over
similar verdicts in the corpus.

## Setup

- **Verdicts**: 4,094 with valid sentencing range (drugs + weapon), after
  removing 9 cross-ID duplicates + 1 underscore-prefix quirk + 37 co-defendants
  (data-leakage fix).
- **Pair pool**: 140,961 LLM-scored pairs (Hybrid-Full GPT-4.1 + extension
  for corrected internal citations + external co-citations R≤50, k≥2).
- **Methods compared (4)**:
  - Hybrid-Full — LLM-as-judge V6 score
  - Gemini-embedding-001 — text cosine similarity
  - TF-IDF (word 1-2 ngrams, cosine)
  - Random-K — within-domain shuffle (chance floor)
- **Pipeline**: top-K kNN with no threshold; aggregation = median (drugs) /
  softmax (weapon); optional σ-filter at Q50 of σ_combined for selective
  prediction.

## Layout

| Dir | What |
|---|---|
| `predictions/` | All prediction outputs. Top-level files are the canonical paper-ready results (top-K=10, no threshold, on 4,094 verdicts). `topk10_clean/` has per-verdict predictions. `qwk_thresholds/` and `qwk_t1_thresholds/` are threshold-based ablations. `old/` archives intermediate runs. |
| `audit/` | Data-cleaning audit trail: lists of removed duplicates, co-defendant clusters, threshold tables. |

## Main result

**`predictions/k_sweep_clean.png`** — MAE / IoU / Coverage as a function of K
∈ {1, 3, 5, 10, 20, 50, 100}. **Hybrid-Full is best in 28/28 cells** (4 reps × 7 K
values × 2 domains × 2 metrics).

## Headline table (top-10, σ-filter, BH-FDR Wilcoxon)

| Method | drugs MAE_lo | drugs MAE_hi | drugs IoU | weapon MAE_lo | weapon MAE_hi | weapon IoU |
|---|---:|---:|---:|---:|---:|---:|
| **Hybrid-Full** | **3.14** | **4.64** | **.609** | **4.25** | **6.56** | **.606** |
| Gemini | 3.28 | 4.93 | .594 | 4.95 | 7.50 | .566 |
| TF-IDF | 3.30 | 5.08 | .590 | 5.45 | 8.10 | .555 |
| Random-K | 4.20 | 6.30 | .530 | 9.71 | 15.16 | .451 |

Hybrid-Full beats Gemini, TF-IDF, Random-K significantly (p<0.05 BH-FDR) on
weapon all 4 cells; on drugs all cells beat Random-K, ties with Gemini/TF-IDF.

Source: `predictions/comparison_topk10_clean.csv`,
        `predictions/wilcoxon_topk10_clean.csv`,
        `predictions/aurc_topk10_clean.csv`.

## Audit trail

`audit/` contains documentation of data-cleaning decisions:
- `true_duplicates.json` — 10 cross-ID duplicate verdicts removed
- `codefendant_clusters.json`, `codefendants_dropped.json` — 37 co-defendants
  removed (data leakage fix)
- `qwk_thresholds.csv`, `f1_thresholds.csv` — F1 / QWK-Oracle thresholds
  computed on task-1 GT (kept for reference, but final results use top-K
  with no threshold).
- `suspect_duplicates_full_scan.json` — full TF-IDF dup scan (all 4,094 verdicts).
