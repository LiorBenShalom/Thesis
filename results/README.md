# Results

All paper-ready results, organized by task.

## Layout

| Path | Task |
|---|---|
| `1_similarity/` | **Task 1** — Similarity prediction (ordinal scale 1-3). |
| `2_sentencing_range/` | **Task 2** — Sentencing-range prediction via kNN. |

Each subdirectory has its own README with details.

## High-level findings

**Task 1**: N=13 model panel × 7 representations.
Manual is the strongest representation (QWK-CV 0.83 drugs / 0.76 weapon),
with Hybrid-Full and Hybrid-Manual close behind. All LLM-reps significantly
beat random + 4 embedding baselines on QWK / Spearman / C-index.

**Task 2**: Hybrid-Full (LLM-as-judge) > Gemini > TF-IDF > Random-K on
top-10 kNN at every K ∈ {1, 3, 5, 10, 20, 50, 100}. Best results with
σ-filter: weapon MAE_high = 6.56 (vs 8.10 for TF-IDF, p < 0.0001).
