# Archived data — superseded batch processing artifacts

## What's still current (top of `data_per_domain/`):

- `verdicts_clean.csv` (symlink to `data_master_final/`) — the canonical 4,133 verdicts
- `similarity_scores.csv` — 84,915 LLM-scored pairs (after dedup)
- `similarity_scores_combined.csv` — extended set with corrected internal +
  external co-citations (140,961 pairs after dedup)
- `similarity_scores_{gemini,tfidf,random}_combined.csv` — baseline scores
- `prediction_results/` — final prediction results (top-10 + AURC + Wilcoxon)
- `network_analysis/` — citation network exports
- `emb_cache_gemini/` — Gemini embedding cache (used by baselines)
- `*.json` — duplicates audit, codefendants audit
- `master_inventory.csv`, `extracted_citations.csv` — derived data still in use

## What's archived in `old/`

### Batch-processing dirs (intermediates that produced `similarity_scores.csv`)

| Dir | Size | What it had |
|---|---:|---|
| `similarity_batch/` | 963MB | original GPT-4.1 batch JSONL inputs + raw outputs |
| `similarity_batch_extended/` | 614MB | the +59K extended batch (corrected internal + external co-cites). The final scored CSV is preserved at `prediction_results/` and `similarity_scores_combined.csv`. |
| `similarity_batch_rawfacts/` | 117MB | alternative batch with raw facts (not used) |
| `similarity_batch_schema/` | 47MB | schema-only batch (not used) |
| `similarity_batch_supplement/` | 16MB | supplement batch — already merged |
| `similarity_retry/` | 72MB | retry batches — already merged |
| `similarity_retry2/` | 63MB | second retries — already merged |

### Older domain analyses

| Dir | Size | What it had |
|---|---:|---|
| `appeals/` | 74MB | Appeal-court verdict analysis (not in main pipeline) |
| `unknown/` | 25MB | Unknown-classification reclassification work |

### Scratch CSVs (per-source ablations)

- `similarity_scores_orig_plus_internal.csv` (5MB)
- `similarity_scores_orig_plus_external.csv` (~5MB)
- `similarity_scores_orig_plus_both.csv` (6MB)

These were intermediate ablation outputs from the per-source analysis. The
canonical version is `similarity_scores_combined.csv` at the top level.

## Note

All these dirs were UNTRACKED in git (gitignored or never added). Moving
them here only affects local filesystem organization — no change to GitHub.
