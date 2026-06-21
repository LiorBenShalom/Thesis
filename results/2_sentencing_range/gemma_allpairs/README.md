# Gemma all-pairs — case similarity & sentence-range prediction (weapon)

Self-hosted **Gemma-4-31B-AWQ** case-similarity scored on **ALL weapon pairs** (no pre-filter),
then used for sentence-range prediction, compared against the GPT-with-filter pipeline and all baselines.
Run completed 2026-06-21. Full write-up: **`GEMMA_REPORT.md`**.

## Headline
- LLM similarity **alone** (all-pairs, no filter/retriever) **beats every classical baseline** —
  including the task-trained supervised embedding (weapon MAE_lo 13.85 → **13.05**).
- It does **not** beat GPT-with-filter (12.11) — the gap is model (AWQ) + a *marginal, not-significant*
  filter refinement (bootstrap CI on the MAE difference includes 0; Wilcoxon directional p≪0.001).
- Conclusion: the **LLM score is the strong predictor**; the retriever adds only a tiny, statistically
  borderline polish on top.

## Data
- `data/gemma_weapon_schema_FINAL.csv.gz` — **1,476,621** unique weapon pairs (ok=1,474,900, no_features=1,718, error=3).
  Columns: `verdict_1, verdict_2, domain, similarity_score (0-100), status, n_out_tokens`.
  Gunzip before use: `gunzip -k data/gemma_weapon_schema_FINAL.csv.gz`.
- `sentencing_range_weapon_gemma_vs_gpt.csv` — full K-sweep MAE table (3 methods).

## Scripts
**Generation** (how the similarity was produced, on RunPod vLLM):
- `scripts/score_pairs.py` — all-pairs scorer (`--all-domain weapon --score-only --features features_schema_weapon.json`); checkpoint/resume; multi-range `--shard`.
- `scripts/prompts.py`, `scripts/local_client.py` — score-only prompt + round-robin vLLM client.

**Analysis** (sentence-range prediction; leave-one-out kNN, weapon):
- `scripts/dedup.py` — dedup raw scored CSV → FINAL (frozenset pair, prefer ok).
- `scripts/predict_range.py` — main MAE: GPT-filtered vs Gemma all-pairs vs Gemma-on-filter-pool.
- `scripts/fair_compare.py` — same comparison on the identical query set (apples-to-apples).
- `scripts/predict_thresh.py` — similarity-threshold variants (≥60/80/90…) + controlled K test.
- `scripts/predict_wmedian.py` — score-weighted median/mean grid.
- `scripts/analyze_missed.py` — are there helpful "good twins" outside the filter? (yes, but redundant/unfindable).
- `scripts/sig_test.py` — paired bootstrap CI + Wilcoxon for the filter effect.

## Reproduce
Ground-truth ranges from `simcse_cuda_bundle/data/supervised_data.csv` (weapon: 1,719 cases with ranges);
GPT-filtered comparison from `experiments/data_per_domain/similarity_scores_combined.csv`.
Scripts use absolute paths to those sources on the author's machine.
