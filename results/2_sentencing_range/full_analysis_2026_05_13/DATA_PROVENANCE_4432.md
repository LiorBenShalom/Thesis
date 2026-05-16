# DATA PROVENANCE — 4,432 rebuild (2026-05-16)

> **Single source of truth: on exactly which data, and how, every result/table in this bundle was produced.**
> Every CSV/plot/table in this bundle dated 2026-05-16 or later was produced on the **4,432 dataset** described here.
> Results from the **prior 3,898 dataset are SUPERSEDED** (kept only in `*.bak_*` / `_bak_3898_*`).

---

## 1. The dataset — `supervised_data.csv` = 4,432 verdicts

| | value |
|---|---|
| total | **4,432** |
| drugs | 2,713 |
| weapon | 1,719 |
| nulls | 0 (verdict, domain, indictment_facts, sentencing_range_low/high) |
| H-Full coverage | 4,432 / 4,432 |
| schema | `verdict, domain, indictment_facts, sentencing_range_low, sentencing_range_high` |
| range units | months. low: min 0 / median 12 / max 504. high: min 2 / median 30 / max 999 |
| backup of prior 3,898 | `simcse_cuda_bundle/data/supervised_data.csv.bak_2026-05-16` |

### Composition (how 4,432 was assembled)
```
3,898  prior supervised set (domain-swap-corrected 2026-05-11)
+ 235  clean verdicts with H-Full that were not in the prior set
        (192 newly H-Full-extracted drugs + 43 already had H-Full)
+ 303  tal-data NEW survivors (215 drugs + 88 weapon)
−   4  duplicate canonical_id
= 4,432
```

### Sources
- **A — `innovation_submission/data_master_final/verdicts_clean.csv`** (4,133 → augmented to 4,432): domain∈{drugs,weapon}, sentencing range at confidence "גבוהה", dedup by canonical_id, outliers flagged. The 303 tal rows were appended here with `citations_json` normalized by `normalize_verdict_name` (same convention as the original 4,133; `citations_count == len(citations_json)`).
- **B — tal-data** (`/Users/liorb/tal-data/{drugs,wep}`, 4,691 raw docx): header citation → `normalize_verdict_name` → Hebrew canonical id (header-only, identical to the original pipeline; no full-text fallback). 413 truly-new (not in full-master) → `3_extract_sentencing_range.py` (GPT few-shot) → **303 high-conf survivors** → `1_extract_indictment_facts.py` (**model: gpt-4-turbo**, original `gpt-4-turbo-preview` was deprecated by OpenAI) → `2_extract_citations.py` (gpt-4.1-mini + `models/citation_classifier` BERT).

---

## 2. Supporting artifacts (all rebuilt to 4,432; all backed up `.bak_2026-05-16`)

| artifact | rows | how |
|---|---|---|
| `simcse_cuda_bundle/data/hybrid_full_cache.json` | **4,433** | prior 3,942 + 192 + 303 new H-Full. Extraction = `batch_hfull.py` via OpenAI Batch: schema `gpt-4.1-mini` → enrich `gpt-4.1`. 0 `__error` stubs. |
| `experiments/data_per_domain/master_inventory.csv` | **9,400** | prior 9,122 + 278 surgical append (range/confidence/domain/year/citations_count) so 4,432/4,432 covered |
| `experiments/data_per_domain/similarity_scores_combined.csv` | **219,381** | prior 140,961 + **78,420 new** (`source_batch=supervised_4432_union_2026_05_16`) |
| `experiments/data_per_domain/network_analysis/citation_pair_types.csv` | **219,381** | `classify_citation_types.py` over the 219,381 pair set. schema = `citation_type` (NOT `rel_type`). 4,432/4,432 + 534/534 new covered |
| `simcse_cuda_bundle/outputs_supervised_filtered/verdict_embeddings_{dom}_topk_fold{1-5}_offenseFiltered.npy` (+ index) | drugs (2,713,768) / weapon (1,719,768) per fold | retrained 2026-05-16, AWS A10G; prior 3,898 embeddings in `_bak_3898_2026-05-16/` |

### The 78,420 new LLM-scored pairs — exactly how
- **Candidate set = UNION** of:
  - (A) supervised **top-20** neighbors, **test-query × train-neighbor**, over the **new filtered 5-fold embeddings**, all 5 folds × both domains (leakage-safe: test held out of training);
  - (B) **646** new citation-network candidate pairs (1-hop edges touching a new verdict, from the rebuilt `citation_edges_in_set.csv`);
  - minus pairs already in `similarity_scores_combined` (140,961). Net-new = 78,573; 78,420 scored (153 skipped — missing feature/domain).
- **Scorer** (unchanged method): `build_similarity_batch_supervised.py` machinery — model **gpt-4.1**, V6 system prompts (`SYSTEM_DRUGS`/`SYSTEM_WEAPON`), temperature 0.1, H-Full features as input. 5 batch chunks, **0 failures**.

---

## 3. Model / CV method (unchanged from the 3,898 pipeline)

- Backbone DictaBERT-base, MultipleNegativesRankingLoss (InfoNCE).
- 5-fold CV, verdict-level split, **seed=42**, each verdict in test exactly once. Split is generated deterministically inside `train_supervised_filtered.py` from `supervised_data.csv` (so the 4,432 produces a fresh 5-fold split).
- Filtered model: positive pairs = top-20 Euclidean neighbors on (low,high) that **also share ≥1 offense label**, backfill to K=20 within 12-month cap.
- Test verdicts encoded by the train-only fold model.

---

## 4. Per-result "which data / how" map

All scripts under `scripts/` read the artifacts in §1–§2 (4,432 state). Output CSVs land in `/tmp/` then are copied into `data/` (numbers) and `plots/` (figures) of this bundle.

| result file | script | data + method |
|---|---|---|
| `rigor_per_query_errors.csv` (36,670 rows) | `rigor_phase_a.py` | per-query MAE for M1–M9 on 4,432, K=10, new filtered embeddings + 219,381 LLM scores + citation_pair_types(4,432) |
| `rigor_mae_with_ci.csv`, `rigor_paired_diffs.csv`, quartile/year-cluster | `rigor_phase_b.py` | bootstrap 95% CI (B=2,000) + paired Wilcoxon on the Phase-A 4,432 per-query errors |
| reality tables (random EXACT, LLM-bucket→gap, citation-type→gap) | `thesis_story_part1.py` | 4,432; random baseline = EXACT mean over all C(n,2) pairs from `master_inventory` (drugs n=2,628 / weapon n=1,768) |
| sweeps / deep / plots | `comprehensive_sweep.py`, `pool_size_sweep.py`, `deep_analysis.py`, `deeper_analysis.py`, `rigor_plots.py`, … | same 4,432 inputs |

Every regenerated table/plot carries the date 2026-05-16 and corresponds to the 4,432 state. If a file is older, it is the superseded 3,898 result.

---

## 5. Headline result change vs the prior 3,898 run

- **Limitation RESOLVED:** sup+LLM now **significantly** beats random+LLM on both domains (drugs Δ=−1.03, p=4.8e-7; weapon Δ=−1.94, p=1.2e-4). On 3,898 this was non-significant (Wilcoxon p=0.84) — the central documented limitation.
- **Narrative shift:** citation+LLM now beats sup+LLM on drugs (Δ=+0.71, p=7.9e-13, significant) and ties on weapon (CI includes 0). Previously sup+LLM was the headline.
- sup+LLM still significantly beats TF-IDF / BM25 / offense-matched / sup-only on both domains.

---
Generated 2026-05-16.
