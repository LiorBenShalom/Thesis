# Figures Index — `results_paper_qwk/`

_Last cleanup: 2026-04-27. Headline metrics: **C-index** + **Spearman ρ** (threshold-free, ranking-based, the natural fit for the ordinal 1–3 task)._

---

## 1. Headline figure for the thesis (1 file)

| File | What it shows |
|---|---|
| [fig_manual_significance.png](fig_manual_significance.png) | Slide-style summary: **Manual significantly better than every other rep in 48/48 tests** (2 domains × 4 ordinal metrics × 6 alternatives) after BH-FDR. Stronger than the binary equivalent (46/48). |

---

## 2. Per-rep distribution + significance — **C-index & Spearman only** (4 files)

These are the cleanest per-metric figures for the paper. Two views per metric:

| Metric | Box+strip with CLD letters | Critical Difference (Demšar) |
|---|---|---|
| C-index (primary) | [fig_cld_c_index.png](fig_cld_c_index.png) | [fig_cd_c_index.png](fig_cd_c_index.png) |
| Spearman ρ (twin) | [fig_cld_spearman.png](fig_cld_spearman.png) | [fig_cd_spearman.png](fig_cd_spearman.png) |

**How to read CLD:** reps sharing a letter are NOT significantly different (Wilcoxon two-sided, BH-FDR α=0.05). Reps without a shared letter are significantly different.
**How to read CD:** average rank across 11 models (1=best). Horizontal black bars at the bottom connect reps NOT significantly different (Nemenyi).

Use CLD for the body of the thesis; CD as supplementary or methodology section.

---

## 3. Archive — superseded figures

### `_archive/old_v1_apr18_19/` (15 files)
Original Apr 18–19 figures: bar charts, heatmaps (`fig1`–`fig12`). Replaced by the Apr 27 set above. Kept for traceability.

### `_archive/exploratory_apr27/` (16 files)
Intermediate Apr 27 generations:
- `fig_tiers_*` (5) — early tier bar charts; replaced by box+strip
- `fig_dist_*` (5) — box+strip with stars vs Manual only; replaced by CLD
- `fig_cd_qwk_oracle/cv/ordinal_auc.png` (3) — CD diagrams for non-headline metrics
- `fig_cld_qwk_oracle/cv/ordinal_auc.png` (3) — CLD for non-headline metrics

These are kept in case you want to discuss QWK or Ordinal AUC explicitly in the thesis (e.g. methodology comparison section). Pull them back into the root if needed.

---

## Why C-index + Spearman are the headline

| Property | C-index | Spearman ρ |
|---|---|---|
| Threshold-free | ✓ | ✓ |
| Uses all 100 pairs per domain | ✓ | ✓ |
| Tier-2 (Schema/Hybrids) > Tier-3 significant | **18/18 (100%)** | **17/18 (94%)** |
| Manual > all others significant | 12/12 | 12/12 |

By comparison, F1 only flagged 10/36 of the same Tier-2 vs Tier-3 contrasts as significant — the binary metrics hide the ranking-quality gap that C-index and Spearman expose.

---

## Source data and scripts

- Tabular data: [full_results_qwk.csv](full_results_qwk.csv), [stabilised/full_results_stabilised.csv](stabilised/full_results_stabilised.csv)
- Pre-computed pairwise tests: `significance_*.csv`
- Headline summary: [MANUAL_VS_OTHERS.md](MANUAL_VS_OTHERS.md)
- Full per-rep significance (legacy): [REPORT_QWK.md](REPORT_QWK.md)
- Generation scripts (in `/tmp/` — move to `src/analysis/` if you want them under version control):
  - `make_research_figures.py` — box+strip + CD diagrams
  - `make_significance_heatmaps.py` — CLD figures (despite the file name)
  - `build_qwk_significance_slide.py` — headline slide
