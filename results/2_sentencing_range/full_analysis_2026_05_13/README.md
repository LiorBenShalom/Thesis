# Sentencing Range Prediction — Full Analysis (2026-05-13)

ניתוח מקיף לחיזוי טווח עונש בפסקי דין פליליים בעברית (drugs + weapon).
**4,432 verdicts** (2,713 drugs + 1,719 weapon), 5-fold CV, 9 שיטות, bootstrap 95% CI.

> ⚠️ **גרסת נתונים 4,432 (2026-05-16).** כל ה-CSV/plots/טבלאות רצו על הדטה הזה. גרסת 3,898 הקודמת superseded (`data/_bak_3898_2026-05-16/`, `plots/_bak_3898_2026-05-16/`).
> **על איזה דטה ואיך כל טבלה הופקה** → [`DATA_PROVENANCE_4432.md`](DATA_PROVENANCE_4432.md).
> שינויים מול 3,898: (1) sup+LLM מנצח random+LLM **מובהקית** (limitation נפתר); (2) citation+LLM > sup+LLM על drugs (מובהק), תיקו weapon.

## 📑 תיעוד — התחילי כאן

> ⭐ **[MASTER_LOG.md](MASTER_LOG.md)** — ה-single source of truth. כל שיטה (M1-M9) וכל ניסוי (E1-E11): מה נבחר, איך, למה, כמה, מה נתן. + הקשת הנרטיבית + ה-bottom-line table. **אם משהו מבולבל — תקראי את זה קודם.**

- **[METHODOLOGY_PART1.md](METHODOLOGY_PART1.md)** — הסבר line-by-line על איך חושב כל מספר בטבלאות "Random baseline" (EXACT), "LLM bucket → gap", "Citation type → gap".
- **[METHODOLOGY_BASELINES.md](METHODOLOGY_BASELINES.md)** — line-by-line של 3 ה-baselines: offense-matched random (M2), TF-IDF+Ridge (M3), BM25 (M4) — מה הדרישה, איך נבנה, איך נבחרו המועמדים, מה נתן.

## הסיפור המלא — 3 שלבים

### ① הבעיה
לחזות טווח עונש (low_months, high_months) לכל פסק דין מתוך k=10 פסקי דין דומים. שיטה אידיאלית: LLM scoring של דמיון. בעיה: ~4M זוגות אפשריים = **~$3,770 לציין הכל**. אי-אפשר.

### ② הצורך — פילטור
ניסינו 9 שיטות פילטור + חיזוי, החל מ-baselines פשוטים ועד שיטות מודרניות:
- **Baselines**: global_median, offense_matched_random, TF-IDF+Ridge, BM25
- **Retrieval+LLM**: random+LLM, citation+LLM, supervised_only, supervised+LLM
- **Upper bound**: LLM-best (top-K מכל הLLM-scored pool)

### ③ מהפילטור לחיזוי
המסקנה הסטטיסטית (paired bootstrap CI + Wilcoxon):
- **Supervised + LLM** הפילטור המעשי (drugs MAE-lo 5.69, weapon 13.03, 100% cov)
- מנצח **מובהק** TF-IDF / BM25 / offense-matched / sup-only בשני ה-domains
- **מנצח גם את random+LLM מובהקית** (drugs Δ=−0.86 p=6.2e-6 · weapon Δ=−2.27 p=5.2e-4) — ה-limitation של 3,898 (p=0.84) **נפתר**
- **citation+LLM מדויק יותר**: drugs 5.26 (Δ=+0.74 vs sup, p=3.6e-15 מובהק), weapon 12.35 (תיקו)
- **תקרה**: LLM-best 4.97/11.85 — דורש לציין הכל

## מה יש בתיקייה

### `data/` — 32 קבצי CSV (כולל RAW DATA)
תוצאות מספריות גולמיות. ה-CSVs הקריטיים:

| קובץ | תוכן |
|---|---|
| ⭐ **`rigor_raw_per_query_K.csv`** | **242,435 רשומות (85MB)** — RAW DATA: לכל query × method × K שמורים true/pred/err + picked neighbors (JSON) + mean LLM. **משם אפשר לבנות כל גרף**. K∈{1,3,5,7,10,15,20,30,50}. |
| `rigor_per_query_errors.csv` | **33,853 רשומות** (query × method × err_lo/err_hi בK=10 בלבד). הbasis של ה-Phase B הקודם. |
| `rigor_mae_with_ci.csv` | MAE per method × domain × low/high עם bootstrap 95% CI (B=2,000) |
| `rigor_paired_diffs.csv` | Paired bootstrap differences בין השיטות + Wilcoxon p-values |
| `rigor_quartile_ci.csv` | MAE per quartile (Q1-Q4) עם CIs — בדיקת "median regressor" |
| `rigor_year_cluster.csv` | Year-clustered bootstrap — בדיקת temporal confounds |
| `rigor_hardest_cases.csv` | 30 קייסים הקשים ביותר — לniqui error analysis |
| `sweep_pool_size.csv` | **גלגל הסיפור**: pool=10→all, MAE יורד מונוטונית |
| `sweep_K.csv` | K∈{1,3,5,...,50} sensitivity |
| `sweep_source.csv` | Source-set subsampling (25%, 50%, 75%, 100%) |
| `sweep_min_k.csv` | Coverage vs MAE tradeoff |
| `deep_recall.csv` | Recall@K_oracle — supervised pool ⊃ LLM-best |
| `deep_pool_quality.csv` | Mean LLM score בpool כפונקציה של pool size |
| `deep_calibration.csv` | % cases within ±6 months |
| `deep_quartile.csv` | Per-quartile MAE × pool size |
| `deep_pareto.csv` | Cost-quality Pareto frontier ב-USD |
| `deep_hybrid.csv` | Supervised ∪ Citation hybrid pool |
| `deeper_weighted.csv` | LLM-weighted median vs simple median |
| `deeper_confidence.csv` | Top-1 LLM score → error correlation |
| `deeper_win_analysis.csv` | Win analysis: sup+LLM vs LLM-from-all |
| `deeper_overlap.csv` | Sup pool ⊃ citation 1hop |
| `comparison_5fold_mae.csv` | השוואה baseline vs filtered model 5-fold |
| `comparison_spearman.csv` | Spearman vs human-similarity GT |
| `comparison_citation_overlap.csv` | Overlap עם citation pool |

### `plots/` — 16 גרפים
Visualizations שעל ה-CSVs. ה-grafs הקריטיים לתזה:

**ה-headline** (להציג ראשון):
- `plot_pool_richness_headline.png` — MAE כפונקציה של pool size, drugs+weapon, monotonic ירידה

**ה-rigorous** (ה-academic backbone):
- `plot_rigor_forest_mae.png` — Forest plot של MAE עם 95% CIs לכל 9 השיטות
- `plot_rigor_paired_diffs.png` — Paired differences vs sup+LLM, עם significance
- `plot_rigor_quartile.png` — Per-quartile MAE עם CIs (defends median-regressor critique)
- `plot_rigor_year_cluster.png` — Year-cluster vs per-query CI width (temporal confounds)

**ה-mechanism plots** (איך זה עובד):
- `plot_deep_recall.png` — Sup pool ⊃ LLM-best, רקול עולה עם pool size
- `plot_deep_pool_quality.png` — Pool צר → high LLM score concentration
- `plot_deep_pareto.png` — Cost vs Quality Pareto curve
- `plot_deep_quartile.png` — Per-quartile MAE × pool size
- `plot_deep_calibration.png` — % within ±6 months
- `plot_deep_hybrid.png` — Hybrid sup ∪ citation
- `plot_pool_richness.png` — 4-panel version of pool richness

**ה-sweeps** (sensitivity):
- `plot_K_sweep.png` — K=1..50
- `plot_source_sweep.png` — source-set 25-100%
- `plot_min_k_coverage.png` — coverage/MAE tradeoff
- `plot_headline.png` — 4-bar summary
- `plot_K_sweep.png` — K=1..50 (sensitivity)

### `scripts/` — 14 סקריפטים
הקוד שייצר את כל הניתוחים. הקריטיים:

**Phase A v2 (NEW — RAW DATA generator)**:
- `rigor_phase_a_v2.py` — מחשב per-query × method × K dataset שלם (242K רשומות) כולל picked neighbors. **זה המסד הגולמי לכל פלוט עתידי.**
- `rigor_plotting_examples.py` — 5 מתכוני plotting מוכנים: MAE vs K, error distribution, scatter pred-vs-true, MAE by year, confidence calibration. השתמשי כתבנית.

**Phase A (original — K=10 only)**:
- `rigor_phase_a.py` — מחשב per-query MAE לכל 9 השיטות (כולל TF-IDF, BM25, offense-matched). מוגבל לK=10.

**Phase B (analysis)**:
- `rigor_phase_b.py` — Bootstrap CIs, paired tests, quartile, year-cluster, error analysis

**Sweep scripts**:
- `comprehensive_sweep.py` — K, source-set, min_k
- `pool_size_sweep.py` — ה-headline: pool=10→all monotonic

**Deep analyses**:
- `deep_analysis.py` — 7 ניתוחים: recall, pool quality, calibration, quartile, marginal, Pareto, hybrid
- `deeper_analysis.py` — 4 ניתוחים נוספים: weighted median, confidence, win analysis, overlap

**Plotting**:
- `generate_plots.py` — בסיסיים
- `plot_pool_size.py` — pool richness
- `plot_deep.py` — deep analyses
- `rigor_plots.py` — Forest plots עם CIs

**Comparisons**:
- `compare_filtered_vs_baseline.py` — בין המודלים (baseline vs filtered embedding)
- `llm_value_across_filters.py` — תרומת LLM לכל פילטור

## 🔑 ה-bottom-line numbers (לabstract)

**גרסה אחת רשמית: 4,432 (corpus==eval; full n=2,713 drugs / 1,719 weapon) — [`DATA_PROVENANCE_4432.md`](DATA_PROVENANCE_4432.md)**

| Method | Drugs MAE-lo [95% CI] | Drugs MAE-hi | Weapon MAE-lo | Weapon MAE-hi |
|---|---|---|---|---|
| Global median | 8.50 [8.11, 8.91] | 13.98 | 17.47 | 26.16 |
| Offense-matched random | 8.72 [8.37, 9.12] | 14.35 | 18.23 | 27.30 |
| TF-IDF + Ridge | 7.39 [7.08, 7.77] | 12.05 | 16.38 | 23.87 |
| BM25 | 6.65 [6.31, 7.01] | 10.60 | 15.26 | 21.77 |
| Random + LLM | 6.38 [5.99, 6.79] | 10.10 | 15.03 | 21.87 |
| Supervised cosine | 5.89 [5.58, 6.22] | 9.38 | 13.85 | 20.52 |
| **★ Supervised + LLM** | **5.69 [5.39, 6.01]** | **9.12** | **13.03** | **19.24** |
| Citation + LLM | 5.26 [4.89, 5.69] | 8.13 | 12.35 | 18.06 |
| LLM-best (UB) | 4.97 [4.68, 5.29] | 7.72 | 11.85 | 17.07 |

## 🚨 LIMITATIONS שצריך לדווח עליהם

1. ~~Random+LLM ≈ Sup+LLM~~ **נפתר ב-4,432**: sup+LLM מנצח random+LLM מובהקית (drugs p=6.2e-6 · weapon p=5.2e-4). היה limitation ב-3,898 (p=0.84).
1b. **Citation+LLM > Sup+LLM על drugs** (Δ=+0.74, p=3.6e-15) — sup+LLM הוא ה-best *מעשי* (100% cov), לא ה-best מוחלט.
2. **Year confounding ב-weapon** — year-cluster CI כפול מ-per-query.
3. **Q4 weapon errors** — קייסים עם 100-300 חודשי מאסר אינם פתירים עם הdata הקיים.
4. **Citation+LLM coverage** — רק 79-90% (חסר ב-10-21% מהקייסים).
5. **LLM-best UB** דורש $3,770 לציין הכל — לא deployable.
6. **No judge-clustering** — אין `judge_id` בנתונים הסטרוקטוריים.
7. **Hebrew tokenization של BM25** משמש whitespace tokenization נאיבית.

## איך לרוץ מחדש

```bash
cd new_try/experiments/results/2_sentencing_range/full_analysis_2026_05_13/scripts/
# Phase A — חישוב per-query errors (~3 דקות)
python3 rigor_phase_a.py
# Phase B — סטטיסטיקה (~1 דקה)
python3 rigor_phase_b.py
# Plots
python3 rigor_plots.py
python3 plot_pool_size.py
python3 plot_deep.py
```

הפלטים יסתיים אל `/tmp/` ויצריך להעתיק חזרה ל-`data/` ו-`plots/`.

## נתונים שמהם הסקריפטים נשענים

- `simcse_cuda_bundle/data/supervised_data.csv` — 3,898 verdicts (drugs+weapon)
- `simcse_cuda_bundle/outputs_supervised_filtered/verdict_embeddings_*_topk_fold*_offenseFiltered.npy` — embeddings 5-fold
- `experiments/data/sentencing_range-old/hfull_features/hybrid_full_cache.json` — H-Full features
- `experiments/data_per_domain/master_inventory.csv` — סנטנציות + year
- `experiments/data_per_domain/similarity_*.csv` — 375K LLM scores מ-6 batches
- `experiments/data_per_domain/network_analysis/citation_pair_types.csv` — citation graph

---

עודכן: 2026-05-13
