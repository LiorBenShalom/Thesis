# Sentencing Range Prediction — Full Analysis (2026-05-13)

ניתוח מקיף לחיזוי טווח עונש בפסקי דין פליליים בעברית (drugs + weapon).
3,898 verdicts, 5-fold CV, 9 שיטות נבחנות עם bootstrap 95% CI.

## 📑 תיעוד מתודולוגיה
- **[METHODOLOGY_PART1.md](METHODOLOGY_PART1.md)** — הסבר line-by-line על איך חושב כל מספר בטבלאות "Random baseline", "LLM bucket → gap", "Citation type → gap".

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
- **Supervised + LLM rerank** הוא הפילטור המעשי המנצח (drugs MAE-lo 6.11, weapon 12.50)
- מנצח **באופן מובהק** את TF-IDF (Δ=-2.01), BM25 (Δ=-0.92), offense-matched random (Δ=-3.91)
- מוסיף LLM rerank על supervised: **שיפור מובהק קטן** (Δ=-0.27, p<1e-9)
- **לא מנצח את random+LLM באופן חד-משמעי** (CI כולל את 0 ב-Wilcoxon p=0.84) — limitation אקדמאית
- **תקרה תיאורטית**: LLM-best 5.18/8.09 — שיפור נוסף של ~17% אבל דורש לציין הכל

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

| Method | Drugs MAE-lo [95% CI] | Drugs MAE-hi | Weapon MAE-lo | Weapon MAE-hi |
|---|---|---|---|---|
| Global median | 8.43 [7.99, 8.89] | 14.08 | 16.67 | 25.46 |
| TF-IDF + Ridge | 7.56 [7.20, 7.94] | 12.48 | 15.58 | 23.13 |
| BM25 | 6.82 [6.45, 7.24] | 11.04 | 14.54 | 21.08 |
| Random + LLM | 6.34 [5.98, 6.74] | 10.11 | 13.44 | 19.70 |
| Supervised cosine | 6.33 [5.98, 6.71] | 10.22 | 13.54 | 20.29 |
| Citation + LLM | 6.11 [5.68, 6.58] | 9.46 | 12.88 | 19.59 |
| **★ Supervised + LLM** | **6.11 [5.77, 6.48]** | **9.91** | **12.50** | **18.48** |
| LLM-best (UB) | 5.18 [4.86, 5.53] | 8.09 | 11.15 | 16.09 |

## 🚨 LIMITATIONS שצריך לדווח עליהם

1. **Random+LLM ≈ Sup+LLM (Wilcoxon p=0.84)** — לא ניתן להוכיח חד-משמעית שהsupervised filter עדיף על random.
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
