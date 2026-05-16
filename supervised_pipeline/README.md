# Supervised Embedding Pipeline — Documentation Hub

מסמך מרכזי לכל מה שקשור למודל ה-supervised contrastive embedding לחיזוי טווח עונש.
**עד היום זה לא היה מתועד בגיט** — הקוד והנתונים ישבו ב-`simcse_cuda_bundle/` שהוא **מחוץ ל-git root** (`experiments/`). מסמך זה מסדר את הכל.

---

## 0. מפת ניווט — מה נמצא איפה

| נכס | מיקום בגיט | גודל | הערה |
|---|---|---|---|
| **קוד אימון** | `supervised_pipeline/code/` | 40KB | train + run scripts |
| **נתוני אימון** | `supervised_pipeline/data/supervised_data.csv` | 9.3MB | 3,898 verdicts |
| **לוגי אימון** | `supervised_pipeline/logs/` | 500KB | 5-fold × 2 domains |
| **Baseline embeddings** | `simcse_outputs/supervised/` | 81MB | מודל ללא פילטר (קיים מראש בגיט) |
| **Filtered embeddings** | `simcse_outputs/supervised_filtered/` | 59MB | המודל המסונן (חדש בגיט) |
| **Model checkpoints** | ❌ לא בגיט | 17GB | `simcse_cuda_bundle/outputs_*/model_*` — מקומי בלבד |
| **H-Full cache** | `data/sentencing_range-old/hfull_features/hybrid_full_cache.json` | 7MB | קיים בגיט |

---

## 1. שני המודלים — Baseline vs Filtered

| | **Baseline** | **Filtered** (offense-aware) |
|---|---|---|
| Backbone | DictaBERT-base | DictaBERT-base (זהה) |
| Loss | MultipleNegativesRankingLoss (InfoNCE) | זהה |
| Positive pairs | top-20 שכנים ב-Euclidean של (low, high) | top-20 שכנים **שגם חולקים ≥1 סעיף עבירה** + backfill cap 12 חודש |
| קוד | `code/train_supervised.py` | `code/train_supervised_filtered.py` |
| Runner | `code/run_5fold_cv.sh` | `code/run_5fold_cv_filtered.sh` |
| Embeddings | `simcse_outputs/supervised/` | `simcse_outputs/supervised_filtered/` |
| Suffix | `_topk_fold{1-5}` | `_topk_fold{1-5}_offenseFiltered` |

**ההבדל היחיד**: אילו זוגות חיוביים מאמנים את המודל. ה-filtered מסנן זוגות שלא חולקים סעיף עבירה — מנקה את אות האימון מ"דמיון מקרי בעונש".

---

## 2. הנתונים — `data/supervised_data.csv`

3,898 פסקי דין (2,305 drugs + 1,593 weapon). Schema מלא ב-[SCHEMA.md](data/SCHEMA.md).

המקור: נגזר מ-`innovation_submission/data_master/verdicts_hebrew.csv` (8,446) → סינון ל-drugs/weapon עם טווח עונש בביטחון "גבוהה". פירוט שרשרת ה-derivation ב-SCHEMA.md.

⚠️ **הערת domain swap (2026-05-11)**: 74 verdicts drugs→weapon + 1 weapon→drugs תוקנו (sיווג domain שגוי שזוהה דרך H-Full schema mismatch). הגרסה הנוכחית של ה-CSV כבר מתוקנת. הגיבוי `supervised_data.csv.bak_pre_domain_swap_2026_05_11` נשאר מקומי בלבד (gitignored).

---

## 3. תהליך האימון

```bash
# מקומי / SSH (A10 GPU). ראה code/run_5fold_cv_filtered.sh
cd simcse_cuda_bundle/   # מקומי — לא בגיט
source .venv/bin/activate
./run_5fold_cv_filtered.sh
```

ה-runner מבצע 10 אימונים: {drugs, weapon} × fold{1..5}. כל אימון:
1. `load_data(domain, fold)` — 5-fold CV split (seed=42), כל verdict ב-test בדיוק פעם אחת
2. `build_positive_pairs(...)` — top-20 Euclidean → סינון offense-overlap → backfill עד K=20 בתוך 12 חודש
3. אימון DictaBERT עם InfoNCE
4. הצפנת **כל** 3,898 verdicts (test מוצפנים ע"י מודל train-only)
5. שמירת `verdict_embeddings_{dom}_topk_fold{f}_offenseFiltered.npy` + index CSV

זמן: ~3-5 שעות ל-10 האימונים על A10.

ה-logs המלאים ב-`logs/`. ה-embeddings ב-`simcse_outputs/supervised_filtered/`.

---

## 4. מבנה ה-embeddings

לכל (domain, fold):
- `verdict_embeddings_{dom}_topk_fold{f}_offenseFiltered.npy` — מטריצה (N_verdicts, 768) float32, L2-normalized
- `verdict_index_{dom}_topk_fold{f}_offenseFiltered.csv` — `verdict, domain, split` (train/test)

הצריכה: `emb[i]` מתאים ל-`verdict` בשורה i של ה-index CSV. cosine = `emb @ emb.T`.

קבצים ללא `fold` (`verdict_embeddings_drugs_topk_offenseFiltered.npy`) = ריצת smoke יחידה (limit=500), לא לשימוש בניתוח הסופי.

---

## 5. מי משתמש ב-embeddings אלה

**כל** הניתוח ב-`results/2_sentencing_range/full_analysis_2026_05_13/` נשען על ה-filtered embeddings. ראה `full_analysis_2026_05_13/README.md`.

המסקנה הקצרה (5-fold CV, bootstrap 95% CI):
- Supervised+LLM: drugs MAE-lo **6.11 [5.77, 6.48]**, weapon **12.50 [11.42, 13.70]**
- מנצח TF-IDF/BM25/offense-matched מובהקית (p<1e-13)

---

## 6. שחזור — צ'קליסט

1. `data/supervised_data.csv` — נתוני האימון ✓ בגיט
2. `code/train_supervised_filtered.py` — קוד האימון ✓ בגיט
3. `code/run_5fold_cv_filtered.sh` — runner ✓ בגיט
4. H-Full cache → `../data/sentencing_range-old/hfull_features/hybrid_full_cache.json` ✓ בגיט
5. Embeddings (תוצאה) → `../simcse_outputs/supervised_filtered/` ✓ בגיט
6. Model checkpoints → ❌ לא בגיט (17GB). יש לאמן מחדש מ-(1)+(2)+(3) אם צריך את המשקלים.

**Reproducibility מלא של ה-prediction analysis אפשרי בלי המשקלים** — רק צריך את ה-embeddings (5) שכבר בגיט.

---

עודכן: 2026-05-16
