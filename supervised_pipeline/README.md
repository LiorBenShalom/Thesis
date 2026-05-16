# Supervised Embedding Pipeline — Documentation Hub

מסמך מרכזי לכל מה שקשור למודל ה-supervised contrastive embedding לחיזוי טווח עונש.
**עד היום זה לא היה מתועד בגיט** — הקוד והנתונים ישבו ב-`simcse_cuda_bundle/` שהוא **מחוץ ל-git root** (`experiments/`). מסמך זה מסדר את הכל.

---

## 0. מפת ניווט — מה נמצא איפה

| נכס | מיקום בגיט | גודל | הערה |
|---|---|---|---|
| **קוד אימון** | `supervised_pipeline/code/` | 40KB | train + run scripts |
| **נתוני אימון** | `supervised_pipeline/data/supervised_data.csv` | ~11MB | **4,432 verdicts** (2,713 drugs + 1,719 weapon) |
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

**4,432 פסקי דין** (2,713 drugs + 1,719 weapon). אין nulls. כל 4,432 מכוסים ב-H-Full cache (4,433 keys). Schema מלא ב-[SCHEMA.md](data/SCHEMA.md).

המקורות: (א) `innovation_submission/data_master_final/verdicts_clean.csv` — הקורפוס הנקי (4,133, high-conf range, dedup); (ב) tal-data — פסקי דין גולמיים נוספים שעברו את אותו פַייפליין (header→Hebrew-id → range high-conf → facts gpt-4-turbo → citations). הסט = איחוד כל פס"ד נקי עם כיסוי H-Full, dedup לפי `verdict`. הרכב: 3,898 (קודם) + 235 + 303 − 4 כפילויות = 4,432.

⚠️ **domain swap (2026-05-11)**: 74 drugs→weapon + 1 weapon→drugs — נשמר בגרסה הנוכחית.

⚠️ **גרסת נתונים 4,432 (2026-05-16)**: ה-embeddings וכל ניתוח ה-prediction (סעיפים 4-5) מבוססים כרגע על גרסת 3,898 הקודמת. אימון 5-fold מחדש על 4,432 + ניקוד LLM + ניתוח מלא — **בתהליך**. המספרים בסעיף 5 יתעדכנו בסיום.

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
4. הצפנת **כל** verdicts ה-domain (test מוצפנים ע"י מודל train-only)
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

המסקנה הקצרה **(גרסת 3,898 — לפני הרחבת הנתונים ל-4,432; re-run בתהליך)** (5-fold CV, bootstrap 95% CI):
- Supervised+LLM: drugs MAE-lo **6.11 [5.77, 6.48]**, weapon **12.50 [11.42, 13.70]**
- מנצח TF-IDF/BM25/offense-matched מובהקית (p<1e-13)
- ⏳ ניתוח מחדש על 4,432 (embeddings חדשים + ניקוד LLM) יעדכן מספרים אלה.

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
