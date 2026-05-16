# `supervised_data.csv` — Schema

**4,432 שורות** (2,713 drugs + 1,719 weapon). אין nulls באף עמודה. כל 4,432 מכוסים ב-H-Full cache.

## עמודות

| עמודה | dtype | תיאור |
|---|---|---|
| `verdict` | str | מזהה ייחודי canonical (לדוגמה `תפח_20623-11-12`). המפתח לcross-reference עם embeddings index, H-Full cache, LLM scores |
| `domain` | str | `drugs` או `weapon`. **מתוקן** אחרי domain-swap 2026-05-11 (74+1 verdicts) |
| `indictment_facts` | str | טקסט עובדות כתב האישום בעברית. הקלט למודל ה-embedding (DictaBERT) ול-TF-IDF/BM25 baselines |
| `sentencing_range_low` | float | קצה תחתון של מתחם הענישה, **בחודשים**. ה-ground truth ל-MAE-low |
| `sentencing_range_high` | float | קצה עליון של מתחם הענישה, בחודשים. ה-ground truth ל-MAE-high |

## התפלגות הtarget

| | min | median | max |
|---|---|---|---|
| `sentencing_range_low` | 0 | 12 | 504 |
| `sentencing_range_high` | 2 | 30 | 999 |

ה-max=999 (weapon) הם outliers — קייסי מאסר עולם / מצטבר. אלו ה-Q4 הקשים שמפילים MAE (ראה METHODOLOGY/full_analysis).

## שרשרת ה-derivation

המקורות הנוכחיים: (א) `innovation_submission/data_master_final/verdicts_clean.csv` — הקורפוס הנקי (4,133, high-conf range, dedup), (ב) tal-data — פסקי דין גולמיים נוספים שעברו את אותו פַייפליין (range → facts → citations). הסט הסופי = כל פס"ד נקי עם כיסוי H-Full.

```
מקור A: verdicts_clean.csv (4,133 — domain∈{drugs,weapon}, range high-conf, dedup)
מקור B: tal-data raw docx → header→Hebrew-id (413 חדשים) → range high-conf (303 ניצולים)
        → facts (gpt-4-turbo) → citations (citation_classifier)
        │
        │  union, gate: כיסוי ב-hybrid_full_cache.json (H-Full),  dedup לפי verdict
        ▼
   4,432 verdicts  ←  supervised_data.csv  (2,713 drugs + 1,719 weapon)
        │  domain-swap fix 2026-05-11 (74 drugs→weapon, 1 weapon→drugs) — נשמר
        ▼
   supervised_data.csv (current)
```

הרכב: 3,898 (גרסה קודמת) + 235 (פס"ד נקיים עם H-Full שלא היו בסט) + 303 (ניצולי tal-data) − 4 כפילויות = **4,432**.

## Cross-reference keys

- `verdict` ↔ `simcse_outputs/supervised_filtered/verdict_index_*.csv` (אותו ID)
- `verdict` ↔ `data/sentencing_range-old/hfull_features/hybrid_full_cache.json` (key = verdict ID)
- `(verdict_1, verdict_2)` ↔ `data_per_domain/similarity_scores_*.csv` (LLM pairs)
- `verdict` ↔ `data_per_domain/master_inventory.csv` (`canonical_id`) — למטא-דאטה נוספת (year, sentencing_classification)

## גרסאות

| גרסה | מתי | שינוי |
|---|---|---|
| pre-swap | עד 2026-05-11 | סיווג domain מקורי (עם 75 שגיאות) |
| 3,898 | 2026-05-11 | domain מתוקן, 3,898 verdicts |
| **current (4,432)** | 2026-05-16 | +235 פס"ד נקיים עם H-Full +303 ניצולי tal-data. גיבוי `supervised_data.csv.bak_2026-05-16` (מקומי, gitignored) |
