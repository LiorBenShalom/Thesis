# `supervised_data.csv` — Schema

**3,898 שורות** (2,305 drugs + 1,593 weapon). אין nulls באף עמודה.

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
| `sentencing_range_low` | 0 | 12 | 384 |
| `sentencing_range_high` | 2 | 30 | 999 |

ה-max=999 (weapon) הם outliers — קייסי מאסר עולם / מצטבר. אלו ה-Q4 הקשים שמפילים MAE (ראה METHODOLOGY/full_analysis).

## שרשרת ה-derivation

```
innovation_submission/data_master/verdicts_hebrew.csv   (8,446 verdicts, raw)
        │  filter: domain ∈ {drugs, weapon}
        │          sentencing_range_low.notna()
        │          sentencing_confidence == "גבוהה"
        ▼
   5,186 verdicts (high-confidence range)
        │  further dedup + indictment_facts.notna()
        ▼
   3,898 verdicts  ←  supervised_data.csv
        │  domain-swap fix 2026-05-11 (74 drugs→weapon, 1 weapon→drugs)
        ▼
   supervised_data.csv (current — domain corrected)
```

## Cross-reference keys

- `verdict` ↔ `simcse_outputs/supervised_filtered/verdict_index_*.csv` (אותו ID)
- `verdict` ↔ `data/sentencing_range-old/hfull_features/hybrid_full_cache.json` (key = verdict ID)
- `(verdict_1, verdict_2)` ↔ `data_per_domain/similarity_scores_*.csv` (LLM pairs)
- `verdict` ↔ `data_per_domain/master_inventory.csv` (`canonical_id`) — למטא-דאטה נוספת (year, sentencing_classification)

## גרסאות

| גרסה | מתי | שינוי |
|---|---|---|
| pre-swap | עד 2026-05-11 | סיווג domain מקורי (עם 75 שגיאות) |
| **current** | 2026-05-11+ | domain מתוקן. גיבוי pre-swap = `supervised_data.csv.bak_pre_domain_swap_2026_05_11` (מקומי, gitignored) |
