# Dataset Report — `data_per_domain/`

**Generated:** 2026-04-25
**Location:** `new_try/experiments/data_per_domain/`

הסט הסופי, מאוחד, מנורמל לעברית, ללא כפילויות, לאחר חילוץ מתחמים ובניית רשת ציטוטים.

---

## 1. מבנה התיקיות

```
data_per_domain/
├── drugs/
│   ├── csv/                  ← 3,720 קבצי CSV של פסקי דין
│   ├── docx/                 ← 3,714 קבצי DOCX מקור
│   └── inventory.csv         ← רשומה לכל תיק
├── weapon/
│   ├── csv/                  ← 3,693
│   ├── docx/                 ← 3,688
│   └── inventory.csv
├── appeals/                  ← 1,391 verdicts (DOCX בלבד ברובם)
├── unknown/                  ← 318 verdicts (לא ניתן לסווג)
├── master_inventory.csv      ← הכל יחד עם metadata מלא
├── extracted_citations.csv   ← לוג חילוץ ציטוטים מ-headers
└── DUPLICATES_REPORT.md      ← דו"ח חזרות + מקורות
```

**כל הקבצים אמיתיים — לא symlinks.**

---

## 2. ספירה כללית — 9,122 verdicts ייחודיים

| Domain | Total | עם CSV | עם DOCX | עם BOTH |
|---|---|---|---|---|
| **drugs** | 3,720 | 3,720 | 3,714 | 3,714 |
| **weapon** | 3,693 | 3,693 | 3,688 | 3,688 |
| appeals | 1,391 | 7 | 1,391 | 7 |
| unknown | 318 | 271 | 68 | 21 |

---

## 3. הסט הראשי — drugs+weapon עם מתחם עונש

לאחר extraction חוזר על 787 תיקי NaN פוסט-2012:

| | Drugs | Weapon | Total |
|---|---|---|---|
| Total verdicts | 3,720 | 3,693 | 7,413 |
| **עם מתחם עונש (high-conf)** | **2,466** | **1,811** | **4,277** |
| % עם מתחם | 66% | 49% | 58% |

### חלוקת המתחם

| Domain | Low (median) | High (median) |
|---|---|---|
| drugs | 9 חודש | 24 חודש |
| weapon | 18 חודש | 40 חודש |

### תיקים בלי מתחם (3,136)

| קטגוריה | תיקים | סיבה |
|---|---|---|
| pre-2012 | ~1,613 | החוק לא דרש מתחם |
| NEGATIVE classifier | ~1,495 | אין מתחם בטקסט |
| POSITIVE conf. בינונית | 27 | ביטחון נמוך |

---

## 4. רשת ציטוטים — 4,277 התיקים עם מתחם

### ציטוטים ישירים

| | |
|---|---|
| סה"כ ציטוטים | 48,319 |
| **ציטוטים לתיקים אחרים בסט (in-set)** | **7,861 (16.3%)** |
| תיקים שמצטטים ≥1 in-set | 2,100 (49%) |
| תיקים שמצטטים ≥3 in-set | **1,040 (24%)** |

### זוגות מועמדים — ניתוח רשת

(זוגות מאותו דומיין בלבד; cross-domain לא נספרים)

| רמת רשת | Drugs | Weapon | Total |
|---|---|---|---|
| **1-hop** (ציטוט ישיר) | 1,990 | 3,246 | **5,236** |
| **2-hop forward** (A→B→C) | 1,380 | 3,808 | 5,188 |
| **Co-citation** (A→B←X) | 16,330 | 60,403 | 76,733 |
| **UNION (כל הרשת)** | **19,113** | **65,980** | **85,093** |

### כיסוי

תיקים שיש להם לפחות מועמד אחד דרך הרשת: **2,433 / 4,118 (59%)**

הרשת מתפתחת מ-5,236 ל-85,093 מועמדים — פי 16. רוב התוספת מ-co-citation (תיקים שמצטטים את אותו תקדים).

---

## 5. ציר הזמן (drugs+weapon)

| תקופה | Total | עם מתחם | % |
|---|---|---|---|
| **לפני 2012** | 1,697 | 84 | 5% |
| **2012+** | 5,753 | 4,200 | 73% |

תיקון 113 לחוק העונשין (2012) חייב הגדרת מתחם. ולכן:
- pre-2012: רוב התיקים אין מתחם (החוק לא דרש)
- post-2012: רוב התיקים עם מתחם

---

## 6. צינור הבנייה (8 סקריפטים תחת `innovation_submission/scripts/`)

1. `build_per_domain.py` — סריקה מקיפה של כל 24+ תיקיות בthesis, איסוף קבצים
2. `extract_citations_all.py` — פתיחת כל docx וחילוץ ה-citation מה-header
3. `rename_files_to_canonical.py` — שינוי שמות לcanonical Hebrew
4. `extract_missing_ranges.py` — חילוץ מתחמים על post-2012 NaN
5. `rename_docx_by_citation.py` — פונקציות נרמול canonical (268 ראשי תיבות)
6. `normalize_citations.py` — נרמול JSON של ציטוטים
7. `build_final_dataset.py` — דדאופ + בניית master
8. `finalize_dataset.py` — סידור עמודות + README

---

## 7. שימוש — Loading

```python
import pandas as pd
master = pd.read_csv("data_per_domain/master_inventory.csv")

# All drugs+weapon with range
df = master[
    master['domain'].isin(['drugs','weapon'])
    & master['sentencing_range_low'].notna()
    & (master['sentencing_confidence']=='גבוהה')
]
# 4,277 verdicts ready for sentence range prediction

# Read full text of a verdict
import os
text = pd.read_csv(f"data_per_domain/{row['domain']}/{row['csv_path']}")['text'].str.cat(sep='\n')
```

---

## 8. סיכום חד

✅ **9,122 unique verdicts** עברית מנורמלת, ללא כפילויות, עם קבצים אמיתיים  
✅ **7,413 drugs+weapon** = הסט הראשי לעבודה  
✅ **4,277 עם מתחם** = הסט המקסימלי לחיזוי טווח עונש  
✅ **5,236 זוגות citation 1-hop** = candidates ראשונים  
✅ **85,093 זוגות citation network** = candidates מורחבים  
✅ **כל הקבצים בעברית canonical**, סינכרון מלא בין master, csv, docx
