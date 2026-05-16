# MASTER LOG — חיזוי טווח עונש: מה, איך, למה, כמה, מה נתן

מסמך-אב יחיד. כל שיטה וכל ניסוי מתועד בפורמט קבוע. **זה ה-source of truth — אם משהו סותר, זה גובר.**

---

## 0. שאלת המחקר

> האם מודל דמיון בין פסקי דין עוזר לחזות את טווח הענישה (low_months, high_months)?

**הצורה הניתנת להפרכה**: לכל verdict q, חזה (low, high) מתוך K שכנים. אם דמיון *לא* עוזר — שיטות מבוססות-דמיון לא ינצחו baseline נאיבי. אם כן — הן ינצחו, מובהקית.

**הנתונים**: 3,898 פסקי דין (2,305 drugs + 2,593... → 1,593 weapon), 5-fold CV, כל verdict ב-test בדיוק פעם אחת.

---

## 1. רישום השיטות (METHOD REGISTRY)

כל שיטה: **מה** = מה היא בקצרה | **איך** = הצינור הטכני | **למה** = מה היא בודקת | **כמה** = MAE-low drugs/weapon (5-fold, bootstrap 95% CI) | **מה נתן** = המסקנה.

### M1 · Global median
- **מה**: ניבוי קבוע = החציון של הdomain (drugs: low=9/high=24; weapon: 18/40).
- **איך**: אין שכנים. `pred = median(all train low/high של הdomain)`.
- **למה**: ה-floor המוחלט. "ניבוי עיוור" — אי אפשר לעשות גרוע מזה עם מידע.
- **כמה**: drugs **8.43 [7.99, 8.89]** · weapon **16.67 [15.32, 18.18]**
- **מה נתן**: ה-reference. כל שיטה אחרת נמדדת ביחס לזה.

### M2 · Offense-matched random
- **מה**: K אקראיים מתוך verdicts שחולקים ≥1 סעיף עבירה עם q.
- **איך**: H-Full → offense set לכל verdict → דגום K=10 random מאלה שחופפים → median.
- **למה**: בודק אם **שיתוף סעיף עבירה לבדו** מספיק לחיזוי (baseline rule-based שהמועצה דרשה).
- **כמה**: drugs **8.53 [8.09, 8.96]** · weapon **17.65 [16.02, 19.30]**
- **מה נתן**: ❌ **גרוע מ-global median!** שיתוף סעיף לבדו לא נושא signal. צריך מנגנון בחירה *בתוך* הקבוצה.

### M3 · TF-IDF + Ridge
- **מה**: רגרסיה ישירה מהטקסט ל-(low, high). לא retrieval.
- **איך**: TF-IDF char n-grams (3-5) על indictment_facts → Ridge(α=10) חוזה low ו-high בנפרד. fit על fold-train.
- **למה**: baseline ML סטנדרטי שכל reviewer ידרוש. "אולי לא צריך דמיון בכלל, רק רגרסיה?"
- **כמה**: drugs **7.56 [7.20, 7.94]** · weapon **15.58 [14.50, 16.66]**
- **מה נתן**: עדיף על median אבל **מפסיד מובהקית לכל שיטות הדמיון** (p<1e-40). מעניין: overshoots קייסים קלים (Q1 MAE=11), מצוין ב-Q3. → regressor מטעה, לא פתרון.

### M4 · BM25
- **מה**: retrieval לקסיקלי קלאסי + median של top-K.
- **איך**: tokenize ב-whitespace → BM25Okapi על fold-train → top-K → median.
- **למה**: baseline retrieval סטנדרטי. "אולי lexical match מספיק, בלי deep learning?"
- **כמה**: drugs **6.82 [6.45, 7.24]** · weapon **14.54 [13.49, 15.61]**
- **מה נתן**: 🟡 **הפתעה — חזק מאוד**. קרוב ל-supervised, מנצח TF-IDF. אבל sup+LLM עדיין מנצח אותו מובהקית (drugs Δ=-0.92, p<1e-13).

### M5 · Random + LLM
- **מה**: 50 candidates אקראיים → LLM מדרג → top-K → median.
- **איך**: דגום 50 random מ-fold-train → ציון LLM לכל (q, c) → sort → top-10 → median.
- **למה**: מבודד את תרומת ה-LLM. "כמה ה-LLM לבדו מציל pool גרוע?"
- **כמה**: drugs **6.34 [5.98, 6.74]** · weapon **13.44 [12.17, 14.79]**
- **מה נתן**: ⚠️ **קריטי** — כמעט זהה ל-sup+LLM. ה-LLM rerank הוא העובד העיקרי; הפילטר רק מספק pool.

### M6 · Citation + LLM
- **מה**: candidates מרשת הציטוטים (1hop+2hop+cocite) → LLM rerank → top-K.
- **איך**: כל verdict שמקושר ציטוטית ל-q → ציון LLM → top-10 → median.
- **למה**: רשת ציטוט = "דמיון משפטי לפי המערכת עצמה". baseline טבעי.
- **כמה**: drugs **6.11 [5.68, 6.58]** · weapon **12.88 [11.68, 14.10]**
- **מה נתן**: 🟢 איכותי מאוד — שווה ל-sup+LLM. **אבל coverage 79-90% בלבד** (תיקים בלי ציטוטים נופלים) → לא scalable כפילטר יחיד.

### M7 · Supervised alone (filtered model, cosine top-K)
- **מה**: top-K הקרובים ב-cosine של ה-embedding המסונן. בלי LLM.
- **איך**: DictaBERT מסונן (offense-aware contrastive) → cosine(q, train) → top-10 → median.
- **למה**: בודק כמה המודל לבדו תופס — בלי עלות LLM כלל.
- **כמה**: drugs **6.33 [5.98, 6.71]** · weapon **13.54 [12.36, 14.86]**
- **מה נתן**: ✓ 25% הורדה מ-median, **inference חינמי לחלוטין**. ה-baseline המעשי-זול.

### M8 · Supervised + LLM rerank ★ (השיטה המוצעת)
- **מה**: top-100 cosine מהמסונן → LLM מדרג → top-K → median.
- **איך**: cosine top-100 → ציון LLM לכל אחד → sort → top-10 → median.
- **למה**: השילוב — פילטר זול מצמצם 4M→100, LLM עושה את הבחירה הסופית.
- **כמה**: drugs **6.11 [5.77, 6.48]** · weapon **12.50 [11.42, 13.70]**
- **מה נתן**: ✅ **הפתרון המעשי**. מנצח מובהקית את TF-IDF (Δ=-2.01), BM25 (Δ=-0.92), offense-matched (Δ=-3.91), ו-sup_only (Δ=-0.27, p<1e-9). 100% coverage.

### M9 · LLM-best (upper bound)
- **מה**: top-K לפי ציון LLM מתוך *כל* fold-train (בלי שום פילטר).
- **איך**: ציון LLM לכל (q, train_v) → sort → top-10 → median.
- **למה**: ה-ceiling התיאורטי. "מה היה אם היה תקציב לציין הכל?"
- **כמה**: drugs **5.18 [4.86, 5.53]** · weapon **11.15 [10.21, 12.19]**
- **מה נתן**: 🏆 התקרה. שיפור נוסף ~17% מעל M8 — אבל דורש ~$3,770 לציין את כל 4M הזוגות. **לא ריאלי בפרודקשן.**

---

## 2. רישום הניסויים (EXPERIMENT REGISTRY)

כל ניסוי: **שאלה** → **מה עשינו** → **מה מצאנו** → **ההחלטה שהתקבלה**.

### E1 · ביקורת overlap בזוגות האימון
- **שאלה**: האם top-20 הזוגות החיוביים שמאמנים את ה-baseline חולקים סעיף עבירה?
- **מה עשינו**: חישבנו offense-set מ-H-Full לכל verdict, בדקנו חפיפה בזוגות.
- **מה מצאנו**: ~30% מהזוגות **לא חולקים אף סעיף** (drugs 30.7%, weapon 32.2%, eligible).
- **ההחלטה**: ה-baseline מתאמן על "דמיון מקרי בעונש" → לבנות מודל מסונן.

### E2 · ביקורת + תיקון domain misclassification
- **שאלה**: למה ל-505 verdicts אין offense-set?
- **מה עשינו**: ניתוח 3-כיווני (schema format + indictment text).
- **מה מצאנו**: 74 drugs בעצם weapon, 1 weapon בעצם drugs (סיווג שגוי). 198 הנשק/סם הוא לא הסעיף החמור. 85 manual-review.
- **ההחלטה**: בוצע SWAP ל-74+1 (לא מחיקה — המשתמשת בחרה). `supervised_data.csv` עודכן. גיבוי נשמר.

### E3 · אימון המודל המסונן (5-fold)
- **שאלה**: האם פילטר offense-overlap משפר את ה-embedding?
- **מה עשינו**: `train_supervised_filtered.py` — top-20 Euclidean → סינון offense-overlap → backfill עד K=20 בתוך 12 חודש. אומן 5-fold × 2 domains על AWS A10.
- **מה מצאנו**: ב-dry-run, drugs שומר 95% מ-anchors, weapon 80%.
- **ההחלטה**: המודל המסונן הוא ה-embedding לכל הניתוח הבא.

### E4 · השוואה filtered vs baseline
- **שאלה**: המסונן באמת טוב יותר?
- **מה עשינו**: MAE 5-fold, Spearman מול human-GT, citation overlap, LLM score של top-K.
- **מה מצאנו**: drugs MAE שווה (6.33≈6.33), weapon נפגע (+1.5). אבל **Spearman השתפר** (drugs 0.44→0.51, weapon 0.20→0.24) ו-**LLM mean של top-K עלה** (drugs 44→49, weapon 39→46).
- **ההחלטה**: המסונן לומד דמיון "משפטי" יותר אמיתי גם אם MAE-weapon מעט נפגע. ממשיכים איתו.

### E5 · ציון LLM batch חדש
- **שאלה**: מה ה-LLM אומר על ה-top-K של המסונן?
- **מה עשינו**: 46,198 זוגות חדשים → gpt-4.1 + V6 prompt + H-Full → batch (~$128). דה-דופ נגד 267K קיימים + 5fold_v2 in-flight.
- **מה מצאנו**: pool ה-LLM גדל ל-375,658 זוגות.
- **ההחלטה**: כל הניתוח הבא משתמש ב-pool של 375K.

### E6 · K sweep
- **שאלה**: K=10 הוא האופטימום?
- **מה עשינו**: K ∈ {1,3,5,7,10,15,20,30,50} לכל שיטה.
- **מה מצאנו**: sup_llm peaks ב-K=5-10; llm_best peaks ב-K=15-20; כולם יציבים מ-K=5 עד K=20.
- **ההחלטה**: K=10 בחירה defensible (יציב, סטנדרטי). דווח sweep כ-sensitivity.

### E7 · Source-set sweep
- **שאלה**: כמה דטה צריך? מה אם רק 25%?
- **מה עשינו**: subsample 25/50/75/100% מ-fold-train.
- **מה מצאנו**: sup_llm יציב מאוד (25%: 6.09 ≈ 100%: 6.11). robustness חזק.
- **ההחלטה**: המודל לא תלוי בכמות דטה ענקית — חיזוק לסיפור.

### E8 · min_k sweep
- **שאלה**: מה אם דורשים מינימום K שכנים לחיזוי?
- **מה עשינו**: min_k ∈ {1,3,5,10}.
- **מה מצאנו**: ל-sup/sup_llm (100% cov) — אין שינוי. ל-citation — strict min_k מוריד coverage ל-47% אבל "משפר" MAE מלאכותית (מסנן קייסים קשים).
- **ההחלטה**: **חשיפת bias** — ה-citation+LLM "6.90 על 50%" המדווח בעבר היה מטעה. דווח הכל ב-min_k=1 (100% coverage) להוגנות.

### E9 · Pool-size sweep ★ (ה-headline)
- **שאלה**: ככל שנותנים ל-LLM pool עשיר יותר — הוא משתפר?
- **מה עשינו**: sup top-N → LLM rerank → top-10, N ∈ {10,20,50,100,200,500,1000,all}.
- **מה מצאנו**: **MAE יורד מונוטונית**. drugs: 6.33→5.27 (-17%); weapon: 13.54→10.84 (-20%).
- **ההחלטה**: זה ה-narrative — "LLM הוא ה-workhorse; המודל המסונן הוא ה-candidate funnel; ככל שה-pool גדול → LLM טוב יותר".

### E10 · ניתוחי עומק
- **שאלה**: איך בדיוק ה-funnel עובד?
- **מה עשינו**: 7+4 ניתוחים — recall@K_oracle, pool quality, calibration, per-quartile, marginal, Pareto, hybrid, weighted median, confidence, win analysis, citation-recall.
- **מה מצאנו (עיקרי)**:
  - Recall: sup pool=100 תופס 40% מ-LLM-best; pool=1000 → 84-95%.
  - Pareto: pool=1000 משיג ~99% מ-"all" ב-55% עלות.
  - Per-quartile: לא median-regressor — Q4 הקשים משתפרים הכי הרבה עם pool.
  - Win analysis: LLM-from-all מנצח את sup+LLM ב-46% מהמקרים → pool גדול עוזר ישירות.
- **ההחלטה**: הוכחת המנגנון של "pool richness". ניתנת לתזה כ-mechanism section.

### E11 · ניתוח סטטיסטי קפדני
- **שאלה**: ההבדלים מובהקים? מה מול baselines אמיתיים?
- **מה עשינו**: bootstrap 95% CI (B=2,000), paired Wilcoxon, הוספת TF-IDF/BM25/offense-matched, year-clustered bootstrap.
- **מה מצאנו**:
  - sup_llm מנצח TF-IDF/BM25/offense-matched/sup_only — **מובהק** (p<1e-9 עד 1e-84).
  - sup_llm vs random_llm: **Wilcoxon p=0.84 — לא מובהק** (limitation!).
  - sup_llm vs citation_llm: CI כולל 0 — לא חד-משמעי.
  - year-cluster: weapon CI מתרחב פי 3 → confound שנתי.
- **ההחלטה**: הסיפור הכן — sup+LLM מנצח baselines קלאסיים מובהקית, אבל ≈ random+LLM. ה-LLM הוא הסיגנל; הפילטר הוא enabler לעלות. דווח את ה-limitations.

---

## 3. הקשת הנרטיבית — איך כל צעד קידם

```
שאלה: דמיון עוזר לחיזוי עונש?
  │
  ├─ E1: גילינו ש-30% מזוגות האימון רעש  ──────────►  בנינו מודל מסונן (E3)
  │                                                      │
  ├─ E2: תיקנו 75 סיווגי domain שגויים  ──────────────►  data נקי יותר
  │                                                      │
  ├─ E4: מסונן = Spearman↑ + LLM-score↑  ─────────────►  ממשיכים עם מסונן
  │                                                      │
  ├─ E5: ציינו 46K זוגות חדשים ($128)  ──────────────►  pool LLM = 375K
  │                                                      │
  ├─ E9: ✦ pool גדול → MAE↓ מונוטונית  ✦  ───────────►  ה-HEADLINE
  │       (LLM הוא ה-workhorse, פילטר = funnel)         │
  │                                                      │
  ├─ E10: המנגנון — recall, Pareto, quartile  ───────►  הוכחת ה-funnel
  │                                                      │
  └─ E11: bootstrap CIs + baselines אמיתיים  ────────►  ה-honest bottom line
```

---

## 4. ה-Bottom Line (הטבלה היחידה שצריך לזכור)

5-fold CV, K=10, bootstrap 95% CI:

| שיטה | Drugs MAE-lo [CI] | Weapon MAE-lo [CI] | מנצח את median? | תפקיד בתזה |
|---|---|---|---|---|
| Global median | 8.43 [7.99, 8.89] | 16.67 [15.32, 18.18] | — | ה-floor |
| Offense-matched random | 8.53 [8.09, 8.96] | 17.65 [16.02, 19.30] | ❌ לא | rule-based נכשל |
| TF-IDF + Ridge | 7.56 [7.20, 7.94] | 15.58 [14.50, 16.66] | ✓ | baseline נחות |
| BM25 | 6.82 [6.45, 7.24] | 14.54 [13.49, 15.61] | ✓ | baseline חזק מפתיע |
| Supervised alone | 6.33 [5.98, 6.71] | 13.54 [12.36, 14.86] | ✓ | זול, 0 LLM |
| Random + LLM | 6.34 [5.98, 6.74] | 13.44 [12.17, 14.79] | ✓ | מראה ש-LLM הוא העובד |
| Citation + LLM | 6.11 [5.68, 6.58] | 12.88 [11.68, 14.10] | ✓ | איכותי, 79% cov |
| **★ Supervised + LLM** | **6.11 [5.77, 6.48]** | **12.50 [11.42, 13.70]** | ✓ | **הפתרון המעשי** |
| LLM-best (UB) | 5.18 [4.86, 5.53] | 11.15 [10.21, 12.19] | ✓ | תקרה ($3,770) |

**משפט הסיכום**: דמיון *כן* עוזר — כל שיטות הדמיון מנצחות את ה-median מובהקית. ה-LLM הוא מקור הסיגנל העיקרי; המודל המסונן הוא הפילטר הזול שמאפשר להפעיל אותו ב-scale (4M זוגות → 100 לכל query, ~$120 במקום $3,770), ב-100% coverage שאף baseline אחר לא נותן.

**ה-limitations (לדווח ביושר)**:
1. sup+LLM ≈ random+LLM (לא מובהק, Wilcoxon p=0.84) — ה-LLM, לא הפילטר, הוא הסיגנל.
2. weapon: year-confound (CI ×3).
3. Q4 (עונשים 100-300 חודש) — בלתי-פתיר עם הdata הקיים.
4. LLM-best UB דורש $3,770.

---

## 5. מפת קבצים — לאיזה ניסוי שייך מה

| ניסוי | סקריפט | נתונים | גרף |
|---|---|---|---|
| E1 (offense overlap) | `scripts/` (analyze_offense_*) | — | — |
| E6-E8 (sweeps) | `scripts/comprehensive_sweep.py` | `data/sweep_{K,source,min_k}.csv` | `plots/plot_{K,source,min_k}*.png` |
| E9 (pool-size) ★ | `scripts/pool_size_sweep.py` | `data/sweep_pool_size.csv` | `plots/plot_pool_richness_headline.png` |
| E10 (deep) | `scripts/deep_analysis.py`, `deeper_analysis.py` | `data/deep_*.csv`, `deeper_*.csv` | `plots/plot_deep_*.png` |
| E11 (rigor) | `scripts/rigor_phase_a.py` + `rigor_phase_b.py` | `data/rigor_*.csv` | `plots/plot_rigor_*.png` |
| Part 1 (variance reality) | `scripts/thesis_story_part1.py` | `data/story_*.csv` | `plots/plot_story_part1_variance.png` |
| RAW DATA לכל פלוט | `scripts/rigor_phase_a_v2.py` | `data/rigor_raw_per_query_K.csv` (242K rows) | (תבנית: `rigor_plotting_examples.py`) |
| Supervised pipeline | `../../../supervised_pipeline/` | `supervised_data.csv` | — |

---

עודכן: 2026-05-16
