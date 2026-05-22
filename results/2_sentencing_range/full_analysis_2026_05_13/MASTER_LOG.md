# MASTER LOG — חיזוי טווח עונש: מה, איך, למה, כמה, מה נתן

מסמך-אב יחיד. כל שיטה וכל ניסוי מתועד בפורמט קבוע. **זה ה-source of truth — אם משהו סותר, זה גובר.**

---

> ## ⚠️ גרסת נתונים — קרא לפני כל מספר במסמך
>
> **כל התוצאות רצו על גרסה אחת רשמית: 4,432 פסקי דין** (corpus = eval; full-coverage methods n=2,713 drugs + 1,719 weapon). **11 שיטות** (כולל SimCSE). pool ה-LLM-scored = **254,952 זוגות** (combined). עודכן 2026-05-18.
> **בדיוק על איזה דטה ואיך כל טבלה הופקה** → [`DATA_PROVENANCE_4432.md`](DATA_PROVENANCE_4432.md).
> תוצאות 3,898 superseded. גודל pool ה-LLM גדל בשלבים: 140,961 (3,898-era) → 219,381 (+78,420 union top-20) → **254,952** (+35,571 SimCSE top-20). כל 11 השיטות חושבו על ה-pool הסופי הזה.
>
> ### Bottom-line (גרסה אחת 4,432, pool 254,952, 5-fold CV, bootstrap 95% CI)
> | שיטה | Drugs MAE-lo [CI] | Weapon MAE-lo [CI] | n (dr/we) |
> |---|---|---|---|
> | Global median | 8.50 [8.11, 8.91] | 17.47 [15.99, 19.03] | 2713/1719 |
> | Offense-matched random | 8.70 [8.34, 9.08] | 18.44 [16.75, 20.26] | 2517/1387 |
> | TF-IDF + Ridge | 7.39 [7.08, 7.75] | 16.38 [15.19, 17.67] | 2713/1719 |
> | SimCSE alone | 7.46 [7.11, 7.85] | 16.08 [14.76, 17.55] | 2713/1719 |
> | BM25 | 6.65 [6.30, 7.03] | 15.26 [14.14, 16.49] | 2713/1719 |
> | Random + LLM | 6.66 [6.30, 7.04] | 14.97 [13.61, 16.39] | 2579/1700 |
> | Supervised alone | 5.89 [5.56, 6.22] | 13.85 [12.63, 15.23] | 2713/1719 |
> | SimCSE + LLM | 5.74 [5.43, 6.07] | 13.40 [12.28, 14.60] | 2713/1719 |
> | **★ Supervised + LLM** | **5.69 [5.38, 6.03]** | **12.97 [11.78, 14.26]** | 2713/1719 |
> | Citation + LLM | 5.33 [4.96, 5.74] | 12.39 [11.18, 13.70] | 2352/1630 |
> | LLM-best (UB) | 5.12 [4.84, 5.45] | 12.12 [11.06, 13.35] | 2713/1719 |
>
> ### שינויי נרטיב מול 3,898 (קריטי לתזה — מאושר על הגרסה האחת)
> 1. **ה-limitation המרכזי נפתר**: sup+LLM **מנצח מובהקית** את random+LLM בשני ה-domains (drugs Δ=−1.22, p=7.1e-13 · weapon Δ=−2.40, p=1.4e-6). ב-3,898 לא-מובהק (p=0.84).
> 2. **citation+LLM מנצח את sup+LLM**: drugs Δ=+0.68 (p=6.0e-14, מובהק) · weapon תיקו (CI כולל 0).
> 3. **SimCSE+LLM ≈ Supervised+LLM — תיקו סטטיסטי בשני ה-domains** (drugs Δ=+0.02 p=0.84 ✗ · weapon Δ=+0.01 CI כולל 0 ✗). כלומר retrieval *לא-מפוקח* + LLM שקול ל-supervised, ב-100% coverage, ללא תוויות ענישה. **SimCSE לבד חלש** (מובהק גרוע מ-sup_only: Δ≈−2.1/−2.6). SimCSE+LLM מנצח random+LLM מובהקית; citation+LLM מנצח אותו ב-drugs (תיקו weapon).
> sup+LLM עדיין מנצח מובהקית את TF-IDF/BM25/offense-matched/sup-only/SimCSE-only.

---

## 0. שאלת המחקר

> האם מודל דמיון בין פסקי דין עוזר לחזות את טווח הענישה (low_months, high_months)?

**הצורה הניתנת להפרכה**: לכל verdict q, חזה (low, high) מתוך K שכנים. אם דמיון *לא* עוזר — שיטות מבוססות-דמיון לא ינצחו baseline נאיבי. אם כן — הן ינצחו, מובהקית.

**הנתונים**: 4,432 פסקי דין (2,713 drugs + 1,719 weapon), 5-fold CV (seed=42), כל verdict ב-test בדיוק פעם אחת. (גרסת 3,898 קודמת — superseded. פירוט מלא: [`DATA_PROVENANCE_4432.md`](DATA_PROVENANCE_4432.md).)

---

## 0.5 המציאות — האם יש בכלל signal? (מקדים לכל השיטות)

לפני שמודדים שיטות חיזוי, צריך לבסס שיש **signal** בדמיון בין תיקים. הניסוי: לכל זוג verdicts, מה הפער בעונש (|Δlow|, |Δhigh|), כפונקציה של "כמה הם דומים"? אם דמיון אמיתי → פער קטן יותר.
*(מתודולוגיה line-by-line: [METHODOLOGY_PART1.md](METHODOLOGY_PART1.md). קוד: `scripts/thesis_story_part1.py`. נתונים [4,432]: `data/story_llm_gaps.csv` (452,225 זוגות, pool 254,952), `data/story_citation_gaps.csv` (219,381 זוגות). גרף: `plots/plot_story_part1_variance.png`. random EXACT: drugs 12.96/21.11, weapon 27.38/41.20.)*

### 1️⃣ השונות הבסיסית — Random baseline (אין שום סינון דמיון)

ה"רעש" הטבעי — ההפרש הממוצע בין כל זוג תיקים. **EXACT** — ממוצע מדויק על *כל* C(n,2) הזוגות (drugs: 2,987,790; weapon: 1,398,628), לא דגימה:

| Domain | \|Δlow\| ממוצע | \|Δhigh\| ממוצע |
|---|---|---|
| **Drugs** | 12.9 חודשים | 21.1 |
| **Weapon** | 26.0 | 39.9 |

זו ה-baseline שאי אפשר לעשות עליה גרוע יותר. "ניבוי עיוור" בלי שום אינפורמציה.

### 2️⃣ נדמה את המציאות עם LLM — תיקים דומים → פער קטן יותר

לכל bucket של ציון LLM (0-100) — מה הפער הממוצע בעונש?

**DRUGS** (201,790 pairs):

| LLM bucket | n pairs | \|Δlow\| mean | \|Δhigh\| mean |
|---|---|---|---|
| 0-24 (לא דומים) | 37,762 | 11.5 | 18.5 |
| 25-49 | 93,108 | 9.1 | 14.9 |
| 50-74 | 51,548 | 7.3 | 11.7 |
| 75-89 | 17,218 | 6.9 | 10.3 |
| **90-100 (זהים כמעט)** | 2,154 | **5.2** | **7.8** |

**WEAPON** (166,140 pairs):

| LLM bucket | n pairs | \|Δlow\| mean | \|Δhigh\| mean |
|---|---|---|---|
| 0-24 | 41,772 | 25.0 | 36.9 |
| 25-49 | 79,340 | 17.7 | 25.8 |
| 50-74 | 33,838 | 15.7 | 22.7 |
| 75-89 | 10,911 | 16.2 | 23.1 |
| **90-100** | 279 | **7.3** | **10.8** |

**אות מובהק** — ככל ש-LLM מציין דמיון גבוה יותר, הפער בעונש קטן באופן מונוטוני. ב-drugs: זוגות בציון 90+ קרובים פי **2.5** מ-random. ב-weapon: פי **3-4**.

### 3️⃣ דרך אחרת לייצג מציאות — ציטוטים

תיקים שמצטטים אחד את השני (1hop) הם "דומים" משפטית לפי מערכת המשפט עצמה:

**DRUGS**:

| Citation type | n pairs | \|Δlow\| | \|Δhigh\| |
|---|---|---|---|
| **1hop** (ישיר) | 740 | **6.7** | **10.5** |
| 2hop (דרך תיק שלישי) | 2,113 | 9.0 | 14.5 |
| cocite (יחד בתיק אחר) | 47,031 | 13.0 | 21.5 |
| none | 606 | 14.1 | 22.2 |

**WEAPON**:

| Citation type | n pairs | \|Δlow\| | \|Δhigh\| |
|---|---|---|---|
| **1hop** | 543 | **11.8** | **18.3** |
| 2hop | 1,197 | 15.8 | 20.9 |
| cocite | 72,738 | 20.7 | 30.0 |
| none | 3,037 | 30.6 | 42.5 |

אותו דפוס — ככל שהקשר המשפטי הדוק יותר, הפער קטן. **1hop drugs: 6.7/10.5 חודשים — חצי מ-random.**

### איך חושב — מה / כמה / למה לכל טבלה

**ההגדרות המשותפות**:
- **Universe**: 4,118 verdicts. סינון מ-`master_inventory.csv`: `domain ∈ {drugs,weapon}` AND `sentencing_range_low.notna()` AND `sentencing_confidence == "גבוהה"` AND dedup על `canonical_id`.
- **`rng_lo[v]`, `rng_hi[v]`**: הtarget של verdict v (חודשים), נטענים ל-dict.
- **`|Δ| בין זוג (a,b)`**: `d_lo = |rng_lo[a] - rng_lo[b]|`, `d_hi = |rng_hi[a] - rng_hi[b]|`. **זה לא error — זה המרחק בין שני העונשים.** קטן = השניים קרובים בעונש.

**טבלה 1 (Random) — מה/כמה/למה**:
- *מה*: ההפרש הממוצע בין 2 verdicts כלשהם (= Gini mean difference).
- *איך*: **EXACT** — `Σ_{i<j} |x_i - x_j| / C(n,2)` ע"י broadcasting וקטורי `np.abs(x[:,None]-x[None,:]).sum()/2/C(n,2)`. דטרמיניסטי לחלוטין, ~0.04s. (גרסה קודמת השתמשה ב-50K Monte-Carlo sample; אומת ש-sample==exact ±0.1 חודש, הוחלף ל-exact כדי להסיר את שאלת "למה רק 50K".)
- *כמה*: **כל** הזוגות — drugs C(2445,2)=2,987,790; weapon C(1673,2)=1,398,628.
- *למה*: ה-floor. מה זה "ניבוי עיוור" — בלי שום הנחת דמיון. כל שיטה חייבת לנצח את זה.

**טבלה 2 (LLM bucket) — מה/כמה/למה**:
- *מה*: הפער בעונש כפונקציה של ציון הדמיון שה-LLM נתן לזוג.
- *איך*: (1) טוען 6 קבצי LLM scores → **554,004** rows. (2) dedup ע"י `tuple(sorted([v1,v2]))` → **452,225** זוגות עם target לשני הצדדים (pool combined = 254,952). (3) חלוקה ל-5 buckets `[0,25) [25,50) [50,75) [75,90) [90,101)`. (4) ממוצע d_lo/d_hi בכל bucket.
- *כמה*: drugs 201,790 זוגות (סך כל ה-buckets), weapon 166,140. ה-n בכל שורת bucket בטבלה.
- *למה*: בודק אם ה-LLM "יודע" דמיון שמתאם לדמיון-בעונש. מונוטוניות = signal.
- *הערה*: bucket 90-100 כולל ציון 100 (hi=101). מימין-סגור: `[lo, hi)`.

**טבלה 3 (Citation) — מה/כמה/למה**:
- *מה*: הפער בעונש כפונקציה של סוג הקשר הציטוטי.
- *איך*: טוען `citation_pair_types.csv` (עמודה `citation_type`). dedup זהה. ממוצע d_lo/d_hi לכל סוג.
  - `1hop` = a מצטט את b ישירות (או הפוך)
  - `2hop` = יש verdict שלישי שמקשר ביניהם
  - `cocite` = a,b מצוטטים יחד בverdict רביעי
  - `none` = אין קשר
- *כמה*: drugs 50,490 זוגות סה"כ (740+2,113+47,031+606), weapon ~78,575. ה-n בכל שורה.
- *למה*: רשת הציטוט = "דמיון משפטי לפי המערכת". baseline טבעי שלא דורש LLM.

**דוגמת חישוב יד אחת** (זוג drugs בbucket 90-100):
A=(low=24, high=60), B=(low=20, high=56) → d_lo=|24-20|=4, d_hi=|60-56|=4. ממוצע ל-2,154 הזוגות בbucket = 5.2/7.8.

### מה זה נתן (המסקנה של חלק 0.5)

✅ **יש signal אמיתי.** דמיון (לפי LLM *או* לפי 1hop ציטוט) מתואם הדוק עם דמיון בעונש, מונוטונית. זה **מה שמצדיק** את כל שאר העבודה — אם לא היה signal, אף שיטה לא הייתה עוזרת.

⚠️ **cocite ≈ random** (drugs 13.0 ≈ 12.9) — קו-ציטוט לבדו לא נושא signal. רק 1hop/2hop כן.

→ עכשיו, אחרי שביססנו שיש signal, השאלה היא: **איזו שיטה מנצלת אותו הכי טוב?** (סעיף 1 ואילך).

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
*(תיעוד line-by-line מלא: [METHODOLOGY_BASELINES.md](METHODOLOGY_BASELINES.md) §M2)*
- **מה**: K=10 אקראיים מתוך verdicts שחולקים ≥1 סעיף עבירה עם q.
- **איך**:
  1. H-Full → offense-set לכל verdict. drugs: דגלי `section_6/7/13/14/19` + `other_drug_offense` (yesno = לא ב-{"","לא","nan","None","0"}). weapon: regex על offense_number+offense_type+additional_offenses ל-`144(א/ב/ב2/ג/ז)`,`145`,`146`.
  2. מועמדים = train verdicts ש-`offense_set(t) ∩ offense_set(q) ≠ ∅` (חיתוך ≥1, **לא** זהות מלאה).
  3. דגום K=10 אקראיים, `seed = hash(q)+1` (דטרמיניסטי פר-query). אם <K → קח את כולם.
  4. median(low/high של הנבחרים).
- **דרישה**: ל-q חייב offense-set לא-ריק, אחרת אין ניבוי (לא נכנס ל-MAE).
- **למה**: ה-baseline שהמועצה (First Principles) דרשה — בודק אם **שיתוף סעיף עבירה לבדו** מספיק. אם כן → המודל הוא רק `GROUP BY offense_type` (תוצאת שנות ה-90).
- **כמה**: drugs **8.53 [8.09, 8.96]** (n=2,121 — 184 נפלו עם offense-set ריק) · weapon **17.65 [16.02, 19.30]** (n=1,272 — 321 נפלו)
- **מה נתן**: ❌ **גרוע מ-global median!** (8.53>8.43, 17.65>16.67). שיתוף סעיף לבדו לא נושא signal — שני תיקי "§7 החזקה" יכולים להיות 3 חודשים מול 5 שנים. **מפריך** את "המודל = GROUP BY offense": sup+LLM מנצח אותו ב-Δ=-3.91 (p<1e-84), כלומר המודל לומד signal אמיתי הרבה מעבר לסיווג סעיף.
- **Overlap sweep** (≥1/≥2/≥3/exact, min_k=1 עקבי, seed דטרמיניסטי — ראה METHODOLOGY_BASELINES §M2): גם דרישת התאמה הדוקה יותר לא עוזרת. ≥2 *מחמיר* (drugs 8.96, weapon 19.55), exact ≈ median (drugs 8.52). שני תיקים עם set-סעיפים זהה רחוקים בעונש כמו זוג אקראי → הסעיף לא קובע עונש בשום רמת דיוק. (נבדק גם min_k=10 → מסקנה זהה.)

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
- **מה מצאנו**: pool ה-LLM גדל ל-452,225 זוגות (combined = 254,952).
- **ההחלטה**: כל הניתוח (11 שיטות) משתמש ב-pool הסופי הזה.

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
  ├─ E5: ציינו 46K זוגות חדשים ($128)  ──────────────►  pool LLM = 452K (combined 254,952)
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

**גרסה אחת רשמית: 4,432 (corpus==eval; full n=2,713 drugs / 1,719 weapon)**, 5-fold CV, K=10, **pool LLM=254,952**, bootstrap 95% CI (B=2,000), **11 שיטות**. מקור: [`DATA_PROVENANCE_4432.md`](DATA_PROVENANCE_4432.md).

| שיטה | Drugs MAE-lo [CI] | Weapon MAE-lo [CI] | מנצח את median? | תפקיד בתזה |
|---|---|---|---|---|
| Global median | 8.50 [8.11, 8.91] | 17.47 [15.99, 19.03] | — | ה-floor |
| Offense-matched random | 8.70 [8.34, 9.08] | 18.44 [16.75, 20.26] | ❌ לא | rule-based נכשל |
| TF-IDF + Ridge | 7.39 [7.08, 7.75] | 16.38 [15.19, 17.67] | ✓ | baseline נחות |
| SimCSE alone | 7.46 [7.11, 7.85] | 16.08 [14.76, 17.55] | ❌ ≈median | retrieval לא-מפוקח לבד — חלש |
| BM25 | 6.65 [6.30, 7.03] | 15.26 [14.14, 16.49] | ✓ | baseline חזק מפתיע |
| Random + LLM | 6.66 [6.30, 7.04] | 14.97 [13.61, 16.39] | ✓ | sup+LLM מנצח אותו **מובהק** |
| Supervised alone | 5.89 [5.56, 6.22] | 13.85 [12.63, 15.23] | ✓ | זול, 0 LLM |
| SimCSE + LLM | 5.74 [5.43, 6.07] | 13.40 [12.28, 14.60] | ✓ | **≈ sup+LLM (תיקו), 100% cov, 0 תוויות** |
| **★ Supervised + LLM** | **5.69 [5.38, 6.03]** | **12.97 [11.78, 14.26]** | ✓ | **הפתרון המעשי, 100% cov** |
| Citation + LLM | 5.33 [4.96, 5.74] | 12.39 [11.18, 13.70] | ✓ | **המנצח על drugs (מובהק)** |
| LLM-best (UB) | 5.12 [4.84, 5.45] | 12.12 [11.06, 13.35] | ✓ | תקרה ($3,770) |

**משפט הסיכום**: דמיון *כן* עוזר — כל שיטות הדמיון-מבוסס-LLM מנצחות את ה-median מובהקית. **sup+LLM מנצח את random+LLM מובהקית בשני ה-domains** (drugs Δ=−1.22 p=7.1e-13, weapon Δ=−2.40 p=1.4e-6) — הפילטר עצמו תורם. citation+LLM הכי מדויק על drugs (מובהק) ותיקו weapon. **SimCSE+LLM שקול סטטיסטית ל-sup+LLM** (תיקו שני domains) — retrieval לא-מפוקח + LLM מספיק, ב-100% coverage וללא תוויות ענישה.

**ה-limitations (לדווח ביושר)**:
1. ~~sup+LLM ≈ random+LLM~~ **נפתר ב-4,432**: sup+LLM מנצח את random+LLM מובהקית (drugs Δ=−1.22 p=7.1e-13 · weapon Δ=−2.40 p=1.4e-6). היה limitation ב-3,898 (Wilcoxon p=0.84).
2. citation+LLM > sup+LLM על drugs (Δ=+0.68, p=6.0e-14) — sup+LLM אינו ה-best filter, אלא ה-best *מעשי* (100% cov; citation+LLM drugs 87% / weapon 95% cov).
3. SimCSE+LLM ≈ sup+LLM (תיקו, p=0.84/CI) — לא ניתן להעדיף את ה-supervised filter על retrieval לא-מפוקח+LLM. **SimCSE לבד** מובהק גרוע מ-sup_only.
4. weapon: year-confound (CI ×3).
5. Q4 (עונשים 100-300 חודש) — בלתי-פתיר עם הdata הקיים.
6. LLM-best UB דורש $3,770.

---

## 4b. בדיקות חוסן ושיטות מול ה-floor — *median floor* ו-*Upper-Bound proof*

### ההתפלגות של טווח-העונש (set קנוני 4,432) — *מקור*

| | mean | std | median | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| drugs LOW (n=2,713) | 12.2 | 13.5 | **9** | 16 | 30 | 36 | 56 | 240 |
| drugs HIGH | 28.2 | 20.8 | **24** | 36 | 55 | 65 | 96 | 300 |
| weapon LOW (n=1,719) | 27.8 | 35.4 | **18** | 30 | 55 | 84 | 214 | 504 |
| weapon HIGH | 53.1 | 53.5 | **40** | 60 | 96 | 144 | 276 | 999 |

זנב ימני כבד (`mean >> median`, p99 גבוה פי-3–10), במיוחד weapon. גרפים: [`plots/plot_sentencing_range_distribution.png`](plots/plot_sentencing_range_distribution.png), [`plots/plot_sentencing_range_cdf.png`](plots/plot_sentencing_range_cdf.png).

### Median-floor (random baseline מחמיר) — חישוב על *כל* C(n,2) הזוגות

| | #זוגות | MEAN \|Δ\| (רשמי) | **MEDIAN \|Δ\|** (חוסן) |
|---|---|---|---|
| drugs LOW | 3,678,828 | 12.6 | **8.0** |
| drugs HIGH | 3,678,828 | 20.4 | **14.0** |
| weapon LOW | 1,476,621 | 27.3 | **14.0** |
| weapon HIGH | 1,476,621 | 40.9 | **24.0** |

הצדקה: ה-MAE עצמו הוא ממוצע → ה-floor הרשמי חייב להיות mean-vs-mean. החציון מובא **כבדיקת חוסן מחמירה** (מנטרל את הזנב הקיצוני). נתונים: [`data/floor_mean_vs_median_robustness.csv`](data/floor_mean_vs_median_robustness.csv).

### % הקטנת רעש — כל שיטה מול שני ה-floors (MAE-lo)

| שיטה | drugs ↓ mean | drugs ↓ **median** | weapon ↓ mean | weapon ↓ **median** |
|---|---|---|---|---|
| llm_best (UB) | 59% | **36%** | 56% | **13%** |
| citation_llm | 58% | **33%** | 55% | **12%** |
| sup_llm | 55% | 29% | 52% | 7% |
| simcse_llm | 55% | 28% | 51% | 4% |
| sup_only | 53% | 26% | 49% | 1% |
| simcse_only | 41% | 7% | 41% | **−15%** ❌ |
| tfidf_ridge | 41% | 8% | 40% | **−17%** ❌ |
| random_llm | 47% | 17% | 45% | **−7%** ❌ |

**תובנת robustness:** כל שיטות ה-+LLM (sup/citation/simcse/llm_best) מנצחות **גם** את ה-floor המחמיר בשני ה-domains. שיטות retrieval-בלבד (simcse_only/tfidf) ו-random+LLM **נכשלות מול ה-median-floor ב-weapon**. כלומר רוב ה-50%+ מול ה-mean-floor מגיע מזיהוי נכון של תיקי-זנב; השיפור על תיקים *טיפוסיים* מתון, במיוחד weapon (Q4 limitation). גרף: [`plots/plot_floor_mean_vs_median.png`](plots/plot_floor_mean_vs_median.png).

### למה llm_best ≥ citation? — *הוכחה מבנית + ממצא empirically*

**טענה:** llm_best ≥ citation_llm **בהבנייה** (upper bound). הפער **לא מובהק סטטיסטית** — citation כמעט מגיע לתקרה.

**הוכחה מבנית (אלגברית):**
- שני המודלים משתמשים *באותו* LLM scorer (gpt-4.1, V6, אותו prompt) ובאותו הליך בחירה (top-K לפי ציון LLM).
- ההבדל היחיד הוא **קבוצת המועמדים** ממנה ה-LLM בוחר את ה-top-K:
  - **llm_best**: top-K מתוך *כל* train עם ציון LLM (כל ה-pool: 254,952 זוגות).
  - **citation_llm**: top-K מתוך תת-קבוצה — train שמחובר לשאילתה בגרף הציטוטים (1-hop) *וגם* בעל ציון LLM.
- מתקיים `(citation-reachable ∩ scored) ⊆ entire scored pool`.
- בחירת ה-top-K-לפי-ציון מ-superset היא חלשה-עדיפה-או-שווה לבחירה מ-subset → **llm_best ≥ citation_llm תמיד**, ללא קשר לנתונים.
- מסקנה: llm_best היא **תקרה תיאורטית** של "מה citation+LLM היה משיג אילו ה-retrieval שלו לא היה מוגבל" — לא שיטה מתחרה.

**ראיה מנגנונית — citation מגיעה רק לכ-4% מהשכנים שה-LLM היה בוחר עם גישה מלאה** (`deeper_overlap.csv`):

| domain | recall של שכני llm_best ב-citation 1-hop @K=10 | @K=100 | @K=500 |
|---|---|---|---|
| drugs | **3.99%** | 18.9% | 46.4% |
| weapon | **5.61%** | 26.7% | 60.3% |

ובנוסף — כיסוי citation: drugs **87%** / weapon **95%** מהשאילתות בלבד יש להן ולו מועמד-ציטוט אחד עם ציון.

**ההפתעה — הפער לא מובהק** (paired bootstrap + Wilcoxon על שאילתות משותפות):

| domain | n משותף | citation MAE-lo | llm_best MAE-lo | Δ (citation−UB) | 95% CI | Wilcoxon p | מובהק? |
|---|---|---|---|---|---|---|---|
| drugs | 2,352 | 5.33 | 5.12 | +0.21 | **[−0.04, +0.51]** | 0.80 | **✗ NOT sig** |
| weapon | 1,630 | 12.39 | 12.12 | +0.43 | **[−0.25, +1.11]** | 0.21 | **✗ NOT sig** |

**הפרשנות (לתזה):** למרות ש-citation **משליכה ~96% מהשכנים שה-LLM היה מעדיף** עם גישה גלובלית, הדיוק שלה **שקול סטטיסטית** לתקרה. כלומר ה-4-6% שהיא *כן* מגיעה אליהם — ציטוטים שיפוטיים מצביעים על תיקים תקדימיים/עובדתיים-קרובים — הם *בדיוק* האות החזק. ה-LLM-rerank מעל הסט הקטן והאיכותי הזה מחלץ כמעט את כל האות הזמין. **citation+LLM היא essentially-optimal** ב-$120 במקום ה-$3,770 של ה-oracle.

נתונים: [`data/upperbound_vs_citation_proof.csv`](data/upperbound_vs_citation_proof.csv).

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
