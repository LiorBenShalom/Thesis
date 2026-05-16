# מתודולוגיה — חלק 1: השונות במציאות

> ⚠️ נתונים עדכניים = **4,432** (2026-05-16). הלוגיקה זהה; הערכים העדכניים ב-`data/*.csv` ו-[`DATA_PROVENANCE_4432.md`](DATA_PROVENANCE_4432.md). random EXACT 4,432: drugs 12.96/21.11, weapon 27.38/41.20.

תיעוד מלא של איך חושב כל מספר בטבלאות **"השונות הבסיסית"**, **"LLM bucket → gap"**, ו**"Citation type → gap"**.

**הסקריפט המקור**: [`scripts/thesis_story_part1.py`](scripts/thesis_story_part1.py)
**גרף**: [`plots/plot_story_part1_variance.png`](plots/plot_story_part1_variance.png)
**CSVs**:
- [`data/story_llm_gaps.csv`](data/story_llm_gaps.csv) — 367,930 שורות (כל LLM-scored pair × |Δsentencing|)
- [`data/story_citation_gaps.csv`](data/story_citation_gaps.csv) — 128,575 שורות (כל citation pair × |Δsentencing|)

---

## הגדרות בסיסיות (משותפות לכל הטבלאות)

### 1. ה-Universe של פסקי דין שאנחנו עובדים עליו

**מקור**: [`data_per_domain/master_inventory.csv`](../../data_per_domain/master_inventory.csv)

**מסנן** (קוד `thesis_story_part1.py` שורות 19-25):
```python
m = pd.read_csv("master_inventory.csv", usecols=[
    "canonical_id", "domain",
    "sentencing_range_low", "sentencing_range_high",
    "sentencing_confidence"
])
m = m[m.domain.isin(["drugs","weapon"])
      & m.sentencing_range_low.notna()
      & (m.sentencing_confidence == "גבוהה")
      ].drop_duplicates("canonical_id")
```

**תוצאה**: **4,118 verdicts** (drugs + weapon, עם טווח עונש תקף, ביטחון "גבוהה").

### 2. השדות שמשמשים כ-ground truth
```python
rng_lo[verdict_id] = sentencing_range_low   # ה-low_months של פסק הדין
rng_hi[verdict_id] = sentencing_range_high  # ה-high_months
```

הם דחוסים ל-dict ב-RAM למהירות.

### 3. מה זה "|Δ sentencing| בין שני פסקי דין"?
לכל **זוג** (q, p):
```python
d_lo = |rng_lo[q] - rng_lo[p]|   # פער בlow_months
d_hi = |rng_hi[q] - rng_hi[p]|   # פער בhigh_months
```

זה לא error, זה פשוט המרחק בין שני העונשים. הקטן יותר → השניים קרובים בעונש.

---

## טבלה 1️⃣ — Random Baseline

### המספרים שדווחו (EXACT — על כל הזוגות)

| Domain | \|Δlow\| ממוצע | \|Δhigh\| ממוצע | # זוגות |
|---|---|---|---|
| **Drugs** | 12.9 (12.89) | 21.1 (21.11) | C(2445,2) = 2,987,790 |
| **Weapon** | 26.0 (26.01) | 39.9 (39.90) | C(1673,2) = 1,398,628 |

### איך זה חושב

**שיטה: ממוצע מדויק על כל C(n,2) הזוגות** (= Gini mean difference). לא דגימה.

**קוד** (`thesis_story_part1.py`):
```python
for dom in ("drugs", "weapon"):
    vs = [v for v in rng_lo if dom_of.get(v) == dom]
    lo = np.array([rng_lo[v] for v in vs], dtype=float)
    hi = np.array([rng_hi[v] for v in vs], dtype=float)
    n = len(lo)
    total_pairs = n * (n - 1) // 2
    # Σ_{i<j} |x_i - x_j| / C(n,2). המטריצה סופרת i<j וגם i>j → /2
    exact_lo = np.abs(lo[:, None] - lo[None, :]).sum() / 2.0 / total_pairs
    exact_hi = np.abs(hi[:, None] - hi[None, :]).sum() / 2.0 / total_pairs
```

**מילים**:
1. בחירת אוכלוסיית הdomain (drugs או weapon)
2. בניית מטריצת המרחקים `|x_i - x_j|` לכל הזוגות ע"י broadcasting (n×n)
3. סכימה / 2 (כי המטריצה סימטרית, סופרת כל זוג פעמיים) / C(n,2)
4. זה בדיוק הממוצע על כל הזוגות — דטרמיניסטי, ~0.04 שניות

**הערה היסטורית**: הגרסה הראשונה השתמשה בדגימת Monte-Carlo של 50,000 זוגות (seed=42). אומת שהדגימה זהה ל-exact עד **±0.02 חודש (drugs) / ±0.15 (weapon)** — SE של אומדן 50K = 0.07-0.17. הוחלף ל-exact כי הוא מהיר באותה מידה, דטרמיניסטי לחלוטין, ומסיר את השאלה "למה רק 50K?". ההבדל היחיד בעיגול: weapon |Δhigh| 39.8 → **39.9**.

**אומר מה?** ככה נראה המרחק הממוצע בין שני verdicts **שאין שום סיבה להעריך שהם דומים**. זה ה-floor של "ניבוי עיוור".

### דוגמה לחישוב יד אחד
שתי verdict ב-drugs:
- A: (low=6, high=24)
- B: (low=18, high=36)
- d_lo = |6 - 18| = 12 חודש
- d_hi = |24 - 36| = 12 חודש

ממוצע על **כל 2,987,790 הזוגות** = 12.89 / 21.11 ב-drugs.

---

## טבלה 2️⃣ — LLM Bucket → Sentencing Gap

### המספרים שדווחו (DRUGS)

| LLM bucket | n pairs | \|Δlow\| mean | \|Δhigh\| mean |
|---|---|---|---|
| 0-24 | 37,762 | 11.5 | 18.5 |
| 25-49 | 93,108 | 9.1 | 14.9 |
| 50-74 | 51,548 | 7.3 | 11.7 |
| 75-89 | 17,218 | 6.9 | 10.3 |
| 90-100 | 2,154 | 5.2 | 7.8 |

(WEAPON זהה בשיטה, ראה גרף)

### איך זה חושב

#### שלב 1: לטעון 375K זוגות עם ציון LLM

**מקורות** (`thesis_story_part1.py` שורות 46-58):
```python
for path in [
    "similarity_scores_combined.csv",                            # 140,961
    "similarity_batch_5fold/results/similarity_scores_5fold.csv", #  63,836
    "similarity_batch_simcse/results/similarity_scores_simcse.csv",      # 48,832
    "similarity_batch_supervised/results/similarity_scores_supervised.csv", # 14,065
    "similarity_batch_5fold_v2/results/similarity_scores_5fold_v2.csv",   # 61,766
    "similarity_batch_filtered/results/similarity_scores_filtered.csv",   # 46,198
]:
    df = pd.read_csv(path)
    for r in df.itertuples():
        if pd.notna(r.similarity_score):
            llm_pairs.append((r.verdict_1, r.verdict_2, r.similarity_score, r.domain))
```

**Schema של כל CSV**: `verdict_1, verdict_2, domain, similarity_score` (ערך 0-100).

**איחוד**: 375,658 unique pairs.

#### שלב 2: dedupe לפי `(verdict_1, verdict_2)` כסט ממוין

**קוד** (שורות 63-73):
```python
seen = set()
for v1, v2, score, dom in llm_pairs:
    if v1 not in rng_lo or v2 not in rng_lo: continue
    a, b = sorted([v1, v2])
    if (a, b) in seen: continue
    seen.add((a, b))
    rows.append({
        "v1": a, "v2": b, "domain": dom, "llm_score": score,
        "d_lo": abs(rng_lo[a] - rng_lo[b]),
        "d_hi": abs(rng_hi[a] - rng_hi[b]),
    })
df_llm = pd.DataFrame(rows)
```

חשוב: זוג מופיע פעם אחת בלבד גם אם הוא ב-2 batches. סורט ע"י `tuple(sorted([v1,v2]))`.

**תוצאה**: **367,930 unique pairs** (חלק מ-375K לא עברו כי אחד מהverdicts לא היה ב-`rng_lo`).

#### שלב 3: חלוקה ל-buckets לפי llm_score

**Buckets** (שורה 80):
```python
buckets = [(0, 25), (25, 50), (50, 75), (75, 90), (90, 101)]
```

מימינים בקצוות: כל bucket `[lo, hi)`. **bucket 90-100 כולל ציון 100** כי hi=101.

#### שלב 4: ממוצע בכל bucket

**קוד** (שורות 86-91):
```python
for lo_b, hi_b in buckets:
    sub = sub_dom[(sub_dom.llm_score >= lo_b) & (sub_dom.llm_score < hi_b)]
    print(f"  {lo_b}-{hi_b-1}: n={len(sub)}  "
          f"|Δlow| mean = {sub.d_lo.mean():.2f}  "
          f"|Δhi|  mean = {sub.d_hi.mean():.2f}")
```

**אומר מה?** ככל ש-LLM נתן ציון דמיון גבוה יותר → הפער בעונש קטן יותר באופן מונוטוני. **זה ה-signal**.

### דוגמה
זוג של 2 verdicts ב-drugs שקיבל ציון LLM=92 (בbucket 90-100):
- A: סחר בסם, range = (24, 60)
- B: סחר בסם דומה, range = (20, 56)
- |Δlow| = 4 חודש, |Δhigh| = 4 חודש

מצופה: ממוצע ל-2,154 הזוגות האלה ~= 5.2 / 7.8 חודשים.

---

## טבלה 3️⃣ — Citation Type → Sentencing Gap

### המספרים שדווחו (DRUGS)

| Citation type | n pairs | \|Δlow\| | \|Δhigh\| |
|---|---|---|---|
| 1hop | 740 | 6.7 | 10.5 |
| 2hop | 2,113 | 9.0 | 14.5 |
| cocite | 47,031 | 13.0 | 21.5 |
| none | 606 | 14.1 | 22.2 |

### איך זה חושב

**המקור**: [`data_per_domain/network_analysis/citation_pair_types.csv`](../../data_per_domain/network_analysis/citation_pair_types.csv)

**Schema**: `verdict_1, verdict_2, domain, similarity_score, citation_type`.

**Citation types**:
- **`1hop`** = verdict_1 ציטט את verdict_2 (או הפוך) ישירות
- **`2hop`** = יש verdict שלישי שמצטט את שניהם, או שניהם מצטטים אותו
- **`cocite`** = יחד מצוטטים בפסק דין רביעי (קו-ציטוט)
- **`none`** = אין קשר ציטוט

**הקוד** (`thesis_story_part1.py` שורות 96-110):
```python
cit_df = pd.read_csv("citation_pair_types.csv")
seen = set()
for r in cit_df.itertuples():
    v1, v2 = r.verdict_1, r.verdict_2
    if v1 not in rng_lo or v2 not in rng_lo: continue
    a, b = sorted([v1, v2])
    if (a, b) in seen: continue
    seen.add((a, b))
    rows.append({
        "v1": a, "v2": b, "domain": r.domain, "cit_type": r.citation_type,
        "d_lo": abs(rng_lo[a] - rng_lo[b]),
        "d_hi": abs(rng_hi[a] - rng_hi[b]),
    })
df_cit = pd.DataFrame(rows)  # 128,575 rows
```

לאחר מכן:
```python
for ct in ("1hop", "2hop", "cocite", "none"):
    sub = sub_dom[sub_dom.cit_type == ct]
    print(f"  {ct}: n={len(sub)}  "
          f"|Δlow| mean = {sub.d_lo.mean():.2f}  "
          f"|Δhi|  mean = {sub.d_hi.mean():.2f}")
```

**אומר מה?** 1hop קרוב פי 2 מ-random (drugs: 6.7 לעומת 12.9), 2hop קרוב פי 1.4, ו-cocite דומה ל-random (cocite=13 ≈ random=12.9).

**מסקנה קריטית**: cocite **כמעט לא נושא signal** — שני verdicts שצוטטו ביחד לא בהכרח דומים בעונש. רק 1hop ו-2hop נושאים signal משמעותי.

---

## בדיקת השוואה מהירה

| Source | Drugs \|Δlow\| | Drugs \|Δhigh\| | Weapon \|Δlow\| | Weapon \|Δhigh\| |
|---|---|---|---|---|
| Random (EXACT, כל הזוגות) | 12.9 | 21.1 | 26.0 | 39.9 |
| LLM 0-24 | 11.5 | 18.5 | 25.0 | 36.9 |
| LLM 25-49 | 9.1 | 14.9 | 17.7 | 25.8 |
| LLM 50-74 | 7.3 | 11.7 | 15.7 | 22.7 |
| LLM 75-89 | 6.9 | 10.3 | 16.2 | 23.1 |
| LLM 90-100 | **5.2** | **7.8** | **7.3** | **10.8** |
| Citation 1hop | 6.7 | 10.5 | 11.8 | 18.3 |
| Citation 2hop | 9.0 | 14.5 | 15.8 | 20.9 |
| Citation cocite | 13.0 | 21.5 | 20.7 | 30.0 |
| Citation none | 14.1 | 22.2 | 30.6 | 42.5 |

**3 ה-takeaways**:
1. ה-LLM יודע לזהות similarity שתואמת לסmilarity בעונש (מונוטוני)
2. Citation 1hop משווה ל-LLM 90+ באיכות (drugs), אבל יש רק 740 כאלה (לעומת 2,154 ב-LLM 90+)
3. Cocite ≈ random → רעש, לא signal

---

## להריץ מחדש

```bash
cd .../full_analysis_2026_05_13/scripts/
python3 thesis_story_part1.py

# Outputs:
#   /tmp/story_llm_gaps.csv
#   /tmp/story_citation_gaps.csv

python3 plot_story_part1.py

# Output:
#   /tmp/thesis_plots/plot_story_part1_variance.png
```

(או לעדכן הpath בסקריפט.)
