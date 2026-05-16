# מתודולוגיה — Baselines (M2 offense-matched, M3 TF-IDF, M4 BM25)

> ⚠️ נתונים עדכניים = **4,432** (2026-05-16). הלוגיקה זהה; ערכים עדכניים ב-`data/rigor_mae_with_ci.csv` ו-[`DATA_PROVENANCE_4432.md`](DATA_PROVENANCE_4432.md). 4,432: offense-matched drugs 8.77 / weapon 18.41 · TF-IDF 7.57 / 16.47 · BM25 6.75 / 15.23.

תיעוד line-by-line של 3 ה-baselines. **קוד מקור**: `scripts/rigor_phase_a.py`.
לכל baseline: מה הדרישה · איך נבנה · מה נבדק · איך נבחרו המועמדים · מה נתן.

הגדרות משותפות (כמו בכל הניתוח):
- 5-fold CV, כל verdict ב-test בדיוק פעם אחת. K=10.
- חיזוי = `median(low של K הנבחרים)`, `median(high)`.
- MAE עם bootstrap 95% CI (B=2,000), Wilcoxon paired.

---

## M2 · Offense-matched random ⭐ (ה-baseline שהמועצה דרשה)

### מה הדרישה (why this baseline exists)
ה-First Principles במועצה טען: *"הטענה 'דמיון עוזר' היא unfalsifiable בלי baseline שמבוסס-חוקה. אם 'שלוף K אקראיים מאותו סעיף עבירה' מגיע לאותו MAE — אז כל המודל הוא רק `GROUP BY offense_type`."* → M2 בודק בדיוק את זה: **האם שיתוף סעיף עבירה לבדו מספיק?**

### שלב 1: בניית offense-set לכל verdict

**מקור**: `data/sentencing_range-old/hfull_features/hybrid_full_cache.json` (H-Full, gpt-4.1 structured extraction).

**עזר** `yesno(v)` — ערך נחשב "כן" אם הוא **לא** ב-`{"", "לא", "nan", "None", "0", "0.0"}`:
```python
def yesno(v):
    s = str(v).strip()
    return s not in ("", "לא", "nan", "None", "0", "0.0")
```

**Drugs** (`drugs_offense_set`) — דגלי סעיפים מ-H-Full:
```python
def drugs_offense_set(feats):
    s = set()
    for sec in ("6","7","13","14","19"):
        if yesno(feats.get(f"section_{sec}")): s.add(f"sec_{sec}")
    if yesno(feats.get("other_drug_offense")): s.add("other")
    return s
```
הסעיפים: **6**=יבוא/יצוא · **7**=החזקה/שימוש · **13**=סחר/ייצור · **14**=מחמירה · **19**=נסיבה מחמירה · **other**=עבירת סם אחרת.

**Weapon** (`weapon_offense_set`) — regex על טקסט (offense_number+offense_type+additional_offenses):
```python
WPAT = [(r"144\s*\(\s*א\s*\)","144a"), (r"144\s*\(\s*ב\s*2\s*\)","144b2"),
        (r"144\s*\(\s*ב\s*\)","144b"), (r"144\s*\(\s*ג\s*\)","144c"),
        (r"144\s*\(\s*ז\s*\)","144g"), (r"\b145\b","145"), (r"\b146\b","146")]
def weapon_offense_set(feats):
    blob = " ".join(str(feats.get(k,"")) for k in
                    ("offense_number","offense_type","additional_offenses"))
    return {label for pat,label in WPAT if re.search(pat, blob)}
```
תת-סעיפי §144 לחוק העונשין: **144(א)**=החזקה · **144(ב)**=נשיאה/הובלה · **144(ב2)**=סחר · **144(ג)** · **144(ז)**=תחמושת/אבזר · **145**=ייצור · **146**=יבוא/יצוא.

הופך ל-dict `verdict_offenses[v] = {set of labels}` לכל 3,898 verdicts.

### שלב 2: הבחירה — איך נבחרו המועמדים

```python
q_off = verdict_offenses.get(q, set())          # offense-set של ה-query
if q_off:                                        # ← דרישה: query חייב offense-set לא-ריק
    off_match_cands = [t for t in train_ids
                       if t != q and (verdict_offenses.get(t,set()) & q_off)]
    #                                            ↑ חיתוך-קבוצות ≥1 (חולקים סעיף כלשהו)
    if len(off_match_cands) >= K:
        rng2 = np.random.default_rng((hash(q)+1) % 2**32)   # seed פר-query → דטרמיניסטי
        sampled_idx = rng2.permutation(len(off_match_cands))[:K]
        picked = [off_match_cands[i] for i in sampled_idx]  # K אקראיים מהמתאימים
    else:
        picked = off_match_cands                  # פחות מ-K → קח את כולם
    # → predict = median(low/high של picked)
```

### הדרישות (requirements) במדויק
1. **ל-query חייב להיות offense-set לא ריק** (`if q_off`). אם ריק → אין ניבוי → לא נכנס ל-MAE.
2. **מועמד = חולק ≥1 סעיף** עם ה-query (חיתוך-קבוצות לא ריק). **לא** דורש זהות מלאה — מספיק סעיף משותף אחד.
3. **K=10 אקראיים** מהמתאימים, **seeded ע"י `hash(q)+1`** → דטרמיניסטי (אותם 10 בכל run).
4. אם <K מתאימים → לוקחים את כולם (n_actual < K).

### מה נבדק / כמה
- **n משתתפים**: drugs **2,121** (לא 2,305), weapon **1,272** (לא 1,593). ההפרש = verdicts עם offense-set ריק שנפלו (drugs ~184, weapon ~321 — פערי H-Full extraction או תיקי weapon שלא תחת §144).
- **כמה**: drugs MAE-lo **8.53 [8.09, 8.96]** · weapon **17.65 [16.02, 19.30]**.

### מה נתן (המסקנה)
❌ **גרוע מ-global median!** (drugs 8.53 > 8.43; weapon 17.65 > 16.67).
- **המשמעות**: שיתוף סעיף עבירה לבדו **לא נושא signal**. שני תיקי "§7 החזקה" יכולים להיות 3 חודשים מול 5 שנים — תלוי בכמות/תפקיד/נסיבות, לא בסעיף.
- **חשיבות לתזה**: זה ה-baseline הקריטי שעונה למועצה. הוא **מפריך** את ההשערה ש"המודל הוא רק GROUP BY offense". sup+LLM מנצח אותו ב-Δ=-3.91 drugs / -6.01 weapon (p<1e-48) — כלומר המודל לומד **הרבה מעבר** לשיתוף סעיף.

### Overlap-threshold sweep — "אולי צריך התאמה הדוקה יותר?"

שאלה מקדימה ל-reviewer: M2 דורש חיתוך **≥1**. מה אם נדרוש ≥2, ≥3, או set **זהה לחלוטין**?
*(קוד: `scripts/offense_matched_overlap_sweep.py` · נתונים: `data/offense_matched_overlap_sweep.csv` · bootstrap 95% CI, B=2,000 · **min_k=1, עקבי עם M1-M9** · seed דטרמיניסטי ע"י `hashlib.md5` — מתוקן מ-`hash()` של Python שמלוח פר-תהליך)*

| Overlap | Drugs cov | Drugs MAE-lo [CI] | Weapon cov | Weapon MAE-lo [CI] |
|---|---|---|---|---|
| **≥1** (=M2) | 100% | 8.63 [8.19, 9.08] | 100% | 17.88 [16.26, 19.60] |
| **≥2** | 63% | 8.96 [8.40, 9.52] | 40% | 19.55 [17.09, 22.47] |
| **≥3** | 33% | 8.34 [7.67, 9.08] | 11% | 18.85 [15.60, 22.55] |
| **exact** (set זהה) | 99% | 8.52 [8.09, 8.95] | 100% | 17.06 [15.50, 18.76] |

*(reference: median drugs 8.43 / weapon 16.67 · sup+LLM 6.11 / 12.50)*

**מה נתן**: בשום רמת חומרה offense-matching לא עובד.
- **≥2 מחמיר** (drugs 8.96, weapon 19.55) — פילטר הדוק בוחר תיקים עם פרופיל-עבירות מורכב יותר = תת-קבוצה רועשת. ומאבד 37-60% coverage.
- **≥3 / exact** חוזרים ל-~median (drugs 8.3-8.5 / weapon 17-19), **אף פעם לא מתקרבים ל-sup+LLM** (6.11/12.50). exact הכי טוב מבין ה-variants — ועדיין שווה לניבוי עיוור.
- **הפרשנות**: שני תיקים עם קבוצת-סעיפים **זהה לחלוטין** רחוקים בעונש כמו זוג אקראי. הסעיף לא קובע את העונש, **בשום רמת דיוק** — הכמות/תפקיד/נסיבות כן.
- **לתזה**: מקדים תשובה ל-"אולי offense-matching הדוק יותר?" — בדקנו ≥1/≥2/≥3/exact, אף אחד לא עובד. רק מודל נלמד תופס את ה-signal.

**הערות מתודולוגיות**:
- *בחירת min_k*: השתמשנו **min_k=1** (query עם ≥1 מועמד מקבל ניבוי) — **עקבי עם כל M1-M9**. בסף ≥1 (ה-headline) זה זהה ל-min_k=10 (0% מה-queries עם <10 מועמדים). נבדק גם **min_k=10** (כל ניבוי = median מלא של 10): המסקנה **זהה** — offense-matching נכשל בכל סף. אומת ש-99% drugs / 91% weapon יש להם ≥10 מועמדים ממילא גם ב-≥3, כך שה-"median רועש" אינו ההסבר.
- *Coverage*: יורד ב-≥2/≥3 כי ל-query רבים **אין אף תיק ב-train שחולק 2-3 סעיפים** (פרופיל-עבירה נדיר), לא בגלל min_k. ה-67% שנשמטו ב-≥3 drugs = פרופילים נדירים, הוצאו לגמרי מה-MAE.
- *Determinism*: התיקון מ-`hash(q)` ל-`hashlib.md5` הפך את הסקריפט reproducible לחלוטין (2 ריצות → מספרים זהים). לפני התיקון היו תנודות ±0.05-0.5 בין ריצות (PYTHONHASHSEED), אך המסקנה הייתה invariant.

---

## M3 · TF-IDF + Ridge regression

### מה הדרישה
baseline ML סטנדרטי שכל reviewer ידרוש: *"אולי לא צריך retrieval/דמיון בכלל — רק רגרסיה ישירה מהטקסט?"*

### איך נבנה (fit לכל fold בנפרד — אין leakage)
```python
valid_train = [v for v in train_ids if v in v_to_text and v in rng_lo]
train_texts = [indictment_facts[v] for v in valid_train]
tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5),
                        min_df=2, max_features=50_000)
X_train = tfidf.fit_transform(train_texts)        # fit על train-fold בלבד
ridge_lo = Ridge(alpha=10.0).fit(X_train, train_lows)   # רגרסיה ל-low
ridge_hi = Ridge(alpha=10.0).fit(X_train, train_highs)  # רגרסיה ל-high
# test:
X_q = tfidf.transform([q_text])
pred_lo, pred_hi = ridge_lo.predict(X_q)[0], ridge_hi.predict(X_q)[0]
```

### הדרישות / איך נבחר
- **char_wb n-grams (3-5)** — תווים, לא מילים. בחירה מודעת: עברית מורפולוגית עשירה, char n-grams עמידים לנטיות.
- **min_df=2, max_features=50,000** — סינון רעש.
- **Ridge α=10** — רגולריזציה בינונית (לא מכווננת מעבר לזה — baseline, לא מודל מטופח).
- **אין K, אין retrieval** — חיזוי ישיר. לכן ה-row ב-`rigor_raw_per_query_K.csv` הוא `K=None`.

### מה נתן
drugs **7.56 [7.20, 7.94]** · weapon **15.58 [14.50, 16.66]**. עדיף על median, אבל **מפסיד מובהקית לכל שיטות הדמיון** (p<1e-40). ניתוח quartile חשף: TF-IDF overshoots Q1 (MAE=11) אבל מצוין ב-Q3 (3.9) → regressor מטעה, לא פתרון.

---

## M4 · BM25 retrieval

### מה הדרישה
baseline retrieval לקסיקלי קלאסי: *"אולי lexical overlap מספיק, בלי deep learning?"*

### איך נבנה
```python
from rank_bm25 import BM25Okapi
train_tokens = [t.split() for t in train_texts]   # tokenize ב-whitespace
bm25 = BM25Okapi(train_tokens)
# test:
q_tokens = q_text.split()
scores = bm25.get_scores(q_tokens)
top = np.argsort(-scores)[:K]                     # top-K לפי BM25
picked = [valid_train[i] for i in top]
# → median(low/high של picked)
```

### הדרישות / איך נבחר
- **whitespace tokenization** — נאיבי לעברית (אין lemmatization). **limitation מודע** — לכן BM25 כנראה underperform-יחסי לפוטנציאל.
- **BM25Okapi** — הגרסה הסטנדרטית של הספרייה `rank_bm25`, פרמטרי ברירת מחדל (k1=1.5, b=0.75).
- top-K=10 לפי score → median.

### מה נתן
🟡 **הפתעה** — drugs **6.82 [6.45, 7.24]**, weapon **14.54 [13.49, 15.61]**. חזק מאוד (קרוב ל-supervised, מנצח TF-IDF). אבל sup+LLM עדיין מנצח מובהקית (drugs Δ=-0.92, p<1e-13). מסר: lexical retrieval בעברית עובד מפתיע-של-טוב — אבל המודל+LLM עדיף.

---

## טבלת סיכום — 3 ה-baselines מול median ו-sup+LLM

| Method | Drugs MAE-lo [CI] | מנצח median? | sup+LLM Δ מולו | מה נתן |
|---|---|---|---|---|
| Global median | 8.43 [7.99, 8.89] | — | — | floor |
| **Offense-matched random** | **8.53 [8.09, 8.96]** | ❌ **לא** | -3.91 (p<1e-84) | rule-based **נכשל** — שיתוף סעיף ≠ signal |
| TF-IDF + Ridge | 7.56 [7.20, 7.94] | ✓ | -2.01 (p<1e-51) | regressor נחות+מטעה |
| BM25 | 6.82 [6.45, 7.24] | ✓ | -0.92 (p<1e-13) | חזק מפתיע, עדיין מפסיד |
| ★ Supervised + LLM | 6.11 [5.77, 6.48] | ✓ | — | הפתרון המעשי |

**ה-takeaway המרכזי**: offense-matched הוא ה-baseline שמפריך את "המודל הוא רק GROUP BY offense" — והוא **נכשל גרוע מ-median**, מה שמוכיח שהמודל לומד signal אמיתי הרבה מעבר לסיווג סעיף.

---

עודכן: 2026-05-16
