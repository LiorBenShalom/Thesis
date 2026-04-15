"""
Structured LLM Comparison Experiment
======================================
Pre-process the two feature vectors into a structured comparison
(shared features, unique-to-each, value differences), then ask
GPT-4.1 to evaluate each legal dimension and decide.

Combines the structured approach (systematic, interpretable) with
LLM semantic understanding (handles synonyms, partial matches,
legal reasoning about significance).
"""

import os
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception: pass
import json
import re
import time
import argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    os.environ.get("OPENAI_API_KEY", ""),
)
GPT4_MODEL = "gpt-4.1"

BASE_DIR = Path(__file__).resolve().parent.parent

CONCEPT_CSV = {
    "drugs":  BASE_DIR / "code" / "post_process_output" / "similarity_database_hybrid_concepts_drugs.csv",
    "weapon": BASE_DIR / "code" / "post_process_output" / "similarity_database_hybrid_concepts_weapon.csv",
}

RAW_CSV = {
    "drugs":  BASE_DIR / "drugs" / "similarity_database_hybrid_full_gpt.csv",
    "weapon": BASE_DIR / "weapon" / "similarity_database_hybrid_full_gpt.csv",
}

# ------------------------------------------------------------------ #
#  Feature vector parsing and structured comparison
# ------------------------------------------------------------------ #

def flatten_json(obj, prefix=""):
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix} → {k}" if prefix else k
            if isinstance(v, dict):
                items.update(flatten_json(v, new_key))
            else:
                items[new_key] = v
    return items


def format_value(v) -> str:
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return str(v)


def build_structured_comparison(fv1_text: str, fv2_text: str) -> str:
    """Pre-compare two feature vectors and return a structured text comparison."""
    try:
        raw1 = json.loads(fv1_text)
        raw2 = json.loads(fv2_text)
    except (json.JSONDecodeError, TypeError):
        return f"תיק 1:\n{fv1_text}\n\nתיק 2:\n{fv2_text}"

    flat1 = flatten_json(raw1)
    flat2 = flatten_json(raw2)

    keys1 = set(flat1.keys())
    keys2 = set(flat2.keys())
    shared = sorted(keys1 & keys2)
    only1 = sorted(keys1 - keys2)
    only2 = sorted(keys2 - keys1)

    lines = []

    # Shared features
    if shared:
        lines.append("## מאפיינים משותפים (קיימים בשני התיקים):")
        for k in shared:
            v1, v2 = format_value(flat1[k]), format_value(flat2[k])
            match_mark = "✓" if v1 == v2 else "≠"
            lines.append(f"  [{match_mark}] {k}:")
            lines.append(f"      תיק 1: {v1}")
            lines.append(f"      תיק 2: {v2}")

    # Only in case 1
    if only1:
        lines.append(f"\n## מאפיינים ייחודיים לתיק 1 ({len(only1)} מאפיינים):")
        for k in only1:
            lines.append(f"  • {k}: {format_value(flat1[k])}")

    # Only in case 2
    if only2:
        lines.append(f"\n## מאפיינים ייחודיים לתיק 2 ({len(only2)} מאפיינים):")
        for k in only2:
            lines.append(f"  • {k}: {format_value(flat2[k])}")

    # Meta stats
    lines.append(f"\n## סיכום מבני:")
    lines.append(f"  • מאפיינים משותפים: {len(shared)}")
    lines.append(f"  • מתוכם זהים: {sum(1 for k in shared if format_value(flat1[k]) == format_value(flat2[k]))}")
    lines.append(f"  • מתוכם שונים: {sum(1 for k in shared if format_value(flat1[k]) != format_value(flat2[k]))}")
    lines.append(f"  • ייחודיים לתיק 1: {len(only1)}")
    lines.append(f"  • ייחודיים לתיק 2: {len(only2)}")

    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Prompts — multiple versions
# ------------------------------------------------------------------ #

# V1: The original 5-dimension prompt (already ran, kept for reference)
SYSTEM_PROMPT_V1 = """את/ה מומחית לדין הפלילי בישראל. מוצגת בפנייך **השוואה מובנית** בין שני תיקים פליליים.
ההשוואה כבר פורקה עבורך: מאפיינים משותפים (עם סימון [✓] אם זהים, [≠] אם שונים), ומאפיינים ייחודיים לכל תיק.

המשימה: הערכת דמיון מהותי — האם תיק אחד יכול לשמש כתקדים ענייני לשני.

נתחי כל ממד בנפרד:
1. **סוג העבירה וחומרתה** — האם מדובר באותו סוג עבירה? באותה רמת חומרה?
2. **תפקיד הנאשם ומעורבותו** — יזם/סוחר לעומת שליח/מחזיק? מעורבות פעילה לעומת פסיבית?
3. **שיטת הביצוע (MO)** — דפוס דומה? אמצעים דומים?
4. **נסיבות הליבה** — כמויות, שימוש בנשק, תכנון, נסיבות מחמירות/מקלות?
5. **ישימות כתקדים** — האם בית משפט יראה את שני התיקים כרלוונטיים זה לזה לצורכי ענישה?

חשוב:
- העובדה שלשני התיקים יש מאפיינים ייחודיים שונים **לא בהכרח** אומרת שהם לא דומים — כל תיק מתועד בצורה שונה.
- התמקדי בפרמטרים המשפטיים המהותיים, לא בפרטים טכניים/ביורוקרטיים.
- השתייכות לאותו תחום (סמים/נשק) לבדה **לא מספיקה** לדמיון מהותי.

פורמט תשובה:
1. ניתוח קצר (3-5 משפטים) של כל ממד
2. שורה אחרונה: FINAL_VERDICT: דומים / לא דומים"""


# V2: Regular prompt (identical to similarity_experiment.py unified binary)
# with structured input — isolates the effect of input format
SYSTEM_PROMPT_REGULAR = """את/ה מסייע/ת משפטית מומחית בתחום הדין הפלילי הישראלי.

מוצגים בפניך שני תיאורי פיצ'רים מובנים על תיק, בפורמט של השוואה מובנית (מאפיינים משותפים עם סימון [✓] אם זהים / [≠] אם שונים, ומאפיינים ייחודיים לכל תיק).
עליך לקבוע את רמת הדמיון המהותי ביניהם, לפי הקריטריונים הבאים:

0 = לא דומים:
    • סוג העבירות או דפוס ההתנהגות שונים
    • נסיבות מרכזיות שונות (למשל: שימוש עצמי מול סחר מתמשך, סוג סם מאוד שונה, תפקידים שונים לגמרי)
    • לא היית מצפה שבית משפט ישתמש באחד כתקדים ענייני לשני

1 = דומים (קיים דמיון מהותי):
    • יש דמיון מהותי בסוג העבירה או בדפוס ההתנהגות
    • נסיבות דומות או דמיון במבנה הסיפור העובדתי (איך בוצעה העבירה, מול מי, באיזה הקשר)
    • היית מצפה שבית משפט ישתמש באחד כתקדים ענייני לשני

חשוב מאוד:
• העובדה שלשני התיקים יש מאפיינים ייחודיים שונים לא בהכרח אומרת שהם לא דומים — כל תיק מתועד בצורה שונה.
• התייחס רק לעובדות / הפיצ'רים המתוארים לפניך.
• ענה אך ורק עם ספרה אחת: 0 או 1."""


# V3: General "think like a lawyer" prompt — domain-agnostic
SYSTEM_PROMPT_GENERAL = """את/ה משפטנ/ית מומחה לדין הפלילי. מוצגת בפנייך **השוואה מובנית** בין שני תיקים פליליים.
ההשוואה כבר פורקה עבורך: מאפיינים משותפים (עם סימון [✓] אם זהים, [≠] אם שונים), ומאפיינים ייחודיים לכל תיק.

המשימה: הערכת דמיון מהותי — האם תיק אחד יכול לשמש כתקדים ענייני לשני לצורכי ענישה.

## איך לנתח
לא כל המאפיינים שווים. השלב הראשון שלך הוא **לזהות מה מהותי ומה שולי** בתיקים שלפנייך.

דוגמאות למאפיינים שבדרך כלל **מהותיים**: סוג העבירה הספציפי, סוג וכמות החומר (סם/נשק), תפקיד הנאשם (יזם/בעלים לעומת שליח/מחזיק), דפוס ההתנהגות (סחר חד-פעמי לעומת רשת מתמשכת), נסיבות מחמירות או מקלות מרכזיות.

דוגמאות למאפיינים שבדרך כלל **שוליים**: מיקום ספציפי (כתובת, עיר), תאריך מדויק, דגם רכב, שם שוטר/יחידה, אמצעי תקשורת (וואטסאפ/טלפון). למשל: מיקום המעבדה לא חשוב — אבל עצם **קיום** מעבדה כן חשוב מאוד.

## כללי הכרעה
- העובדה ששני תיקים שייכים לאותו תחום כללי (סמים/נשק) **לא מספיקה** לדמיון מהותי.
- העובדה שלתיקים יש מאפיינים ייחודיים שונים **לא בהכרח** מעידה על חוסר דמיון — כל תיק מתועד בצורה שונה.
- זהה מהם 3-5 המאפיינים הקריטיים ביותר **לתיקים הספציפיים שלפנייך**, ובדוק אם הם דומים.

פורמט תשובה:
1. זהה את המאפיינים הקריטיים (2-3 משפטים)
2. נתח את הדמיון/שוני במאפיינים הקריטיים (3-5 משפטים)
3. שורה אחרונה: FINAL_VERDICT: דומים / לא דומים"""



# V4: 5-dimensions + asymmetric cost (recall-oriented)
SYSTEM_PROMPT_V4_LENIENT = """את/ה מומחית לדין הפלילי בישראל. מוצגת בפנייך **השוואה מובנית** בין שני תיקים פליליים.
ההשוואה כבר פורקה עבורך: מאפיינים משותפים (עם סימון [✓] אם זהים, [≠] אם שונים), ומאפיינים ייחודיים לכל תיק.

המשימה: הערכת דמיון מהותי — האם תיק אחד יכול לשמש כתקדים ענייני לשני.

נתחי כל ממד בנפרד:
1. **סוג העבירה וחומרתה** — האם מדובר באותו סוג עבירה? באותה רמת חומרה?
2. **תפקיד הנאשם ומעורבותו** — יזם/סוחר לעומת שליח/מחזיק? מעורבות פעילה לעומת פסיבית?
3. **שיטת הביצוע (MO)** — דפוס דומה? אמצעים דומים?
4. **נסיבות הליבה** — כמויות, שימוש בנשק, תכנון, נסיבות מחמירות/מקלות?
5. **ישימות כתקדים** — האם בית משפט יראה את שני התיקים כרלוונטיים זה לזה לצורכי ענישה?

חשוב:
- העובדה שלשני התיקים יש מאפיינים ייחודיים שונים **לא בהכרח** אומרת שהם לא דומים — כל תיק מתועד בצורה שונה.
- התמקדי בפרמטרים המשפטיים המהותיים, לא בפרטים טכניים/ביורוקרטיים.
- השתייכות לאותו תחום (סמים/נשק) לבדה **לא מספיקה** לדמיון מהותי.
- **כלל הכרעה**: אם לפחות 3 מתוך 5 הממדים מצביעים על דמיון — סווגי כ"דומים". אם יש ספק — העדיפי "דומים", כי עדיף להציע תקדים שמעט פחות רלוונטי מאשר לפספס תקדים רלוונטי.

פורמט תשובה:
1. ניתוח קצר (3-5 משפטים) של כל ממד
2. שורה אחרונה: FINAL_VERDICT: דומים / לא דומים"""


# V5: 5-dimensions + continuous score (0-100) for threshold optimization
SYSTEM_PROMPT_V5_SCORE = """את/ה מומחית לדין הפלילי בישראל. מוצגת בפנייך **השוואה מובנית** בין שני תיקים פליליים.
ההשוואה כבר פורקה עבורך: מאפיינים משותפים (עם סימון [✓] אם זהים, [≠] אם שונים), ומאפיינים ייחודיים לכל תיק.

המשימה: הערכת דמיון מהותי — עד כמה תיק אחד יכול לשמש כתקדים ענייני לשני?

נתחי כל ממד בנפרד:
1. **סוג העבירה וחומרתה** — האם מדובר באותו סוג עבירה? באותה רמת חומרה?
2. **תפקיד הנאשם ומעורבותו** — יזם/סוחר לעומת שליח/מחזיק? מעורבות פעילה לעומת פסיבית?
3. **שיטת הביצוע (MO)** — דפוס דומה? אמצעים דומים?
4. **נסיבות הליבה** — כמויות, שימוש בנשק, תכנון, נסיבות מחמירות/מקלות?
5. **ישימות כתקדים** — האם בית משפט יראה את שני התיקים כרלוונטיים זה לזה לצורכי ענישה?

חשוב:
- העובדה שלשני התיקים יש מאפיינים ייחודיים שונים **לא בהכרח** אומרת שהם לא דומים — כל תיק מתועד בצורה שונה.
- התמקדי בפרמטרים המשפטיים המהותיים, לא בפרטים טכניים/ביורוקרטיים.
- השתייכות לאותו תחום (סמים/נשק) לבדה **לא מספיקה** לדמיון מהותי.

פורמט תשובה:
1. ניתוח קצר (2-3 משפטים) של כל ממד
2. שורה אחרונה בדיוק בפורמט: SIMILARITY_SCORE: X
   כאשר X הוא מספר שלם בין 0 ל-100.
   0 = שונים לחלוטין, 100 = זהים כמעט. ציון מעל 50 = תקדים רלוונטי."""


USER_TEMPLATE = """להלן השוואה מובנית בין שני תיקים פליליים:

{comparison}

האם התיקים דומים מהותית?"""

USER_TEMPLATE_SCORE = """להלן השוואה מובנית בין שני תיקים פליליים:

{comparison}

מהו ציון הדמיון המהותי (0-100)?"""

USER_TEMPLATE_SCORE_RAW = """להלן שני תיקים פליליים עם פיצ'רים מובנים:

תיק 1:
{fv1}

תיק 2:
{fv2}

מהו ציון הדמיון המהותי (0-100)?"""


# V6: Same as v5 (5 dims + score) but with raw concept JSONs, no structured comparison
SYSTEM_PROMPT_V6_SCORE_RAW_wep = """את/ה מומחית לדין הפלילי בישראל. מוצגים בפנייך **פיצ'רים מובנים** של שני תיקים פליליים.

המשימה: הערכת דמיון מהותי — עד כמה תיק אחד יכול לשמש כתקדים ענייני לשני?

נתחי כל ממד בנפרד:
1. **סוג העבירה וחומרתה** — האם מדובר באותו סוג עבירה? באותה רמת חומרה?
2. **תפקיד הנאשם ומעורבותו** — יזם/סוחר לעומת שליח/מחזיק? מעורבות פעילה לעומת פסיבית?
3. **שיטת הביצוע (MO)** — דפוס דומה? אמצעים דומים?
4. **נסיבות הליבה** — כמות תחמושת (סדר גודל), שימוש בנשק, תכנון, נסיבות מחמירות/מקלות?
5. **ישימות כתקדים** — האם בית משפט יראה את שני התיקים כרלוונטיים זה לזה לצורכי ענישה?

חשוב:
- התמקדי בפרמטרים המשפטיים המהותיים, לא בפרטים טכניים/ביורוקרטיים.
- השתייכות לאותו תחום (סמים/נשק) לבדה **לא מספיקה** לדמיון מהותי.
- הבחנה בין 144(א) (החזקה בלבד) ל-144(ב) (נשיאה/הובלה) היא מהותית — תיק שהוא 144(א) לבדו כמעט אף פעם לא מהווה תקדים לתיק 144(ב).


פורמט תשובה:
1. ניתוח קצר (2-3 משפטים) של כל ממד
2. שורה אחרונה בדיוק בפורמט: SIMILARITY_SCORE: X
   כאשר X הוא מספר שלם בין 0 ל-100.
   0 = שונים לחלוטין, 100 = זהים כמעט. ציון מעל 50 = תקדים רלוונטי."""

SYSTEM_PROMPT_V6_SCORE_RAW_drugs = """את/ה מומחית לדין הפלילי בישראל. מוצגים בפנייך **פיצ'רים מובנים** של שני תיקים פליליים.

המשימה: הערכת דמיון מהותי — עד כמה תיק אחד יכול לשמש כתקדים ענייני לשני?

נתחי כל ממד בנפרד:
1. **סוג העבירה וחומרתה** — האם מדובר באותו סוג עבירה? באותה רמת חומרה?
2.  סוג הסם וכמותו — האם מדובר בסוג דומה בחומרתו? בכמויות דומות בסדר גודל?
3. **שיטת הביצוע (MO)** — דפוס דומה? אמצעים דומים?
4. **נסיבות הליבה** — מעבדה, תכנון, נסיבות מחמירות/מקלות?
5. **ישימות כתקדים** — האם בית משפט יראה את שני התיקים כרלוונטיים זה לזה לצורכי ענישה?

חשוב:
- התמקדי בפרמטרים המשפטיים המהותיים, לא בפרטים טכניים/ביורוקרטיים.
- השתייכות לאותו תחום (סמים/נשק) לבדה **לא מספיקה** לדמיון מהותי.

פורמט תשובה:
1. ניתוח קצר (2-3 משפטים) של כל ממד
2. שורה אחרונה בדיוק בפורמט: SIMILARITY_SCORE: X
   כאשר X הוא מספר שלם בין 0 ל-100.
   0 = שונים לחלוטין, 100 = זהים כמעט. ציון מעל 50 = תקדים רלוונטי."""

# Map prompt names to prompts
PROMPT_VERSIONS = {
    "v1_5dim":       SYSTEM_PROMPT_V1,
    "v2_regular":    SYSTEM_PROMPT_REGULAR,
    "v3_general":    SYSTEM_PROMPT_GENERAL,
    "v4_lenient":    SYSTEM_PROMPT_V4_LENIENT,
    "v5_score":      SYSTEM_PROMPT_V5_SCORE,
    "v6_score_raw":  SYSTEM_PROMPT_V6_SCORE_RAW_wep,
}


# ------------------------------------------------------------------ #
#  API call
# ------------------------------------------------------------------ #

def call_gpt(system_prompt: str, user_prompt: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GPT4_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 1200,
    }

    for attempt in range(4):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            wait = 3 * (2 ** attempt)
            print(f"  ⚠️  Connection error attempt {attempt+1}/4, retrying in {wait}s...")
            time.sleep(wait)
        except requests.exceptions.HTTPError as e:
            sc = e.response.status_code if e.response else None
            if sc == 429:
                wait = 30 * (2 ** attempt)
                print(f"  ⚠️  Rate limit, waiting {wait}s...")
                time.sleep(wait)
            elif sc and 500 <= sc < 600:
                wait = 5 * (2 ** attempt)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Failed after 4 attempts")


# ------------------------------------------------------------------ #
#  Parse verdict
# ------------------------------------------------------------------ #

def parse_verdict(text: str) -> int | None:
    """Extract binary verdict from LLM response."""
    text = text.strip()

    # Simple digit response (from regular prompt: just "0" or "1")
    if text in ("0", "1"):
        return int(text)

    # Look for FINAL_VERDICT
    m = re.search(r'FINAL_VERDICT\s*:\s*(.*)', text)
    if m:
        v = m.group(1).strip()
        if "לא דומים" in v or "לא" in v.split()[:3]:
            return 0
        if "דומים" in v:
            return 1
        if v.strip() in ("0", "1"):
            return int(v.strip())

    # Look for bare 0 or 1 at end
    last_line = text.strip().split('\n')[-1].strip()
    if last_line in ("0", "1"):
        return int(last_line)

    # Fallback: search last 200 chars
    tail = text[-200:]
    if "לא דומים" in tail:
        return 0
    if "דומים" in tail:
        return 1

    return None


def parse_score(text: str) -> float | None:
    """Extract continuous similarity score (0-100) from LLM response."""
    m = re.search(r'SIMILARITY_SCORE\s*:\s*(\d+)', text)
    if m:
        return float(m.group(1))
    # Fallback: last number in last line
    last_line = text.strip().split('\n')[-1]
    nums = re.findall(r'\b(\d{1,3})\b', last_line)
    if nums:
        val = float(nums[-1])
        if 0 <= val <= 100:
            return val
    return None


def parse_score_v6(text: str) -> float | None:
    """v6 multimodel: only accept an explicit SIMILARITY_SCORE line (no fallback heuristics)."""
    m = re.search(r'SIMILARITY_SCORE\s*:\s*(\d+)', text)
    if m:
        return float(m.group(1))
    return None


def find_best_threshold(scores: np.ndarray, labels: np.ndarray,
                        n_steps: int = 200) -> tuple[float, float]:
    """Find threshold that maximizes F1 score."""
    best_f1, best_thr = 0.0, 50.0
    for thr in np.linspace(0, 100, n_steps):
        preds = (scores >= thr).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1


def loo_threshold(scores: np.ndarray, labels: np.ndarray,
                  n_steps: int = 200) -> dict:
    """Leave-One-Out threshold evaluation for unbiased metrics."""
    n = len(scores)
    loo_preds = np.zeros(n, dtype=int)
    for i in range(n):
        train_scores = np.delete(scores, i)
        train_labels = np.delete(labels, i)
        thr, _ = find_best_threshold(train_scores, train_labels, n_steps)
        loo_preds[i] = int(scores[i] >= thr)

    return {
        "accuracy": round(accuracy_score(labels, loo_preds), 4),
        "f1": round(f1_score(labels, loo_preds, zero_division=0), 4),
        "precision": round(precision_score(labels, loo_preds, zero_division=0), 4),
        "recall": round(recall_score(labels, loo_preds, zero_division=0), 4),
        "cm": confusion_matrix(labels, loo_preds, labels=[0, 1]).tolist(),
    }


# ------------------------------------------------------------------ #
#  Main experiment
# ------------------------------------------------------------------ #

def run_experiment(domain: str, use_concepts: bool = True, task: str = "binary_0",
                   prompt_version: str = "v3_general") -> dict:
    csv_map = CONCEPT_CSV if use_concepts else RAW_CSV
    csv_path = csv_map[domain]
    repr_name = "concept" if use_concepts else "raw_hybrid"
    system_prompt = PROMPT_VERSIONS[prompt_version]
    is_score_mode = prompt_version in ("v5_score", "v6_score_raw")
    is_raw_input = prompt_version == "v6_score_raw"

    print(f"\n{'='*65}")
    print(f"  Structured LLM Comparison — {domain.upper()} ({repr_name})")
    print(f"  Prompt: {prompt_version}  |  Task: {task}  |  Model: {GPT4_MODEL}")
    print(f"  Mode: {'continuous score' if is_score_mode else 'binary verdict'}")
    print(f"{'='*65}")

    df = pd.read_csv(csv_path)
    labels = df[f"similarity_{task}"].values.astype(int)
    print(f"  Pairs: {len(df)}  |  Pos: {labels.sum()}  |  Neg: {(1-labels).sum()}")

    raw_outputs = []   # score (float) or binary pred (int) or None
    responses = []

    for idx, row in df.iterrows():
        if is_raw_input:
            user_prompt = USER_TEMPLATE_SCORE_RAW.format(
                fv1=row["feature_vector_1"], fv2=row["feature_vector_2"]
            )
        else:
            comparison = build_structured_comparison(
                row["feature_vector_1"], row["feature_vector_2"]
            )
            if is_score_mode:
                user_prompt = USER_TEMPLATE_SCORE.format(comparison=comparison)
            else:
                user_prompt = USER_TEMPLATE.format(comparison=comparison)

        try:
            response = call_gpt(system_prompt, user_prompt)
            if is_score_mode:
                out = parse_score(response)
            else:
                out = parse_verdict(response)
        except Exception as e:
            print(f"  ❌ Pair {idx}: {e}")
            response = ""
            out = None

        raw_outputs.append(out)
        responses.append(response)

        if (idx + 1) % 10 == 0:
            valid = sum(1 for o in raw_outputs if o is not None)
            print(f"  Processed {idx+1}/{len(df)} pairs ({valid} valid)")

    # Build valid arrays
    valid_mask = np.array([o is not None for o in raw_outputs])
    n_valid = valid_mask.sum()
    print(f"\n  Valid responses: {n_valid}/{len(df)}")

    out_dir = BASE_DIR / domain / f"results_{domain}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"structured_llm_{prompt_version}_{repr_name}_{task}"

    if is_score_mode and n_valid > 0:
        scores = np.array([o for o in raw_outputs if o is not None])
        y_true = labels[valid_mask]

        # Global best threshold
        best_thr, best_f1 = find_best_threshold(scores, y_true)
        y_pred_best = (scores >= best_thr).astype(int)
        acc_best = accuracy_score(y_true, y_pred_best)
        prec_best = precision_score(y_true, y_pred_best, zero_division=0)
        rec_best = recall_score(y_true, y_pred_best, zero_division=0)
        cm = confusion_matrix(y_true, y_pred_best, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        print(f"\n  --- Global best threshold: {best_thr:.1f} ---")
        print(f"  Accuracy: {acc_best:.3f}  |  F1: {best_f1:.3f}  |  Prec: {prec_best:.3f}  |  Rec: {rec_best:.3f}")
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

        # LOO evaluation
        print("\n  Running LOO threshold evaluation...")
        loo = loo_threshold(scores, y_true)
        loo_cm = np.array(loo["cm"])
        loo_tn, loo_fp, loo_fn, loo_tp = loo_cm.ravel()
        print(f"  LOO: F1={loo['f1']:.3f}  Acc={loo['accuracy']:.3f}  "
              f"Prec={loo['precision']:.3f}  Rec={loo['recall']:.3f}")
        print(f"  LOO: TP={loo_tp}  FP={loo_fp}  FN={loo_fn}  TN={loo_tn}")

        # Score distribution
        print(f"\n  Score stats: mean={scores.mean():.1f}  std={scores.std():.1f}  "
              f"min={scores.min():.0f}  max={scores.max():.0f}  median={np.median(scores):.0f}")
        print(f"  Pos scores (GT=1): mean={scores[y_true==1].mean():.1f}  std={scores[y_true==1].std():.1f}")
        print(f"  Neg scores (GT=0): mean={scores[y_true==0].mean():.1f}  std={scores[y_true==0].std():.1f}")

        results = {
            "domain": domain, "representation": repr_name, "task": task,
            "model": GPT4_MODEL, "method": "structured_llm_comparison",
            "prompt_version": prompt_version,
            "n_pairs": len(df), "n_valid": int(n_valid),
            "best_threshold": round(best_thr, 2),
            "global_best": {
                "f1": round(best_f1, 4), "accuracy": round(acc_best, 4),
                "precision": round(prec_best, 4), "recall": round(rec_best, 4),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            },
            "loo": loo,
            "score_stats": {
                "mean": round(scores.mean(), 2), "std": round(scores.std(), 2),
                "pos_mean": round(scores[y_true==1].mean(), 2),
                "neg_mean": round(scores[y_true==0].mean(), 2),
            },
        }

        # Save score predictions
        pred_df = df[["verdict_1", "verdict_2", f"similarity_{task}"]].copy()
        pred_df["score"] = raw_outputs
        pred_df["response"] = responses
        pred_df.to_csv(out_dir / f"{tag}_preds.csv", index=False, encoding="utf-8-sig")

    else:
        # Binary mode
        pred_arr = np.array([o if o is not None else -1 for o in raw_outputs])
        valid_mask_int = pred_arr >= 0
        n_valid = valid_mask_int.sum()

        if n_valid > 0:
            y_true = labels[valid_mask_int]
            y_pred = pred_arr[valid_mask_int]

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

            print(f"  Accuracy: {acc:.3f}  |  F1: {f1:.3f}  |  Prec: {prec:.3f}  |  Rec: {rec:.3f}")
            print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        else:
            acc = f1 = prec = rec = 0
            tp = fp = fn = tn = 0

        results = {
            "domain": domain, "representation": repr_name, "task": task,
            "model": GPT4_MODEL, "method": "structured_llm_comparison",
            "prompt_version": prompt_version,
            "n_pairs": len(df), "n_valid": int(n_valid),
            "accuracy": round(acc, 4), "f1": round(f1, 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        }

        pred_df = df[["verdict_1", "verdict_2", f"similarity_{task}"]].copy()
        pred_df["prediction"] = raw_outputs
        pred_df["response"] = responses
        pred_df.to_csv(out_dir / f"{tag}_preds.csv", index=False, encoding="utf-8-sig")

    with open(out_dir / f"{tag}_stats.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  Saved to: {out_dir / tag}_*")
    return results


def main():
    parser = argparse.ArgumentParser(description="Structured LLM comparison experiment")
    parser.add_argument("--domain", choices=["drugs", "weapon", "both"], default="both")
    parser.add_argument("--repr", choices=["concept", "raw"], default="concept")
    parser.add_argument("--task", default="binary_0")
    parser.add_argument("--prompt", choices=list(PROMPT_VERSIONS.keys()), default="v3_general",
                        help="Prompt version: v1_5dim, v2_regular, v3_general, v4_lenient, v5_score, v6_score_raw")
    args = parser.parse_args()

    domains = ["drugs", "weapon"] if args.domain == "both" else [args.domain]
    use_concepts = args.repr == "concept"

    all_results = []
    for d in domains:
        res = run_experiment(d, use_concepts=use_concepts, task=args.task,
                            prompt_version=args.prompt)
        all_results.append(res)

    print(f"\n{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    for r in all_results:
        print(f"  {r['domain']:8s} | F1={r['f1']:.3f}  Acc={r['accuracy']:.3f}  "
              f"Prec={r['precision']:.3f}  Rec={r['recall']:.3f}  "
              f"({r['n_valid']}/{r['n_pairs']} valid)")


if __name__ == "__main__":
    main()
