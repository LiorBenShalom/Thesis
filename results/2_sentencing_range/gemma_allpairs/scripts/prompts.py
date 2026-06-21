"""Canonical v6 similarity prompts + parser — copied VERBATIM from
experiments/src/scoring/structured_llm_comparison_experiment.py so that local
scoring is method-identical to the paper. The ONLY thing that changes vs the
paper is the model backend (local Gemma instead of the API)."""
import re
import math

# ------------------------------------------------------------------ #
# User template (identical to USER_TEMPLATE_SCORE_RAW)
# ------------------------------------------------------------------ #
USER_TEMPLATE_SCORE_RAW = """להלן שני תיקים פליליים עם פיצ'רים מובנים:

תיק 1:
{fv1}

תיק 2:
{fv2}

מהו ציון הדמיון המהותי (0-100)?"""


# ------------------------------------------------------------------ #
# System prompts — weapon + drugs (identical to SYSTEM_PROMPT_V6_SCORE_RAW_*)
# ------------------------------------------------------------------ #
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

SYSTEM_BY_DOMAIN = {
    "drugs":  SYSTEM_PROMPT_V6_SCORE_RAW_drugs,
    "weapon": SYSTEM_PROMPT_V6_SCORE_RAW_wep,
}


# ------------------------------------------------------------------ #
# SCORE-ONLY (== the EXACT sentencing-experiment prompt).
# This is NOT an invented variant: it is the prompt actually used to score every
# similarity pair in the sentencing-range prediction experiment (extracted
# verbatim, byte-identical, from build_similarity_batch_*.py into
# prompts_sentencing.py). Its answer format is a SINGLE line `SIMILARITY_SCORE: X`
# with no analysis — i.e. "a number, no explanation". It also uses a slightly
# different USER template than the v6-multimodel one (no "להלן..." header).
# Use this mode to replicate the sentencing experiment locally.
# ------------------------------------------------------------------ #
from prompts_sentencing import (  # noqa: E402
    SYSTEM_DRUGS as SYSTEM_SENT_drugs,
    SYSTEM_WEAPON as SYSTEM_SENT_wep,
    USER_TEMPLATE_SENTENCING,
    SYSTEM_BY_DOMAIN as SYSTEM_SCORE_ONLY_BY_DOMAIN,
)


# ------------------------------------------------------------------ #
# Parser + validator (identical to parse_score_v6 / validate_score)
# ------------------------------------------------------------------ #
def parse_score_v6(text):
    """Only accept an explicit SIMILARITY_SCORE line (no fallback heuristics)."""
    if not text:
        return None
    m = re.search(r'SIMILARITY_SCORE\s*:\s*(\d+)', text)
    if m:
        return float(m.group(1))
    return None


def parse_score_bare(text):
    """For score-only mode: accept a bare integer. Prefer an explicit
    SIMILARITY_SCORE: line if present, else the FIRST standalone integer in
    [0,100]. Robust to the model adding a stray word or a trailing period."""
    if not text:
        return None
    m = re.search(r'SIMILARITY_SCORE\s*:\s*(\d+)', text)
    if m:
        v = float(m.group(1))
        return v if 0 <= v <= 100 else None
    for tok in re.findall(r'\d+', text):
        v = float(tok)
        if 0 <= v <= 100:
            return v
    return None


def select(domain, score_only):
    """Return (system_prompt, user_template, parser) for the chosen mode+domain.

    score_only=True  -> the EXACT sentencing prompt (one-line SIMILARITY_SCORE,
                        no analysis) + the sentencing USER template.
    score_only=False -> the v6-multimodel prompt (5-dim analysis + score line).
    """
    if score_only:
        sysmap, template, parser = SYSTEM_SCORE_ONLY_BY_DOMAIN, USER_TEMPLATE_SENTENCING, parse_score_bare
    else:
        sysmap, template, parser = SYSTEM_BY_DOMAIN, USER_TEMPLATE_SCORE_RAW, parse_score_v6
    return sysmap.get(domain, sysmap["drugs"]), template, parser


def validate_score(s):
    if s is None:
        return False
    try:
        v = float(s)
    except (TypeError, ValueError):
        return False
    if math.isnan(v):
        return False
    return 0.0 <= v <= 100.0
