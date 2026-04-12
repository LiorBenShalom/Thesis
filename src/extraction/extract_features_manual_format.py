"""
Automated Feature Extraction — Direct Manual-Format Output
==========================================================
Extracts features from court verdict texts using OpenAI API, outputting
DIRECTLY in the same Hebrew-key JSON format as the manual ground-truth
annotation CSV (similarity_database_fe.csv).

Two domains:
  - drugs  → סמים/כמויות + דגלי עבירה (כן/לא לכל קטגוריה) + תפקיד כשני שדות נפרדים + שאר השדות
  - weapon → רשת סוגי נשק + דגלי סוג עבירה / אופן החזקה + שאר השדות

Each feature uses a dedicated prompt plus OpenAI **Structured Outputs** (`response_format`:
`json_schema` עם `strict: true`) — אין `json_object` כללי ואין פענוח JSON ידני (regex וכו');
רק `json.loads` על תשובת המודל והוצאת ערכים לפי מפתחות.

Usage:
  export OPENAI_API_KEY="sk-..."
  python extract_features_manual_format.py --domain drugs
  python extract_features_manual_format.py --domain weapon
  python extract_features_manual_format.py --domain both
"""

from __future__ import annotations

import argparse
import json
import os
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception: pass
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

DEFAULT_MODEL = "gpt-4.1"
CACHE_FILENAME = "feature_cache_manual_format.json"
_FALLBACK_KEY = os.environ.get("OPENAI_API_KEY", "")

client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY", _FALLBACK_KEY)
        client = OpenAI(api_key=api_key)
    return client


def _call_gpt(
    prompt: str,
    system: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    json_schema: Optional[Dict[str, Any]] = None,
) -> str:
    """json_schema: OpenAI Structured Outputs — חובה לכל חילוץ; name/strict/schema."""
    kwargs: Dict[str, Any] = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if json_schema is None:
        raise ValueError("כל קריאת _call_gpt חייבת json_schema (Structured Outputs)")
    kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}
    resp = _get_client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def _openai_json_schema(name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """מעטפת OpenAI Structured Outputs (json_schema ב-call)."""
    return {"name": name, "strict": True, "schema": schema}


def _loads_structured(raw: str) -> Dict[str, Any]:
    """פלט ממודל עם response_format json_schema — JSON תקין בלבד; בלי פענוח/תיקון ידני של טקסט."""
    if not raw or not str(raw).strip():
        return {}
    try:
        return json.loads(str(raw).strip())
    except json.JSONDecodeError:
        return {}


def _strip_empty_feature_values(features: Dict[str, Any]) -> Dict[str, Any]:
    """Drop keys whose values are empty/whitespace (aligns with GT: present keys are never empty)."""
    out: Dict[str, Any] = {}
    for k, v in features.items():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        out[k] = v
    return out


def _apply_drugs_structured_defaults_like_legacy(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    ברירות מחדל כמו ב-apply_manual_drugs_defaults.py (CSV legacy ידני):
    מכירה לסוכן / מעבדה / עבירות נלוות → «לא» אם חסר; תפקיד בעלות סמים → «בעל הסמים»;
    אם מעבדה == «לא» → תפקיד_בעלות_מעבדה == «לא רלוונטי» (לא «לא בעל המעבדה»).
    """
    out: Dict[str, Any] = dict(features)
    for k, _ in DRUG_OFFENSE_FLAG_KEYS:
        v = out.get(k)
        if v not in ("כן", "לא"):
            out[k] = "לא"
    for k in ("מכירה לסוכן", "מעבדה", "עבירות נלוות כן/לא"):
        v = out.get(k)
        if v not in ("כן", "לא"):
            out[k] = "לא"
    role = str(out.get("תפקיד_בעלות_סמים", "")).strip()
    if role not in ("בעל הסמים", "לא בעל הסמים"):
        out["תפקיד_בעלות_סמים"] = "בעל הסמים"
    if out.get("מעבדה") == "לא":
        out["תפקיד_בעלות_מעבדה"] = "לא רלוונטי"
    else:
        lab_role = str(out.get("תפקיד_בעלות_מעבדה", "")).strip()
        if lab_role not in ("בעל המעבדה", "לא בעל המעבדה", "לא רלוונטי"):
            out["תפקיד_בעלות_מעבדה"] = "לא רלוונטי"
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  DRUGS DOMAIN — 6 Features
# ═══════════════════════════════════════════════════════════════════════════════

DRUGS_SYSTEM = (
    "אתה מנתח פסקי דין בעברית בתחום עבירות סמים מסוכנים. "
    "תפקידך: לחלץ עובדות משפטיות מדויקות לשדות אנוטציה מובנים (פלט JSON בלבד לפי הסכימה). "
    "קרא את כל קטע הטקסט שסופק. "
    "העדף עובדות מכתב האישום (ומתוקנו), מעובדות המקרה, מהודאות ומהרשעה — לגבי הנאשם והעבירות **בתיק הנדון**. "
    "אל תסתמך על ציטוטים מפסקי דין אחרים, על עבירות של נאשמים אחרים, או על רקע שלא נקבע כעובדה לגבי התיק. "
    "אם אין בסיס בטקסט — ברירות המחדל של השדה (לא השערות). "
    "אין טקסט חופשי מחוץ ל-JSON."
)

DRUGS_USER_PREFIX = """משימה: חילוץ עובדות לשדות האנוטציה (JSON לפי הסכימה בלבד).

מיקוד עובדתי (בסדר עדיפות): כתב האישום וגרסאותיו • עובדות המקרה / מה שמוצג כעובדתי • הודאות והרשעה — כולם לגבי **הנאשם והאירועים בתיק זה**.
התעלם מציטוטי תקדים, מדיונים בעבירות של נאשמים אחרים, וממידע שלא נקבע כעובדה לתיק.

קרא את **כל** הטקסט למטה לפני מילוי השדות."""


# עבירות סמים — דגל נפרד לכל תיבת סימון (כן/לא), בלי מחרוזת מאוחדת ובלי "אחר"
DRUG_OFFENSE_FLAG_KEYS: List[Tuple[str, str]] = [
    ("עבירה_יבוא_סחר", "יבוא/סחר (סחר/ייבוא/הברחה לפי הטופס)"),
    ("עבירה_החזקה", "החזקה שלא לצריכה עצמית"),
    ("עבירה_ייצור", "ייצור (ייצור/גידול)"),
    ("עבירה_כלים", "כלים (סעיף 10)"),
    ("עבירה_ניסיון_סחר", "ניסיון לסחר/ייבוא"),
    ("עבירה_ניסיון_ייצור", "ניסיון לייצור/גידול"),
    ("עבירה_19", "סעיף 19 מופרש (ייבוא/ייצוא/הברחה)"),
]


def _schema_drug_offense_flags() -> Dict[str, Any]:
    props = {k: {"type": "string", "enum": ["כן", "לא"]} for k, _ in DRUG_OFFENSE_FLAG_KEYS}
    keys = list(props.keys())
    return _openai_json_schema(
        "drug_offense_flags",
        {"type": "object", "properties": props, "required": keys, "additionalProperties": False},
    )


def _schema_drug_type_quantity() -> Dict[str, Any]:
    return _openai_json_schema(
        "drug_type_quantity",
        {
            "type": "object",
            "properties": {
                "סוג הסם, כמות": {
                    "type": "string",
                    "description": "פורמט הטופס; ריק אם אין סמים מפורטים",
                }
            },
            "required": ["סוג הסם, כמות"],
            "additionalProperties": False,
        },
    )


def _schema_yes_no_field(field_key: str, schema_name: str) -> Dict[str, Any]:
    return _openai_json_schema(
        schema_name,
        {
            "type": "object",
            "properties": {field_key: {"type": "string", "enum": ["כן", "לא"]}},
            "required": [field_key],
            "additionalProperties": False,
        },
    )


def _schema_drug_role() -> Dict[str, Any]:
    return _openai_json_schema(
        "drug_role",
        {
            "type": "object",
            "properties": {
                "תפקיד_בעלות_סמים": {"type": "string", "enum": ["בעל הסמים", "לא בעל הסמים"]},
                "תפקיד_בעלות_מעבדה": {
                    "type": "string",
                    "enum": ["בעל המעבדה", "לא בעל המעבדה", "לא רלוונטי"],
                },
            },
            "required": ["תפקיד_בעלות_סמים", "תפקיד_בעלות_מעבדה"],
            "additionalProperties": False,
        },
    )


def _extract_drug_offense_flags(text: str, model: str) -> Dict[str, str]:
    """עבירות סמים — דגל כן/לא לכל קטגוריה מוגדרת מראש (לא מחרוזת אחת, לא «אחר»)."""
    keys_lines = "\n".join(f'"{key}": "כן" או "לא" — {desc}' for key, desc in DRUG_OFFENSE_FLAG_KEYS)
    prompt = f"""{DRUGS_USER_PREFIX}

שדה לחילוץ: "עבירות"
סוג השדה: תיבות סימון — כל תיבה עצמאית.

⚠️ חשוב מאוד:
- ענה **רק** בקטגוריות המוגדרות למטה. **אל תוסיף שדה "אחר"** ואל תפרט טקסט חופשי.
- אם התוכן בגזר הדין דומה לקטגוריה קיימת — סמן אותה בלבד; **אל תפתח קטגוריה חדשה**.
- לכל מפתח: רק "כן" או "לא".

המפתחות (כולם חובה ב-JSON):
{keys_lines}

כללים:
- התבסס רק על כתב האישום / הרשעה בגזר הדין הנוכחי.
- "עבירה_19" = "כן" רק אם סעיף 19 (ייבוא/ייצוא/הברחה) מוזכר מפורשות בהקשר הרלוונטי.

טקסט פסק הדין (קרא במלואו):
{text}

ענה ב-JSON עם כל המפתחות לעיל (ערכים: "כן" או "לא" בלבד).
"""
    resp = _call_gpt(prompt, DRUGS_SYSTEM, model=model, json_schema=_schema_drug_offense_flags())
    data = _loads_structured(resp)
    if not data:
        return {k: "לא" for k, _ in DRUG_OFFENSE_FLAG_KEYS}
    return {k: data.get(k, "לא") for k, _ in DRUG_OFFENSE_FLAG_KEYS}


def _extract_drug_type_quantity(text: str, model: str) -> str:
    """סוג הסם, כמות — in the exact manual GT format.

    GT / manual format instruction:
      "כמות-יחידת מידה-סוג הסם, (לדוגמא: 50-ק"ג-קנאבוס-, 5-גרם-קוקאין)
       *בין כל סם לשים פסיק ורווח"
    """
    prompt = f"""{DRUGS_USER_PREFIX}

שדה לחילוץ: "סוג הסם, כמות"
הנחיית פורמט (כמו ב-GT): "כמות-יחידת מידה-סוג הסם, (לדוגמא: 50-ק"ג-קנאבוס-, 5-גרם-קוקאין) *בין כל סם לשים פסיק ורווח"

הפורמט הנדרש (חובה לעקוב בדיוק):
כמות-יחידת_מידה-שם_הסם
אם יש מספר סמים: הפרד בפסיק ורווח ", "

שמות סמים מותרים (שם הסם בלבד — לא יחידת מידה):
• קוקאין
• MDMA — גם באבקה/גבישים וגם בטבליות; הסימון MDMAמ ב-GT הוא לעיתים סימון נוסף, לא «סוג סם» נפרד מטבליות
• קנבוס / קנאביס / קנביס (מריחואנה)
• חשיש
• LSDמ (LSD, אל.אס.די, בולים)
• מתאמפטמין
• KETAMINE (קטמין)
• קתינון
• מתילמקאתינון
• האיוואסקה

יחידות מידה מותרות (כמו ק״ג וגרם — לא חלק מ«שם הסם»):
• גרם
• ק"ג (קילוגרם)
• טבליות
• כדורים
• בולים (בדרך כלל LSD)
• עציצים / שתילים (שתילי קנבוס)

דוגמאות מתוך נתוני אמת (שים לב לפורמט!):
• "13189-טבליות-MDMAמ, 1301.44-גרם-MDMAמ"
• "700-גרם-KETAMINE"
• "888-גרם-קוקאין"
• "235-עציצים-קנבוס, 72.8-ק"ג,קנבוס"
• "30.11-גרם-LSDמ, 1,255.48-גרם-קנבוס, 689.77-גרם-חשיש, 52.616-גרם-MDMA"
• "4.922-ק"ג-קוקאין"
• "8363-כדורים-MDMAמ"
• "1.865-ק"ג-מתאמפטמין"
• "140-ק"ג-קנביס"
• "1029.44-גרם-קוקאין"
• "31243-טבליות-MDMAמ, 0.9-גרם-קנאביס"

כללים:
- רשום רק סמים שמוזכרים מפורשות בגזר הדין עם כמויות.
- השתמש בכמות המדויקת שמופיעה בטקסט.
- אם כתוב קילוגרם → כתוב ק"ג (אל תמיר לגרם).
- אם כתוב גרם → כתוב גרם.
- הפורמט הוא: כמות-יחידה-שם_הסם (שלושה חלקים מופרדים במקף).
- כמו באנוטציה הידנית (GT): אם יש כמה אזכורים לאותו סם באותה יחידה (למשל כמה חבילות גרם MDMA שנתפסו בנפרד) — רשום כל כמות כשלישיה נפרדת, הפרד בין השלישיות בפסיק ורווח ", ". אל תאחד לסכום אחד אלא אם בטקסט מפורש "בסך הכל" / סה"כ לאותה עסקה או חבילה אחת.
- אם יש גם טבליות וגם אבקה של אותו סם — שתי שלישיות נפרדות (למשל MDMAמ לטבליות ו-MDMA לגרם), כמו בדוגמאות ב-GT.

טקסט פסק הדין (קרא במלואו):
{text}

ענה בפורמט JSON:
{{"סוג הסם, כמות": "..."}}
"""
    resp = _call_gpt(
        prompt, DRUGS_SYSTEM, model=model, max_tokens=1500, json_schema=_schema_drug_type_quantity()
    )
    data = _loads_structured(resp)
    return (data.get("סוג הסם, כמות") or "").strip()


def _extract_drug_additional_offenses(text: str, model: str) -> str:
    """עבירות נלוות כן/לא

    GT: binary כן/לא
    """
    prompt = f"""{DRUGS_USER_PREFIX}

שדה לחילוץ: "עבירות נלוות כן/לא" (חובה במסגרת האנוטציה)
סוג השדה: בחירה יחידה (כן/לא).

הסבר: האם בנוסף לעבירות הסמים, הנאשם הורשע גם בעבירות נלוות שאינן עבירות סמים?

עבירות נלוות כוללות למשל:
- קשירת קשר (סעיף 499)
- עבירות אלימות (תקיפה, חבלה)
- עבירות נשק
- עבירות תנועה
- הלבנת הון
- שיבוש הליכי משפט

עבירות שאינן נחשבות "נלוות" (כי הן עבירות סמים):
- סחר/החזקה/ייצור/ייבוא של סם מסוכן
- החזקת כלים לצריכת סמים

ענה:
• "כן" — יש עבירות נלוות שאינן עבירות סמים
• "לא" — אין עבירות נלוות (ברירת מחדל)

טקסט פסק הדין (קרא במלואו):
{text}

ענה בפורמט JSON:
{{"עבירות נלוות כן/לא": "..."}}
"""
    resp = _call_gpt(
        prompt,
        DRUGS_SYSTEM,
        model=model,
        json_schema=_schema_yes_no_field("עבירות נלוות כן/לא", "drug_additional_offenses"),
    )
    data = _loads_structured(resp)
    return data.get("עבירות נלוות כן/לא", "לא")


def _extract_drug_laboratory(text: str, model: str) -> str:
    """מעבדה

    GT: binary כן/לא (no asterisk = not required, default = לא)
    """
    prompt = f"""{DRUGS_USER_PREFIX}

שדה לחילוץ: "מעבדה"
סוג השדה: בחירה יחידה (כן/לא). שדה אופציונלי — ברירת מחדל: "לא".

הסבר: האם הייתה מעורבת מעבדה לייצור/הכנת סמים בתיק?

חפש אזכורים של:
- מעבדה / מעבדת סמים
- מכשור ייצור / ציוד כימי
- תהליך ייצור / הפקה של סם

ענה:
• "כן" — מוזכרת מעבדה או ציוד ייצור
• "לא" — לא מוזכרת מעבדה (ברירת מחדל)

טקסט פסק הדין (קרא במלואו):
{text}

ענה בפורמט JSON:
{{"מעבדה": "..."}}
"""
    resp = _call_gpt(prompt, DRUGS_SYSTEM, model=model, json_schema=_schema_yes_no_field("מעבדה", "drug_lab"))
    data = _loads_structured(resp)
    return data.get("מעבדה", "לא")


def _extract_drug_sold_to_agent(text: str, model: str) -> str:
    """מכירה לסוכן

    GT: binary כן/לא (not required, default = לא)
    """
    prompt = f"""{DRUGS_USER_PREFIX}

שדה לחילוץ: "מכירה לסוכן"
סוג השדה: בחירה יחידה (כן/לא). שדה אופציונלי — ברירת מחדל: "לא".

הסבר: האם הנאשם מכר סמים לסוכן סמוי / סוכן משטרתי / מקור משטרתי?

חפש אזכורים של:
- "סוכן סמוי" / "סוכן חשאי" / "קצין חשאי"
- "מקור משטרתי" / "מודיע"
- "רכישה מבוקרת" / "עסקה מבוקרת"

ענה:
• "כן" — מוזכר סוכן סמוי או מכירה/עסקה מבוקרת
• "לא" — לא מוזכר (ברירת מחדל)

טקסט פסק הדין (קרא במלואו):
{text}

ענה בפורמט JSON:
{{"מכירה לסוכן": "..."}}
"""
    resp = _call_gpt(
        prompt, DRUGS_SYSTEM, model=model, json_schema=_schema_yes_no_field("מכירה לסוכן", "drug_agent")
    )
    data = _loads_structured(resp)
    return data.get("מכירה לסוכן", "לא")


def _extract_drug_role_structured(text: str, model: str) -> Dict[str, str]:
    """תפקיד — שני מימדים נפרדים (לא מחרוזת אחת): בעלות סמים + בעלות מעבדה."""
    prompt = f"""{DRUGS_USER_PREFIX}

מימד 1 — בעלות על הסמים (בחר **אחת** בלבד):
• "בעל הסמים" — סוחר/ייבואן/בעלים של הסמים
• "לא בעל הסמים" — שליח/מוביל/מתווך/שומר ולא בעלים

מימד 2 — בעלות על מעבדה — **רק אם** מוזכרת מעבדה/ייצור בבית חולים או ציוד ייצור בתיק. אחרת השאר null:
• "בעל המעבדה"
• "לא בעל המעבדה"

⚠️ ענה ב-JSON בלבד לפי המסגרת הטכנית (enum):
- "תפקיד_בעלות_מעבדה": אם אין שום אזכור למעבדה → "לא רלוונטי"

כללים:
- ברירת מחדל לבעלות סמים אם לא ברור: "לא בעל הסמים"

טקסט פסק הדין (קרא במלואו):
{text}
"""
    resp = _call_gpt(prompt, DRUGS_SYSTEM, model=model, json_schema=_schema_drug_role())
    data = _loads_structured(resp)
    if not data:
        return {"תפקיד_בעלות_סמים": "בעל הסמים", "תפקיד_בעלות_מעבדה": "לא רלוונטי"}
    out: Dict[str, str] = {"תפקיד_בעלות_סמים": data["תפקיד_בעלות_סמים"]}
    if data.get("תפקיד_בעלות_מעבדה") != "לא רלוונטי":
        out["תפקיד_בעלות_מעבדה"] = data["תפקיד_בעלות_מעבדה"]
    return out


def extract_drugs_features(text: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """חילוץ פיצ'רי סמים: דגלי עבירה נפרדים, תפקיד כשני שדות, כמויות כמחרוזת GT."""
    features: Dict[str, Any] = {}

    features.update(_extract_drug_offense_flags(text, model))

    drug_qty = _extract_drug_type_quantity(text, model)
    if drug_qty:
        features["סוג הסם, כמות"] = drug_qty

    features["עבירות נלוות כן/לא"] = _extract_drug_additional_offenses(text, model)
    features["מעבדה"] = _extract_drug_laboratory(text, model)
    features["מכירה לסוכן"] = _extract_drug_sold_to_agent(text, model)

    features.update(_extract_drug_role_structured(text, model))

    features = _strip_empty_feature_values(features)
    features = _apply_drugs_structured_defaults_like_legacy(features)
    from drug_offense_categories import canonical_offense_label_from_drug_flag_dict

    features["עבירה"] = canonical_offense_label_from_drug_flag_dict(features)
    return features


# ═══════════════════════════════════════════════════════════════════════════════
#  WEAPON DOMAIN — up to 18 Features
# ═══════════════════════════════════════════════════════════════════════════════

WEAPON_SYSTEM = (
    "אתה מנתח פסקי דין בעברית בתחום עבירות נשק. "
    "תפקידך: לחלץ עובדות משפטיות מדויקות לשדות אנוטציה מובנים (פלט JSON בלבד לפי הסכימה). "
    "קרא את כל קטע הטקסט שסופק. "
    "העדף עובדות מכתב האישום, מתיאור העובדות/התפיסה/האחזקה ומהרשעה — לגבי **הנאשם, הנשק והאירוע בתיק הנדון**. "
    "התעלם מציטוטי תקדים, מעבירות של אחרים, ומסעיפים שמוזכרים שלא בהקשר הרשעה של הנאשם הנוכחי. "
    "אם אין בסיס בטקסט — ברירות המחדל של השדה (לא השערות). "
    "אין טקסט חופשי מחוץ ל-JSON."
)

WEAPON_USER_PREFIX = """משימה: חילוץ עובדות לשדות האנוטציה (JSON לפי הסכימה בלבד).

מיקוד עובדתי: כתב האישום (ומתוקנו) • עובדות המקרה ותיאור הנשק/התפיסה/האחזקה • הודאות והרשעה — לגבי **הנאשם והמעשה בתיק זה**.
אל תמלא לפי אזכורים עקיפים, תקדים או עבירות של נאשמים אחרים.

קרא את **כל** הטקסט למטה לפני מילוי השדות."""


def _schema_weapon_offense_number() -> Dict[str, Any]:
    return _openai_json_schema(
        "weapon_offense_number",
        {
            "type": "object",
            "properties": {
                "מספר עבירה": {
                    "type": "string",
                    "description": 'למשל "144 א", "144 ב", "144 א, 144 ב"; ריק אם לא חל',
                }
            },
            "required": ["מספר עבירה"],
            "additionalProperties": False,
        },
    )


def _extract_weapon_offense_number(text: str, model: str) -> str:
    """מספר עבירה

    GT: סעיפי הרשעה — 144 א | 144 ב
    """
    prompt = f"""{WEAPON_USER_PREFIX}

שדה לחילוץ: "מספר עבירה"
סוג השדה: תיבות סימון (checkboxes).

האפשרויות:
☐ 144 א — סעיף 144(א) לחוק העונשין: החזקת/נשיאת נשק ללא רישיון
☐ 144 ב — סעיף 144(ב): סחר/ייצור/ייבוא/ייצוא נשק ללא רישיון

⚠️ כללים קריטיים:
- רשום רק סעיפים שהנאשם הנוכחי בתיק הזה הורשע בהם בפועל.
- חפש את ההרשעה בניסוחים כמו: "נותן את הדין בגין...", "הורשע ב...", "הודה ב...", "כתב האישום המתוקן..."
- אל תרשום סעיפים שמוזכרים בהקשרים אחרים:
  × לא מפסקי דין אחרים שמצוטטים כתקדים
  × לא מתנאי מאסר על תנאי
  × לא מכותרות/רשימות חוקים בראש פסק הדין
  × לא מדיון בעבירות של נאשמים אחרים
- כתוב "144 א" (עם רווח לפני א), לא "144(א)".
- אם הנאשם הורשע בשני הסעיפים → "144 א, 144 ב"

טקסט פסק הדין (קרא במלואו):
{text}

ענה בפורמט JSON:
{{"מספר עבירה": "..."}}
"""
    resp = _call_gpt(prompt, WEAPON_SYSTEM, model=model, json_schema=_schema_weapon_offense_number())
    data = _loads_structured(resp)
    return (data.get("מספר עבירה") or "").strip()


WEAPON_OFFENSE_FLAG_KEYS: List[Tuple[str, str]] = [
    ("סוג_עבירה_החזקה", "החזקה נשק"),
    ("סוג_עבירה_נשיאה", "נשיאת נשק"),
    ("סוג_עבירה_הובלה", "הובלת נשק"),
    ("סוג_עבירה_סחר", "סחר בנשק"),
    ("סוג_עבירה_ניסיון_סחר", "ניסיון לסחר"),
    ("סוג_עבירה_ביצוע_עבירות", "ביצוע עבירות בנשק"),
    ("סוג_עבירה_ייצור", "ייצור נשק"),
]


def _schema_weapon_offense_flags() -> Dict[str, Any]:
    props = {k: {"type": "string", "enum": ["כן", "לא"]} for k, _ in WEAPON_OFFENSE_FLAG_KEYS}
    keys = list(props.keys())
    return _openai_json_schema(
        "weapon_offense_type_flags",
        {"type": "object", "properties": props, "required": keys, "additionalProperties": False},
    )


def _extract_weapon_offense_type_flags(text: str, model: str) -> Dict[str, str]:
    """סוג עבירת נשק — דגל כן/לא לכל תיבה; בלי «אחר», רק קטגוריות מוגדרות."""
    keys_lines = "\n".join(f'"{key}": "כן" או "לא" — {desc}' for key, desc in WEAPON_OFFENSE_FLAG_KEYS)
    prompt = f"""{WEAPON_USER_PREFIX}

סוג העבירה = תיבות סימון עצמאיות. לכל מפתח: **רק** "כן" או "לא".

⚠️ השתמש **רק** בקטגוריות המוגדרות. **אל תוסיף "אחר"** ואל תכתוב טקסט חופשי.
אם התיאור בגזר הדין דומה לאחת הקטגוריות — סמן אותה בלבד.

המפתחות:
{keys_lines}

כללים:
- התבסס על כתב האישום / הרשעה של הנאשם בתיק זה.

טקסט פסק הדין (קרא במלואו):
{text}

ענה ב-JSON עם כל המפתחות לעיל.
"""
    resp = _call_gpt(prompt, WEAPON_SYSTEM, model=model, json_schema=_schema_weapon_offense_flags())
    data = _loads_structured(resp)
    if not data:
        return {k: "לא" for k, _ in WEAPON_OFFENSE_FLAG_KEYS}
    return {k: data.get(k, "לא") for k, _ in WEAPON_OFFENSE_FLAG_KEYS}


WEAPON_STORAGE_FLAG_KEYS: List[Tuple[str, str]] = [
    ("אופן_החזקה_בבית", "בבית"),
    ("אופן_החזקה_ברכב", "ברכב"),
    ("אופן_החזקה_על_גופו", "על גופו"),
    ("אופן_החזקה_מוסלק", "מוסלק - מוסתר"),
    ("אופן_החזקה_סמוך_לבית", "סמוך לבית"),
]


def _schema_weapon_storage() -> Dict[str, Any]:
    props = {k: {"type": "string", "enum": ["כן", "לא"]} for k, _ in WEAPON_STORAGE_FLAG_KEYS}
    keys = list(props.keys())
    return _openai_json_schema(
        "weapon_storage_flags",
        {"type": "object", "properties": props, "required": keys, "additionalProperties": False},
    )


def _extract_weapon_storage_flags(text: str, model: str) -> Dict[str, str]:
    """אופן החזקת הנשק — דגל לכל מיקום; לא מחרוזת אחת."""
    keys_lines = "\n".join(f'"{key}": "כן" או "לא" — {desc}' for key, desc in WEAPON_STORAGE_FLAG_KEYS)
    prompt = f"""{WEAPON_USER_PREFIX}

שאלה: אופן החזקת הנשק (תיבות סימון).

לכל מיקום מהרשימה — רק "כן" אם רלוונטי, "לא" אחרת.
**אל תוסיף "אחר"** — אם אין התאמה, כל המפתחות "לא".

המפתחות:
{keys_lines}

הערה: אם יש מרדף — תאר מצב לפני המרדף.

טקסט פסק הדין (קרא במלואו):
{text}

ענה ב-JSON עם כל המפתחות.
"""
    resp = _call_gpt(prompt, WEAPON_SYSTEM, model=model, json_schema=_schema_weapon_storage())
    data = _loads_structured(resp)
    if not data:
        return {k: "לא" for k, _ in WEAPON_STORAGE_FLAG_KEYS}
    return {k: data.get(k, "לא") for k, _ in WEAPON_STORAGE_FLAG_KEYS}


# All 14 weapon types from the manual annotation grid
_WEAPON_TYPE_GRID = [
    ("סוג הנשק [אקדח]", "אקדח", "אקדח, פיסטול, אקדח מסוג"),
    ("סוג הנשק [תת מקלע]", "תת מקלע", "תת מקלע (לא מאולתר)"),
    ("סוג הנשק [תת מקלע מאולתר]", "תת מקלע מאולתר", "תת מקלע מאולתר, 'קרלו', קרלו"),
    ("סוג הנשק [בקבוק תבערה]", "בקבוק תבערה", "בקבוק תבערה, מולוטוב"),
    ("סוג הנשק [מטען חבלה]", "מטען חבלה", "מטען חבלה, מטען נפץ"),
    ("סוג הנשק [רימון רסס]", "רימון רסס", "רימון רסס, רימון עשן"),
    ("סוג הנשק [רובה סער ]", "רובה סער", "רובה סער, M16, M4, קלצ'ניקוב, AK-47"),
    ("סוג הנשק [רימון הלם/גז]", "רימון הלם/גז", "רימון הלם, רימון גז"),
    ("סוג הנשק [טיל לאו]", "טיל לאו", "טיל לאו, LAW"),
    ("סוג הנשק [טיל מטאדור]", "טיל מטאדור", "טיל מטאדור"),
    ("סוג הנשק [רובה צייד]", "רובה צייד", "רובה צייד"),
    ("סוג הנשק [רובה צלפים]", "רובה צלפים", "רובה צלפים"),
    ("סוג הנשק [מטען חבלה מאולתר]", "מטען חבלה מאולתר", "מטען חבלה מאולתר, מטען מאולתר"),
    ("סוג הנשק [רובה סער מאולתר]", "רובה סער מאולתר", "רובה סער מאולתר, M16 מאולתר"),
]


def _schema_weapon_types() -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    for gt_key, _, _ in _WEAPON_TYPE_GRID:
        props[gt_key] = {"type": "integer", "minimum": 0, "maximum": 6}
    props["סוג הנשק - אם לא נמצא בטבלה"] = {"type": "string", "description": "ריק אם אין"}
    keys = list(props.keys())
    return _openai_json_schema(
        "weapon_types_grid",
        {"type": "object", "properties": props, "required": keys, "additionalProperties": False},
    )


def _extract_weapon_types(text: str, model: str) -> Dict[str, Any]:
    """רשת סוגי נשק (GT) + שדה 'אם לא נמצא בטבלה'.

    Form: 14-row grid (weapon type × quantity 1–6), plus free text "סוג הנשק - אם לא נמצא בטבלה".
    """
    type_list = "\n".join(
        f"- {heb_name} ({aliases})" for _, heb_name, aliases in _WEAPON_TYPE_GRID
    )
    prompt = f"""{WEAPON_USER_PREFIX}

שדה לחילוץ: "סוג הנשק"
הקשר / דוגמה: "לדוגמה: 'אקדח מסוג יריחו', 'תת מקלע מאולתר', 'אקדח'. *אפשר להכניס מספר אפשריות"
סוג השדה: טבלת רשת — כל סוג נשק × כמות (1 עד 6).

סוגי נשק ברשת:
{type_list}

בנוסף, יש שדה חופשי: "סוג הנשק - אם לא נמצא בטבלה" — לנשק שלא ברשימה.

כללים:
- לכל סוג נשק, רשום את הכמות הכוללת (מספר שלם). אם לא נמצא → 0.
- אם מוזכרים מספר נשקים מאותו סוג במקומות שונים בגזר הדין (למשל שני אקדחים שנתפסו בנפרד) — סכם לכמות הכוללת באותו סוג ברשת. אל תכפול את אותו נשק אם הוא מתואר פעמיים באותה תפיסה.
- לשדה "אם לא נמצא בטבלה" → תיאור חופשי, או "" אם אין.
- התבסס רק על מה שכתוב בגזר הדין.

טקסט פסק הדין (קרא במלואו):
{text}

המפתחות ב-JSON חייבים להיות **בדיוק** שמות השדות מהטופס (כפי שבמסגרת הטכנית), למשל "סוג הנשק [אקדח]" ולא השם הקצר בלבד.
רשום 0 לסוגים שלא נמצאו.
"""
    resp = _call_gpt(
        prompt, WEAPON_SYSTEM, model=model, max_tokens=1000, json_schema=_schema_weapon_types()
    )
    data = _loads_structured(resp)
    if not data:
        return {}

    result: Dict[str, Any] = {}
    for gt_key, _, _ in _WEAPON_TYPE_GRID:
        val = data.get(gt_key, 0)
        try:
            val = int(val)
        except (TypeError, ValueError):
            val = 0
        if val > 0:
            result[gt_key] = float(val)

    other = data.get("סוג הנשק - אם לא נמצא בטבלה", "")
    if str(other).strip():
        result["סוג הנשק - אם לא נמצא בטבלה"] = str(other).strip()

    return result


WEAPON_STATUS_ALLOWED: Tuple[str, ...] = (
    "נשק עם כדור בקנה",
    "נשק עם מחסנית בהכנס",
    "נשק מופרד מתחמושת",
    "נשק מפורק",
    "תקין",
    "תקול",
)


def _weapon_status_response_schema() -> Dict[str, Any]:
    """Structured Outputs: enum בלבד."""
    return _openai_json_schema(
        "weapon_status_single_choice",
        {
            "type": "object",
            "properties": {
                "סטטוס הנשק": {
                    "type": "string",
                    "description": "סטטוס יחיד לפי הפרומפט",
                    "enum": list(WEAPON_STATUS_ALLOWED),
                }
            },
            "required": ["סטטוס הנשק"],
            "additionalProperties": False,
        },
    )


WEAPON_PURPOSE_ENUM: Tuple[str, ...] = (
    "בצע כסף",
    "הגנה עצמית",
    "חתונה",
    "סכסוך",
    "תדמית",
    "לא צוין",
)


def _schema_weapon_how_obtained() -> Dict[str, Any]:
    return _openai_json_schema(
        "weapon_how_obtained",
        {
            "type": "object",
            "properties": {
                "אופן קבלת הנשק": {
                    "type": "string",
                    "enum": ["רכש", "מצא", "גנב", "מאחר", "ייצר", "אחר", "לא ידוע"],
                }
            },
            "required": ["אופן קבלת הנשק"],
            "additionalProperties": False,
        },
    )


def _schema_weapon_ammunition() -> Dict[str, Any]:
    return _openai_json_schema(
        "weapon_ammunition",
        {
            "type": "object",
            "properties": {"כמות תחמושת": {"type": "string", "description": "ריק אם אין"}},
            "required": ["כמות תחמושת"],
            "additionalProperties": False,
        },
    )


def _schema_weapon_purpose() -> Dict[str, Any]:
    return _openai_json_schema(
        "weapon_purpose",
        {
            "type": "object",
            "properties": {
                "מטרה-סיבת העבירה": {"type": "string", "enum": list(WEAPON_PURPOSE_ENUM)},
            },
            "required": ["מטרה-סיבת העבירה"],
            "additionalProperties": False,
        },
    )


def _schema_weapon_additional_offenses() -> Dict[str, Any]:
    return _openai_json_schema(
        "weapon_additional_offenses",
        {
            "type": "object",
            "properties": {"עבירות נוספות": {"type": "string", "description": "ריק אם אין"}},
            "required": ["עבירות נוספות"],
            "additionalProperties": False,
        },
    )


def _schema_weapon_planning() -> Dict[str, Any]:
    return _openai_json_schema(
        "weapon_planning",
        {
            "type": "object",
            "properties": {"תכנון": {"type": "string", "enum": ["כן", "לא"]}},
            "required": ["תכנון"],
            "additionalProperties": False,
        },
    )


def _schema_weapon_use() -> Dict[str, Any]:
    return _openai_json_schema(
        "weapon_use",
        {
            "type": "object",
            "properties": {
                "שימוש": {
                    "type": "string",
                    "description": 'למשל "לא", "כן,ירי", שילובים כמו ב-GT',
                }
            },
            "required": ["שימוש"],
            "additionalProperties": False,
        },
    )


def _extract_weapon_status(text: str, model: str) -> str:
    """סטטוס הנשק — קריאת GPT אחת; הפלט כפוף ל-schema (enum) בלבד, בלי סריקת גזר דין בפייתון אחרי המודל."""
    prompt = f"""{WEAPON_USER_PREFIX}

שאלה: "סטטוס הנשק"
סוג: בחירה יחידה בלבד — **סטטוס אחד** מתוך הרשימה שבמסגרת הטכנית (enum). **אסור** "אחר".

עקרונות (חשובים — כדי להתאים למסמכי אמת):
1) **תקין** — כשאין תיאור **קונקרטי** של מצב טעינה / מיקום מחסנית / הפרדה נשק–תחמושת, או כשכתוב רק שהנשק תקין/פעיל בלי פירוט. אל תדרג ל"נשק עם כדור בקנה" בגלל מילה "ירי"/"ירה" אם היא מתייחסת לעבירה אחרת, לאירוע אחר, או ללא קישור ישיר **למצב הנשק בעת האחזקה/גילוי/מעצר** של אותו נשק.

2) **נשק עם כדור בקנה** — רק אם **מפורש** אחד מאלה **באותו הקשר** של הנשק הנדון: "טעון ודרוך", "כדור בקנה", נשק בפועל יורה/ירה **ממצב טעון**, או תיאור שקול. אם בגזר הדין יש **כמה מצבים בזמן** (למשל תקול ואחר כך תוקן וירה) — בחר את **החמור ביותר** לפי ההיררכיה למטה (ירי מנשק טעון מנצח על תקול קודם באותו סיפור).

3) **נשק עם מחסנית בהכנס** — רק כשמפורש שהמחסנית **בתוך** הנשק **ויש בה כדורים** (מחסנית טעונה בהכנס). אם המחסנית **ריקה** (0 כדורים) או שלא נאמר שיש בה כדורים — **לא** לבחור זאת; העדף "נשק מופרד מתחמושת".

4) **נשק מופרד מתחמושת** — נשק ותחמושת/מחסניות **בנפרד**; או מחסנית ריקה בנשק; או תיאור שקול ("הופרד", "ללא כדור בקנה", "מחסניות ריקות").

5) **נשק מפורק** — מפורק (ללא צירוף "תקול" בלבד — אם גם תקול מפורש, ראה למטה).

6) **תקול** — לא פועל / מקולקל, בלי תיאור ירי/טעינה סותר באותו קטע רלוונטי.

7) אם **גם** "מפורק" **וגם** "תקול" חלים — ענה **"תקול"** (ערך יחיד מהרשימה).

היררכיה כשיש כמה מצבים **באותו סיפור על אותו נשק** (מהחמור לפחות):
1. נשק עם כדור בקנה
2. נשק עם מחסנית בהכנס (מחסנית טעונה בהכנס)
3. נשק מופרד מתחמושת
4. נשק מפורק
5. תקול
6. תקין

טקסט פסק הדין (קרא במלואו):
{text}
"""
    resp = _call_gpt(
        prompt,
        WEAPON_SYSTEM,
        model=model,
        json_schema=_weapon_status_response_schema(),
    )
    data = _loads_structured(resp)
    return data.get("סטטוס הנשק", "תקין")


def _extract_weapon_how_obtained(text: str, model: str) -> str:
    """אופן קבלת הנשק

    GT אפשרויות: מאחר | מצא | גנב | רכש | ייצר | אחר
    הערה: אם יש מרדף — לפני המרדף.
    """
    prompt = f"""{WEAPON_USER_PREFIX}

שדה לחילוץ: "אופן קבלת הנשק"
הקשר / דוגמה: "...נמסר לנאשם ע"י פלוני אלמוני ליד רכבו..." — כיצד הנאשם השיג את הנשק.
         "הערה: אם יש מרדף אז לפני המרדף"
סוג השדה: בחירה יחידה.

האפשרויות:
○ רכש — קנה/רכש את הנשק (כולל רכישה מאדם אחר בתמורה כספית)
○ מצא — מצא את הנשק (באקראי, ברחוב, במקום נטוש)
○ גנב — גנב את הנשק
○ מאחר — קיבל מאדם אחר (לא קנה — קיבל, נמסר לו, הופקד אצלו, לשמירה)
○ ייצר — ייצר/הרכיב את הנשק בעצמו
○ אחר — דרך אחרת (למשל: "עבודה" — הנשק קשור לעבודתו כשומר/חייל)

דוגמאות מתוך נתוני אמת:
• "רכש"
• "מצא"
• "גנב"
• "מאחר"
• "עבודה"
• "ייצר"

כללים:
- "רכש" = שילם כסף. "מאחר" = קיבל בלי לשלם.
- אם לא ברור כלל — בחר "לא ידוע" (לא לנחש).

טקסט פסק הדין (קרא במלואו):
{text}

ענה בפורמט JSON:
{{"אופן קבלת הנשק": "..."}}
"""
    resp = _call_gpt(prompt, WEAPON_SYSTEM, model=model, json_schema=_schema_weapon_how_obtained())
    data = _loads_structured(resp)
    v = data.get("אופן קבלת הנשק", "לא ידוע")
    return "" if v == "לא ידוע" else v


def _extract_weapon_ammunition(text: str, model: str) -> str:
    """כמות תחמושת

    GT: טקסט חופשי; דוגמה: מחסנית ריקה → ניסוח כמו ב-GT
    """
    prompt = f"""{WEAPON_USER_PREFIX}

שדה לחילוץ: "כמות תחמושת"
הקשר / דוגמה: "אם יש מחסנית ריקה, נכתוב 'מחסנית ריקה ובה 0 כדורים'"
סוג השדה: טקסט חופשי.

דוגמאות מתוך נתוני אמת:
• "מחסנית ובה 10 כדורים"
• "מחסנית ובה 7 כדורים"
• "מחסנית ובה 0 כדורים"
• "2 מחסניות, 11 כדורים"
• "3 מחסניות"
• "75 כדורים"
• "כדור"
• "מחסנית + 17 כדורים"
• "מחסנית + 2 כדורים"
• "מחסנית ובה 0 כדורים, 2 מחסניות"
• "מחסנית ריקה ובה 0 כדורים + 1 כדור"
• "מחסנית ריקה ובה 0 כדורים + מחסנית ריקה ובה 0 כדורים + 26 כדורים"
• "מספר מחסניות"
• "2 מחסניות, מאות כדורים, 2 מחסניות ובהן 0 כדורים"

כללים:
- רשום מחסניות וכדורים כפי שמתוארים בגזר הדין.
- מחסנית ריקה → "מחסנית ריקה ובה 0 כדורים" או "מחסנית ובה 0 כדורים".
- השתמש ב-"כדורים" (לא "קליעים" או "תחמושת").
- אם אין תחמושת כלל → השאר ריק.

טקסט פסק הדין (קרא במלואו):
{text}

ענה בפורמט JSON:
{{"כמות תחמושת": "..."}}
אם אין תחמושת — מחרוזת ריקה.
"""
    resp = _call_gpt(prompt, WEAPON_SYSTEM, model=model, json_schema=_schema_weapon_ammunition())
    data = _loads_structured(resp)
    return (data.get("כמות תחמושת") or "").strip()


def _extract_weapon_purpose(text: str, model: str) -> str:
    """מטרה-סיבת העבירה

    GT: קטגוריות מוגדרות; אם יש סחר והמטרה לא צוינה — ברירת מחדל בצע כסף
    """
    prompt = f"""{WEAPON_USER_PREFIX}

שדה לחילוץ: "מטרה-סיבת העבירה"
הקשר / דוגמה: "..להחזקה עצמית..", "..להתרברב בחתונות.."
         "אם יש סחר, המטרה הדיפולטיבית הינו בצע כסף"
סוג השדה: בחירה יחידה (אפשר גם אחר).

האפשרויות:
○ בצע כסף — מטרה כספית: סחר בנשק, מכירה, שוד, סחיטה. ברירת מחדל כאשר יש סחר בנשק.
○ הגנה עצמית — הנאשם טען שהנשק לצורך הגנה/ביטחון אישי
○ חתונה — ירי באוויר בחתונה/חגיגה/אירוע שמח
○ סכסוך — הנשק קשור לסכסוך, ריב, נקמה, עימות
○ תדמית — צורך חברתי, הפגנת כוח, יוקרה, הרתעה

⚠️ בחר **רק** אחת מהאפשרויות למעלה. **אל תוסיף "אחר"** ואל תכתוב טקסט חופשי — אם התיאור דומה לאחת הקטגוריות, בחר אותה בלבד.

כללים:
- התבסס רק על מה שכתוב בגזר הדין.
- אם יש סחר בנשק והמטרה לא צוינה → ברירת מחדל: "בצע כסף"
- אם אין מידע כלל על המטרה → "לא צוין"
- אם אין התאמה טובה — בחר את **הקרובה ביותר** מהרשימה ב-enum

טקסט פסק הדין (קרא במלואו):
{text}

ענה בפורמט JSON:
{{"מטרה-סיבת העבירה": "..."}}
"""
    resp = _call_gpt(prompt, WEAPON_SYSTEM, model=model, json_schema=_schema_weapon_purpose())
    data = _loads_structured(resp)
    v = data.get("מטרה-סיבת העבירה", "לא צוין")
    return "" if v == "לא צוין" else v


def _extract_weapon_additional_offenses(text: str, model: str) -> str:
    """עבירות נוספות

    GT: טקסט חופשי (אופציונלי)
    """
    prompt = f"""{WEAPON_USER_PREFIX}

שדה לחילוץ: "עבירות נוספות"
סוג השדה: טקסט חופשי (לא חובה).

הסבר: רשום עבירות נוספות שהנאשם הורשע בהן, מעבר לעבירת הנשק (סעיף 144).

דוגמאות מתוך נתוני אמת:
• "קשירת קשר לביצוע פשע"
• "ירי מנשק חם"
• "ירי מנשק חם באזור מגורים"
• "הפרעה לשוטר במילוי תפקידו לפי סעיף 275 לחוק"
• "נהיגה ללא רישיון, נהיגה ללא ביטוח וללא רישיון רכב"
• "עבירות של פציעה, לפי סעיף 334 לחוק העונשין, התשל"ז-1977"
• "סיוע לנסיון לתקיפה לפי סעיפים 379 + 25 + 31 לחוק"
• "ניסיון לחבלה חמורה, רכישת נשק"

כללים:
- רשום רק עבירות שאינן סעיף 144 (לא עבירת הנשק עצמה).
- כלול סעיפי חוק אם מוזכרים בטקסט.
- אם אין עבירות נוספות → השאר ריק.

טקסט פסק הדין (קרא במלואו):
{text}

ענה בפורמט JSON:
{{"עבירות נוספות": "..."}}
אם אין עבירות נוספות — מחרוזת ריקה.
"""
    resp = _call_gpt(prompt, WEAPON_SYSTEM, model=model, json_schema=_schema_weapon_additional_offenses())
    data = _loads_structured(resp)
    return (data.get("עבירות נוספות") or "").strip()


def _extract_weapon_planning(text: str, model: str) -> str:
    """תכנון

    GT: binary כן/לא, default=לא
    """
    prompt = f"""{WEAPON_USER_PREFIX}

שדה לחילוץ: "תכנון"
הקשר / דוגמה: "ברירת מחדל - לא"
סוג השדה: בחירה יחידה (כן/לא).

ענה:
• "כן" — יש ראיות לתכנון מראש: תכנון רכישת נשק, תכנון פיגוע/שוד, סחר מאורגן
• "לא" — אין ראיות לתכנון, מדובר בהחזקה/נשיאה בלבד (ברירת מחדל)

כללים:
- החזקת נשק בלבד ללא פעולה מתוכננת = "לא"
- סחר בנשק = "כן" (יש תכנון מסחרי)
- ירי מתוכנן / פיגוע / שוד = "כן"
- נשיאת/הובלת נשק בלבד = "לא"
- ברירת מחדל: "לא"

טקסט פסק הדין (קרא במלואו):
{text}

ענה בפורמט JSON:
{{"תכנון": "..."}}
"""
    resp = _call_gpt(prompt, WEAPON_SYSTEM, model=model, json_schema=_schema_weapon_planning())
    data = _loads_structured(resp)
    return data.get("תכנון", "לא")


def _extract_weapon_use(text: str, model: str) -> str:
    """שימוש

    GT: לא | כן,ירי | ניסיון לירי… | זריקת רימון | נקירה | מטען | אחר
    """
    prompt = f"""{WEAPON_USER_PREFIX}

שדה לחילוץ: "שימוש"
סוג השדה: בחירה יחידה (או אחר).

האפשרויות:
○ לא — לא השתמש בנשק בפועל, רק החזקה/נשיאה (ברירת מחדל)
○ כן,ירי — ירה בנשק (שים לב: בלי רווח אחרי הפסיק)
○ ניסיון לירי ללא הצלחה — ניסה לירות אך לא הצליח (תקלת נשק, עצר ברגע האחרון)
○ זריקת רימון — זרק רימון
○ נקירת נשק — נקירת נשק (הצגת הנשק כאיום)
○ הפעלת מטען — הפעלת מטען חבלה
○ אחר — שימוש אחר

דוגמאות מתוך נתוני אמת:
• "כן,ירי"
• "זריקת רימון"
• "ניסיון לירי ללא הצלחה, זריקת רימון"

כללים:
- אם מוזכר "ירי" או "ירה" → "כן,ירי"
- אם מוזכר זריקת רימון → "זריקת רימון"
- יכול להיות שילוב: "ניסיון לירי ללא הצלחה, זריקת רימון"
- רק החזקה/נשיאה ללא שימוש → "לא"

טקסט פסק הדין (קרא במלואו):
{text}

ענה בפורמט JSON:
{{"שימוש": "..."}}
"""
    resp = _call_gpt(prompt, WEAPON_SYSTEM, model=model, json_schema=_schema_weapon_use())
    data = _loads_structured(resp)
    return (data.get("שימוש") or "לא").strip()


def extract_weapon_features(text: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """Extract all weapon features in direct manual-format Hebrew keys.

    Only keys with non-empty values are included (matching manual sparsity).
    """
    features: Dict[str, Any] = {}

    offense_num = _extract_weapon_offense_number(text, model)
    if offense_num:
        features["מספר עבירה"] = offense_num

    features.update(_extract_weapon_offense_type_flags(text, model))

    weapon_types = _extract_weapon_types(text, model)
    features.update(weapon_types)

    status = _extract_weapon_status(text, model)
    features["סטטוס הנשק"] = status

    features.update(_extract_weapon_storage_flags(text, model))

    how_obtained = _extract_weapon_how_obtained(text, model)
    if how_obtained:
        features["אופן קבלת הנשק"] = how_obtained

    ammo = _extract_weapon_ammunition(text, model)
    if ammo:
        features["כמות תחמושת"] = ammo

    purpose = _extract_weapon_purpose(text, model)
    if purpose:
        features["מטרה-סיבת העבירה"] = purpose

    additional = _extract_weapon_additional_offenses(text, model)
    if additional:
        features["עבירות נוספות"] = additional

    planning = _extract_weapon_planning(text, model)
    features["תכנון"] = planning

    use = _extract_weapon_use(text, model)
    if use and use != "לא":
        features["שימוש"] = use

    return _strip_empty_feature_values(features)


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def _json_default(o: Any):
    if hasattr(o, "item") and callable(o.item):
        try:
            return o.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def get_unique_verdicts(manual_csv_path: str) -> Set[str]:
    df = pd.read_csv(manual_csv_path)
    v1_col = "verdict_1" if "verdict_1" in df.columns else df.columns[0]
    v2_col = "verdict_2" if "verdict_2" in df.columns else df.columns[1]
    return set(df[v1_col].unique()) | set(df[v2_col].unique())


def build_verdict_lookup(verdict_csv_dir: str, verdict_ids: Set[str]) -> Dict[str, str]:
    lookup = {}
    vdir = Path(verdict_csv_dir)
    for vid in verdict_ids:
        fpath = vdir / f"{vid}.csv"
        if not fpath.exists():
            continue
        try:
            df = pd.read_csv(fpath)
            full_text = "\n".join(df["text"].dropna().astype(str).tolist())
            lookup[vid] = full_text
        except Exception as e:
            print(f"  Warning: could not read {fpath}: {e}")
    return lookup


def run_extraction(
    domain: str,
    base_path: str,
    checkpoint_every: int = 10,
    model: str = DEFAULT_MODEL,
    sleep_between: float = 0.5,
    max_verdicts: Optional[int] = None,
    artifact_tag: Optional[str] = None,
    refetch: bool = False,
) -> str:
    print(f"\n{'='*60}")
    print(f"Feature Extraction (Manual Format) — {domain.upper()}")
    print(f"  model={model}")
    print(f"  refetch={refetch}")
    if max_verdicts is not None:
        print(f"  max_verdicts={max_verdicts}")
    print(f"{'='*60}\n")

    tag = artifact_tag
    if tag is None and max_verdicts is not None:
        tag = f"smoke{max_verdicts}"
    tag_suffix = f"_{tag}" if tag else ""

    manual_csv = Path(base_path) / "similarity_database_fe.csv"
    verdict_csv_dir = Path(base_path) / "verdict_csv"
    output_csv = Path(base_path) / f"similarity_database_fe_manual_format{tag_suffix}.csv"
    cache_path = Path(base_path) / CACHE_FILENAME.replace(".json", f"{tag_suffix}.json")

    print(f"  Manual CSV (GT): {manual_csv}")
    print(f"  Verdict CSV dir: {verdict_csv_dir}")
    print(f"  Output CSV:      {output_csv}")
    print(f"  Cache:           {cache_path}")

    feature_cache: Dict[str, Any] = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            feature_cache = json.load(f)
        print(f"  Loaded {len(feature_cache)} cached features")

    df_manual = pd.read_csv(manual_csv)
    v1_col = "verdict_1" if "verdict_1" in df_manual.columns else df_manual.columns[0]
    v2_col = "verdict_2" if "verdict_2" in df_manual.columns else df_manual.columns[1]
    unique_verdicts = set(df_manual[v1_col].unique()) | set(df_manual[v2_col].unique())
    print(f"  Found {len(unique_verdicts)} unique verdicts in GT")

    extraction_set = set(unique_verdicts)
    if max_verdicts is not None:
        picked: List[str] = []
        seen: Set[str] = set()
        for _, row in df_manual.iterrows():
            for col in (v1_col, v2_col):
                vid = row[col]
                if vid not in seen:
                    seen.add(vid)
                    picked.append(vid)
                    if len(picked) >= max_verdicts:
                        break
            if len(picked) >= max_verdicts:
                break
        extraction_set = set(picked[:max_verdicts])
        print(f"  Limited to {len(extraction_set)} verdict IDs")

    verdict_lookup = build_verdict_lookup(str(verdict_csv_dir), extraction_set)
    print(f"  Full verdict text available: {len(verdict_lookup)} / {len(extraction_set)}")
    missing = extraction_set - set(verdict_lookup.keys())
    if missing:
        print(f"  Missing verdict files: {missing}")

    if refetch:
        n_drop = sum(1 for vid in extraction_set if vid in feature_cache)
        for vid in extraction_set:
            feature_cache.pop(vid, None)
        print(f"  Refetch: removed {n_drop} entries from cache")

    extract_fn = extract_drugs_features if domain == "drugs" else extract_weapon_features
    to_process = [v for v in sorted(extraction_set) if v not in feature_cache and v in verdict_lookup]
    print(f"  Verdicts to process: {len(to_process)}")
    print(f"  Already cached:      {len([v for v in extraction_set if v in feature_cache])}")

    new_count = 0
    for vid in tqdm(to_process, desc=f"Extracting {domain}"):
        text = verdict_lookup.get(vid, "")
        if not text:
            continue
        try:
            feature_cache[vid] = extract_fn(text, model)
        except Exception as e:
            print(f"\n  Failed {vid}: {e}")
            feature_cache[vid] = {}

        new_count += 1
        if new_count % checkpoint_every == 0:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(feature_cache, f, ensure_ascii=False, indent=2, default=_json_default)
            print(f"\n  Checkpoint: {len(feature_cache)} verdicts cached")

        if sleep_between > 0:
            time.sleep(sleep_between)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(feature_cache, f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"\n  Final cache: {len(feature_cache)} verdicts")

    print("  Building output CSV...")
    if max_verdicts is not None:
        mask = df_manual[v1_col].isin(extraction_set) & df_manual[v2_col].isin(extraction_set)
        df_pair = df_manual.loc[mask].copy()
    else:
        df_pair = df_manual

    records = []
    for _, row in df_pair.iterrows():
        v1 = row[v1_col]
        v2 = row[v2_col]
        feat1 = feature_cache.get(v1, {})
        feat2 = feature_cache.get(v2, {})
        records.append({
            "verdict_1": v1,
            "verdict_2": v2,
            "similarity_scale": row["similarity_scale"],
            "similarity_binary_0": row["similarity_binary_0"],
            "similarity_binary_1": row["similarity_binary_1"],
            "feature_vector_1": json.dumps(feat1, ensure_ascii=False, default=_json_default),
            "feature_vector_2": json.dumps(feat2, ensure_ascii=False, default=_json_default),
        })

    df_out = pd.DataFrame(records)
    df_out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n  Output saved: {output_csv}")
    print(f"  Total pairs:  {len(df_out)}")
    return str(output_csv)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

BASE_PATHS = {
    "weapon": "/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/weapon/",
    "drugs": "/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/drugs/",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract features in manual GT format (Hebrew keys)"
    )
    parser.add_argument("--domain", choices=["weapon", "drugs", "both"], default="both")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--max-verdicts", type=int, default=None)
    parser.add_argument("--artifact-tag", type=str, default=None)
    parser.add_argument("--refetch", action="store_true")
    args = parser.parse_args()

    domains = ["weapon", "drugs"] if args.domain == "both" else [args.domain]
    output_paths = {}
    for dom in domains:
        output_paths[dom] = run_extraction(
            dom, BASE_PATHS[dom],
            checkpoint_every=args.checkpoint,
            model=args.model,
            sleep_between=args.sleep,
            max_verdicts=args.max_verdicts,
            artifact_tag=args.artifact_tag,
            refetch=args.refetch,
        )

    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print("="*60)
    for dom, path in output_paths.items():
        print(f"  {dom}: {path}")
