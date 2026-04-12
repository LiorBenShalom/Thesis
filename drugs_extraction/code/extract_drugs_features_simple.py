"""
Simple drugs case feature extraction from docx files using GPT-4.1.
One GPT call per feature, structured JSON schema output.
Mirrors the structure of extract_weapon_features_simple.py.
"""

import json
import os
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception: pass
import sys
import csv
from openai import OpenAI

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    os.environ.get("OPENAI_API_KEY", ""),
)
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL = "gpt-4.1"
SYSTEM_PROMPT = """אתה עוזר משפטי מומחה בניתוח כתבי אישום ופסקי דין בתיקי סמים בעברית.
קרא בדקדקנות את תוכן התיק והיצמד להוראות בדיוק. תן תשובות מדויקות בפורמט JSON בלבד.

כלל קריטי: חלץ מידע אך ורק מעובדות כתב האישום ומדברי השופט בגזר הדין.
אל תחלץ מידע מתסקיר שירות המבחן, מטיעוני הסנגור, או מכל מקור אחר.
אם מידע מסוים לא מופיע בעובדות כתב האישום או בדברי השופט - הוא לא קיים לצורך החילוץ.
חלץ רק מידע על העבירה של הנאשם עצמו, לא של נאשמים אחרים."""


# ── helpers ──────────────────────────────────────────────────────────────

def read_docx(path: str) -> str:
    """
    Read docx text using lxml to capture text from Word SmartTag elements
    (metric values that python-docx's high-level API misses).
    """
    import zipfile
    from lxml import etree

    W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    with zipfile.ZipFile(path) as z:
        xml_bytes = z.read("word/document.xml")
    root = etree.fromstring(xml_bytes)

    lines = []
    for para in root.iter(f"{{{W}}}p"):
        parts = []
        for node in para.iter():
            if node.tag == f"{{{W}}}t":
                parts.append(node.text or "")
        text = "".join(parts).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _call_gpt(prompt: str, schema: dict, schema_name: str, max_tokens: int = 500) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=max_tokens,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        },
    )
    return json.loads(resp.choices[0].message.content)


# ── JSON schemas per feature ─────────────────────────────────────────────

SCHEMA_OFFENSE_NUMBER = {
    "type": "object",
    "properties": {
        "סעיף_6":  {"type": "string", "description": "ריק '' אם הסעיף לא הופיע. אחרת '1' אם הופיע ללא אות, או האות/האותיות שמלוות אותו (למשל 'א', 'ג', 'א_ג'). סעיף 6 = ייצור."},
        "סעיף_7":  {"type": "string", "description": "ריק '' אם הסעיף לא הופיע. אחרת '1' או האות (למשל 'א', 'ג', 'א_ג'). סעיף 7 = החזקה/שימוש."},
        "סעיף_10": {"type": "string", "description": "ריק '' אם הסעיף לא הופיע. אחרת '1' או האות. סעיף 10 = ניסיון."},
        "סעיף_13": {"type": "string", "description": "ריק '' אם הסעיף לא הופיע. אחרת '1' או האות. סעיף 13 = ייצוא/ייבוא/סחר."},
        "סעיף_14": {"type": "string", "description": "ריק '' אם הסעיף לא הופיע. אחרת '1' או האות. סעיף 14 = הספקה/אספקה."},
        "סעיף_19": {"type": "string", "description": "ריק '' אם הסעיף לא הופיע. אחרת '1' או האות. סעיף 19 = החזקה שלא לצריכה עצמית. אם מופיע 19א - לסמן 'א'."},
        "סעיף_21": {"type": "string", "description": "ריק '' אם הסעיף לא הופיע. אחרת '1' או האות. סעיף 21 = הכנת מקום/דירה לסמים."},
        "סעיף_22": {"type": "string", "description": "ריק '' אם הסעיף לא הופיע. אחרת '1' או האות. סעיף 22 = הסתת קטין/מתן סם לקטין."},
        "סעיף_61": {"type": "string", "description": "ריק '' אם הסעיף לא הופיע. אחרת '1' או האות (למשל 'א_ג' עבור 61(א)(ג)). סעיף 61 לחוק העונשין."},
        "עבירת_סמים_אחרת": {"type": "string", "description": "מלל חופשי קצר רק אם הנאשם הואשם בעבירת סמים שאינה ברשימה לעיל. מחרוזת ריקה אם אין."},
    },
    "required": ["סעיף_6", "סעיף_7", "סעיף_10", "סעיף_13", "סעיף_14", "סעיף_19", "סעיף_21", "סעיף_22", "סעיף_61", "עבירת_סמים_אחרת"],
    "additionalProperties": False,
}

SCHEMA_SIDE_OFFENSES = {
    "type": "object",
    "properties": {
        "עבירות_נלוות": {"type": "integer", "description": "1 אם הנאשם הואשם בעבירות נוספות שאינן עבירות סמים (לדוגמה: נשק, פציעה, גניבה, הפרת אמונים), 0 אחרת."},
    },
    "required": ["עבירות_נלוות"],
    "additionalProperties": False,
}

SCHEMA_DRUG_TYPE = {
    "type": "object",
    "properties": {
        "LSD":             {"type": "array", "items": {"type": "string"}, "description": "רשימת כמויות LSD שנמצאו ברשות הנאשם, בפורמט 'מספר-יחידה' (למשל '154-בולים' או '1.75-מיליליטר')."},
        "METHAMPHETAMINE": {"type": "array", "items": {"type": "string"}, "description": "רשימת כמויות מתאמפטמין בפורמט 'מספר-יחידה' (למשל '1865-גרם')."},
        "האיוואסקה":       {"type": "array", "items": {"type": "string"}, "description": "רשימת כמויות איוואסקה בפורמט 'מספר-יחידה'."},
        "קתינון":          {"type": "array", "items": {"type": "string"}, "description": "רשימת כמויות קתינון בפורמט 'מספר-יחידה'."},
        "קטמין":           {"type": "array", "items": {"type": "string"}, "description": "רשימת כמויות קטמין בפורמט 'מספר-יחידה' (למשל '700-גרם')."},
        "חשיש":            {"type": "array", "items": {"type": "string"}, "description": "רשימת כמויות חשיש בפורמט 'מספר-יחידה'."},
        "מתילמקאתינון":    {"type": "array", "items": {"type": "string"}, "description": "רשימת כמויות מתילמקאתינון בפורמט 'מספר-יחידה'."},
        "קנבוס_בשתילים":   {"type": "array", "items": {"type": "string"}, "description": "שתילי קנבוס חיים/שתולים בעציצים. אם יש משקל בגרם → 'X-גרם'. אם רק ספירה → 'X-שתילים'. קנבוס מעובד/יבש/נמכר → שייך ל'קנבוס'."},
        "קנבוס":           {"type": "array", "items": {"type": "string"}, "description": "קנבוס מעובד/יבש/פירורים/נמכר (לא שתילים חיים). פורמט 'מספר-יחידה' (בעיקר גרם)."},
        "MDMA":            {"type": "array", "items": {"type": "string"}, "description": "רשימת כמויות MDMA בפורמט 'מספר-יחידה' (יכול להיות 'X-טבליות' וגם 'Y-גרם' באותה רשימה)."},
        "קוקאין":          {"type": "array", "items": {"type": "string"}, "description": "רשימת כמויות קוקאין בפורמט 'מספר-יחידה' (למשל '69.09-גרם')."},
        "אחר":             {"type": "array", "items": {"type": "string"}, "description": "רשימה לסמים שאינם ברשימה לעיל. כל איבר בפורמט 'שם_הסם:מספר-יחידה'. רשימה ריקה אם הכל ברשימה."},
    },
    "required": ["LSD", "METHAMPHETAMINE", "האיוואסקה", "קתינון", "קטמין", "חשיש", "מתילמקאתינון", "קנבוס_בשתילים", "קנבוס", "MDMA", "קוקאין", "אחר"],
    "additionalProperties": False,
}

SCHEMA_LAB = {
    "type": "object",
    "properties": {
        "מעבדה": {"type": "integer", "description": "1 אם מתוארת מעבדת סמים, או כלים שמרמזים על קיומה של מעבדה (לדוגמה: מאזניים אלקטרוניים בלבד אינם מעבדה; ציוד עיבוד/הפקה/בידוד כן מעיד על מעבדה). ברירת מחדל: 0."},
    },
    "required": ["מעבדה"],
    "additionalProperties": False,
}

SCHEMA_ROLE = {
    "type": "object",
    "properties": {
        "בעל_הסמים":  {"type": "integer", "description": "1 אם הסמים בבעלות הנאשם או שהוא היצרן/המוכר. 0 רק אם נאמר/נרמז במפורש שהסמים הועברו אליו לצורך הובלה/שמירה/סיוע ואינם שלו. ברירת מחדל: 1."},
        "בעל_המעבדה": {"type": "integer", "description": "1 אם קיימת מעבדה והיא בבעלות הנאשם. 0 אם אין מעבדה כלל, או אם המעבדה קיימת אך נאמר/נרמז שאינה בבעלותו. ברירת מחדל: 1 אם יש מעבדה, 0 אם אין."},
    },
    "required": ["בעל_הסמים", "בעל_המעבדה"],
    "additionalProperties": False,
}

SCHEMA_UNDERCOVER = {
    "type": "object",
    "properties": {
        "מכירה_לסוכן": {"type": "integer", "description": "1 אם מתוארת מכירת סמים לסוכן משטרתי / סוכן סמוי / מתחזה. ברירת מחדל: 0."},
    },
    "required": ["מכירה_לסוכן"],
    "additionalProperties": False,
}


# ── per-feature extraction functions ─────────────────────────────────────

def extract_offense_number(text: str) -> dict:
    prompt = f"""קרא בדקדקנות את גזר הדין הבא והיצמד להוראות.
חלץ את סעיפי החוק שבהם הואשם הנאשם בלבד (לפי פקודת הסמים המסוכנים בעיקר).

דגשים:
- רק לפי העבירות שהואשם בהן הנאשם! התעלם מסעיפים של נאשמים אחרים.
- רק מעובדות כתב האישום או דברי השופט.
- כל סעיף יכול להופיע עם אות לידו (למשל "סעיף 19א", "סעיף 7(א)(ג)", "סעיף 6(ב)").
- אם הסעיף הופיע ללא אות - השדה מקבל ערך '1'.
- אם הסעיף הופיע עם אות אחת - השדה מקבל את האות (למשל 'א').
- אם הסעיף הופיע עם שתי אותיות - השדה מקבל אותן עם קו תחתון (למשל 'א_ג' עבור 7(א)(ג)).
- אם הסעיף לא הופיע כלל - השדה ריק ''.
- שים לב: סעיף 19 ו-19א נחשבים שניהם 'סעיף_19' עם האות 'א' (לא קיים שדה נפרד ל-19א).

שדה עבירת_סמים_אחרת:
- מלא רק אם הנאשם הואשם בעבירת סמים שאינה ברשימה לעיל (למשל סעיף שאינו מופיע).
- אל תכלול עבירות שאינן עבירות סמים (אלו שייכות ל"עבירות נלוות", לא כאן).
- מחרוזת ריקה אם אין.

תוכן התיק:
{text}"""
    return _call_gpt(prompt, SCHEMA_OFFENSE_NUMBER, "offense_number", max_tokens=400)


def extract_side_offenses(text: str) -> dict:
    prompt = f"""קרא בדקדקנות את גזר הדין הבא והיצמד להוראות.
האם הנאשם הואשם בעבירות שאינן עבירות סמים?

דגשים:
- בדוק את כתב האישום הראשי וכן כל תיק שצורף לגזר הדין (תיק מצורף / ת"פ מצורף).
- עבירות נלוות אופייניות: נשק, פציעה, גניבה, הסעת שוהה בלתי חוקי, הפרת אמונים, אלימות, איומים, קשירת קשר לפי חוק העונשין, עבירות תעבורה, וכו'.
- קשירת קשר לפי סעיף 499 לחוק העונשין (לא לפי פקודת הסמים) = עבירה נלווית.
- עבירות לפי פקודת הסמים המסוכנים בלבד אינן עבירות נלוות.
- אם בתיק מצורף יש עבירה שאינה סמים — זה נחשב כן.
- 1 אם יש לפחות עבירה אחת שאינה עבירת סמים (בכתב האישום הראשי או בתיק מצורף), 0 אחרת.

תוכן התיק:
{text}"""
    return _call_gpt(prompt, SCHEMA_SIDE_OFFENSES, "side_offenses", max_tokens=200)


def extract_drug_type(text: str) -> dict:
    prompt = f"""קרא בדקדקנות את גזר הדין הבא והיצמד להוראות.
חלץ את סוגי הסמים והכמויות שהיו קשורים בעבירה של הנאשם עצמו בכתב האישום הנוכחי.

דגשים כלליים:
- רק מעובדות כתב האישום הנוכחי או דברי השופט. רק לגבי הנאשם עצמו - לא נאשמים אחרים.
- כל סוג סם מקבל רשימה של מחרוזות בפורמט 'מספר-יחידה'. אם אין - רשימה ריקה [].
- היחידות המקובלות: גרם, ק"ג, מיליליטר, טבליות, בולים, שתילים, יחידות.
- ק"ג → המר לגרם (×1000) ואחד עם שאר הגרמים לערך אחד.

כלל קריטי - לא מתיקים קודמים / פסיקה מצוטטת:
- אם מופיע בתיק תיאור של תיקים קודמים של הנאשם (לדוגמה: "הרשעות קודמות", "תיק שנדון בעבר", "כתב אישום קודם", "חזרה לסורו", "עברו הפלילי"), **התעלם לחלוטין מהסמים והכמויות שמופיעים שם**.
- חלץ אך ורק סמים המופיעים בעובדות כתב האישום הנוכחי שעליו ניתן גזר הדין.
- גם תסקיר שירות המבחן ופסיקה מצוטטת (פסקי דין אחרים שמצוטטים) אינם רלוונטיים - אל תחלץ מהם סמים.


כלל קריטי - ריבוי עבירות / אישומים בכתב האישום הנוכחי:
- אם בכתב האישום הנוכחי יש כמה אישומים (אישום ראשון/שני/שלישי) או ריבוי עסקאות → סכם את הסמים מ**כל האישומים והעסקאות יחד** באותה קטגוריה ויחידה.
- אותו סם + אותה יחידה (גרם או טבליות וכו') = איבר אחד מסוכם. דוגמה: אישום 1: קוקאין 20.09 גרם + אישום 2: קוקאין 49 גרם → קוקאין: ['69.09-גרם'] (לא שני איברים).
- כלל קריטי — גם אם בתיק מפורטות כמויות רבות של אותו סם (למשל תפיסות שונות, חלקים שונים) — **תמיד סכם לאיבר אחד** באותה יחידה. אסור להחזיר רשימה של כמויות נפרדות מאותו סוג ויחידה.
- אותו סם + יחידות שונות = איברים נפרדים. דוגמה: MDMA 1301.44 גרם + MDMA 13189 טבליות → ['1301.44-גרם', '13189-טבליות'].

כלל קריטי - בחירת הערך כשיש גם כמות וגם משקל:
- אם לאותו סם/אותם פריטים מופיעים גם כמות יחידות וגם משקל בגרם → רשום **רק** את המשקל בגרם, **אל תוסיף** גם את הספירה.
- דוגמה: "20 שתילי קנבוס במשקל 400 גרם" → קנבוס_בשתילים: ['400-גרם'] בלבד. **לא** ['400-גרם', '20-שתילים'].
- דוגמה: "2 יחידות מתילמקאתינון במשקל 1.906 גרם" → מתילמקאתינון: ['1.906-גרם'] בלבד. **לא** ['1.906-גרם', '2-יחידות'].
- דוגמה: "1246 שתילים... ק"ג נטו קנבוס בכמות 140 ק"ג" → כל ה-140 ק"ג הם הסם הסופי (הפקה/ייבוש), לא שתילים → קנבוס: ['140000-גרם'] (מכיוון שהסם כבר מעובד).
- **חריג**: אם יש שני ערכים של **צורות שונות** של אותו סם (למשל: MDMA בטבליות + MDMA נוזלי בגרם) → שני איברים נפרדים כי מדובר בשתי צורות שונות.
- שים לב: זו בחירת **ערך** (משקל ולא ספירה), לא בחירת **קטגוריה**. הקטגוריה נקבעת לפי הצורה הפיזית של הסם.

כלל קריטי - לקרוא את המשפט/הפסקה עד הסוף:
- לעיתים תיאור הסם מתחיל בכמות שתילים/יחידות ומסתיים בסוף המשפט או הפסקה במשקל הכולל.
- חובה לקרוא את המשפט עד סופו ואת הפסקה כולה. לפעמים יש סיכום של כל הסמים בסוף.

כלל קריטי - הבחנה בין 'קנבוס' ל-'קנבוס_בשתילים' - שתי קטגוריות נפרדות לחלוטין:
הדפוס פשוט וחד:
- אם בכתב האישום מתואר **שתילים חיים / שתולים בעציצים / גדלים** (כלומר: בעת התפיסה הם עדיין שתילים, לא יובשו) → הקטגוריה היא **קנבוס_בשתילים**.
- בתוך הקטגוריה 'קנבוס_בשתילים', בחירת **הערך**:
  * אם מתואר **משקל בגרם/ק"ג לשתילים** → רשום את המשקל בגרם. דוגמה: "170 ק"ג שהיה שתול במאות עציצים" → קנבוס_בשתילים: ['170000-גרם'].
  * אם מתואר **רק ספירת שתילים בלי משקל** → רשום את הספירה בשתילים. דוגמה: "5 שתילי קנבוס" → קנבוס_בשתילים: ['5-שתילים'].
- אם מתואר קנבוס **שאינו בצורת שתיל** (מעובד / יבש / מפוזר / נמכר / נתפס באריזה) → הקטגוריה היא **קנבוס**, עם משקל בגרם.
- **מקרה מיוחד - מעבדה שגידלה שתילים ויבשה אותם**: אם השתילים **כבר יובשו/הופקו** ומתואר משקל של סם מסוג קנבוס (לדוגמה: "גידלו שתילים... יבשו... סם קנבוס בכמות 140 ק"ג"), ובזמן התפיסה אין עוד שתילים חיים - רשום רק את הכמות הסופית כ-קנבוס, ולא את ספירת השתילים המקורית.
- **לעולם אל תאחד בין שתי הקטגוריות.** אם בתיק יש גם שתילים חיים וגם קנבוס מעובד - שתי רשומות נפרדות. דוגמה: "180 ק\"ג שתילי קנבוס + 7.48 ק\"ג קנבוס מעובד" → קנבוס_בשתילים: ['180000-גרם'], קנבוס: ['7480-גרם']. **לא** לאחד.
- שדה 'קנבוס' לא יכיל לעולם 'שתילים' כיחידה. שדה 'קנבוס_בשתילים' יכיל גרם או שתילים בלבד.

כלל פורמט:
- רק ספרות (לא 'שלושה' / 'מאה'): '154-בולים', '71.52-גרם', '50-טבליות'.
- בלי פסיקים בתוך מספר ('1255.48-גרם' ולא '1,255.48-גרם').
- ללא רווחים בתוך הפורמט.

שדה 'אחר':
- רק אם יש סם שאינו ברשימת ה-11 הסוגים. כתוב כל איבר בפורמט 'שם_הסם:מספר-יחידה'.
- רשימה ריקה אם כל הסמים נמצאו ברשימה.

תוכן התיק:
{text}"""
    return _call_gpt(prompt, SCHEMA_DRUG_TYPE, "drug_type", max_tokens=1200)


def extract_lab(text: str) -> dict:
    prompt = f"""קרא בדקדקנות את גזר הדין הבא והיצמד להוראות.
האם מתוארת מעבדת סמים, או כלים שמרמזים על קיומה של מעבדה?

דגשים:
- רק מעובדות כתב האישום או דברי השופט.
- מעבדה = מקום/ציוד לעיבוד/הפקה/בידוד/ייצור של סם (לדוגמה: ציוד זיקוק, כלי כימיה, חומרי גלם, מיכלי תהליך, תאי גידול מתוחכמים).
- מאזניים אלקטרוניים, שקיות אריזה, או חיתוך/אריזה לבד - **אינם** מעבדה.
- שתילי קנבוס בעציצים בבית - אלו לא מעבדה אלא גידול.
- ברירת מחדל: 0. רק אם יש סימנים ברורים למעבדה - 1.

תוכן התיק:
{text}"""
    return _call_gpt(prompt, SCHEMA_LAB, "lab", max_tokens=200)


def extract_role(text: str) -> dict:
    prompt = f"""קרא בדקדקנות את גזר הדין הבא והיצמד להוראות.
חלץ את תפקיד הנאשם ביחס לסמים ולמעבדה.

דגשים:
- רק מעובדות כתב האישום או דברי השופט.

בעל_הסמים:
- ברירת מחדל: 1.
- 0 רק אם ברור מהעובדות שהנאשם פעל כשליח/מתווך/נושא עבור אחר, והסמים אינם שלו.

מקרים שבהם בעל_הסמים=0:
- הנאשם ייבא/הביא סמים עבור אחר (מזמין/שולח) — הסמים שייכים לאחר, הנאשם רק הוביל.
- הנאשם קיבל מלאי מאחר על מנת לבצע עסקאות עבורו — "האחר העביר לנאשם סמים כמלאי".
- הנאשם מכר/העביר סמים שקיבל מאחר ומסר לו את התמורה — מתווך/שליח.
- נאמר במפורש שהסמים אינם שייכים לנאשם (של מישהו אחר, שכירות, פיקדון).

מקרים שבהם בעל_הסמים=1:
- הנאשם הוא היצרן / הגדל / המוכר העצמאי.
- הנאשם קנה את הסמים לצורך עצמו (שימוש עצמי או מכירה עצמאית).
- הנאשם החזיק סמים בביתו לצריכה עצמית.
- המכירה לסוכן בוצעה על ידי הנאשם עצמו מרצונו ולטובתו (גם אם הסם הגיע מספק).

בעל_המעבדה:
- אם אין בכלל מעבדה בתיק → בעל_המעבדה=0.
- אם יש מעבדה: ברירת מחדל=1, 0 רק אם נאמר/נרמז במפורש שהמעבדה אינה בבעלותו (לדוגמה: של שותף, שכרה אחר).

תוכן התיק:
{text}"""
    return _call_gpt(prompt, SCHEMA_ROLE, "role", max_tokens=250)


def extract_undercover(text: str) -> dict:
    prompt = f"""קרא בדקדקנות את גזר הדין הבא והיצמד להוראות.
האם מתוארת מכירת סמים לסוכן משטרתי?

דגשים:
- רק מעובדות כתב האישום או דברי השופט.
- מכירה לסוכן = מכירה ל"סוכן משטרתי", "סוכן סמוי", "מתחזה" של המשטרה.
- מסירה לסוכן בתמורה לכסף או בתמורה לדבר אחר נחשבת מכירה.
- ניסיון מכירה שנקטע נחשב גם כן.
- ברירת מחדל: 0.

תוכן התיק:
{text}"""
    return _call_gpt(prompt, SCHEMA_UNDERCOVER, "undercover", max_tokens=200)


# ── main extraction ─────────────────────────────────────────────────────

FEATURE_EXTRACTORS = [
    ("מספר_עבירה",     extract_offense_number),
    ("עבירות_נלוות",   extract_side_offenses),
    ("סוג_הסם",         extract_drug_type),
    ("מעבדה",           extract_lab),
    ("תפקיד",           extract_role),
    ("מכירה_לסוכן",    extract_undercover),
]


def extract_all_features(text: str) -> dict:
    results = {}
    for name, func in FEATURE_EXTRACTORS:
        print(f"  [{name}]...", end=" ", flush=True)
        try:
            results[name] = func(text)
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            results[name] = None
    return results


def process_single(docx_path: str) -> dict:
    print(f"Reading: {docx_path}")
    text = read_docx(docx_path)
    print(f"  Text length: {len(text)} chars")
    return extract_all_features(text)


def _flatten_value(key: str, val, flat: dict) -> None:
    """Flatten a feature value into csv columns. Lists are JSON-encoded."""
    if isinstance(val, dict):
        for sub_key, sub_val in val.items():
            if isinstance(sub_val, list):
                flat[f"{key}__{sub_key}"] = json.dumps(sub_val, ensure_ascii=False)
            else:
                flat[f"{key}__{sub_key}"] = sub_val
    elif isinstance(val, list):
        flat[key] = json.dumps(val, ensure_ascii=False)
    else:
        flat[key] = val


def process_directory(dir_path: str, output_csv: str):
    docx_files = sorted(
        f for f in os.listdir(dir_path) if f.endswith(".docx") and not f.startswith("~")
    )
    if not docx_files:
        print(f"No docx files found in {dir_path}")
        return

    print(f"Found {len(docx_files)} docx files")
    results = []

    for fname in docx_files:
        path = os.path.join(dir_path, fname)
        try:
            features = process_single(path)
            features["_filename"] = fname
            results.append(features)
        except Exception as e:
            print(f"  ERROR on {fname}: {e}")
            results.append({"_filename": fname, "_error": str(e)})

    # Flatten nested dicts for CSV
    if results:
        flat_rows = []
        for r in results:
            flat = {"filename": r.get("_filename", "")}
            for key, val in r.items():
                if key.startswith("_"):
                    continue
                _flatten_value(key, val, flat)
            flat_rows.append(flat)

        all_cols = []
        for row in flat_rows:
            for k in row:
                if k not in all_cols:
                    all_cols.append(k)

        with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=all_cols)
            writer.writeheader()
            writer.writerows(flat_rows)

        print(f"\nResults saved to {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file:    python extract_drugs_features_simple.py file.docx")
        print("  Directory:      python extract_drugs_features_simple.py dir/ output.csv")
        sys.exit(1)

    path = sys.argv[1]

    if os.path.isfile(path):
        features = process_single(path)
        print("\n" + json.dumps(features, ensure_ascii=False, indent=2))
    elif os.path.isdir(path):
        output = sys.argv[2] if len(sys.argv) > 2 else "drugs_features_output.csv"
        process_directory(path, output)
    else:
        print(f"Path not found: {path}")
        sys.exit(1)
