"""
GPT-based feature extraction for drugs cases.
One call per feature, structured JSON output.
Mirrors the manual GT format from:
  experiments/data/drugs/full_gt.xlsx - drugs-fe_gt (1).csv

Usage:
  python extract_drugs_features.py <docx_file>
  python extract_drugs_features.py <directory> [output_checkpoint.json]
"""

from __future__ import annotations

import csv
import json
import os
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception: pass
import re
import sys
import time
from pathlib import Path

import docx
import openai

# ── OpenAI client ────────────────────────────────────────────────────────────
_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
client = openai.OpenAI(api_key=_API_KEY)
GPT_MODEL = "gpt-4.1"


def read_docx(path: str) -> str:
    """Read docx text including smartTag elements (Word metric converters)."""
    import zipfile
    from lxml import etree

    W  = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    ST = "urn:schemas-microsoft-com:office:smarttags"

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


def call_gpt(system: str, user: str) -> dict:
    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


# ── Feature extractors ───────────────────────────────────────────────────────

def extract_offense_number(text: str) -> dict:
    system = """את/ה מומחית לדין הפלילי בישראל בתחום עבירות סמים.
מוצג בפנייך פסק דין. עליך לזהות את סעיפי עבירות הסמים שבהם הואשם (והורשע) הנאשם בלבד.

ענה ב-JSON עם המבנה הבא:
{
  "סעיף_6": 0/1,         // ייצור
  "סעיף_7": "",          // ריק, "א", "ג", "א_ג" - לפי האות שמופיעה בסעיף
  "סעיף_13": 0/1,        // כלים
  "סעיף_14": 0/1,        // ייבוא/ייצוא
  "סעיף_19": 0/1,        // סחר (19 ללא א)
  "סעיף_19א": 0/1,       // ייבוא/סחר (19א)
  "סעיף_21": 0/1,
  "סעיף_22": 0/1,
  "סעיף_61_א_ג": 0/1,   // 61(א)(ג)
  "עבירת_סמים_אחרת": ""  // מלל חופשי אם יש עבירת סמים שאינה ברשימה, אחרת ריק
}

חשוב: רק עבירות שהנאשם הורשע בהן לפי פקודת הסמים המסוכנים. אל תכלול עבירות שנגד הנאשם.
"""
    return call_gpt(system, f"פסק הדין:\n\n{text[:6000]}")


def extract_side_offenses(text: str) -> dict:
    system = """את/ה מומחית לדין הפלילי בישראל.
מוצג בפנייך פסק דין בעבירות סמים. עליך לזהות האם הנאשם הורשע גם בעבירות נוספות שאינן מפקודת הסמים המסוכנים.

ענה ב-JSON:
{
  "עבירות_נלוות": 0/1   // 1=כן, 0=לא
}
"""
    return call_gpt(system, f"פסק הדין:\n\n{text[:6000]}")


def extract_drug_type(text: str) -> dict:
    system = """את/ה מומחית לדין הפלילי בישראל בתחום עבירות סמים.
מוצג בפנייך פסק דין. עליך לזהות את סוגי הסמים וכמויותיהם הקשורים לעבירה של הנאשם עצמו.

ענה ב-JSON עם המבנה הבא — עבור כל סוג סם, רשימה של מחרוזות בפורמט "כמות-יחידה":
{
  "LSD": [],
  "METHAMPHETAMINE": [],
  "האיוואסקה": [],
  "קתינון": [],
  "קטמין": [],
  "חשיש": [],
  "מתילמקאתינון": [],
  "קנבוס_בשתילים": [],
  "קנבוס": [],
  "MDMA": [],
  "קוקאין": [],
  "אחר": []
}

כללים חשובים:
- **קרא כל משפט עד סופו לפני חילוץ כמות.** לעיתים משפט מתחיל ב"X שתילים" ומסתיים ב"במשקל כולל של Y גרם נטו" — הכמות הנכונה היא Y גרם, לא X שתילים.
- **אם מצויין גם מספר שתילים/יחידות וגם משקל בגרם/ק"ג — קח רק את המשקל.**
- **קנבוס_בשתילים**: השתמש רק כאשר מדובר בשתילי צמח חי ואין משקל. אם יש משקל — רשום תחת קנבוס.
- **סכום כל הכמויות** של אותו סם מכל המופעים בפסק הדין (כולל עסקאות שונות), ולא רק אחד מהם.
- קילוגרם → המר לגרם (×1000) ואחד עם שאר הגרמים לערך אחד.
- פורמט כל איבר: "כמות-יחידה" למשל "71.52-גרם", "50-טבליות".
- רשימה מכילה איבר אחד לכל יחידת מידה (גרם ו-טבליות יכולים להיות יחד, אבל גרם מופיע פעם אחת בלבד).
- אם לא ידוע או לא מוזכר — רשימה ריקה [].
- אל תכלול סמים שהיו שייכים לאדם אחר.

דוגמא 1: "גידל עשרות שתילים במשקל כולל של 837.76 גרם נטו" → קנבוס: ["837.76-גרם"] (לא שתילים!)
דוגמא 2: "170,000 גרם" ו-"10.2 גרם" → ["170010.2-גרם"] ולא שני ערכים.
דוגמא 3: "700 שתילים במשקל 180 ק"ג" + "7.48 ק"ג קנבוס נפרד" → קנבוס: ["187480-גרם"]
"""
    return call_gpt(system, f"פסק הדין:\n\n{text[:8000]}")


def extract_lab(text: str) -> dict:
    system = """את/ה מומחית לדין הפלילי בישראל.
מוצג בפנייך פסק דין בעבירות סמים. האם מתוארת מעבדת סמים או כלים המרמזים על קיומה של מעבדה?

ענה ב-JSON:
{
  "מעבדה": 0/1   // 1=כן, 0=לא
}
ברירת מחדל: 0
"""
    return call_gpt(system, f"פסק הדין:\n\n{text[:6000]}")


def extract_role(text: str) -> dict:
    system = """את/ה מומחית לדין הפלילי בישראל בתחום עבירות סמים.
מוצג בפנייך פסק דין. עליך לזהות את תפקיד הנאשם.

ענה ב-JSON:
{
  "בעל_הסמים": 0/1,      // 1 אם הסמים שלו, 0 אם הועברו אליו ולא ברשותו המלאה
  "בעל_המעבדה": 0/1      // 1 רק אם יש מעבדה והיא שלו; 0 אם אין מעבדה או לא שלו
}

הנחיות:
- אם נאמר שהסמים הועברו לנאשם לצורך הובלה/שמירה בלבד — בעל_הסמים=0
- אם הנאשם הוא הבעלים/המוכר/היצרן — בעל_הסמים=1
- ברירת מחדל: בעל_הסמים=1, בעל_המעבדה=0
"""
    return call_gpt(system, f"פסק הדין:\n\n{text[:6000]}")


def extract_undercover(text: str) -> dict:
    system = """את/ה מומחית לדין הפלילי בישראל.
מוצג בפנייך פסק דין בעבירות סמים. האם מתוארת מכירת סמים לסוכן משטרתי?

ענה ב-JSON:
{
  "מכירה_לסוכן": 0/1   // 1=כן, 0=לא
}
ברירת מחדל: 0
"""
    return call_gpt(system, f"פסק הדין:\n\n{text[:6000]}")


# ── Combine all features ─────────────────────────────────────────────────────

def extract_all_features(text: str) -> dict:
    features = {}

    print("  → מספר עבירה", end="", flush=True)
    features["מספר_עבירה"] = extract_offense_number(text)
    time.sleep(0.3)

    print(", עבירות נלוות", end="", flush=True)
    features["עבירות_נלוות"] = extract_side_offenses(text)
    time.sleep(0.3)

    print(", סוג הסם", end="", flush=True)
    features["סוג_הסם"] = extract_drug_type(text)
    time.sleep(0.3)

    print(", מעבדה", end="", flush=True)
    features["מעבדה"] = extract_lab(text)
    time.sleep(0.3)

    print(", תפקיד", end="", flush=True)
    features["תפקיד"] = extract_role(text)
    time.sleep(0.3)

    print(", מכירה לסוכן", end="", flush=True)
    features["מכירה_לסוכן"] = extract_undercover(text)
    time.sleep(0.3)

    print()
    return features


# ── Directory processing ─────────────────────────────────────────────────────

def process_directory(dir_path: str, checkpoint_file: str):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, encoding="utf-8") as f:
            results = json.load(f)
        print(f"Loaded checkpoint: {len(results)} verdicts done")
    else:
        results = {}

    docx_files = sorted(Path(dir_path).glob("*.docx"))
    todo = [f for f in docx_files if f.stem not in results]
    print(f"Total docx: {len(docx_files)} | Done: {len(results)} | Remaining: {len(todo)}")

    for i, docx_path in enumerate(todo):
        vid = docx_path.stem
        print(f"\n[{i+1}/{len(todo)}] {vid}")
        try:
            text = read_docx(str(docx_path))
            features = extract_all_features(text)
            results[vid] = features
        except Exception as e:
            print(f"  FAILED: {e}")
            results[vid] = {"_error": str(e)}

        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(results)} verdicts saved to {checkpoint_file}")
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file:  python extract_drugs_features.py file.docx")
        print("  Directory:    python extract_drugs_features.py dir/ [checkpoint.json]")
        sys.exit(1)

    path = sys.argv[1]

    if os.path.isdir(path):
        checkpoint = sys.argv[2] if len(sys.argv) > 2 else "drugs_extracted_features.json"
        process_directory(path, checkpoint)
    else:
        text = read_docx(path)
        features = extract_all_features(text)
        print("\n" + json.dumps(features, ensure_ascii=False, indent=2))
