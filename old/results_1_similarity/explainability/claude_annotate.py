"""
Annotate similarity-explanation CSVs with Claude Opus 4.7 acting as a human annotator.

Fills 4 columns per row:
  נאמנות_עובדתית (1-3): factual faithfulness of the explanation to the feature vectors
  רלוונטיות_משפטית (1-3): legal relevance of the factors discussed
  שלמות (1-3): completeness — coverage of important similarity/difference dimensions
  הערות: short free-text annotator note

Reads/writes in place. Saves after each row (crash-safe). Skips rows already annotated.
"""

import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic


BASE = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/experiments/explainability_annotation/hybrid_full")

FILES = {
    "claude_sonnet_4_6": {
        "drugs":  BASE / "explainability_drugs_claude_sonnet_4_6.csv",
        "weapon": BASE / "explainability_weapon_claude_sonnet_4_6.csv",
    },
    "gemma4_31b_or": {
        "drugs":  BASE / "explainability_drugs_gemma4_31b_or.csv",
        "weapon": BASE / "explainability_weapon_gemma4_31b_or.csv",
    },
    "gpt4": {
        "drugs":  BASE / "explainability_drugs_gpt4.csv",
        "weapon": BASE / "explainability_weapon_gpt4.csv",
    },
}

MODEL = "claude-opus-4-7"

DOMAIN_LABEL_HEB = {"drugs": "סמים", "weapon": "נשק"}

SYSTEM_PROMPT = """את/ה משמש/ת כמתייג/ת אנושי/ת מומחה/ית בתחום המשפט הפלילי הישראלי (גזרי דין בעבירות סמים ונשק).

המשימה: בכל זוג של גזרי דין, המודל קיבל שני feature vectors (מיצוי מובנה של עובדות) וניסח הסבר השוואתי של דמיון בעברית, עם ציון דמיון מספרי (0–100). תפקידך להעריך את איכות ההסבר של המודל על סקאלה 1–3 בשלושה ממדים, ולהוסיף הערה מילולית קצרה.

### שלושה ממדי הערכה (סקאלה 1–3):

**1) נאמנות_עובדתית** — האם ההסבר מציג נכונה את העובדות מה-feature vectors?
  - 3 = ההסבר מדויק עובדתית לחלוטין. כל טענה ניתנת לאימות מול ה-vectors. אין הזיות, אין סילוף, אין הכללות לא מבוססות.
  - 2 = רוב הטענות נכונות, אך יש שגיאה עובדתית קטנה אחת, סילוף קל, או הכללה יתרה של פרט אחד. לא פוגע מהותית בהבנה.
  - 1 = שגיאות עובדתיות מהותיות. ההסבר כולל הזיות, סילוף חמור של פרטים, או טענות שסותרות את ה-vectors.

**2) רלוונטיות_משפטית** — האם ההסבר מתמקד בגורמים בעלי משקל משפטי-עונשי בתחום הספציפי (סמים/נשק)?
  ממדים רלוונטיים נפוצים: סוג העבירה וסעיף החוק, חומרה, תפקיד הנאשם (עיקרי/שולי), סוג וכמות החומר (סם/נשק), שיטת ביצוע (MO), נסיבות מחמירות/מקלות, עבירות נלוות, שימוש.
  - 3 = ההסבר מתמקד כמעט כולו בממדים בעלי משקל משפטי מובהק. אין משקל לפרטים זניחים או "רעש".
  - 2 = ההסבר כולל ממדים רלוונטיים, אך גם פרטים פחות רלוונטיים משפטית (למשל: תאריכים מדויקים, שמות שותפים) שמקבלים משקל מוגזם.
  - 1 = ההסבר מתמקד בעיקר בפרטים חסרי משקל משפטי או מפספס את הממדים המהותיים שהיו צריכים להוביל את ההשוואה.

**3) שלמות** — האם ההסבר מכסה את כל ממדי הדמיון/ההבדל המשמעותיים שעולים מה-vectors?
  - 3 = ההסבר מכסה את כל הממדים המשמעותיים. לא הוחמץ גורם משפטי משמעותי שעולה מה-vectors.
  - 2 = מרבית הממדים מכוסים, אך הוחמץ ממד משני אחד או שניים.
  - 1 = הוחמצו ממדים מרכזיים. ההסבר שטחי, חלקי, או מתרכז בהיבט אחד תוך התעלמות מהיתר.

### הערה מילולית (הערות):
טקסט חופשי קצר בעברית (1–3 משפטים): הצדק/י את הציונים — מה עבד טוב, מה חסר או שגוי. התייחס/י לדוגמה קונקרטית מההסבר אם אפשר.

### עקרונות עבודה:
- את/ה מתייג/ת את **איכות ההסבר** בלבד — לא את נכונות ציון הדמיון של המודל ולא את הדמיון האמיתי בין התיקים.
- ה-GT (סקייל 1/2/3) ניתן לך כאינפורמציה — אבל אל תתני לו להטות את הציון שלך להסבר. הסבר טוב יכול לתאר נכונה למה שני תיקים דומים גם אם בפועל הציון של המודל שגוי.
- בעת קריאת feature vectors: שים/י לב להעדר שדות (לא כל תיק כולל את כל השדות), ולקודי עבירה (144 לנשק, 19/13 לסמים).
- החזר/י JSON תקני לפי הסכמה שתסופק, ללא טקסט נוסף.
"""


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "נאמנות_עובדתית": {"type": "integer", "enum": [1, 2, 3]},
        "רלוונטיות_משפטית": {"type": "integer", "enum": [1, 2, 3]},
        "שלמות": {"type": "integer", "enum": [1, 2, 3]},
        "הערות": {"type": "string", "minLength": 1},
    },
    "required": ["נאמנות_עובדתית", "רלוונטיות_משפטית", "שלמות", "הערות"],
    "additionalProperties": False,
}


def load_api_key() -> str:
    env_path = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/.env")
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip().lower() in ("antropic_api_key", "anthropic_api_key"):
            return v.strip().strip('"').strip("'")
    raise RuntimeError("No (a)nthropic_api_key found in .env")


def build_user_message(row: dict, domain: str) -> str:
    domain_heb = DOMAIN_LABEL_HEB[domain]
    return f"""להלן זוג גזרי דין בתחום {domain_heb} שההסבר של המודל עליהם דורש תיוג.

**GT (סקייל 1–3, תיוג אנושי של דמיון אמיתי):** {row['GT']}
**ציון דמיון שהמודל חזה (0–100):** {row['model_score']}
**תחזית סקייל של המודל לפי thresholds (1–3):** {row['model_pred_scale']}

---
**feature_vector_1** (תיק {row['verdict_1']}):
{row['feature_vector_1']}

---
**feature_vector_2** (תיק {row['verdict_2']}):
{row['feature_vector_2']}

---
**הסבר המודל לדמיון בין השניים:**
{row['explanation']}

---
הערך/י את איכות ההסבר בשלושה ממדים (1–3) וכתוב/י הערה קצרה. החזר/י JSON בלבד."""


def is_annotated(row: dict) -> bool:
    for col in ("נאמנות_עובדתית", "רלוונטיות_משפטית", "שלמות"):
        v = (row.get(col) or "").strip()
        if not v:
            return False
        try:
            n = int(float(v))
            if n not in (1, 2, 3):
                return False
        except (ValueError, TypeError):
            return False
    return True


def call_claude(client: anthropic.Anthropic, row: dict, domain: str) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        output_config={
            "effort": "xhigh",
            "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
        },
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": build_user_message(row, domain)}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    parsed = json.loads(text)
    return {
        "parsed": parsed,
        "usage": response.usage,
        "stop_reason": response.stop_reason,
    }


def annotate_file(path: Path, domain: str, client: anthropic.Anthropic, limit: int | None, workers: int = 10):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    pending = [(i, r) for i, r in enumerate(rows) if not is_annotated(r)]
    if limit is not None:
        pending = pending[:limit]

    print(f"\n=== {path.name} — {len(pending)}/{len(rows)} rows to annotate (workers={workers})")
    if not pending:
        return

    totals = {"in": 0, "cache_write": 0, "cache_read": 0, "out": 0}
    t0 = time.time()
    lock = threading.Lock()
    n_done = [0]  # mutable counter

    def task(idx_row):
        i, row = idx_row
        pair_id = row.get("pair_id", "?")
        try:
            result = call_claude(client, row, domain)
        except anthropic.APIError as e:
            return ("api", pair_id, str(e), None, None)
        except (json.JSONDecodeError, KeyError) as e:
            return ("parse", pair_id, str(e), None, None)
        return ("ok", pair_id, None, i, result)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(task, ir) for ir in pending]
        for fut in as_completed(futures):
            kind, pair_id, err, i, result = fut.result()
            if kind != "ok":
                with lock:
                    n_done[0] += 1
                    print(f"  [{n_done[0]}/{len(pending)}] pair={pair_id}  {kind.upper()} ERROR: {err}")
                continue
            parsed = result["parsed"]
            u = result["usage"]
            with lock:
                rows[i]["נאמנות_עובדתית"] = str(parsed["נאמנות_עובדתית"])
                rows[i]["רלוונטיות_משפטית"] = str(parsed["רלוונטיות_משפטית"])
                rows[i]["שלמות"] = str(parsed["שלמות"])
                rows[i]["הערות"] = parsed["הערות"]
                totals["in"] += u.input_tokens
                totals["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
                totals["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
                totals["out"] += u.output_tokens
                n_done[0] += 1
                cur = n_done[0]
                # Persist (crash-safe) — under lock so no concurrent writers
                with open(path, "w", encoding="utf-8-sig", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
                    w.writeheader()
                    w.writerows(rows)
                elapsed = time.time() - t0
                rate = cur / elapsed if elapsed else 0
                print(
                    f"  [{cur}/{len(pending)}] pair={pair_id}  "
                    f"F={parsed['נאמנות_עובדתית']} R={parsed['רלוונטיות_משפטית']} C={parsed['שלמות']}  "
                    f"({rate:.2f} rows/s)"
                )

    # Cost estimate (Opus 4.7: $5/M input, $25/M output, $6.25/M cache_write, $0.50/M cache_read)
    cost = (
        totals["in"] * 5e-6
        + totals["cache_write"] * 6.25e-6
        + totals["cache_read"] * 0.5e-6
        + totals["out"] * 25e-6
    )
    print(
        f"\n  Totals: in={totals['in']:,}  cache_write={totals['cache_write']:,}  "
        f"cache_read={totals['cache_read']:,}  out={totals['out']:,}  "
        f"est_cost=${cost:.2f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_model", choices=["claude_sonnet_4_6", "gemma4_31b_or", "gpt4"], default="claude_sonnet_4_6",
                    help="Which target model's explanations to annotate")
    ap.add_argument("--domain", choices=["drugs", "weapon", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None, help="Annotate at most N rows per file (for testing)")
    ap.add_argument("--workers", type=int, default=10, help="Concurrent API calls")
    args = ap.parse_args()

    os.environ["ANTHROPIC_API_KEY"] = load_api_key()
    client = anthropic.Anthropic()

    domains = ["drugs", "weapon"] if args.domain == "both" else [args.domain]
    for d in domains:
        annotate_file(FILES[args.source_model][d], d, client, args.limit, args.workers)


if __name__ == "__main__":
    main()
