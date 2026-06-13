"""Sharper, non-arbitrary answer to "how much is beyond the manual schema".

NO clustering, NO distance threshold. Each distinct RAW field-name is mapped by an
LLM DIRECTLY onto the FIXED manual schema (drugs: 6 fields; weapon: 11 conceptual
fields = the 8 weapon-type columns collapsed to one + 10 others) or to "OTHER".
Two names that mean the same thing simply land on the same fixed target — no merge
decision is needed, so there is no arbitrary parameter.

Stage 1 (this file, `map`): name -> schema field | OTHER, aggregated by
  (a) distinct names and (b) coverage-mass (verdicts containing the name).
Stage 2 (`patterns`): a SEPARATE LLM pass over ONLY the OTHER names, asked to let
  recurring themes EMERGE (no fixed count), each theme backed by its field-names.
"""
from __future__ import annotations
import csv, json, os, time, argparse
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from common import load_rep, FOCUS, DOMAINS

HERE = Path(__file__).parent
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
MODEL = "gpt-5.2"
MAX_CONCURRENCY = 50


def _llm_json(client, sys_prompt, user_prompt, max_tokens=16000):
    """Call the model and parse a JSON object. gpt-5.* reject custom temperature
    and use max_completion_tokens; older models use temperature=0 + max_tokens."""
    kw = dict(model=MODEL,
              response_format={"type": "json_object"},
              messages=[{"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}])
    if MODEL.startswith("gpt-5"):
        kw["max_completion_tokens"] = max_tokens
    else:
        kw["temperature"] = 0
        kw["max_tokens"] = max_tokens
    last = None
    for attempt in range(5):
        try:
            r = client.chat.completions.create(**kw)
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise last

SCHEMA = {
    "drugs": {
        "מכירה_לסוכן": "מכירה/עסקה מול סוכן משטרתי סמוי",
        "מעבדה": "מעבדת סמים/גידול: קיומה, ציודה, מיקומה, פירוטה",
        "סוג_הסם_וכמות": "איזה סם וכמה: סוג/שם הסם, משקל, כמות, יחידות, שתילים, חבילות, מינון",
        "עבירה": "סוג עבירת הסמים וסעיפי החוק (יבוא/סחר/החזקה/ייצור, מספרי סעיפים)",
        "עבירות_נלוות": "קיום עבירות נוספות/נלוות",
        "תפקיד": "תפקיד הנאשם בעבירה (בעלים/שליח/מתווך/בעל מעבדה/מבצע)",
    },
    "weapon": {
        "סוג_הנשק_וכמות": "סוג כלי הנשק וכמותם (אקדח/תת-מקלע/רובה/רימון/מאולתר, ספירה, דגם, קליבר)",
        "אופן_החזקת_הנשק": "כיצד/היכן הוחזק או אוחסן הנשק (ברכב/בבית/על הגוף)",
        "אופן_קבלת_הנשק": "כיצד הושג הנשק (רכש/מצא/גנב/עבודה)",
        "כמות_תחמושת": "כמות תחמושת/מחסניות/כדורים",
        "מטרה_סיבת_העבירה": "המטרה או המניע לעבירה (בצע כסף/הגנה/סכסוך)",
        "מספר_עבירה": "מספר סעיף החוק של עבירת הנשק (למשל 144(ב))",
        "סוג_עבירה": "סוג עבירת הנשק (החזקה/נשיאה/הובלה/סחר/ייצור)",
        "סטטוס_הנשק": "מצב/תקינות הנשק (תקין/מופרד מתחמושת/טעון/מאולתר)",
        "עבירות_נוספות": "עבירות נוספות שאינן עבירות נשק",
        "שימוש": "האם והיכן נעשה שימוש בנשק (ירי)",
        "תכנון": "האם העבירה תוכננה מראש",
    },
}


import re
# Deterministic overrides for LITERAL manual-schema column names that the LLM maps
# inconsistently. These are the schema's own fields — no judgment needed. (gpt-5.2
# wrongly sent ~half of the "סוג הנשק [אקדח]" weapon-type columns to OTHER.)
OVERRIDES = {
    "drugs": [],
    "weapon": [(re.compile(r"^סוג[_ ]הנשק"), "סוג_הנשק_וכמות")],
}


def apply_overrides(domain, cache, names):
    for n in names:
        for pat, fld in OVERRIDES[domain]:
            if pat.search(n):
                cache[n] = fld
                break
    return cache


def _key():
    p = HERE.parents[1] / "experiments" / ".env"
    for line in p.read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    return os.environ.get("OPENAI_API_KEY", "")


def _client():
    from openai import OpenAI
    return OpenAI(api_key=_key())


def collect(domain):
    """Return name -> {rep: coverage(verdict count)} and a sample value per name."""
    cov = defaultdict(lambda: defaultdict(int))
    sample = {}
    for rep in FOCUS:
        for d in load_rep(domain, rep).values():
            seen = set()
            for k, v in d.items():
                name = str(k).strip()
                if not name or name in seen:
                    continue
                seen.add(name)
                cov[name][rep] += 1
                if name not in sample and v not in (None, "", []):
                    sample[name] = str(v)[:50]
    return cov, sample


def stage_map(domain):
    cache_path = OUT / f"direct_map_{domain}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    cov, sample = collect(domain)
    names = sorted(cov)
    todo = [n for n in names if n not in cache]
    if todo:
        client = _client()
        fields = "\n".join(f"  - {k}: {v}" for k, v in SCHEMA[domain].items())
        hints = {
            "drugs": (
                "הבהרות מיפוי (חובה):\n"
                "- כל שדה על סוג/שם/משקל/כמות/יחידות/שתילים/חבילות/מינון של הסם => סוג_הסם_וכמות.\n"
                "- כל שדה על מעבדה/גידול/ציוד/חדרים/כלים להכנת סם => מעבדה.\n"
                "- כל שדה על סעיפי-חוק/סוג-העבירה (יבוא/סחר/החזקה) => עבירה.\n"
                "- שווי כספי/מחיר/תמורה, מקור/מדינת-מוצא, שיטת הסתרה/הברחה/שילוח, תקשורת, "
                "תאריך/שעה/מיקום, מצב-נפשי/אשמה/נזק => OTHER (אינם בסכמה).\n"
            ),
            "weapon": (
                "הבהרות מיפוי (חובה):\n"
                "- כל שדה על סוג/דגם/קליבר/יצרן/ספירת כלי-נשק, כולל 'סוג הנשק [אקדח]', "
                "'סוג הנשק [תת מקלע]', 'סוג הרימון', 'סוג הנשק - אם לא נמצא בטבלה' => סוג_הנשק_וכמות.\n"
                "- תחמושת/מחסניות/כדורים => כמות_תחמושת.\n"
                "- מספר סעיף החוק => מספר_עבירה; שם/סוג העבירה (החזקה/נשיאה/סחר) => סוג_עבירה.\n"
                "- ירי/שימוש בנשק => שימוש; היכן/כיצד אוחסן => אופן_החזקת_הנשק; כיצד הושג => אופן_קבלת_הנשק.\n"
                "- תאריך/שעה/מיקום, גיל/שנת-לידה, רכב/רישוי, רישיון, מצב-נפשי/אשמה/נזק, "
                "תכנון מורחב, מעורבים/ארגון, תמורה/מחיר => OTHER (אינם בסכמה).\n"
            ),
        }[domain]
        sys = (
            "אתה ממפה שמות-שדה שחולצו מגזרי-דין על סכמה ידנית קבועה.\n"
            f"שדות הסכמה (דומיין {domain}):\n{fields}\n\n" + hints +
            "לכל שם-שדה החזר את שם שדה-הסכמה היחיד שאליו הוא שייך מבחינת תוכן (גם אם הניסוח "
            "שונה), או \"OTHER\" אם התוכן אינו מכוסה ע\"י אף שדה סכמה. בלבד אחת מהאפשרויות.\n"
            "החזר JSON: {\"items\":[{\"name\":\"<השם המדויק כפי שניתן>\",\"field\":\"<שם-שדה|OTHER>\"}]}"
        )
        B = 40
        batches = [todo[i:i + B] for i in range(0, len(todo), B)]

        def run_batch(batch):
            listing = "\n".join(f'- {n}   (דוגמת ערך: {sample.get(n,"")})' for n in batch)
            obj = _llm_json(client, sys, "שמות לסיווג:\n" + listing)
            return {it["name"].strip(): it.get("field", "OTHER")
                    for it in obj.get("items", []) if it.get("name", "").strip()}

        done = 0
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as ex:
            futs = {ex.submit(run_batch, b): b for b in batches}
            for fut in as_completed(futs):
                try:
                    cache.update(fut.result())
                except Exception as e:
                    print(f"  batch failed: {str(e)[:80]}")
                done += 1
                print(f"  mapped batch {done}/{len(batches)}")
        for n in names:
            cache.setdefault(n, "OTHER")
        cache_path.write_text(json.dumps(cache, ensure_ascii=False))

    cache = apply_overrides(domain, cache, names)  # deterministic, idempotent
    cache_path.write_text(json.dumps(cache, ensure_ascii=False))

    # aggregate
    print("=" * 70); print("DOMAIN:", domain)
    valid = set(SCHEMA[domain])
    for rep in FOCUS:
        names_schema = names_other = 0
        mass_schema = mass_other = 0
        fan = Counter(); other_cov = Counter()
        for n, repcov in cov.items():
            c = repcov.get(rep, 0)
            if c == 0:
                continue
            tgt = cache.get(n, "OTHER")
            if tgt in valid:
                names_schema += 1; mass_schema += c; fan[tgt] += 1
            else:
                names_other += 1; mass_other += c; other_cov[n] += c
        tn = names_schema + names_other or 1
        tm = mass_schema + mass_other or 1
        print(f"\n  {rep}:")
        print(f"    distinct names: SCHEMA {names_schema} ({100*names_schema/tn:.0f}%)  "
              f"OTHER {names_other} ({100*names_other/tn:.0f}%)")
        print(f"    coverage-mass:  SCHEMA {100*mass_schema/tm:.0f}%  OTHER {100*mass_other/tm:.0f}%")
    # global schema fan-in (names per schema field, across all reps)
    fan_all = Counter()
    for n in names:
        t = cache.get(n, "OTHER")
        if t in valid:
            fan_all[t] += 1
    print("\n  schema fan-in (distinct RAW names mapped to each schema field, any rep):")
    for f, k in fan_all.most_common():
        print(f"      {k:4d}  {f}")
    n_other = sum(1 for n in names if cache.get(n) not in valid)
    print(f"\n  total distinct names={len(names)}  -> OTHER={n_other} ({100*n_other/len(names):.0f}%)")
    return cov, cache


def stage_patterns(domain, cov, cache):
    """Stage 2: let themes EMERGE from the OTHER bucket (no fixed count)."""
    valid = set(SCHEMA[domain])
    others = [(n, sum(cov[n].values())) for n in cov if cache.get(n) not in valid]
    others.sort(key=lambda x: -x[1])
    client = _client()
    listing = "\n".join(f"{n} ({c})" for n, c in others)
    sys = (
        "להלן שמות-שדה שחולצו מגזרי-דין ושאינם נכללים בסכמה הידנית (כל שם עם מספר התיקים "
        "שבהם הופיע, בסוגריים). זהה את הדפוסים/הנושאים החוזרים שעולים מתוך הרשימה — "
        "**אל תכפה מספר נושאים מראש**, דווח כמה שהדאטה מראה. לכל נושא תן: שם קצר, סך-הכיסוי "
        "(סכום המספרים בסוגריים של שדותיו), ו-5-8 שמות-שדה מייצגים. JSON: "
        "{\"themes\":[{\"name\":...,\"coverage\":<int>,\"examples\":[...]}]}"
    )
    themes = _llm_json(client, sys, listing).get("themes", [])
    (OUT / f"other_patterns_{domain}.json").write_text(
        json.dumps(themes, ensure_ascii=False, indent=1))
    print("=" * 70); print(f"OTHER patterns — {domain}  ({len(others)} residual names)")
    for t in sorted(themes, key=lambda x: -x.get("coverage", 0)):
        print(f"  {t.get('coverage',0):5d}  {t.get('name','?')}")
        print(f"         {', '.join(t.get('examples',[])[:8])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["map", "patterns", "both"], default="map")
    a = ap.parse_args()
    for dom in DOMAINS:
        cov, cache = stage_map(dom)
        if a.stage in ("patterns", "both"):
            stage_patterns(dom, cov, cache)
