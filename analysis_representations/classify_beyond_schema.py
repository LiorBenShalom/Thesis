"""How much information did the automatic extraction add BEYOND the manual schema?

The key question: many distinct extracted concepts (כמות שתילים / כמות קוקאין /
משקל נטו / מספר חבילות ...) all collapse into ONE schema field (סוג הסם, כמות).
So we map every extracted CONCEPT onto the manual schema and decide:
  - SCHEMA  : reducible to an existing manual-schema field (no new information)
  - BEYOND  : a genuinely new substantive dimension (name it)
  - IDENT   : an administrative identifier with no cross-case value
             (dates, times, names, phone numbers, serial numbers, addresses)
Classification uses an LLM (gpt-4.1-mini) for Hebrew legal nuance; cached to disk.
Then we report, per representation, how much of the extracted info is SCHEMA vs BEYOND.
"""
from __future__ import annotations
import json, csv, os, time
from pathlib import Path
from collections import defaultdict, Counter
from common import FOCUS

HERE = Path(__file__).parent
OUT = HERE / "out"
MODEL = "gpt-4.1-mini"

SCHEMA = {
    "drugs": {
        "מכירה_לסוכן": "האם בוצעה מכירה לסוכן משטרתי סמוי",
        "מעבדה": "האם מעורבת מעבדת סמים / גידול (כולל ציוד, מיקום, פירוט המעבדה)",
        "סוג_הסם_וכמות": "סוג הסם וכמותו/משקלו/מספר יחידות/שתילים/חבילות — כל ביטוי של איזה סם וכמה",
        "עבירה": "סוג העבירה וסעיפי החוק (יבוא/סחר/החזקה/ייצור, מספרי סעיפים)",
        "עבירות_נלוות": "האם היו עבירות נוספות/נלוות",
        "תפקיד": "תפקיד הנאשם בעבירה (בעלים/שליח/מתווך/בעל מעבדה)",
    },
    "weapon": {
        "סוג_הנשק_וכמות": "סוג כלי הנשק וכמותם (אקדח/תת-מקלע/רובה/רימון, ספירה, דגם, קליבר)",
        "אופן_החזקת_הנשק": "כיצד/היכן הוחזק או אוחסן הנשק",
        "אופן_קבלת_הנשק": "כיצד הושג הנשק (רכש/מצא/גנב/עבודה) ומחירו",
        "כמות_תחמושת": "כמות התחמושת/מחסניות/כדורים",
        "מטרה_סיבת_העבירה": "המטרה או המניע לעבירה (בצע כסף/הגנה/סכסוך)",
        "מספר_עבירה": "מספר סעיף החוק של העבירה",
        "סוג_עבירה": "סוג עבירת הנשק (החזקה/נשיאה/סחר/ייצור)",
        "סטטוס_הנשק": "מצב/תקינות הנשק (תקין/מופרד מתחמושת/טעון/מאולתר)",
        "עבירות_נוספות": "עבירות נוספות שאינן עבירות נשק",
        "שימוש": "האם והיכן נעשה שימוש בנשק (ירי)",
        "תכנון": "האם העבירה תוכננה מראש",
    },
}


def _key():
    p = HERE.parents[1] / "experiments" / ".env"
    for line in p.read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    return os.environ.get("OPENAI_API_KEY", "")


def classify(domain, concepts):
    cache_path = OUT / f"schema_map_{domain}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [c for c in concepts if str(c["cluster"]) not in cache]
    if todo:
        from openai import OpenAI
        client = OpenAI(api_key=_key())
        fields = "\n".join(f"  - {k}: {v}" for k, v in SCHEMA[domain].items())
        sys = (
            "אתה ממפה מושגי-פיצ'רים שחולצו מגזרי-דין על סכמה ידנית קבועה.\n"
            f"שדות הסכמה הקיימים (דומיין {domain}):\n{fields}\n\n"
            "לכל מושג החזר אחת מהאפשרויות:\n"
            "1. שם שדה-סכמה מהרשימה — אם המושג ניתן לרדוקציה לאותו שדה (גם אם בניסוח/פירוט אחר).\n"
            "   למשל 'כמות שתילים','משקל נטו','מספר חבילות','כמות קוקאין' => סוג_הסם_וכמות.\n"
            "2. \"BEYOND\" — אם זה מידע מהותי שאינו מכוסה ע\"י אף שדה סכמה (למשל שווי כספי, שיטת "
            "הברחה, מקור/מדינת מוצא, אמצעי תקשורת, שיתוף פעולה/מבנה ארגוני, מצב נפשי/אשמה, "
            "נזק, יכולת שליטה).\n"
            "3. \"IDENT\" — מזהה אדמיניסטרטיבי ללא ערך-דמיון: תאריך, שעה, שם אדם, מספר טלפון, "
            "מספר סידורי, כתובת ספציפית, מספר תיק.\n\n"
            "כשמחזירים BEYOND חובה לתת גם 'dimension' = שם-על קצר בעברית לממד החדש "
            "(למשל 'שווי כספי','שיטת הסתרה/הברחה','מקור הנשק/הסם','אמצעי תקשורת',"
            "'מבנה ארגוני/שותפים','מצב נפשי ואשמה','נזק','זירה גאוגרפית').\n"
            "החזר JSON: {\"items\":[{\"id\":<int>,\"cat\":\"<field|BEYOND|IDENT>\","
            "\"dimension\":\"<רק ל-BEYOND>\"}]}"
        )
        B = 40
        for i in range(0, len(todo), B):
            batch = todo[i:i + B]
            listing = "\n".join(
                f'{c["cluster"]}: {c["label"]}  | דוגמאות שמות: {c["members"][:160]}'
                for c in batch)
            for attempt in range(4):
                try:
                    r = client.chat.completions.create(
                        model=MODEL, temperature=0,
                        response_format={"type": "json_object"},
                        messages=[{"role": "system", "content": sys},
                                  {"role": "user", "content": "מושגים לסיווג:\n" + listing}])
                    items = json.loads(r.choices[0].message.content)["items"]
                    for it in items:
                        cache[str(it["id"])] = {"cat": it.get("cat", "BEYOND"),
                                                "dim": it.get("dimension", "")}
                    break
                except Exception as e:
                    if attempt == 3:
                        raise
                    time.sleep(2 * (attempt + 1))
            print(f"  classified {min(i+B,len(todo))}/{len(todo)}")
        # second pass for ids the model silently dropped
        for _round in range(3):
            missing = [c for c in concepts if str(c["cluster"]) not in cache]
            if not missing:
                break
            print(f"  retry {len(missing)} missing ids (round {_round+1})")
            for i in range(0, len(missing), 20):
                batch = missing[i:i + 20]
                listing = "\n".join(
                    f'{c["cluster"]}: {c["label"]}  | דוגמאות שמות: {c["members"][:160]}'
                    for c in batch)
                try:
                    r = client.chat.completions.create(
                        model=MODEL, temperature=0,
                        response_format={"type": "json_object"},
                        messages=[{"role": "system", "content": sys},
                                  {"role": "user", "content": "מושגים לסיווג:\n" + listing}])
                    for it in json.loads(r.choices[0].message.content)["items"]:
                        cache[str(it["id"])] = {"cat": it.get("cat", "BEYOND"),
                                                "dim": it.get("dimension", "")}
                except Exception:
                    pass
        # anything still missing -> default BEYOND/other
        for c in concepts:
            cache.setdefault(str(c["cluster"]), {"cat": "BEYOND", "dim": "אחר"})
        cache_path.write_text(json.dumps(cache, ensure_ascii=False))
    return cache


def main():
    for domain in ["drugs", "weapon"]:
        concepts = list(csv.DictReader(
            open(OUT / f"concepts_{domain}.csv", encoding="utf-8-sig")))
        for c in concepts:
            c["cluster"] = int(c["cluster"])
            for r in FOCUS:
                c[f"cov_{r}"] = int(c[f"cov_{r}"])
        cmap = classify(domain, concepts)

        print("=" * 80); print("DOMAIN:", domain)
        # per-rep: concept-count and COVERAGE-MASS split into SCHEMA / BEYOND / IDENT
        for rep in FOCUS:
            cnt = Counter(); mass = Counter()
            for c in concepts:
                cov = c[f"cov_{rep}"]
                if cov == 0:
                    continue
                cat = mTYPE(cmap.get(str(c["cluster"]),{"cat":"BEYOND","dim":"אחר"})["cat"], domain)
                cnt[cat] += 1
                mass[cat] += cov
            tot_c = sum(cnt.values()); tot_m = sum(mass.values())
            print(f"\n  {rep}:")
            print(f"    concepts:  SCHEMA {cnt['SCHEMA']:4d} ({100*cnt['SCHEMA']/tot_c:2.0f}%)   "
                  f"BEYOND {cnt['BEYOND']:4d} ({100*cnt['BEYOND']/tot_c:2.0f}%)   "
                  f"IDENT {cnt['IDENT']:4d} ({100*cnt['IDENT']/tot_c:2.0f}%)")
            print(f"    coverage-mass: SCHEMA {100*mass['SCHEMA']/tot_m:2.0f}%   "
                  f"BEYOND {100*mass['BEYOND']/tot_m:2.0f}%   IDENT {100*mass['IDENT']/tot_m:2.0f}%")

        # BEYOND dimensions ranked by total coverage across reps
        dim_cov = Counter(); dim_examples = defaultdict(set)
        for c in concepts:
            m = cmap.get(str(c["cluster"]),{"cat":"BEYOND","dim":"אחר"})
            if mTYPE(m["cat"], domain) == "BEYOND":
                tot = sum(c[f"cov_{r}"] for r in FOCUS)
                dim = m["dim"] or "אחר"
                dim_cov[dim] += tot
                dim_examples[dim].add(c["label"])
        print("\n  BEYOND-SCHEMA dimensions (by total coverage across reps):")
        for dim, cov in dim_cov.most_common(15):
            ex = list(dim_examples[dim])[:5]
            print(f"      {cov:5d}  {dim:22s}  e.g. {ex}")

        # schema fan-out: how many distinct concepts collapse into each schema field
        fan = Counter()
        fan_ex = defaultdict(list)
        for c in concepts:
            cat = cmap.get(str(c["cluster"]),{"cat":"BEYOND","dim":"אחר"})["cat"]
            if cat in SCHEMA[domain]:
                fan[cat] += 1
                fan_ex[cat].append(c["label"])
        print("\n  SCHEMA fan-out (distinct concepts that reduce to each schema field):")
        for fld, n in fan.most_common():
            print(f"      {n:4d} concepts -> {fld}   e.g. {fan_ex[fld][:6]}")


def mTYPE(cat, domain):
    if cat == "BEYOND":
        return "BEYOND"
    if cat == "IDENT":
        return "IDENT"
    if cat in SCHEMA[domain]:
        return "SCHEMA"
    return "BEYOND"  # unknown label -> treat as beyond


if __name__ == "__main__":
    main()
