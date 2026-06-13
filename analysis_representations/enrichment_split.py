"""Isolate GPT's actual contribution in the Hybrids by REMOVING the manual core
(which is present by construction: hybrid vector = manual features + GPT additions).
Compares SCHEMA/BEYOND/IDENT coverage-mass for ALL concepts vs ENRICHMENT-only."""
import csv, json
from collections import Counter
from pathlib import Path

csv.field_size_limit(10**9)
OUT = Path(__file__).parent / "out"
NV = {"drugs": 68, "weapon": 101}
SCHEMA = {
    "drugs": {"מכירה_לסוכן", "מעבדה", "סוג_הסם_וכמות", "עבירה", "עבירות_נלוות", "תפקיד"},
    "weapon": {"סוג_הנשק_וכמות", "אופן_החזקת_הנשק", "אופן_קבלת_הנשק", "כמות_תחמושת",
               "מטרה_סיבת_העבירה", "מספר_עבירה", "סוג_עבירה", "סטטוס_הנשק",
               "עבירות_נוספות", "שימוש", "תכנון"},
}


def cat3(c, dom):
    if c == "IDENT":
        return "IDENT"
    return "SCHEMA" if c in SCHEMA[dom] else "BEYOND"


for dom in ["drugs", "weapon"]:
    nv = NV[dom]
    rows = list(csv.DictReader(open(OUT / f"concepts_{dom}.csv", encoding="utf-8-sig")))
    smap = json.loads((OUT / f"schema_map_{dom}.json").read_text(encoding="utf-8"))
    print("=" * 64)
    print(f"DOMAIN {dom}  (concepts={len(rows)})")
    for rep in ["Hybrid-Manual", "Hybrid-Full", "GPT-Free", "GPT-Law"]:
        covc = f"cov_{rep}"
        allm = Counter()
        enrm = Counter()
        n_core = 0
        core = []
        for r in rows:
            cov = int(r[covc])
            if cov == 0:
                continue
            cat = cat3(smap.get(str(r["cluster"]), {"cat": "BEYOND"})["cat"], dom)
            allm[cat] += cov
            if cov >= 0.9 * nv:
                n_core += 1
                core.append((r["label"], cat))
            else:
                enrm[cat] += cov
        ta = sum(allm.values()) or 1
        te = sum(enrm.values()) or 1
        print(f"\n  {rep}: core(>=90% verdicts)={n_core} concepts")
        print(f"    ALL concepts mass:   SCHEMA {100*allm['SCHEMA']/ta:4.0f}%  "
              f"BEYOND {100*allm['BEYOND']/ta:4.0f}%  IDENT {100*allm['IDENT']/ta:4.0f}%")
        print(f"    ENRICHMENT-only mass: SCHEMA {100*enrm['SCHEMA']/te:4.0f}%  "
              f"BEYOND {100*enrm['BEYOND']/te:4.0f}%  IDENT {100*enrm['IDENT']/te:4.0f}%")
        if rep.startswith("Hybrid"):
            print("    injected core:", ", ".join(f"{l}({c})" for l, c in core))
