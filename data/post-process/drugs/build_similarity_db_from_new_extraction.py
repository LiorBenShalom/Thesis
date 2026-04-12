"""
Builds similarity_database_fe_gpt_schema_v2.csv from gpt_extracted_drugs.json
(the new SmartTag-fixed extraction) using the same pair structure as
similarity_database_fe.csv.

Feature vector format (6 keys, same as manual GT):
  מכירה לסוכן, מעבדה, סוג הסם כמות, עבירה, עבירות נלוות כן/לא, תפקיד
"""
from __future__ import annotations
import csv, json
from pathlib import Path

BASE = Path(__file__).parent
EXTRACTED_JSON  = BASE / "gpt_extracted_drugs.json"
GT_PAIRS_CSV    = BASE / "similarity_database_fe.csv"
OUTPUT_CSV      = BASE / "similarity_database_fe_gpt_schema_v2.csv"

DRUG_NAME_MAP = {
    "LSD":              "LSD",
    "METHAMPHETAMINE":  "מתאמפטמין",
    "האיוואסקה":        "איוואסקה",
    "קתינון":           "קתינון",
    "קטמין":            "קטמין",
    "חשיש":             "חשיש",
    "מתילמקאתינון":     "מתילמקאתינון",
    "קנבוס_בשתילים":    "קנבוס (שתילים)",
    "קנבוס":            "קנבוס",
    "MDMA":             "MDMA",
    "קוקאין":           "קוקאין",
    "אחר":              "סם אחר",
}

UNIT_HEB = {
    "גרם":        "גרם",
    "מיליליטר":   "מיליליטר",
    "שתילים":     "שתילים",
    "טבליות":     "טבליות",
    "בולים":      "בולים",
    "יחידות":     "יחידות",
}


def _parse_amount_str(s: str) -> tuple[float | None, str]:
    """Parse '3440-גרם' → (3440.0, 'גרם')"""
    import re
    m = re.match(r"^([\d\.,]+)-(.+)$", s.strip())
    if not m:
        return None, s
    amt = float(m.group(1).replace(",", ""))
    unit = m.group(2).strip()
    return amt, unit


def _format_drug_quantity(sot: dict) -> str:
    parts = []
    for key, heb in DRUG_NAME_MAP.items():
        entries = sot.get(key, [])
        if not entries:
            continue
        # pool entries by unit
        unit_totals: dict[str, float] = {}
        for e in entries:
            amt, unit = _parse_amount_str(e)
            if amt is not None:
                unit_totals[unit] = unit_totals.get(unit, 0) + amt
        for unit, total in unit_totals.items():
            u_heb = UNIT_HEB.get(unit, unit)
            if unit in ("גרם", "מיליליטר"):
                parts.append(f"{heb} במשקל {total:g} {u_heb}")
            elif unit in ("שתילים", "טבליות", "בולים", "יחידות"):
                parts.append(f"{heb} {int(total)} {u_heb}")
            else:
                parts.append(f"{heb} {total:g} {u_heb}")
    return ", ".join(parts)


def _format_offense(mn: dict) -> str:
    parts = []
    if mn.get("סעיף_6"):
        parts.append("ייצור")
    if mn.get("סעיף_7"):
        parts.append("החזקה שלא לצריכה עצמית")
    if mn.get("סעיף_13"):
        parts.append("יבוא/סחר")
    if mn.get("סעיף_14") and "יבוא/סחר" not in parts:
        parts.append("יבוא/סחר")
    if mn.get("סעיף_19") or mn.get("סעיף_19א"):
        parts.append("19")
    if mn.get("סעיף_21"):
        parts.append("כלים")
    if mn.get("סעיף_22"):
        parts.append("ניסיון")
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p); out.append(p)
    return ", ".join(out)


def _format_role(taf: dict) -> str:
    parts = []
    if taf.get("בעל_הסמים"):
        parts.append("בעל הסמים")
    else:
        parts.append("לא בעל הסמים")
    if taf.get("בעל_המעבדה"):
        parts.append("בעל המעבדה")
    else:
        parts.append("לא בעל המעבדה")
    return ", ".join(parts)


def build_feature_vector(gpt: dict) -> dict:
    return {
        "מכירה לסוכן": "כן" if gpt.get("מכירה_לסוכן", {}).get("מכירה_לסוכן") else "לא",
        "מעבדה":        "כן" if gpt.get("מעבדה", {}).get("מעבדה") else "לא",
        "סוג הסם, כמות": _format_drug_quantity(gpt.get("סוג_הסם", {})),
        "עבירה":         _format_offense(gpt.get("מספר_עבירה", {})),
        "עבירות נלוות כן/לא": "כן" if gpt.get("עבירות_נלוות", {}).get("עבירות_נלוות") else "לא",
        "תפקיד":         _format_role(gpt.get("תפקיד", {})),
    }


def main():
    with open(EXTRACTED_JSON, encoding="utf-8") as f:
        extracted = json.load(f)

    rows_out = []
    missing = []
    with open(GT_PAIRS_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v1, v2 = row["verdict_1"], row["verdict_2"]
            g1 = extracted.get(v1)
            g2 = extracted.get(v2)
            if g1 is None:
                missing.append(v1)
            if g2 is None:
                missing.append(v2)
            fv1 = build_feature_vector(g1) if g1 else {}
            fv2 = build_feature_vector(g2) if g2 else {}
            rows_out.append({
                "verdict_1": v1,
                "verdict_2": v2,
                "similarity_scale":    row["similarity_scale"],
                "similarity_binary_0": row["similarity_binary_0"],
                "similarity_binary_1": row["similarity_binary_1"],
                "feature_vector_1": json.dumps(fv1, ensure_ascii=False),
                "feature_vector_2": json.dumps(fv2, ensure_ascii=False),
            })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["verdict_1","verdict_2","similarity_scale",
                                           "similarity_binary_0","similarity_binary_1",
                                           "feature_vector_1","feature_vector_2"])
        w.writeheader()
        w.writerows(rows_out)

    if missing:
        uniq = sorted(set(missing))
        print(f"⚠️  Missing verdicts in extraction ({len(uniq)}): {uniq}")
    print(f"✅ Wrote {len(rows_out)} pairs → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
