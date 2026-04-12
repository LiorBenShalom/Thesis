"""
convert_annotator_to_gt.py (drugs)
Converts the annotator Google-Forms CSV (v2 schema with single "סוג הסם, כמות" column)
→ gt_manual_drugs.csv format (split drug columns).

Each annotator response becomes one output row, so inter-annotator agreement
can be computed directly on a uniform schema.

Usage:
    python convert_annotator_to_gt.py \
        --in  /path/to/v2_annotator_responses.csv \
        --out /path/to/annotator_as_gt.csv
"""
from __future__ import annotations
import argparse
import csv
import pathlib
import re
import unicodedata

DRUG_COLS = [
    "LSD", "METHAMPHETAMINE", "האיוואסקה", "קתינון", "קטמין",
    "חשיש", "מתילמקאתינון", "קנבוס בשתילים", "קנבוס", "MDMA", "קוקאין",
]

DRUG_ALIASES = {
    "קוקאין": "קוקאין",
    "קוקאין,": "קוקאין",
    "mdma": "MDMA",
    "mdmaמ": "MDMA",
    "אקסטזי": "MDMA",
    "lsd": "LSD",
    "methamphetamine": "METHAMPHETAMINE",
    "מת־אמפטמין": "METHAMPHETAMINE",
    "מתאמפטמין": "METHAMPHETAMINE",
    "קנבוס": "קנבוס",
    "קנאביס": "קנבוס",
    "קנביס": "קנבוס",
    "קבנוס": "קנבוס",
    "קנבוסבשתילים": "קנבוס בשתילים",
    "קנאביסבשתילים": "קנבוס בשתילים",
    "קנביסבשתילים": "קנבוס בשתילים",
    "ketamine": "קטמין",
    "cocaine": "קוקאין",
    "cannabis": "קנבוס",
    "hashish": "חשיש",
    "שתילים": "קנבוס בשתילים",
    "חשיש": "חשיש",
    "קטמין": "קטמין",
    "קתינון": "קתינון",
    "מתילמקאתינון": "מתילמקאתינון",
    "מתילמטאקתינון": "מתילמקאתינון",
    "איוואסקה": "האיוואסקה",
    "האיוואסקה": "האיוואסקה",
}

UNIT_NORMALIZE = {
    'ק"ג': 'ק"ג',
    "קג": 'ק"ג',
    "קילו": 'ק"ג',
    "גר'": "גרם",
    "גרם": "גרם",
    "גרמים": "גרם",
    "מ\"ג": "מ\"ג",
    "מג": "מ\"ג",
    "טבליות": "טבליות",
    "טבליה": "טבליות",
    "יחידות": "יחידות",
    "יחידה": "יחידות",
    "שתילים": "שתילים",
    "שתיל": "שתילים",
    "מיליליטר": "מיליליטר",
    "מ\"ל": "מיליליטר",
    "מל": "מיליליטר",
    "נוזל": "מיליליטר",
}

OUTPUT_COLS = [
    "data_source", "מספר תיק", "שם קובץ התיק",
    "עבירות [סעיף 6]", "עבירות [סעיף 7]", "עבירות [סעיף 13]",
    "עבירות [סעיף 14]", "עבירות [סעיף 19]", "עבירות [סעיף 21]",
    "עבירות [סעיף 22]", "עבירות [61(א)(ג)]",
    "עבירת סמים שלא הייתה ברשימה", "עבירות נלוות כן/לא",
] + [f"סוג הסם [{d}]" for d in DRUG_COLS] + [
    "מעבדה", "תפקיד", "מכירה לסוכן", "עונש עיקרי", "ענש נלווה",
    "מתחם ענישה מאשימה (פרקליט)", "מתחם ענישה בא כוח (סנגור)",
    "מתחם ענישה שופט", "הערות מחשבות",
    # extras useful for agreement analysis
    "שם המתייג", "חותמת זמן",
]


def _strip_ws(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _normalize_drug_name(name: str) -> str | None:
    key = _strip_ws(name).lower().strip(",.:;[]()")
    # try exact
    if key in DRUG_ALIASES:
        return DRUG_ALIASES[key]
    # try upper
    for alias, canon in DRUG_ALIASES.items():
        if alias == key or alias == key.rstrip("ם") or alias == key.rstrip("ים"):
            return canon
    # last resort: contains
    for canon in DRUG_COLS:
        if _strip_ws(canon).lower() in key or key in _strip_ws(canon).lower():
            return canon
    return None


def _normalize_unit(u: str) -> str:
    u = u.strip().strip(",.:;[]()")
    return UNIT_NORMALIZE.get(u, u)


def parse_drug_quantity(text: str) -> dict[str, str]:
    """
    Parse free-text like '3.44 - ק"ג - קוקאין, 13189-טבליות-MDMAמ, 1301.44-גרם-MDMA'
    → {"קוקאין": "3.44-ק\"ג", "MDMA": "13189-טבליות, 1301.44-גרם"}
    """
    if not text or not text.strip():
        return {}
    # protect decimal commas: replace comma inside numbers
    SEP = "\u2063"
    t = re.sub(r'(\d),(\d)', lambda m: m.group(1) + SEP + m.group(2), text)
    result: dict[str, list[str]] = {}
    for seg in t.split(","):
        seg = seg.replace(SEP, ",").strip()
        if not seg:
            continue
        # find first number
        m = re.search(r"(\d+(?:\.\d+)?)", seg)
        if not m:
            continue
        amount = m.group(1)
        rest = (seg[:m.start()] + " " + seg[m.end():]).strip()
        # split rest by '-' or whitespace into tokens
        tokens = [tok for tok in re.split(r"[-\s]+", rest) if tok]
        # detect drug name and unit from tokens
        drug = None
        unit = None
        unknown = []
        for tok in tokens:
            nd = _normalize_drug_name(tok)
            if nd and drug is None:
                drug = nd
                continue
            nu = _normalize_unit(tok)
            if nu in UNIT_NORMALIZE.values() and unit is None:
                unit = nu
                continue
            unknown.append(tok)
        # fall-back: entire rest string to detect
        if drug is None:
            drug = _normalize_drug_name(rest)
        if unit is None:
            for tok in unknown:
                nu = _normalize_unit(tok)
                if nu in UNIT_NORMALIZE.values():
                    unit = nu
                    break
        if drug is None:
            # skip unparseable segment
            continue
        piece = f"{amount}-{unit}" if unit else amount
        result.setdefault(drug, []).append(piece)
    return {k: ", ".join(v) for k, v in result.items()}


def convert_row(r: dict) -> dict:
    out = {c: "" for c in OUTPUT_COLS}
    out["data_source"] = "GT"
    out["מספר תיק"] = (r.get("מספר תיק") or "").strip()
    fname = (r.get("שם קובץ התיק") or "").strip()
    # strip .doc/.docx extension
    out["שם קובץ התיק"] = re.sub(r"\.docx?$", "", fname)
    # copy section fields 1:1
    for col in [
        "עבירות [סעיף 6]", "עבירות [סעיף 7]", "עבירות [סעיף 13]",
        "עבירות [סעיף 14]", "עבירות [סעיף 19]", "עבירות [סעיף 21]",
        "עבירות [סעיף 22]", "עבירות [61(א)(ג)]",
        "עבירת סמים שלא הייתה ברשימה", "עבירות נלוות כן/לא",
        "מעבדה", "מכירה לסוכן", "עונש עיקרי", "ענש נלווה",
        "מתחם ענישה מאשימה (פרקליט)", "מתחם ענישה בא כוח (סנגור)",
        "מתחם ענישה שופט", "הערות מחשבות",
    ]:
        out[col] = (r.get(col) or "").strip()
    # תפקיד column may have trailing space in source
    role_val = r.get("תפקיד") or r.get("תפקיד ") or ""
    out["תפקיד"] = role_val.strip()
    # split drug text into per-drug columns
    drug_text = r.get("סוג הסם, כמות") or ""
    per_drug = parse_drug_quantity(drug_text)
    for drug, val in per_drug.items():
        out[f"סוג הסם [{drug}]"] = val
    # annotator metadata (kept for agreement analysis)
    out["שם המתייג"] = (r.get("שם המתייג") or "").strip()
    out["חותמת זמן"] = (r.get("חותמת זמן") or "").strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    with open(args.inp, newline="", encoding="utf-8-sig") as f:
        rows_in = list(csv.DictReader(f))
    print(f"Loaded {len(rows_in)} annotator rows")

    rows_out = [convert_row(r) for r in rows_in]
    unparsed = [r for r, o in zip(rows_in, rows_out)
                if (r.get("סוג הסם, כמות") or "").strip()
                and not any(o.get(f"סוג הסם [{d}]") for d in DRUG_COLS)]
    if unparsed:
        print(f"⚠  {len(unparsed)} rows had drug text but nothing parsed; examples:")
        for r in unparsed[:3]:
            print(f"   - [{r.get('שם קובץ התיק')}] {r.get('סוג הסם, כמות')!r}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_COLS)
        w.writeheader()
        w.writerows(rows_out)
    print(f"✅ Wrote {len(rows_out)} rows → {args.out}")


if __name__ == "__main__":
    main()
