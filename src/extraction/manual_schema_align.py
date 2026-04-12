"""
Map granular extractor dicts (features_extract_* 2.py / to_dict()) into the same JSON keys
as similarity_database_fe.csv (manual annotation) so you can compare content apples-to-apples.

Drugs: 6 Hebrew keys.
Weapon: 18 Hebrew keys (full union used across the GT file).
"""

from __future__ import annotations

from typing import Any, Dict, List

# --- Drugs (manual union) ---
DRUGS_MANUAL_KEYS = [
    "מכירה לסוכן",
    "מעבדה",
    "סוג הסם, כמות",
    "עבירה",
    "עבירות נלוות כן/לא",
    "תפקיד",
]

# --- Weapon: exact key strings as in manual CSV ---
WEAPON_MANUAL_KEYS = [
    "אופן החזקת הנשק",
    "אופן קבלת הנשק",
    "כמות תחמושת",
    "מטרה-סיבת העבירה",
    "מספר עבירה",
    "סוג הנשק - אם לא נמצא בטבלה",
    "סוג הנשק [אקדח]",
    "סוג הנשק [רובה סער ]",
    "סוג הנשק [רובה צייד]",
    "סוג הנשק [רימון הלם/גז]",
    "סוג הנשק [רימון רסס]",
    "סוג הנשק [תת מקלע מאולתר]",
    "סוג הנשק [תת מקלע]",
    "סוג עבירה",
    "סטטוס הנשק",
    "עבירות נוספות",
    "שימוש",
    "תכנון",
]

WEAPON_COUNT_FIELDS: List[tuple[str, str]] = [
    ("סוג הנשק [אקדח]", "pistol"),
    ("סוג הנשק [תת מקלע]", "submachine_gun"),
    ("סוג הנשק [תת מקלע מאולתר]", "improvised_submachine_gun"),
    ("סוג הנשק [רובה צייד]", "hunting_rifle"),
    ("סוג הנשק [רובה סער ]", "assault_rifle"),
    ("סוג הנשק [רימון הלם/גז]", "stun_grenade"),
    ("סוג הנשק [רימון רסס]", "tear_gas_grenade"),
]

DRUG_QUANTITY_FIELDS = [
    "lsd",
    "methamphetamine",
    "ayahuasca",
    "cathinone",
    "ketamine",
    "hashish",
    "methylmethcathinone",
    "cannabis_plants",
    "cannabis",
    "mdma",
    "cocaine",
]


def _blank(v: Any) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    return s == "" or s == "לא"


def _norm_yes_no(v: Any) -> str:
    if v is None:
        return "לא"
    s = str(v).strip().lower()
    if s in ("כן", "yes", "true", "1"):
        return "כן"
    return "לא" if _blank(v) else str(v).strip()


def _weapon_count(v: Any) -> Any:
    if v is None:
        return 0
    if isinstance(v, bool):
        return int(v)
    try:
        x = float(v)
        if x == int(x):
            return int(x)
        return x
    except (TypeError, ValueError):
        return 0


def _build_drug_quantity(g: Dict[str, Any]) -> str:
    summary = g.get("drug_type_quantity_summary")
    if summary and str(summary).strip():
        return str(summary).strip()
    parts: List[str] = []
    for field in DRUG_QUANTITY_FIELDS:
        val = g.get(field)
        if val and str(val).strip() and str(val).strip() != "לא":
            parts.append(str(val).strip())
    return ", ".join(parts)


def _section_active(val: Any) -> bool:
    if val is None:
        return False
    s = str(val).strip().lower()
    if s in ("", "לא", "no", "false"):
        return False
    return True


def _build_drug_offense_label(g: Dict[str, Any]) -> str:
    """
    מיפוי סעיף בפקודה / בפסק → תוויות בווקטור (כמו GT):

    סעיף 6 → ייצור
    סעיף 7(א)(ג) / 7(ג) (נשמר ב-section_7) → החזקה שלא לצריכה עצמית
    סעיף 10 (ב-other_drug_offense) → כלים
    סעיף 13, 14 → יבוא/סחר
    סעיף 19 / 19א (נשמר ב-section_19) → 19
    סעיף 25 לחוק העונשין (penal_25_attempt) → ניסיון
    """
    parts: List[str] = []
    if _section_active(g.get("section_6")):
        parts.append("ייצור")
    if _section_active(g.get("section_7")):
        parts.append("החזקה שלא לצריכה עצמית")
    if _section_active(g.get("section_13")):
        parts.append("יבוא/סחר")
    if _section_active(g.get("section_14")):
        parts.append("יבוא/סחר")
    if _section_active(g.get("section_19")):
        parts.append("19")

    odo_s = str(g.get("other_drug_offense") or "").strip()
    if odo_s and odo_s != "לא" and any(x in odo_s for x in ("10", "כלים", "כלי")):
        parts.append("כלים")

    if _section_active(g.get("penal_25_attempt")):
        parts.append("ניסיון")

    seen = set()
    out: List[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return ", ".join(out)


def _dedupe_csv_tokens(s: str) -> str:
    if not s:
        return s
    tokens = [t.strip() for t in s.split(",") if t.strip()]
    seen = set()
    out: List[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return ", ".join(out)


def granular_to_manual_drugs(g: Dict[str, Any]) -> Dict[str, Any]:
    """Granular DrugFeatureExtractor.to_dict() -> 6 manual keys."""
    from drug_offense_categories import canonical_offense_label_from_granular_dict

    offense = canonical_offense_label_from_granular_dict(g)

    add = g.get("additional_offenses")
    if add and str(add).strip() and str(add).strip() != "לא":
        nl = "כן"
    else:
        nl = "לא"

    return {
        "מכירה לסוכן": _norm_yes_no(g.get("sold_to_agent")),
        "מעבדה": _norm_yes_no(g.get("laboratory")),
        "סוג הסם, כמות": _build_drug_quantity(g),
        "עבירה": offense,
        "עבירות נלוות כן/לא": nl,
        "תפקיד": str(g.get("role") or "").strip(),
    }


def _weapon_extra_types_line(g: Dict[str, Any]) -> str:
    """Types not in the 7 manual count columns — fold into 'סוג הנשק - אם לא נמצא בטבלה'."""
    bits: List[str] = []
    if g.get("other_weapon"):
        bits.append(str(g.get("other_weapon")).strip())
    extras = [
        ("molotov", "בקבוק תבערה"),
        ("explosive", "מטען חבלה"),
        ("lau_missile", "טיל לאו"),
        ("matador_missile", "טיל מטאדור"),
        ("sniper_rifle", "רובה צלפים"),
        ("improvised_explosive", "מטען מאולתר"),
        ("improvised_assault_rifle", "רובה סער מאולתר"),
    ]
    for field, label in extras:
        n = _weapon_count(g.get(field))
        if isinstance(n, (int, float)) and n != 0:
            bits.append(f"{label}: {n}")
    return ", ".join(bits)


def granular_to_manual_weapon(g: Dict[str, Any]) -> Dict[str, Any]:
    """Granular FeatureExtractor.to_dict() -> 18 manual keys."""
    out: Dict[str, Any] = {k: "" for k in WEAPON_MANUAL_KEYS}
    for heb, eng in WEAPON_COUNT_FIELDS:
        out[heb] = _weapon_count(g.get(eng))

    out["אופן החזקת הנשק"] = (g.get("storage_method") or "").strip()
    out["אופן קבלת הנשק"] = (g.get("how_obtained") or "").strip()
    out["כמות תחמושת"] = (g.get("ammunition_quantity") or "").strip()
    out["מטרה-סיבת העבירה"] = (g.get("purpose") or "").strip()
    out["מספר עבירה"] = (g.get("offense_number") or "").strip()
    out["סוג עבירה"] = (g.get("offense_type") or "").strip()
    out["סטטוס הנשק"] = (g.get("weapon_status") or "").strip()
    out["עבירות נוספות"] = (g.get("additional_offenses") or "").strip()
    out["שימוש"] = (g.get("weapon_use") or "").strip()
    out["תכנון"] = (g.get("planning") or "").strip()

    extra = _weapon_extra_types_line(g)
    out["סוג הנשק - אם לא נמצא בטבלה"] = extra

    return out


def align_granular_to_manual(domain: str, granular: Dict[str, Any]) -> Dict[str, Any]:
    if domain == "drugs":
        return granular_to_manual_drugs(granular)
    if domain == "weapon":
        return granular_to_manual_weapon(granular)
    raise ValueError(domain)
