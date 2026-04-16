"""
convert_gpt_cache_to_fe.py  (weapon)
Converts GPT extraction cache → feature-vector CSV compatible with
experiments/data/wep/manual_fe.csv format.

Usage:
    python convert_gpt_cache_to_fe.py \
        --cache  ../cache/eval_weapon_gpt_cache.json \
        --pairs  ../../data/wep/facts.csv \
        --out    ../results/fe_gpt_extracted.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import pathlib

# ── weapon type keys (cache key → display label) ─────────────────────────
WEAPON_DISPLAY = {
    "אקדח":             "אקדח",
    "תת_מקלע":          "תת מקלע",
    "תת_מקלע_מאולתר":   "תת מקלע מאולתר",
    "בקבוק_תבערה":      "בקבוק תבערה",
    "מטען_חבלה":         "מטען חבלה",
    "רימון_רסס":         "רימון רסס",
    "רימון_הלם_גז":     "רימון הלם/גז",
    "טיל_לאו":           "טיל לאו",
    "טיל_מטאדור":        "טיל מטאדור",
    "רובה_צייד":         "רובה צייד",
    "רובה_צלפים":        "רובה צלפים",
    "מטען_חבלה_מאולתר":  "מטען חבלה מאולתר",
    "רובה_סער":          "רובה סער",
    "רובה_סער_מאולתר":   "רובה סער מאולתר",
}

# ── offense-type keys → display ───────────────────────────────────────────
OFFENSE_TYPE_DISPLAY = {
    "החזקת_נשק":           "החזקה נשק",
    "נשיאת_נשק":           "נשיאת נשק",
    "הובלת_נשק":           "הובלת נשק",
    "סחר_בנשק":            "סחר בנשק",
    "ניסיון_לסחר_בנשק":    "ניסיון לסחר בנשק",
    "ירי_באזור_מגורים":    "ירי באזור מגורים",
    "ייצור_נשק":           "ייצור נשק",
    "עבירות_אחרות_בנשק":   "ביצוע עבירות בנשק",
}

# ── weapon status keys → display ─────────────────────────────────────────
STATUS_DISPLAY = {
    "תקין":                    "תקין",
    "תקול":                    "תקול",
    "נשק_מפורק":               "נשק מפורק",
    "נשק_מופרד_מתחמושת":       "נשק מופרד מתחמושת",
    "נשק_עם_מחסנית_בהכנס":     "נשק עם מחסנית בהכנס",
    "נשק_עם_כדור_בקנה":        "נשק עם כדור בקנה",
}

# ── storage method keys → display ────────────────────────────────────────
STORAGE_DISPLAY = {
    "בבית":          "בבית",
    "ברכב":          "ברכב",
    "על_גופו":       "על גופו",
    "מוסלק_מוסתר":   "מוסלק - מוסתר",
    "סמוך_לבית":     "סמוך לבית",
}

# ── how obtained keys → display ───────────────────────────────────────────
HOW_OBTAINED_DISPLAY = {
    "רכש":   "רכש",
    "מאחר":  "מאחר",
    "מצא":   "מצא",
    "גנב":   "גנב",
    "ייצר":  "ייצר",
    "עבודה": "עבודה",
}

# ── purpose keys → display ────────────────────────────────────────────────
PURPOSE_DISPLAY = {
    "בצע_כסף":                       "בצע כסף",
    "הגנה_עצמית":                    "הגנה עצמית",
    "פחד":                           "פחד",
    "ביטחון":                        "ביטחון",
    "חברות_בארגון_פשע_או_טרור":      "חברות בארגון טרור",
    "סכסוך":                         "סכסוך",
    "חתונה":                         "חתונה",
    "תדמית":                         "תדמית",
}


def _build_offense_number(offense_dict: dict) -> str:
    parts = []
    if offense_dict.get("144_א"): parts.append("144 א")
    if offense_dict.get("144_ב"): parts.append("144 ב")
    return ", ".join(parts)


def _build_offense_type(offense_dict: dict) -> str:
    parts = [display for key, display in OFFENSE_TYPE_DISPLAY.items()
             if offense_dict.get(key)]
    return ", ".join(parts)


def _build_weapon_type(weapon_dict: dict) -> dict:
    """Returns dict of 'סוג הנשק [X]': count for present weapons."""
    result = {}
    for cache_key, display in WEAPON_DISPLAY.items():
        val = weapon_dict.get(cache_key, 0)
        if val:
            result[f"סוג הנשק [{display}]"] = float(val)
    other = weapon_dict.get("אחר", "")
    if other:
        result["סוג הנשק - אם לא נמצא בטבלה"] = other
    return result


def _build_status(status_dict: dict) -> str:
    for key, display in STATUS_DISPLAY.items():
        if status_dict.get(key):
            return display
    return ""


def _build_storage(storage_dict: dict) -> str:
    parts = [display for key, display in STORAGE_DISPLAY.items()
             if storage_dict.get(key)]
    return ", ".join(parts)


def _build_how_obtained(obtained_dict: dict) -> str:
    for key, display in HOW_OBTAINED_DISPLAY.items():
        if obtained_dict.get(key):
            return display
    return ""


def _build_purpose(purpose_dict: dict) -> str:
    for key, display in PURPOSE_DISPLAY.items():
        if purpose_dict.get(key):
            return display
    other = purpose_dict.get("אחר", "")
    return other if other else ""


def _build_usage(usage_dict: dict) -> str:
    """Map cache keys to the manual GT format value.
    Cache keys: לא, כן_ירי, ניסיון_לירי_ללא_הצלחה, זריקת_רימון, נקירת_נשק, הפעלת_מטען
    Manual format values: לא, כן,ירי, ניסיון לירי ללא הצלחה, זריקת רימון, נקירת נשק, הפעלת מטען"""
    KEY_TO_VALUE = {
        "כן_ירי":                  "כן,ירי",
        "ניסיון_לירי_ללא_הצלחה":  "ניסיון לירי ללא הצלחה",
        "זריקת_רימון":            "זריקת רימון",
        "נקירת_נשק":              "נקירת נשק",
        "הפעלת_מטען":             "הפעלת מטען",
    }
    if isinstance(usage_dict, dict):
        # Check non-"לא" keys first (priority to actual usage)
        for key, value in KEY_TO_VALUE.items():
            if usage_dict.get(key):
                return value
        if usage_dict.get("לא"):
            return "לא"
    return "לא"


def cache_to_feature_vector(pred: dict) -> dict:
    fv = {}

    # offense number
    off_num = _build_offense_number(pred.get("מספר_עבירה", {}))
    if off_num:
        fv["מספר עבירה"] = off_num

    # offense type
    off_type = _build_offense_type(pred.get("סוג_העבירה", {}))
    if off_type:
        fv["סוג עבירה"] = off_type

    # additional offenses
    side = pred.get("עבירות_נוספות", {})
    if isinstance(side, dict) and side.get("קיימות"):
        fv["עבירות נוספות"] = side.get("פירוט", "כן")

    # weapon type (adds multiple keys)
    fv.update(_build_weapon_type(pred.get("סוג_הנשק", {})))

    # weapon status
    status = _build_status(pred.get("סטטוס_הנשק", {}))
    if status:
        fv["סטטוס הנשק"] = status

    # planning
    tik = pred.get("תכנון")
    if isinstance(tik, dict):
        tik = tik.get("תכנון")
    fv["תכנון"] = "כן" if tik == 1 else "לא"

    # storage
    storage = _build_storage(pred.get("אופן_החזקת_הנשק", {}))
    if storage:
        fv["אופן החזקת הנשק"] = storage

    # how obtained
    obtained = _build_how_obtained(pred.get("אופן_קבלת_הנשק", {}))
    if obtained:
        fv["אופן קבלת הנשק"] = obtained

    # ammunition
    ammo = pred.get("כמות_תחמושת", "") or ""
    if ammo:
        fv["כמות תחמושת"] = ammo

    # purpose
    purpose = _build_purpose(pred.get("מטרה_סיבת_העבירה", {}))
    if purpose:
        fv["מטרה-סיבת העבירה"] = purpose

    # usage
    fv["שימוש"] = _build_usage(pred.get("שימוש", {}))

    return fv


def main():
    BASE = pathlib.Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=pathlib.Path,
                    default=BASE / "cache/eval_weapon_gpt_cache.json")
    ap.add_argument("--pairs", type=pathlib.Path,
                    default=BASE.parent / "data/wep/facts.csv")
    ap.add_argument("--out",   type=pathlib.Path,
                    default=BASE / "results/fe_gpt_extracted.csv")
    args = ap.parse_args()

    with open(args.cache, encoding="utf-8") as f:
        cache: dict = json.load(f)
    print(f"Loaded {len(cache)} cached predictions")

    with open(args.pairs, newline="", encoding="utf-8-sig") as f:
        pairs = list(csv.DictReader(f))
    print(f"Loaded {len(pairs)} pairs")

    missing: set[str] = set()
    rows_out = []
    for row in pairs:
        v1, v2 = row["verdict_1"], row["verdict_2"]
        p1, p2 = cache.get(v1), cache.get(v2)
        if p1 is None: missing.add(v1)
        if p2 is None: missing.add(v2)
        fv1 = cache_to_feature_vector(p1) if p1 else {}
        fv2 = cache_to_feature_vector(p2) if p2 else {}
        rows_out.append({
            "verdict_1":           v1,
            "verdict_2":           v2,
            "similarity_scale":    row["similarity_scale"],
            "similarity_binary_0": row["similarity_binary_0"],
            "similarity_binary_1": row["similarity_binary_1"],
            "feature_vector_1":    json.dumps(fv1, ensure_ascii=False),
            "feature_vector_2":    json.dumps(fv2, ensure_ascii=False),
        })

    if missing:
        print(f"⚠  Missing in cache ({len(missing)}): {sorted(missing)[:10]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=[
            "verdict_1", "verdict_2", "similarity_scale",
            "similarity_binary_0", "similarity_binary_1",
            "feature_vector_1", "feature_vector_2",
        ])
        w.writeheader()
        w.writerows(rows_out)

    print(f"✅ Wrote {len(rows_out)} pairs → {args.out}")


if __name__ == "__main__":
    main()
