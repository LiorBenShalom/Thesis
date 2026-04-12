"""
Build gpt_manual_features.csv — same pair format as similarity_database_fe.csv
but feature vectors come from gpt_extracted_features.json (GPT auto-extraction).

GPT dict keys use underscores; this script converts them to the canonical
Hebrew GT format (spaces, bracket notation for weapon types, etc.).
"""

import csv
import io
import json
import re

MANUAL_FE = "/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/weapon/similarity_database_fe.csv"
GPT_JSON  = "/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/code/gpt_extracted_features.json"
OUT_CSV   = "/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/weapon/gpt_manual_features.csv"


# ── GPT dict → canonical GT-format feature vector ──────────────────────────

def _u2l(s: str) -> str:
    return s.replace("_", " ")


WEAPON_KEY_MAP = {
    "אקדח":               "סוג הנשק [אקדח]",
    "תת_מקלע":            "סוג הנשק [תת מקלע]",
    "תת_מקלע_מאולתר":     "סוג הנשק [תת מקלע מאולתר]",
    "רובה_סער":           "סוג הנשק [רובה סער ]",
    "רובה_סער_מאולתר":    "סוג הנשק [רובה סער מאולתר]",
    "רובה_צייד":          "סוג הנשק [רובה צייד]",
    "רובה_צלפים":         "סוג הנשק [רובה צלפים]",
    "רימון_רסס":          "סוג הנשק [רימון רסס]",
    "רימון_הלם_גז":       "סוג הנשק [רימון הלם/גז]",
    "טיל_לאו":            "סוג הנשק [טיל לאו]",
    "טיל_מטאדור":         "סוג הנשק [טיל מטאדור]",
    "מטען_חבלה":          "סוג הנשק [מטען חבלה]",
    "מטען_חבלה_מאולתר":   "סוג הנשק [מטען חבלה מאולתר]",
    "בקבוק_תבערה":        "סוג הנשק [בקבוק תבערה]",
}

OFFENSE_TYPE_MAP = {
    "החזקת_נשק":          "החזקה נשק",
    "נשיאת_נשק":          "נשיאת נשק",
    "הובלת_נשק":          "הובלת נשק",
    "סחר_בנשק":           "סחר בנשק",
    "ניסיון_לסחר_בנשק":   "ניסיון לסחר",
    "ירי_באזור_מגורים":   "ירי באזור מגורים",
    "ייצור_נשק":          "ייצור נשק",
    "עבירות_אחרות_בנשק":  "עבירות אחרות",
}

PURPOSE_MAP = {
    "בצע_כסף":    "בצע כסף",
    "הגנה_עצמית": "הגנה עצמית",
    "חתונה":      "חתונה",
    "סכסוך":      "סכסוך",
    "תדמית":      "תדמית",
}

STORAGE_ORDER = ["בבית", "ברכב", "על_גופו", "מוסלק_מוסתר", "סמוך_לבית"]
STORAGE_LABEL = {
    "בבית":          "בבית",
    "ברכב":          "ברכב",
    "על_גופו":       "על גופו",
    "מוסלק_מוסתר":  "מוסלק - מוסתר",
    "סמוך_לבית":    "סמוך לבית",
}

USAGE_ORDER = ["כן_ירי", "ניסיון_לירי_ללא_הצלחה", "זריקת_רימון", "נקירת_נשק", "הפעלת_מטען"]
USAGE_LABEL = {
    "כן_ירי":                   "כן,ירי",
    "ניסיון_לירי_ללא_הצלחה":   "ניסיון לירי ללא הצלחה",
    "זריקת_רימון":              "זריקת רימון",
    "נקירת_נשק":               "נקירת נשק",
    "הפעלת_מטען":              "הפעלת מטען",
}


def gpt_to_gt_fv(gpt: dict) -> dict:
    """Convert a GPT feature dict → GT-format feature vector dict."""
    fv = {}

    # מספר עבירה
    offense_num = gpt.get("מספר_עבירה", {})
    parts = []
    if isinstance(offense_num, dict):
        if offense_num.get("144_א") == 1:
            parts.append("144 א")
        if offense_num.get("144_ב") == 1:
            parts.append("144 ב")
    if parts:
        fv["מספר עבירה"] = ", ".join(parts)

    # סוג עבירה
    offense_type = gpt.get("סוג_העבירה", {})
    if isinstance(offense_type, dict):
        labels = [OFFENSE_TYPE_MAP.get(k, _u2l(k)) for k, v in offense_type.items() if v == 1]
        if labels:
            fv["סוג עבירה"] = ", ".join(labels)

    # עבירות נוספות — כולל רק אם קיימות
    extra = gpt.get("עבירות_נוספות", {})
    if isinstance(extra, dict) and extra.get("קיימות") == 1:
        fv["עבירות נוספות"] = "כן"

    # סוג הנשק — bracket format with count 1.0
    weapon_type = gpt.get("סוג_הנשק", {})
    if isinstance(weapon_type, dict):
        for k, v in weapon_type.items():
            if k == "אחר":
                continue
            if v == 1:
                gt_key = WEAPON_KEY_MAP.get(k, f"סוג הנשק [{_u2l(k)}]")
                fv[gt_key] = 1.0

    # סטטוס הנשק
    status = gpt.get("סטטוס_הנשק", {})
    if isinstance(status, dict):
        for k, v in status.items():
            if v == 1:
                fv["סטטוס הנשק"] = _u2l(k)
                break

    # תכנון
    planning = gpt.get("תכנון", 0)
    fv["תכנון"] = "כן" if planning == 1 else "לא"

    # אופן החזקת הנשק
    storage = gpt.get("אופן_החזקת_הנשק", {})
    if isinstance(storage, dict):
        parts = [STORAGE_LABEL[k] for k in STORAGE_ORDER if storage.get(k) == 1]
        # any extra keys not in order
        for k, v in storage.items():
            if v == 1 and k not in STORAGE_ORDER:
                parts.append(_u2l(k))
        if parts:
            fv["אופן החזקת הנשק"] = ", ".join(parts)

    # אופן קבלת הנשק
    obtained = gpt.get("אופן_קבלת_הנשק", {})
    if isinstance(obtained, dict):
        for k, v in obtained.items():
            if v == 1:
                fv["אופן קבלת הנשק"] = _u2l(k)
                break

    # כמות תחמושת
    ammo = gpt.get("כמות_תחמושת", "")
    ammo_str = str(ammo or "").strip()
    # normalize GPT "לא מתואר" variants
    if re.match(r"^(לא מתואר|לא צוין|לא מוזכר|אין|ריק|מחרוזת ריקה)", ammo_str):
        ammo_str = ""
    if ammo_str:
        fv["כמות תחמושת"] = ammo_str

    # מטרה-סיבת העבירה
    purpose = gpt.get("מטרה_סיבת_העבירה", {})
    if isinstance(purpose, dict):
        for k, v in purpose.items():
            if k == "אחר":
                other = str(v or "").strip()
                if other and other != "0":
                    fv["מטרה-סיבת העבירה"] = other
                continue
            if v == 1:
                fv["מטרה-סיבת העבירה"] = PURPOSE_MAP.get(k, _u2l(k))
                break

    # שימוש
    usage = gpt.get("שימוש", {})
    if isinstance(usage, dict) and not usage.get("לא") == 1:
        parts = [USAGE_LABEL[k] for k in USAGE_ORDER if usage.get(k) == 1]
        if parts:
            fv["שימוש"] = ", ".join(parts)

    return fv


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    # Load pairs from manual_fe (same structure, similarity labels)
    with open(MANUAL_FE, newline="", encoding="utf-8-sig") as f:
        content = f.read().lstrip("\ufeff")
    reader = csv.DictReader(io.StringIO(content))
    pairs = list(reader)
    print(f"Loaded {len(pairs)} pairs from manual_fe")

    # Load GPT features
    with open(GPT_JSON, encoding="utf-8") as f:
        gpt_all = json.load(f)
    print(f"Loaded {len(gpt_all)} verdicts from GPT JSON")

    out_rows = []
    skipped = 0
    for row in pairs:
        v1 = row["verdict_1"]
        v2 = row["verdict_2"]

        gpt1 = gpt_all.get(v1, {})
        gpt2 = gpt_all.get(v2, {})

        if gpt1.get("_error") or gpt2.get("_error"):
            skipped += 1
            continue

        fv1 = gpt_to_gt_fv(gpt1)
        fv2 = gpt_to_gt_fv(gpt2)

        out_rows.append({
            "verdict_1":         v1,
            "verdict_2":         v2,
            "similarity_scale":  row["similarity_scale"],
            "similarity_binary_0": row["similarity_binary_0"],
            "similarity_binary_1": row["similarity_binary_1"],
            "feature_vector_1":  json.dumps(fv1, ensure_ascii=False),
            "feature_vector_2":  json.dumps(fv2, ensure_ascii=False),
        })

    print(f"Output: {len(out_rows)} pairs (skipped {skipped} with errors)")

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "verdict_1", "verdict_2", "similarity_scale",
            "similarity_binary_0", "similarity_binary_1",
            "feature_vector_1", "feature_vector_2",
        ])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Saved → {OUT_CSV}")


if __name__ == "__main__":
    main()
