"""
Eval script: run extract_weapon_features_simple.py on GT cases and compare per-feature accuracy.

Usage:
    python eval_weapon_features.py \
        --gt   ../../data/wep/manual-fe-gt.csv \
        --docx ../../../weapon/weapon_docx/ \
        --out  ../results/eval_weapon_results.csv
"""

import argparse
import csv
import json
import os
import re
import sys
import pathlib

sys.path.insert(0, os.path.dirname(__file__))
from extract_weapon_features_simple import (
    read_docx,
    extract_offense_number,
    extract_offense_type,
    extract_additional_offenses,
    extract_weapon_type,
    extract_weapon_status,
    extract_planning,
    extract_storage_method,
    extract_how_obtained,
    extract_ammunition,
    extract_purpose,
    extract_usage,
    apply_conditional_defaults,
)

# ── GT column names ───────────────────────────────────────────────────────
GT_CASE_COL       = "case"
GT_OFFENSE_NUM    = "מספר עבירה"
GT_OFFENSE_TYPE   = "סוג עבירה"
GT_SIDE_OFFENSES  = "עבירות נוספות"
GT_WEAPON_COLS = {
    "אקדח":             "סוג הנשק [אקדח]",
    "תת_מקלע":          "סוג הנשק [תת מקלע]",
    "תת_מקלע_מאולתר":   "סוג הנשק [תת מקלע מאולתר]",
    "בקבוק_תבערה":      "סוג הנשק [בקבוק תבערה]",
    "מטען_חבלה":         "סוג הנשק [מטען חבלה]",
    "רימון_רסס":         "סוג הנשק [רימון רסס]",
    "רימון_הלם_גז":     "סוג הנשק [רימון הלם/גז]",
    "טיל_לאו":           "סוג הנשק [טיל לאו ]",
    "טיל_מטאדור":        "סוג הנשק [טיל מטאדור]",
    "רובה_צייד":         "סוג הנשק [רובה צייד]",
    "רובה_צלפים":        "סוג הנשק [רובה צלפים]",
    "מטען_חבלה_מאולתר":  "סוג הנשק [מטען חבלה מאולתר]",
    "רובה_סער":          "סוג הנשק [רובה סער ]",
    "רובה_סער_מאולתר":   "סוג הנשק [רובה סער מאולתר]",
}
GT_WEAPON_STATUS  = "סטטוס הנשק"
GT_PLANNING       = "תכנון"
GT_STORAGE        = "אופן החזקת הנשק"
GT_HOW_OBTAINED   = "אופן קבלת הנשק"
GT_AMMUNITION     = "כמות תחמושת"
GT_PURPOSE        = "מטרה-סיבת העבירה"
GT_USAGE          = "שימוש"


# ── helpers ───────────────────────────────────────────────────────────────

def _yn(val: str) -> int | None:
    v = (val or "").strip()
    if v == "כן": return 1
    if v == "לא": return 0
    return None


def _norm_set(val: str) -> set[str]:
    """Split comma-separated GT value into a normalized set of strings."""
    if not val or not val.strip():
        return set()
    return {x.strip() for x in val.split(",") if x.strip()}


# ── offense number ────────────────────────────────────────────────────────

def _parse_gt_offense_number(val: str) -> dict:
    """GT: '144 א', '144 ב', '144 א, 144 ב' → {144_א: 0/1, 144_ב: 0/1}"""
    v = (val or "").lower()
    return {
        "144_א": 1 if "144" in v and "א" in v else 0,
        "144_ב": 1 if "144" in v and "ב" in v else 0,
    }


def _compare_offense_number(gt_row: dict, pred: dict) -> dict:
    gt = _parse_gt_offense_number(gt_row.get(GT_OFFENSE_NUM, ""))
    pr = pred.get("מספר_עבירה", {})
    results = {}
    for key in ["144_א", "144_ב"]:
        gv = gt.get(key, 0)
        pv = pr.get(key, 0)
        results[f"מספר_עבירה_{key}"] = {"match": gv == pv, "gt": gv, "pred": pv}
    return results


# ── offense type ──────────────────────────────────────────────────────────

OFFENSE_TYPE_MAP = {
    "החזקה נשק":         "החזקת_נשק",
    "החזקת נשק":         "החזקת_נשק",
    "נשיאת נשק":         "נשיאת_נשק",
    "נשיאה נשק":         "נשיאת_נשק",
    "הובלת נשק":         "הובלת_נשק",
    "הובלה נשק":         "הובלת_נשק",
    "סחר בנשק":          "סחר_בנשק",
    "סחר נשק":           "סחר_בנשק",
    # Both "ניסיון לסחר" and "ניסיון לסחר בנשק" → same key (handles user's duplicate concern)
    "ניסיון לסחר בנשק":  "ניסיון_לסחר_בנשק",
    "ניסיון לסחר":       "ניסיון_לסחר_בנשק",
    "ניסיון סחר בנשק":   "ניסיון_לסחר_בנשק",
    "ירי באזור מגורים":  "ירי_באזור_מגורים",
    "ייצור נשק":         "ייצור_נשק",
    "ביצוע עבירות בנשק": "עבירות_אחרות_בנשק",
}

def _parse_gt_offense_type(val: str) -> dict:
    result = {k: 0 for k in ["החזקת_נשק","נשיאת_נשק","הובלת_נשק","סחר_בנשק",
                               "ניסיון_לסחר_בנשק","ירי_באזור_מגורים","ייצור_נשק","עבירות_אחרות_בנשק"]}
    for part in _norm_set(val):
        key = OFFENSE_TYPE_MAP.get(part)
        if key:
            result[key] = 1
        elif part:
            result["עבירות_אחרות_בנשק"] = 1
    return result


def _compare_offense_type(gt_row: dict, pred: dict) -> dict:
    gt = _parse_gt_offense_type(gt_row.get(GT_OFFENSE_TYPE, ""))
    pr = pred.get("סוג_העבירה", {})
    results = {}
    for key in gt:
        gv = gt[key]
        pv = pr.get(key, 0)
        results[f"סוג_עבירה_{key}"] = {"match": gv == pv, "gt": gv, "pred": pv}
    return results


# ── additional offenses ───────────────────────────────────────────────────

def _compare_additional_offenses(gt_row: dict, pred: dict) -> dict:
    gt_val = (gt_row.get(GT_SIDE_OFFENSES) or "").strip()
    gt_has = 1 if gt_val else 0
    pr = pred.get("עבירות_נוספות", {})
    pv = pr.get("קיימות", 0)
    return {"עבירות_נוספות": {"match": gt_has == pv, "gt": gt_has, "pred": pv}}


# ── weapon type ───────────────────────────────────────────────────────────

IMPROVISED_PAIRS = [
    ("תת_מקלע", "תת_מקלע_מאולתר"),
    ("רובה_סער", "רובה_סער_מאולתר"),
    ("מטען_חבלה", "מטען_חבלה_מאולתר"),
]

def _compare_weapon_type(gt_row: dict, pred: dict) -> dict:
    """Compare weapon types as counts. Partial 0.8 for X ↔ X_מאולתר confusion of same base."""
    pr = pred.get("סוג_הנשק", {})
    # First compute raw counts
    counts = {}
    for pred_key, gt_col in GT_WEAPON_COLS.items():
        gt_raw = (gt_row.get(gt_col) or "").strip()
        try: gv = int(float(gt_raw)) if gt_raw else 0
        except ValueError: gv = 0
        try: pv = int(pr.get(pred_key, 0))
        except (ValueError, TypeError): pv = 0
        counts[pred_key] = {"gt": gv, "pred": pv}

    # Detect מאולתר-vs-base confusion: for each pair, if GT has one and PRED has the other (but not both)
    partial_keys = set()
    for base, imp in IMPROVISED_PAIRS:
        if base not in counts or imp not in counts: continue
        gb, pb = counts[base]["gt"], counts[base]["pred"]
        gi, pi = counts[imp]["gt"], counts[imp]["pred"]
        # GT has base, PRED has imp (or vice versa) — same base weapon, only improvised flag differs
        if (gb > 0 and pb == 0 and gi == 0 and pi > 0) or (gi > 0 and pi == 0 and gb == 0 and pb > 0):
            partial_keys.add(base); partial_keys.add(imp)

    results = {}
    for key, c in counts.items():
        if key in partial_keys:
            results[f"נשק_{key}"] = {"match": 0.8, "gt": c["gt"], "pred": c["pred"]}
        else:
            results[f"נשק_{key}"] = {"match": (c["gt"] == c["pred"]), "gt": c["gt"], "pred": c["pred"]}
    return results


# ── weapon status ─────────────────────────────────────────────────────────

STATUS_MAP = {
    "תקין":                       "תקין",
    "תקול":                       "תקול",
    "נשק מפורק":                  "נשק_מפורק",
    "נשק מפורק, תקול":            "תקול",       # most severe = תקול? actually both; pick תקול
    "נשק מופרד מתחמושת":          "נשק_מופרד_מתחמושת",
    "נשק עם מחסנית בהכנס":        "נשק_עם_מחסנית_בהכנס",
    "נשק עם כדור בקנה":           "נשק_עם_כדור_בקנה",
}
STATUS_KEYS = ["תקין","תקול","נשק_מפורק","נשק_מופרד_מתחמושת","נשק_עם_מחסנית_בהכנס","נשק_עם_כדור_בקנה"]

def _parse_gt_status(val: str) -> str | None:
    v = (val or "").strip()
    return STATUS_MAP.get(v)


def _compare_weapon_status(gt_row: dict, pred: dict) -> dict:
    """Partial match 0.8 for 'תקין ↔ נשק_מופרד_מתחמושת' confusion (both = unloaded/safe)."""
    gt_label = _parse_gt_status(gt_row.get(GT_WEAPON_STATUS, ""))
    pr = pred.get("סטטוס_הנשק", {})
    pred_label = next((k for k in STATUS_KEYS if pr.get(k)), None)

    # Is this the specific תקין ↔ מופרד confusion?
    close_pair = {"תקין", "נשק_מופרד_מתחמושת"}
    is_close_confusion = (gt_label in close_pair and pred_label in close_pair and gt_label != pred_label)

    results = {}
    for key in STATUS_KEYS:
        gv = 1 if gt_label == key else 0
        pv = pr.get(key, 0)
        if is_close_confusion and (key == gt_label or key == pred_label):
            # Give 0.8 partial for both the GT and PRED key (instead of 0/0)
            results[f"סטטוס_{key}"] = {"match": 0.8, "gt": gv, "pred": pv}
        else:
            results[f"סטטוס_{key}"] = {"match": (gv == pv), "gt": gv, "pred": pv}
    return results


# ── planning ──────────────────────────────────────────────────────────────

def _compare_planning(gt_row: dict, pred: dict) -> dict:
    gv = _yn(gt_row.get(GT_PLANNING, ""))
    pv = pred.get("תכנון")
    if isinstance(pv, dict):
        pv = pv.get("תכנון")
    match = (gv == pv) if gv is not None else True
    return {"תכנון": {"match": match, "gt": gv, "pred": pv}}


# ── storage method ────────────────────────────────────────────────────────

STORAGE_KEYS = ["בבית","ברכב","על_גופו","מוסלק_מוסתר","סמוך_לבית"]

def _parse_gt_storage(val: str) -> set[str]:
    parts = _norm_set(val)
    result = set()
    for p in parts:
        if "בבית" in p:   result.add("בבית")
        if "ברכב" in p:   result.add("ברכב")
        if "על גופו" in p: result.add("על_גופו")
        if "מוסלק" in p or "מוסתר" in p: result.add("מוסלק_מוסתר")
        if "סמוך" in p:   result.add("סמוך_לבית")
    return result


def _compare_storage(gt_row: dict, pred: dict) -> dict:
    """אם ההפרש היחיד הוא 'מוסלק_מוסתר' — נותן partial match של 0.8 לאותו שדה (פנלטי 20%)
    במקום mismatch מלא. שאר השדות ב-100% normal."""
    gt_set = _parse_gt_storage(gt_row.get(GT_STORAGE, ""))
    pr = pred.get("אופן_החזקת_הנשק", {})
    raw = {}
    for key in STORAGE_KEYS:
        gv = 1 if key in gt_set else 0
        pv = pr.get(key, 0)
        raw[key] = {"gt": gv, "pred": pv, "binary_match": (gv == pv)}

    # Check if only מוסלק_מוסתר differs (everything else matches)
    other_match = all(raw[k]["binary_match"] for k in STORAGE_KEYS if k != "מוסלק_מוסתר")
    moslak_diff = not raw["מוסלק_מוסתר"]["binary_match"]

    results = {}
    for key in STORAGE_KEYS:
        d = raw[key]
        if key == "מוסלק_מוסתר" and moslak_diff and other_match:
            # All else matches → partial 0.8 for this field
            results[f"החזקה_{key}"] = {"match": 0.8, "gt": d["gt"], "pred": d["pred"]}
        else:
            results[f"החזקה_{key}"] = {"match": d["binary_match"], "gt": d["gt"], "pred": d["pred"]}
    return results


# ── how obtained ──────────────────────────────────────────────────────────

HOW_OBTAINED_MAP = {
    "רכש":   "רכש",
    "מאחר":  "מאחר",
    "מצא":   "מצא",
    "גנב":   "גנב",
    "ייצר":  "ייצר",
    "עבודה": "עבודה",
}

def _compare_how_obtained(gt_row: dict, pred: dict) -> dict:
    gt_val = (gt_row.get(GT_HOW_OBTAINED) or "").strip()
    pr = pred.get("אופן_קבלת_הנשק", {})
    results = {}
    for key in HOW_OBTAINED_MAP:
        gv = 1 if gt_val == key else 0
        pv = pr.get(key, 0)
        results[f"קבלה_{key}"] = {"match": gv == pv, "gt": gv, "pred": pv}
    return results


# ── ammunition ────────────────────────────────────────────────────────────

def _normalize_ammo_text(s: str) -> str:
    """Fix common typos & format variations before parsing."""
    if not s: return ""
    s = s.strip()
    # Common typos: "וסה", "ובם" → "ובה"
    s = re.sub(r'\b(וסה|ובם|ובאה|ובע)\b', 'ובה', s)
    # Normalize "מחסנית ריקה ובה 0 כדורים" ≡ "מחסנית ובה 0 כדורים"
    s = re.sub(r'מחסנית\s+ריקה(\s+ובה\s+0\s+כדורים)?', 'מחסנית ובה 0 כדורים', s)
    # "קליעים" → "כדורים"
    s = re.sub(r'\bקליעים?\b', 'כדורים', s)
    return s


def _count_magazines(s: str) -> int:
    """Count magazines in the string. Supports '3 מחסניות' and 'מחסנית, מחסנית, מחסנית'."""
    if not s: return 0
    s = _normalize_ammo_text(s)
    # Patterns like "3 מחסניות" or "5 מחסניות"
    total = 0
    matched_spans = set()
    for m in re.finditer(r'(\d+)\s*מחסניות?', s):
        total += int(m.group(1))
        matched_spans.add((m.start(), m.end()))
    # Count standalone "מחסנית" occurrences (not preceded by digit)
    for m in re.finditer(r'מחסנית', s):
        if not any(start <= m.start() < end for start, end in matched_spans):
            total += 1
    return total


def _extract_bullet_count(s: str) -> int | None:
    """Extract total bullet count from ammunition string."""
    if not s or not s.strip():
        return 0
    s = _normalize_ammo_text(s)
    if s.strip() == "ללא":
        return 0
    total = 0
    found = False
    for m in re.finditer(r'(\d+)\s*כדורים?', s):
        total += int(m.group(1))
        found = True
    return total if found else None


def _compare_ammunition(gt_row: dict, pred: dict, bullet_tolerance: int = 1, mag_tolerance: int = 0) -> dict:
    """Compare ammo with tolerance (similar to drugs_match's gram tolerance).

    bullet_tolerance: allow ±N bullet difference (default 1, for typo robustness).
    mag_tolerance: 0 (mags should match exactly).
    """
    gt_raw = (gt_row.get(GT_AMMUNITION) or "").strip()
    pv_raw = pred.get("כמות_תחמושת", "") or ""
    gt_bullets = _extract_bullet_count(gt_raw)
    pv_bullets = _extract_bullet_count(pv_raw)
    gt_mags = _count_magazines(gt_raw)
    pv_mags = _count_magazines(pv_raw)

    bullets_match = None
    if gt_bullets is not None and pv_bullets is not None:
        bullets_match = abs(gt_bullets - pv_bullets) <= bullet_tolerance
    mags_match = abs(gt_mags - pv_mags) <= mag_tolerance

    if bullets_match is None:
        match = mags_match
    else:
        match = bullets_match and mags_match
    return {"כמות_תחמושת": {"match": match, "gt": gt_raw, "pred": pv_raw}}


# ── purpose ───────────────────────────────────────────────────────────────

PURPOSE_MAP = {
    "בצע כסף":        "בצע_כסף",
    "הגנה עצמית":     "הגנה_עצמית",
    "פחד":            "פחד",
    "ביטחון":         "ביטחון",
    "חברות בארגון טרור": "חברות_בארגון_פשע_או_טרור",
    "סכסוך":          "סכסוך",
    # Normalize all conflict synonyms to סכסוך
    "ריב":            "סכסוך",
    "מחלוקת":         "סכסוך",
    "הפחדה":          "סכסוך",
    "איום":           "סכסוך",
    "הקנטה":          "סכסוך",
    "סגירת חשבון":    "סכסוך",
    "נקמה":           "סכסוך",
    "גביית חוב":      "סכסוך",
    "סכסוך כספי":     "סכסוך",
    "סכסוך משפחתי":   "סכסוך",
    "אלימות במשפחה":  "סכסוך",
    "חתונה":          "חתונה",
    "תדמית":          "תדמית",
}
PURPOSE_KEYS = ["בצע_כסף","הגנה_עצמית","פחד","ביטחון","חברות_בארגון_פשע_או_טרור","סכסוך","חתונה","תדמית"]

def _compare_purpose(gt_row: dict, pred: dict) -> dict:
    gt_val = (gt_row.get(GT_PURPOSE) or "").strip()
    gt_key = PURPOSE_MAP.get(gt_val)
    pr = pred.get("מטרה_סיבת_העבירה", {})
    results = {}
    for key in PURPOSE_KEYS:
        gv = 1 if gt_key == key else 0
        pv = pr.get(key, 0)
        results[f"מטרה_{key}"] = {"match": gv == pv, "gt": gv, "pred": pv}
    return results


# ── usage ─────────────────────────────────────────────────────────────────

def _compare_usage(gt_row: dict, pred: dict) -> dict:
    gt_val = (gt_row.get(GT_USAGE) or "").strip()
    gt_used = 0 if (not gt_val or gt_val == "לא") else 1
    pr = pred.get("שימוש", {})
    if isinstance(pr, dict):
        pr_used = 0 if pr.get("לא", 0) == 1 else 1
    else:
        pr_used = 0
    return {"שימוש": {"match": gt_used == pr_used, "gt": gt_used, "pred": pr_used}}


# ── main comparison ───────────────────────────────────────────────────────

def compare_row(gt_row: dict, pred: dict) -> dict:
    results = {}
    results.update(_compare_offense_number(gt_row, pred))
    results.update(_compare_offense_type(gt_row, pred))
    results.update(_compare_additional_offenses(gt_row, pred))
    results.update(_compare_weapon_type(gt_row, pred))
    results.update(_compare_weapon_status(gt_row, pred))
    results.update(_compare_planning(gt_row, pred))
    results.update(_compare_storage(gt_row, pred))
    results.update(_compare_how_obtained(gt_row, pred))
    results.update(_compare_ammunition(gt_row, pred))
    results.update(_compare_purpose(gt_row, pred))
    results.update(_compare_usage(gt_row, pred))
    return results


# ── main groups (for top-level summary) ──────────────────────────────────

MAIN_FEATURE_GROUPS = {
    "מספר_עבירה":       ["מספר_עבירה_144_א", "מספר_עבירה_144_ב"],
    "סוג_עבירה":        [f"סוג_עבירה_{k}" for k in ["החזקת_נשק","נשיאת_נשק","הובלת_נשק",
                          "סחר_בנשק","ניסיון_לסחר_בנשק","ירי_באזור_מגורים","ייצור_נשק","עבירות_אחרות_בנשק"]],
    "עבירות_נוספות":    ["עבירות_נוספות"],
    "סוג_הנשק":         [f"נשק_{k}" for k in GT_WEAPON_COLS],
    "סטטוס_הנשק":       [f"סטטוס_{k}" for k in ["תקין","תקול","נשק_מפורק",
                          "נשק_מופרד_מתחמושת","נשק_עם_מחסנית_בהכנס","נשק_עם_כדור_בקנה"]],
    "תכנון":            ["תכנון"],
    "אופן_החזקת_הנשק":  [f"החזקה_{k}" for k in STORAGE_KEYS],
    "אופן_קבלת_הנשק":   [f"קבלה_{k}" for k in HOW_OBTAINED_MAP],
    "כמות_תחמושת":      ["כמות_תחמושת"],
    "מטרה_סיבת_העבירה": [f"מטרה_{k}" for k in PURPOSE_KEYS],
    "שימוש":            ["שימוש"],
}


# ── run eval ──────────────────────────────────────────────────────────────

def run_eval(gt_csv: str, docx_dir: str, out_csv: str):
    cache_path = str(pathlib.Path(out_csv).parent.parent / "cache" / "eval_weapon_gpt_cache.json")

    import json as _json
    gpt_cache: dict = {}
    if os.path.isfile(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            gpt_cache = _json.load(f)
        print(f"Loaded {len(gpt_cache)} cached predictions from {cache_path}")

    with open(gt_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        all_gt = [r for r in reader if r.get("data_source", "").strip() == "GT"]
    # deduplicate by case name (keep first occurrence)
    seen_cases = set()
    gt_rows = []
    for r in all_gt:
        c = r.get(GT_CASE_COL, "").strip()
        if c and c not in seen_cases:
            seen_cases.add(c)
            gt_rows.append(r)
    print(f"GT cases: {len(gt_rows)} (after dedup from {len(all_gt)})")

    all_case_results = []
    feature_stats: dict[str, dict] = {}

    for i, row in enumerate(gt_rows):
        fname = (row.get(GT_CASE_COL) or "").strip()
        docx_path = os.path.join(docx_dir, fname + ".docx")
        docx_missing = not os.path.isfile(docx_path)

        if docx_missing and fname not in gpt_cache:
            print(f"  [{i+1}/{len(gt_rows)}] MISSING docx (no cache): {fname}")
            continue

        print(f"  [{i+1}/{len(gt_rows)}] {fname} ...", end=" ", flush=True)
        try:
            if fname in gpt_cache:
                pred = gpt_cache[fname]
                print("(cached)", end=" ")
            else:
                text = read_docx(docx_path)
                pred = {
                    "מספר_עבירה":         extract_offense_number(text),
                    "סוג_העבירה":         extract_offense_type(text),
                    "עבירות_נוספות":      extract_additional_offenses(text),
                    "סוג_הנשק":           extract_weapon_type(text),
                    "סטטוס_הנשק":         extract_weapon_status(text),
                    "תכנון":              extract_planning(text),
                    "אופן_החזקת_הנשק":   extract_storage_method(text),
                    "אופן_קבלת_הנשק":    extract_how_obtained(text),
                    "כמות_תחמושת":        extract_ammunition(text),
                    "מטרה_סיבת_העבירה":  extract_purpose(text),
                    "שימוש":              extract_usage(text),
                }
                gpt_cache[fname] = pred
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as cf:
                    _json.dump(gpt_cache, cf, ensure_ascii=False, indent=2)
            pred = apply_conditional_defaults(pred)
            comparison = compare_row(row, pred)
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            comparison = {}

        all_case_results.append({"filename": fname, "comparison": comparison})

        for feat, info in comparison.items():
            if feat not in feature_stats:
                feature_stats[feat] = {"correct": 0.0, "total": 0, "disagreements": []}
            feature_stats[feat]["total"] += 1
            m = info.get("match")
            # match may be bool, int, or float (e.g., 0.8 for partial)
            if isinstance(m, bool):
                m_val = 1.0 if m else 0.0
            else:
                m_val = float(m or 0)
            feature_stats[feat]["correct"] += m_val
            if m_val < 1.0:
                feature_stats[feat]["disagreements"].append({
                    "case": fname, "gt": info.get("gt"), "pred": info.get("pred"),
                    "match": m_val,
                })

    # ── compute INVOLVED-ONLY metric (default — fair, ignores trivial negatives) ──
    # A binary question is "involved" only if at least one side answered "yes" (>0 / non-empty).
    # This avoids inflating accuracy by trivially-correct shared zeros.
    inv_per_sub: dict[str, dict] = {}
    for case in all_case_results:
        for feat, info in case["comparison"].items():
            gt_raw = str(info.get("gt", "")).strip()
            pr_raw = str(info.get("pred", "")).strip()
            # Determine "involved" — true if either side has truthy/non-zero/non-empty value
            try: gv = int(float(gt_raw)) if gt_raw else 0
            except ValueError: gv = 1 if gt_raw else 0
            try: pv = int(float(pr_raw)) if pr_raw else 0
            except ValueError: pv = 1 if pr_raw else 0
            involved = (gv > 0 or pv > 0) or (bool(gt_raw) and not gt_raw.replace('.','').isdigit())
            if feat not in inv_per_sub:
                inv_per_sub[feat] = {"correct": 0, "total": 0}
            if involved:
                inv_per_sub[feat]["total"] += 1
                m = info.get("match")
                m_val = 1.0 if (isinstance(m, bool) and m) else (0.0 if isinstance(m, bool) else float(m or 0))
                inv_per_sub[feat]["correct"] += m_val

    # ── main feature summary (involved-only, default) ──
    main_stats: dict[str, dict] = {}
    main_stats_all: dict[str, dict] = {}  # keep "all-binaries" for comparison
    for main_feat, sub_feats in MAIN_FEATURE_GROUPS.items():
        correct = total = 0
        correct_all = total_all = 0
        for sf in sub_feats:
            if sf in inv_per_sub:
                correct += inv_per_sub[sf]["correct"]
                total   += inv_per_sub[sf]["total"]
            if sf in feature_stats:
                correct_all += feature_stats[sf]["correct"]
                total_all   += feature_stats[sf]["total"]
        main_stats[main_feat]     = {"correct": correct,     "total": total}
        main_stats_all[main_feat] = {"correct": correct_all, "total": total_all}

    print("\n" + "=" * 75)
    print(f"{'פיצר ראשי':<28} {'INVOLVED-ONLY':>20} {'(כל בינאריים)':>22}")
    print("=" * 75)
    all_correct = all_total = 0
    all_correct_a = all_total_a = 0
    for main_feat in MAIN_FEATURE_GROUPS:
        s  = main_stats[main_feat]
        sa = main_stats_all[main_feat]
        acc  = s["correct"]  / s["total"]  if s["total"]  else 0
        acca = sa["correct"] / sa["total"] if sa["total"] else 0
        print(f"{main_feat:<28} {acc:>9.1%} ({s['correct']:>3}/{s['total']:<3})  {acca:>9.1%} ({sa['correct']:>4}/{sa['total']:<4})")
        all_correct += s["correct"]; all_total += s["total"]
        all_correct_a += sa["correct"]; all_total_a += sa["total"]
    print("-" * 75)
    overall  = all_correct / all_total if all_total else 0
    overalla = all_correct_a / all_total_a if all_total_a else 0
    print(f"{'סהכ':<28} {overall:>9.1%} ({all_correct:>3}/{all_total:<3})  {overalla:>9.1%} ({all_correct_a:>4}/{all_total_a:<4})")

    # ── sub-feature detail ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'פיצ\'ר משנה':<35} {'דיוק':>10} {'נכון/סה\"כ':>12}")
    print("=" * 60)
    for feat in sorted(feature_stats):
        s = feature_stats[feat]
        acc = s["correct"] / s["total"] if s["total"] else 0
        print(f"{feat:<35} {acc:>9.1%}  {s['correct']:>4}/{s['total']:<4}")

    print("\n--- אי-הסכמות ---")
    for feat in sorted(feature_stats):
        disags = feature_stats[feat]["disagreements"]
        if not disags:
            continue
        print(f"\n[{feat}] ({len(disags)} אי-הסכמות)")
        for d in disags:
            print(f"  {d['case']}: GT={d['gt']}  PRED={d['pred']}")

    # ── write CSVs ─────────────────────────────────────────────────────
    rows_out = []
    for case in all_case_results:
        for feat, info in case["comparison"].items():
            rows_out.append({
                "filename": case["filename"],
                "feature":  feat,
                "match":    int(info.get("match", 0)),
                "gt":       str(info.get("gt", "")),
                "pred":     str(info.get("pred", "")),
            })
    if rows_out:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["filename","feature","match","gt","pred"])
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"\nResults written to {out_csv}")

    summary_path = out_csv.replace(".csv", "_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["level","feature","accuracy_involved","correct_involved","total_involved","accuracy_all","correct_all","total_all"])
        writer.writeheader()
        for main_feat in MAIN_FEATURE_GROUPS:
            s  = main_stats[main_feat]
            sa = main_stats_all[main_feat]
            acc  = s["correct"]  / s["total"]  if s["total"]  else 0
            acca = sa["correct"] / sa["total"] if sa["total"] else 0
            writer.writerow({"level": "main", "feature": main_feat,
                             "accuracy_involved": f"{acc:.3f}", "correct_involved": s["correct"], "total_involved": s["total"],
                             "accuracy_all": f"{acca:.3f}", "correct_all": sa["correct"], "total_all": sa["total"]})
        writer.writerow({"level": "overall", "feature": "סה\"כ",
                         "accuracy_involved": f"{overall:.3f}", "correct_involved": all_correct, "total_involved": all_total,
                         "accuracy_all": f"{overalla:.3f}", "correct_all": all_correct_a, "total_all": all_total_a})
        for feat in sorted(feature_stats):
            s_inv = inv_per_sub.get(feat, {"correct": 0, "total": 0})
            s_all = feature_stats[feat]
            acc_inv = s_inv["correct"] / s_inv["total"] if s_inv["total"] else 0
            acc_all = s_all["correct"] / s_all["total"] if s_all["total"] else 0
            writer.writerow({"level": "sub", "feature": feat,
                             "accuracy_involved": f"{acc_inv:.3f}", "correct_involved": s_inv["correct"], "total_involved": s_inv["total"],
                             "accuracy_all": f"{acc_all:.3f}", "correct_all": s_all["correct"], "total_all": s_all["total"]})
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    BASE = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt",   default=str(BASE.parent / "data/wep/manual-fe-gt.csv"))
    parser.add_argument("--docx", default=str(BASE.parent.parent / "weapon/weapon_docx/"))
    parser.add_argument("--out",  default=str(BASE / "results/eval_weapon_results.csv"))
    args = parser.parse_args()
    run_eval(args.gt, args.docx, args.out)
