"""
Eval script: run extract_drugs_features_simple.py on GT cases and compare per-feature accuracy.
Usage:
    python eval_drugs_features.py \
        --gt   /path/to/gt_manual_drugs.csv \
        --docx /path/to/drugs_docx/ \
        --out  /path/to/output_eval.csv
"""

import argparse
import csv
import json
import os
import sys
import re

# ── import extraction logic ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from extract_drugs_features_simple import (
    read_docx,
    extract_offense_number,
    extract_side_offenses,
    extract_drug_type,
    extract_lab,
    extract_role,
    extract_undercover,
)

# ── GT column names ───────────────────────────────────────────────────────
GT_FILE_COL      = "שם קובץ התיק"
GT_OFF6          = "עבירות [סעיף 6]"
GT_OFF7          = "עבירות [סעיף 7]"
GT_OFF13         = "עבירות [סעיף 13]"
GT_OFF14         = "עבירות [סעיף 14]"
GT_OFF19         = "עבירות [סעיף 19]"
GT_OFF21         = "עבירות [סעיף 21]"
GT_OFF22         = "עבירות [סעיף 22]"
GT_OFF61         = "עבירות [61(א)(ג)]"
GT_OFF_OTHER     = "עבירת סמים שלא הייתה ברשימה"
GT_SIDE          = "עבירות נלוות כן/לא"
GT_LAB           = "מעבדה"
GT_ROLE          = "תפקיד"
GT_UNDERCOVER    = "מכירה לסוכן"
GT_DRUG_COLS = {
    "LSD":             "סוג הסם [LSD]",
    "METHAMPHETAMINE": "סוג הסם [METHAMPHETAMINE]",
    "האיוואסקה":       "סוג הסם [האיוואסקה]",
    "קתינון":          "סוג הסם [קתינון]",
    "קטמין":           "סוג הסם [קטמין]",
    "חשיש":            "סוג הסם [חשיש]",
    "מתילמקאתינון":    "סוג הסם [מתילמקאתינון]",
    "קנבוס_בשתילים":   "סוג הסם [קנבוס בשתילים]",
    "קנבוס":           "סוג הסם [קנבוס]",
    "MDMA":            "סוג הסם [MDMA]",
    "קוקאין":          "סוג הסם [קוקאין]",
}


# ── helpers ───────────────────────────────────────────────────────────────

def norm_offense(val: str) -> str:
    """Normalize offense letter: strip whitespace, replace / with _.
    GT uses '\"\"' (two double-quotes) to mean 'present, no letter' = '1'.
    """
    v = (val or "").strip().replace("/", "_").replace("(", "").replace(")", "")
    # GT marker for "section present, no sub-letter"
    # handles ASCII quotes AND Hebrew gershayim ״ (U+05F4)
    if v in ('""', '"', '״״', '״'):
        return "1"
    # 'א, ג' -> 'א_ג'
    v = re.sub(r",\s*", "_", v)
    return v


def gt_yes_no_to_int(val: str) -> int | None:
    v = (val or "").strip()
    if v == "כן":
        return 1
    if v == "לא":
        return 0
    return None  # missing / unknown


def parse_gt_drug_list(val: str) -> list[str]:
    """
    Parse GT drug value like '[71.52-גרם]' or '[1301.44-גרם, 13189-טבליות]'
    or '[בולים-154,נוזל-1.75 מיליליטר]' into a list of 'num-unit' strings.
    Handles commas inside numbers (thousands separator): '1,255.48-גרם' → '1255.48-גרם'.
    Returns [] for empty.
    """
    v = (val or "").strip()
    if not v:
        return []
    # strip outer brackets, and any stray prefix chars before '['
    # e.g. 'ב[גרם-52.616]' → '[גרם-52.616]'
    bracket_start = v.find("[")
    if bracket_start > 0:
        v = v[bracket_start:]
    v = v.strip("[]")
    if not v:
        return []
    # Split on commas that are NOT between two digits (i.e. not thousands separators).
    # A thousands-separator comma: digit , digit  → keep together.
    # An item separator comma: not between two digits → split.
    import re as _re
    # Replace thousands-separator commas with placeholder, then split on remaining commas
    v_marked = _re.sub(r'(\d),(\d)', r'\1COMMA\2', v)
    items = [x.strip() for x in v_marked.split(",")]
    result = []
    for item in items:
        item = item.strip().replace("COMMA", "").replace(" ", "")
        if item:
            result.append(item)
    return result


def parse_pred_drug_list(json_str) -> list[str]:
    """Parse JSON list from script output."""
    if not json_str:
        return []
    if isinstance(json_str, list):
        lst = json_str
    else:
        try:
            lst = json.loads(json_str)
        except Exception:
            return []
    # normalise: remove spaces
    return [str(x).replace(" ", "") for x in lst]


def norm_drug_val(s: str) -> tuple[float | None, str]:
    """
    Parse 'num-unit' string into (number, unit).
    Handles reversed format 'unit-num' too.
    Returns (None, original) if unparseable.
    """
    s = s.lower().replace(",", "").replace(" ", "").strip()
    parts = s.split("-")
    if len(parts) == 2:
        a, b = parts
        try:
            return float(a), b
        except ValueError:
            pass
        try:
            return float(b), a
        except ValueError:
            pass
    return None, s


def drugs_match(gt_list: list[str], pred_list: list[str], tolerance: float = 1.0) -> bool:
    """
    Compare two drug lists.
    - Both empty = match.
    - Same number of items, same units, numbers within tolerance (default 1 gram).
    """
    if len(gt_list) != len(pred_list):
        return False
    if not gt_list:
        return True
    gt_parsed  = sorted([norm_drug_val(x) for x in gt_list],  key=lambda t: (t[1], t[0] or 0))
    pr_parsed  = sorted([norm_drug_val(x) for x in pred_list], key=lambda t: (t[1], t[0] or 0))
    for (gn, gu), (pn, pu) in zip(gt_parsed, pr_parsed):
        if gu != pu:
            return False
        if gn is None or pn is None:
            if gn != pn:
                return False
        elif abs(gn - pn) > tolerance:
            return False
    return True


def parse_gt_role(val: str):
    """
    GT תפקיד examples:
      'בעל הסמים'
      'לא בעל הסמים'
      'בעל הסמים, בעל המעבדה'
      'לא בעל הסמים, לא בעל המעבדה'
    Returns (owns_drugs: int, owns_lab: int)
    """
    v = (val or "").strip()
    if not v:
        return None, None
    owns_drugs = 0 if "לא בעל הסמים" in v else 1
    owns_lab   = 1 if "בעל המעבדה" in v and "לא בעל המעבדה" not in v else 0
    return owns_drugs, owns_lab


# ── main comparison ───────────────────────────────────────────────────────

def compare_row(gt_row: dict, pred: dict) -> dict:
    """
    Compare a single GT row against extracted predictions.
    Returns a dict with per-feature: match (bool), gt_val, pred_val.
    """
    results = {}

    # ── offense sections ──────────────────────────────────────────────
    for gt_col, pred_key, section_label in [
        (GT_OFF6,   "סעיף_6",  "סעיף_6"),
        (GT_OFF7,   "סעיף_7",  "סעיף_7"),
        (GT_OFF13,  "סעיף_13", "סעיף_13"),
        (GT_OFF14,  "סעיף_14", "סעיף_14"),
        (GT_OFF19,  "סעיף_19", "סעיף_19"),
        (GT_OFF21,  "סעיף_21", "סעיף_21"),
        (GT_OFF22,  "סעיף_22", "סעיף_22"),
        (GT_OFF61,  "סעיף_61", "סעיף_61"),
    ]:
        gt_val   = norm_offense(gt_row.get(gt_col, ""))
        pred_raw = pred.get("מספר_עבירה", {}) or {}
        pred_val = norm_offense(pred_raw.get(pred_key, ""))
        results[section_label] = {
            "gt": gt_val, "pred": pred_val,
            "match": gt_val == pred_val,
        }

    # ── side offenses ─────────────────────────────────────────────────
    gt_side  = gt_yes_no_to_int(gt_row.get(GT_SIDE, ""))
    pred_side_raw = pred.get("עבירות_נלוות", {}) or {}
    pred_side = pred_side_raw.get("עבירות_נלוות")
    results["עבירות_נלוות"] = {
        "gt": gt_side, "pred": pred_side,
        "match": (gt_side is not None) and (gt_side == pred_side),
    }

    # ── lab ───────────────────────────────────────────────────────────
    gt_lab  = gt_yes_no_to_int(gt_row.get(GT_LAB, ""))
    pred_lab_raw = pred.get("מעבדה", {}) or {}
    pred_lab = pred_lab_raw.get("מעבדה")
    results["מעבדה"] = {
        "gt": gt_lab, "pred": pred_lab,
        "match": (gt_lab is not None) and (gt_lab == pred_lab),
    }

    # ── role ──────────────────────────────────────────────────────────
    gt_owns_drugs, gt_owns_lab = parse_gt_role(gt_row.get(GT_ROLE, ""))
    pred_role_raw = pred.get("תפקיד", {}) or {}
    pred_owns_drugs = pred_role_raw.get("בעל_הסמים")
    pred_owns_lab   = pred_role_raw.get("בעל_המעבדה")
    results["בעל_הסמים"] = {
        "gt": gt_owns_drugs, "pred": pred_owns_drugs,
        "match": (gt_owns_drugs is not None) and (gt_owns_drugs == pred_owns_drugs),
    }
    results["בעל_המעבדה"] = {
        "gt": gt_owns_lab, "pred": pred_owns_lab,
        "match": (gt_owns_lab is not None) and (gt_owns_lab == pred_owns_lab),
    }

    # ── undercover ────────────────────────────────────────────────────
    gt_uc  = gt_yes_no_to_int(gt_row.get(GT_UNDERCOVER, ""))
    pred_uc_raw = pred.get("מכירה_לסוכן", {}) or {}
    pred_uc = pred_uc_raw.get("מכירה_לסוכן")
    results["מכירה_לסוכן"] = {
        "gt": gt_uc, "pred": pred_uc,
        "match": (gt_uc is not None) and (gt_uc == pred_uc),
    }

    # ── drug types ────────────────────────────────────────────────────
    pred_drugs = pred.get("סוג_הסם", {}) or {}
    for drug_key, gt_col in GT_DRUG_COLS.items():
        gt_list   = parse_gt_drug_list(gt_row.get(gt_col, ""))
        pred_list = parse_pred_drug_list(pred_drugs.get(drug_key, []))
        match = drugs_match(gt_list, pred_list)
        results[f"סם_{drug_key}"] = {
            "gt": gt_list, "pred": pred_list,
            "match": match,
        }

    return results


def run_eval(gt_csv: str, docx_dir: str, out_csv: str, cache_path: str = None):
    # load cache if exists
    if cache_path is None:
        # default: ../cache/ relative to out_csv location
        import pathlib
        cache_path = str(pathlib.Path(out_csv).parent.parent / "cache" / "eval_drugs_results_gpt_cache.json")
    gpt_cache: dict = {}
    if os.path.isfile(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            gpt_cache = json.load(f)
        print(f"Loaded {len(gpt_cache)} cached predictions from {cache_path}")

    # read GT
    with open(gt_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        gt_rows = [r for r in reader if r.get("data_source", "").strip() == "GT"]

    print(f"GT cases: {len(gt_rows)}")

    all_case_results = []
    feature_stats: dict[str, dict] = {}

    for i, row in enumerate(gt_rows):
        fname = (row.get(GT_FILE_COL) or "").strip()
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
                    "מספר_עבירה":   extract_offense_number(text),
                    "עבירות_נלוות": extract_side_offenses(text),
                    "סוג_הסם":      extract_drug_type(text),
                    "מעבדה":        extract_lab(text),
                    "תפקיד":        extract_role(text),
                    "מכירה_לסוכן":  extract_undercover(text),
                }
                gpt_cache[fname] = pred
                with open(cache_path, "w", encoding="utf-8") as cf:
                    json.dump(gpt_cache, cf, ensure_ascii=False, indent=2)
            comparison = compare_row(row, pred)
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            comparison = {}

        all_case_results.append({
            "filename": fname,
            "comparison": comparison,
        })

        # accumulate stats
        for feat, info in comparison.items():
            if feat not in feature_stats:
                feature_stats[feat] = {"correct": 0.0, "total": 0, "disagreements": []}
            feature_stats[feat]["total"] += 1
            m = info.get("match")
            m_val = 1.0 if (isinstance(m, bool) and m) else (0.0 if isinstance(m, bool) else float(m or 0))
            feature_stats[feat]["correct"] += m_val
            if m_val < 1.0:
                feature_stats[feat]["disagreements"].append({
                    "case": fname,
                    "gt":   info.get("gt"),
                    "pred": info.get("pred"),
                    "match": m_val,
                })

    # ── aggregate 6 main features ─────────────────────────────────────
    MAIN_FEATURE_GROUPS = {
        "מספר_עבירה":   ["סעיף_6", "סעיף_7", "סעיף_13", "סעיף_14", "סעיף_19",
                          "סעיף_21", "סעיף_22", "סעיף_61"],
        "עבירות_נלוות": ["עבירות_נלוות"],
        "סוג_הסם":      [f"סם_{d}" for d in ["LSD","MDMA","METHAMPHETAMINE","האיוואסקה",
                          "חשיש","מתילמקאתינון","קוקאין","קטמין","קנבוס","קנבוס_בשתילים","קתינון"]],
        "מעבדה":        ["מעבדה"],
        "תפקיד":        ["בעל_הסמים", "בעל_המעבדה"],
        "מכירה_לסוכן":  ["מכירה_לסוכן"],
    }
    # ── compute INVOLVED-ONLY metric (default — fair, ignores trivial negatives) ──
    inv_per_sub: dict[str, dict] = {}
    for case in all_case_results:
        for feat, info in case["comparison"].items():
            gt_raw = info.get("gt")
            pr_raw = info.get("pred")
            def _truthy(x):
                if x is None: return False
                if isinstance(x, list): return len(x) > 0
                if isinstance(x, (int, float)): return x != 0
                return bool(str(x).strip())
            involved = _truthy(gt_raw) or _truthy(pr_raw)
            if feat not in inv_per_sub:
                inv_per_sub[feat] = {"correct": 0.0, "total": 0}
            if involved:
                inv_per_sub[feat]["total"] += 1
                m = info.get("match")
                m_val = 1.0 if (isinstance(m, bool) and m) else (0.0 if isinstance(m, bool) else float(m or 0))
                inv_per_sub[feat]["correct"] += m_val

    main_stats: dict[str, dict] = {}
    main_stats_all: dict[str, dict] = {}
    for main_feat, sub_feats in MAIN_FEATURE_GROUPS.items():
        c = t = ca = ta = 0
        for sf in sub_feats:
            if sf in inv_per_sub:
                c += inv_per_sub[sf]["correct"]; t += inv_per_sub[sf]["total"]
            if sf in feature_stats:
                ca += feature_stats[sf]["correct"]; ta += feature_stats[sf]["total"]
        main_stats[main_feat]     = {"correct": c,  "total": t}
        main_stats_all[main_feat] = {"correct": ca, "total": ta}

    print("\n" + "=" * 75)
    print(f"{'פיצר ראשי':<25} {'INVOLVED-ONLY':>20} {'(כל בינאריים)':>22}")
    print("=" * 75)
    all_correct = all_total = 0
    all_correct_a = all_total_a = 0
    for main_feat in MAIN_FEATURE_GROUPS:
        s  = main_stats[main_feat]
        sa = main_stats_all[main_feat]
        acc  = s["correct"]  / s["total"]  if s["total"]  else 0
        acca = sa["correct"] / sa["total"] if sa["total"] else 0
        print(f"{main_feat:<25} {acc:>9.1%} ({s['correct']:>3}/{s['total']:<3})  {acca:>9.1%} ({sa['correct']:>4}/{sa['total']:<4})")
        all_correct += s["correct"]; all_total += s["total"]
        all_correct_a += sa["correct"]; all_total_a += sa["total"]
    print("-" * 75)
    overall  = all_correct / all_total if all_total else 0
    overalla = all_correct_a / all_total_a if all_total_a else 0
    print(f"{'סהכ':<25} {overall:>9.1%} ({all_correct:>3}/{all_total:<3})  {overalla:>9.1%} ({all_correct_a:>4}/{all_total_a:<4})")

    # ── print sub-feature detail ───────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'פיצ\'ר משנה':<30} {'דיוק':>10} {'נכון/סה\"כ':>12}")
    print("=" * 60)
    for feat in sorted(feature_stats):
        s = feature_stats[feat]
        acc = s["correct"] / s["total"] if s["total"] else 0
        print(f"{feat:<30} {acc:>9.1%}  {s['correct']:>4}/{s['total']:<4}")

    print("\n--- אי-הסכמות ---")
    for feat in sorted(feature_stats):
        disags = feature_stats[feat]["disagreements"]
        if not disags:
            continue
        print(f"\n[{feat}] ({len(disags)} אי-הסכמות)")
        for d in disags:
            print(f"  {d['case']}: GT={d['gt']}  PRED={d['pred']}")

    # ── write output CSV ──────────────────────────────────────────────
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
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "feature", "match", "gt", "pred"])
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"\nResults written to {out_csv}")

    # write summary: main features first, then sub-features
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt",   required=True, help="path to gt_manual_drugs.csv")
    parser.add_argument("--docx", required=True, help="directory with .docx files")
    parser.add_argument("--out",  default="eval_drugs_output.csv", help="output CSV path")
    args = parser.parse_args()
    run_eval(args.gt, args.docx, args.out)
