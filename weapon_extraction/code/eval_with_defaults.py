"""
Apply default-value rules to the GT CSV, then re-run eval_weapon_features
to compare accuracy per feature — before vs after defaults.

Defaults (from Entity recipe - Defult_rules.csv):
  - אופן החזקת הנשק: if empty AND offense is נשיאה → "על גופו"
  - סטטוס הנשק:      if empty → (נשק עם כדור בקנה if usage; else תקין)
  - מטרה-סיבת העבירה: if empty AND offense is סחר → "בצע כסף"
  - כמות תחמושת:     if empty → "ללא"
  - תכנון:           if empty → "לא"
  - שימוש:           if empty → "לא"
"""

import csv, os, sys, pathlib, argparse, shutil, subprocess

BASE = pathlib.Path(__file__).resolve().parent.parent
GT_IN  = BASE.parent / "data/wep/manual-fe-gt.csv"
GT_OUT = BASE / "data" / "manual-fe-gt-with-defaults.csv"
RESULTS_DIR = BASE / "results"


def apply_row_defaults(row: dict) -> tuple[dict, dict]:
    """Return (new_row, changes_by_field)."""
    out = dict(row)
    changes = {}
    offense = (row.get("סוג עבירה") or "")
    usage   = (row.get("שימוש") or "").strip()

    def _set(col, val):
        out[col] = val
        changes[col] = changes.get(col, 0) + 1

    if not (row.get("אופן החזקת הנשק") or "").strip():
        if "נשיאת" in offense or "נשיאה" in offense:
            _set("אופן החזקת הנשק", "על גופו")

    if not (row.get("סטטוס הנשק") or "").strip():
        if usage and usage != "לא":
            _set("סטטוס הנשק", "נשק עם כדור בקנה")
        else:
            _set("סטטוס הנשק", "תקין")

    if not (row.get("מטרה-סיבת העבירה") or "").strip():
        if "סחר" in offense:
            _set("מטרה-סיבת העבירה", "בצע כסף")

    if not (row.get("כמות תחמושת") or "").strip():
        _set("כמות תחמושת", "ללא")

    if not (row.get("תכנון") or "").strip():
        _set("תכנון", "לא")

    if not (row.get("שימוש") or "").strip():
        _set("שימוש", "לא")

    return out, changes


def build_gt_with_defaults():
    with open(GT_IN, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    total_changes = {}
    new_rows = []
    for r in rows:
        new_r, ch = apply_row_defaults(r)
        new_rows.append(new_r)
        for k, v in ch.items():
            total_changes[k] = total_changes.get(k, 0) + v

    GT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(GT_OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(new_rows)

    print(f"Defaults applied → {GT_OUT}")
    print(f"Rows: {len(new_rows)}")
    print(f"Changes by field:")
    for k, v in sorted(total_changes.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    return GT_OUT


def main():
    gt_defaults_path = build_gt_with_defaults()

    # Run eval with defaults GT → new summary
    script = pathlib.Path(__file__).parent / "eval_weapon_features.py"
    out_path = RESULTS_DIR / "eval_weapon_results_WITH_DEFAULTS.csv"

    cmd = [
        sys.executable, str(script),
        "--gt", str(gt_defaults_path),
        "--out", str(out_path),
    ]
    print(f"\nRunning: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

    # Compare summaries
    def load_summary(path):
        out = {}
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r["level"] == "main":
                    out[r["feature"]] = (float(r["accuracy"]), int(r["correct"]), int(r["total"]))
        return out

    before = load_summary(RESULTS_DIR / "eval_weapon_results_summary.csv")
    after  = load_summary(str(out_path).replace(".csv", "_summary.csv"))

    print(f"\n{'='*70}")
    print(f"{'Feature':<22} {'before':>10} {'after':>10} {'Δ acc':>10}   {'Δ correct':>12}")
    print(f"{'-'*70}")
    for feat in before:
        if feat in after:
            b_acc, b_c, b_t = before[feat]
            a_acc, a_c, a_t = after[feat]
            d_acc = a_acc - b_acc
            d_c   = a_c - b_c
            print(f"{feat:<22} {b_acc:>9.1%}  {a_acc:>9.1%}  {d_acc:>+9.3%}   {d_c:>+7d}/{a_t}")


if __name__ == "__main__":
    main()
