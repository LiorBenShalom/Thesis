#!/usr/bin/env python3
"""
Per-pair table: manual_fe vs fe_manual_format for binary_0.

For each domain (drugs, weapon): merges GT, continuous scores from v6 preds (same --model),
optimized thresholds from *_stats.json, and full feature_vector_1/2 from both representations.

Output: excel_tables/binary0_<model>_manual_vs_manual_format_<domain>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NEW_TRY = ROOT.parents[1]  # .../new_try
RESULTS = {
    "drugs": ROOT / "drugs" / "results_drugs",
    "weapon": ROOT / "weapon" / "results_weapon",
}
SOURCE_CSV = {
    "drugs": {
        "manual_fe": NEW_TRY / "drugs" / "similarity_database_fe.csv",
        "fe_manual_format": NEW_TRY / "drugs" / "similarity_database_fe_manual_format.csv",
    },
    "weapon": {
        "manual_fe": NEW_TRY / "weapon" / "similarity_database_fe.csv",
        "fe_manual_format": NEW_TRY / "weapon" / "similarity_database_fe_manual_format.csv",
    },
}


def _load_threshold(domain: str, model: str, rep_stem: str) -> float:
    """rep_stem: 'similarity_database_fe' or 'similarity_database_fe_manual_format' etc."""
    pat = f"{rep_stem}_v6score_{model}_binary_0_stats.json"
    p = RESULTS[domain] / pat
    data = json.loads(p.read_text(encoding="utf-8"))
    return float(data["best_threshold"])


def _read_preds(path: Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            k = (row["verdict_1"], row["verdict_2"])
            out[k] = row
    return out


def _read_source_features(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    out: dict[tuple[str, str], tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            k = (row["verdict_1"], row["verdict_2"])
            out[k] = (row["feature_vector_1"], row["feature_vector_2"])
    return out


def build_domain(domain: str, model: str, out_dir: Path) -> Path:
    th_m = _load_threshold(domain, model, "similarity_database_fe")
    th_f = _load_threshold(domain, model, "similarity_database_fe_manual_format")

    preds_m = RESULTS[domain] / f"similarity_database_fe_v6score_{model}_binary_0_preds.csv"
    preds_f = (
        RESULTS[domain]
        / f"similarity_database_fe_manual_format_v6score_{model}_binary_0_preds.csv"
    )
    if not preds_m.is_file():
        raise FileNotFoundError(preds_m)
    if not preds_f.is_file():
        raise FileNotFoundError(preds_f)

    pm = _read_preds(preds_m)
    pf = _read_preds(preds_f)
    fm = _read_source_features(SOURCE_CSV[domain]["manual_fe"])
    ff = _read_source_features(SOURCE_CSV[domain]["fe_manual_format"])

    keys = sorted(set(pm.keys()) & set(pf.keys()) & set(fm.keys()) & set(ff.keys()))

    out_path = out_dir / f"binary0_{model}_manual_vs_manual_format_{domain}.csv"
    fields = [
        "domain",
        "verdict_1",
        "verdict_2",
        "gt_similarity_binary_0",
        "score_manual_fe",
        "score_fe_manual_format",
        "delta_score_fmt_minus_manual",
        "threshold_manual_fe",
        "threshold_fe_manual_format",
        "pred_binary_manual_fe",
        "pred_binary_fe_manual_format",
        "preds_disagree",
        "manual_fe_correct_vs_gt",
        "fe_manual_format_correct_vs_gt",
        "feature_vector_1_manual_fe",
        "feature_vector_2_manual_fe",
        "feature_vector_1_fe_manual_format",
        "feature_vector_2_fe_manual_format",
        "status_manual_fe",
        "status_fe_manual_format",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for k in keys:
            rm, rf = pm[k], pf[k]
            gt = int(rm["similarity_binary_0"])
            if int(rf["similarity_binary_0"]) != gt:
                raise ValueError(f"GT mismatch for {k}: manual {rm['similarity_binary_0']} vs fmt {rf['similarity_binary_0']}")

            try:
                sm = float(rm["score"]) if rm.get("status") == "ok" else float("nan")
            except (TypeError, ValueError):
                sm = float("nan")
            try:
                sf = float(rf["score"]) if rf.get("status") == "ok" else float("nan")
            except (TypeError, ValueError):
                sf = float("nan")

            pred_m = int(sm >= th_m) if sm == sm else -1
            pred_f = int(sf >= th_f) if sf == sf else -1
            fv1m, fv2m = fm[k]
            fv1f, fv2f = ff[k]

            w.writerow(
                {
                    "domain": domain,
                    "verdict_1": k[0],
                    "verdict_2": k[1],
                    "gt_similarity_binary_0": gt,
                    "score_manual_fe": f"{sm:.6g}" if sm == sm else "",
                    "score_fe_manual_format": f"{sf:.6g}" if sf == sf else "",
                    "delta_score_fmt_minus_manual": f"{sf - sm:.6g}" if (sm == sm and sf == sf) else "",
                    "threshold_manual_fe": th_m,
                    "threshold_fe_manual_format": th_f,
                    "pred_binary_manual_fe": pred_m if pred_m >= 0 else "",
                    "pred_binary_fe_manual_format": pred_f if pred_f >= 0 else "",
                    "preds_disagree": (pred_m != pred_f) if (pred_m >= 0 and pred_f >= 0) else "",
                    "manual_fe_correct_vs_gt": (pred_m == gt) if pred_m >= 0 else "",
                    "fe_manual_format_correct_vs_gt": (pred_f == gt) if pred_f >= 0 else "",
                    "feature_vector_1_manual_fe": fv1m,
                    "feature_vector_2_manual_fe": fv2m,
                    "feature_vector_1_fe_manual_format": fv1f,
                    "feature_vector_2_fe_manual_format": fv2f,
                    "status_manual_fe": rm.get("status", ""),
                    "status_fe_manual_format": rf.get("status", ""),
                }
            )

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt4", help="v6 backend id (e.g. gpt4, llama3_70b)")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "excel_tables",
        help="Output directory for CSVs",
    )
    args = ap.parse_args()
    written = []
    for dom in ("drugs", "weapon"):
        p = build_domain(dom, args.model, args.out_dir)
        written.append(p)
        print(f"Wrote {p} ({sum(1 for _ in p.open(encoding='utf-8-sig')) - 1} rows)")
    print("Done.")


if __name__ == "__main__":
    main()
