#!/usr/bin/env python3
"""
Line-by-line validation: every *_binary_0_preds.csv under v6_full_matrix vs source CSV
in new_try/{drugs,weapon}/, plus matching *_stats.json complete/n_failed consistency.

**Strict mode (default):** every row with status=ok must contain a real
`SIMILARITY_SCORE: <int>` line in `response`, and it must match the `score` column
(not merely a number 0–100 from the fallback parser).

Run from anywhere:
  python validate_v6_matrix_data.py
  python validate_v6_matrix_data.py --no-strict-response   # legacy: numeric score only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# new_try/ (parent of code/)
NEW_TRY = Path(__file__).resolve().parents[2]
V6_ROOT = Path(__file__).resolve().parent

# Same as regenerate_v6_tables.py — official v6_full_matrix experiment grid
MATRIX_MODELS = [
    "gpt4",
    "gpt5mini",
    "qwen3_235b",
    "mistral",
    "llama3_70b",
    "gpt52",
    "gpt51_thinking",
    "qwen_hf",
    "gemini_25_pro",
    "gemini_3_flash",
    "gemma3_27b",
]


def _parse_preds_name(name: str) -> tuple[str, str] | None:
    """Return (source_csv_stem, model) e.g. ('similarity_database_fe_gpt_schema', 'gemini_3_flash')."""
    m = re.match(r"^(.+)_v6score_(.+)_binary_0_preds\.csv$", name)
    if not m:
        return None
    return m.group(1), m.group(2)


def _validate_score(v: object) -> bool:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return False
    if np.isnan(x):
        return False
    return 0.0 <= x <= 100.0


# v6 prompts require this exact line; parse_score() may otherwise take a fake number from list items (1., 2., 3.).
_SIM_SCORE_LINE = re.compile(r"SIMILARITY_SCORE\s*:\s*(\d+)", re.IGNORECASE)


def _strict_ok_row(preds_name: str, row: pd.Series) -> list[str]:
    """Return list of error strings if status=ok row fails semantic parse checks."""
    out: list[str] = []
    if str(row.get("status", "")) != "ok":
        return out
    resp = row.get("response")
    if pd.isna(resp):
        resp = ""
    else:
        resp = str(resp)
    m = _SIM_SCORE_LINE.search(resp)
    if not m:
        out.append(
            f"missing SIMILARITY_SCORE line in response (likely truncated or fallback-parse; "
            f"response_len={len(resp)})"
        )
        return out
    declared = float(m.group(1))
    try:
        stored = float(row["score"])
    except (TypeError, ValueError):
        out.append(f"bad score column: {row.get('score')!r}")
        return out
    if abs(declared - stored) > 0.501:
        out.append(f"score mismatch: CSV score={stored} vs SIMILARITY_SCORE in text={declared}")
    return out


def validate_all(*, matrix_only: bool, strict_response: bool) -> int:
    errors: list[str] = []
    warnings: list[str] = []
    skipped_extra: list[str] = []

    for domain in ("drugs", "weapon"):
        rdir = V6_ROOT / domain / f"results_{domain}"
        if not rdir.is_dir():
            errors.append(f"Missing results dir: {rdir}")
            continue
        src_dir = NEW_TRY / domain

        for preds_path in sorted(rdir.glob("*_v6score_*_binary_0_preds.csv")):
            parsed = _parse_preds_name(preds_path.name)
            if not parsed:
                errors.append(f"Bad preds filename: {preds_path.name}")
                continue
            stem, model = parsed
            if matrix_only and model not in MATRIX_MODELS:
                skipped_extra.append(f"{domain}/{preds_path.name}")
                continue
            src_path = src_dir / f"{stem}.csv"
            if not src_path.exists():
                errors.append(f"Missing source CSV for {preds_path.name}: {src_path}")
                continue

            try:
                src = pd.read_csv(src_path)
            except Exception as e:
                errors.append(f"Cannot read source {src_path}: {e}")
                continue

            try:
                pr = pd.read_csv(preds_path)
            except Exception as e:
                errors.append(f"Cannot read preds {preds_path}: {e}")
                continue

            if len(pr) != len(src):
                errors.append(
                    f"{domain}/{preds_path.name}: row count {len(pr)} != source {len(src)}"
                )
                continue

            for col in ("verdict_1", "verdict_2"):
                if col not in pr.columns or col not in src.columns:
                    errors.append(f"{preds_path.name}: missing column {col}")
                    break
            else:
                for i in range(len(src)):
                    a = str(src.iloc[i]["verdict_1"]).strip()
                    b = str(src.iloc[i]["verdict_2"]).strip()
                    pa = str(pr.iloc[i]["verdict_1"]).strip()
                    pb = str(pr.iloc[i]["verdict_2"]).strip()
                    if (a, b) != (pa, pb):
                        errors.append(
                            f"{preds_path.name} row {i}: pair mismatch "
                            f"src=({a},{b}) preds=({pa},{pb})"
                        )
                        break

            if "status" not in pr.columns:
                errors.append(f"{preds_path.name}: missing status column")
            else:
                bad = pr[pr["status"].astype(str) != "ok"]
                if len(bad):
                    errors.append(
                        f"{preds_path.name}: {len(bad)} rows with status != ok "
                        f"(e.g. row {bad.index[0]} status={bad.iloc[0]['status']!r})"
                    )

            if "score" not in pr.columns:
                errors.append(f"{preds_path.name}: missing score column")
            else:
                for i, row in pr.iterrows():
                    if str(row.get("status", "")) != "ok":
                        continue
                    if not _validate_score(row.get("score")):
                        errors.append(
                            f"{preds_path.name} row {i}: invalid score {row.get('score')!r}"
                        )

            # duplicate pairs
            keys = list(
                zip(
                    pr["verdict_1"].astype(str).str.strip(),
                    pr["verdict_2"].astype(str).str.strip(),
                )
            )
            if len(keys) != len(set(keys)):
                errors.append(f"{preds_path.name}: duplicate (verdict_1, verdict_2) rows")

            # empty response (warning only)
            if "response" in pr.columns:
                empty_ok = pr[(pr["status"].astype(str) == "ok") & (pr["response"].isna() | (pr["response"].astype(str).str.strip() == ""))]
                if len(empty_ok):
                    warnings.append(
                        f"{preds_path.name}: {len(empty_ok)} ok rows with empty response"
                    )

            # Strict: ok rows must contain real SIMILARITY_SCORE line matching score column (not fallback parser garbage)
            if strict_response and "response" in pr.columns:
                for i, row in pr.iterrows():
                    if str(row.get("status", "")) != "ok":
                        continue
                    msgs = _strict_ok_row(preds_path.name, row)
                    for msg in msgs:
                        errors.append(f"{preds_path.name} row {i}: {msg}")

            # stats consistency
            stats_path = preds_path.with_name(
                preds_path.name.replace("_preds.csv", "_stats.json")
            )
            if stats_path.exists():
                try:
                    st = json.loads(stats_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as e:
                    errors.append(f"{stats_path.name}: invalid JSON: {e}")
                    continue
                if not st.get("complete"):
                    errors.append(
                        f"{stats_path.name}: complete={st.get('complete')} (expected True)"
                    )
                if st.get("n_failed", -1) != 0:
                    errors.append(
                        f"{stats_path.name}: n_failed={st.get('n_failed')} (expected 0)"
                    )
                if st.get("n_valid") is not None and st["n_valid"] != len(src):
                    errors.append(
                        f"{stats_path.name}: n_valid={st['n_valid']} != source rows {len(src)}"
                    )
                if st.get("n_pairs") is not None and st["n_pairs"] != len(src):
                    errors.append(
                        f"{stats_path.name}: n_pairs={st['n_pairs']} != source rows {len(src)}"
                    )
            else:
                errors.append(f"Missing stats: {stats_path.name}")

    n_matrix = 0
    for d in ("drugs", "weapon"):
        for p in (V6_ROOT / d / f"results_{d}").glob("*_v6score_*_binary_0_preds.csv"):
            prs = _parse_preds_name(p.name)
            if prs and prs[1] in MATRIX_MODELS:
                n_matrix += 1
    n_all = sum(
        1
        for d in ("drugs", "weapon")
        for _ in (V6_ROOT / d / f"results_{d}").glob("*_v6score_*_binary_0_preds.csv")
    )

    print(f"Preds files on disk (all): {n_all}")
    print(f"Preds in official matrix (11 models): {n_matrix}")
    if matrix_only and skipped_extra:
        print(f"Skipped (not in MATRIX_MODELS): {len(skipped_extra)}")
        for s in skipped_extra[:20]:
            print(f"  skip: {s}")
        if len(skipped_extra) > 20:
            print(f"  ... and {len(skipped_extra) - 20} more")
    print(f"Errors: {len(errors)}")
    if strict_response and errors:
        by_model: dict[str, int] = {}
        for e in errors:
            m = re.search(r"_v6score_([^/]+?)_binary_0_preds", e)
            key = m.group(1) if m else "?"
            by_model[key] = by_model.get(key, 0) + 1
        print("  (strict-response errors by model)")
        for k in sorted(by_model.keys(), key=lambda x: -by_model[x]):
            print(f"    {k}: {by_model[k]}")
    for e in errors[:80]:
        print(f"  ERROR: {e}")
    if len(errors) > 80:
        print(f"  ... and {len(errors) - 80} more errors (see summary by model above)")

    print(f"Warnings: {len(warnings)}")
    for w in warnings[:40]:
        print(f"  WARN: {w}")
    if len(warnings) > 40:
        print(f"  ... and {len(warnings) - 40} more warnings")

    if errors:
        return 1
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--all-files",
        action="store_true",
        help="Also validate extra preds (e.g. dicta, nemotron) not in the 11-model matrix",
    )
    ap.add_argument(
        "--no-strict-response",
        action="store_true",
        help="Skip check that ok rows contain SIMILARITY_SCORE: N matching the score column (legacy lenient mode)",
    )
    args = ap.parse_args()
    sys.exit(
        validate_all(
            matrix_only=not args.all_files,
            strict_response=not args.no_strict_response,
        )
    )
