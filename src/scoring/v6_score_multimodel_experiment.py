#!/usr/bin/env python3
"""
v6 score (5 dimensions + SIMILARITY_SCORE 0–100) across multiple LLM backends and representations.

User message: raw JSON feature vectors side-by-side (or indictment facts for facts-only CSVs).
Post-hoc: optimize threshold on F1 + report LOO threshold metrics (same as structured_llm v6).

Usage:
  cd new_try/code
  python v6_score_multimodel_experiment.py --domain both --models gpt4 gpt5mini --reps hybrid_full_gpt
  python v6_score_multimodel_experiment.py --domain drugs --models gpt4 --reps all --limit 5   # smoke test
  python v6_score_multimodel_experiment.py --domain both --reps hybrid_full_gpt --models gpt4 \\
    --output-root ../experiments/v6_hybrid_full_gpt_score_multimodel
  python v6_score_multimodel_experiment.py --domain both --reps hybrid_full_gpt --parallel 4 \\
    --models gpt4 mistral qwen_hf gemini_25_pro

Default representation is hybrid_full_gpt (same flagship hybrid as the main similarity paper).
Optional rep "concepts" exists only to reproduce the concept-reduced CSVs; prior work showed
concept post-processing was neutral vs full hybrid — use hybrid_full_gpt for fair comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Reuse prompts + score utilities
from structured_llm_comparison_experiment import (
    USER_TEMPLATE_SCORE_RAW,
    find_best_threshold,
    loo_threshold,
    parse_score_v6,
)

BASE_DIR = Path(__file__).resolve().parent.parent

# System prompt: same as v6_score_raw (two raw JSON blocks)
from structured_llm_comparison_experiment import SYSTEM_PROMPT_V6_SCORE_RAW_drugs, SYSTEM_PROMPT_V6_SCORE_RAW_wep

SYSTEM_PROMPT_V6_FACTS_drugs = """את/ה מומחית לדין הפלילי בישראל. מוצגות בפנייך **עובדות מכתב אישום** של שני תיקים פליליים.

המשימה: הערכת דמיון מהותי — עד כמה תיק אחד יכול לשמש כתקדים ענייני לשני?

נתחי כל ממד בנפרד:
1. **סוג העבירה וחומרתה** — האם מדובר באותו סוג עבירה? באותה רמת חומרה?
2.  סוג הסם וכמותו — האם מדובר בסוג דומה בחומרתו? בכמויות דומות בסדר גודל?
3. **שיטת הביצוע (MO)** — דפוס דומה? אמצעים דומים?
4. **נסיבות הליבה** — מעבדה, תכנון, נסיבות מחמירות/מקלות?
5. **ישימות כתקדים** — האם בית משפט יראה את שני התיקים כרלוונטיים זה לזה לצורכי ענישה?

חשוב:
- התמקדי בפרמטרים המשפטיים המהותיים, לא בפרטים טכניים/ביורוקרטיים.
- השתייכות לאותו תחום (סמים/נשק) לבדה **לא מספיקה** לדמיון מהותי.

פורמט תשובה:
1. ניתוח קצר (2-3 משפטים) של כל ממד
2. שורה אחרונה בדיוק בפורמט: SIMILARITY_SCORE: X
   כאשר X הוא מספר שלם בין 0 ל-100.
   0 = שונים לחלוטין, 100 = זהים כמעט. ציון מעל 50 = תקדים רלוונטי."""


SYSTEM_PROMPT_V6_FACTS_wep = """את/ה מומחית לדין הפלילי בישראל. מוצגות בפנייך **עובדות מכתב אישום** של שני תיקים פליליים.

המשימה: הערכת דמיון מהותי — עד כמה תיק אחד יכול לשמש כתקדים ענייני לשני?

נתחי כל ממד בנפרד:
1. **סוג העבירה וחומרתה** — האם מדובר באותו סוג עבירה? באותה רמת חומרה?
2. **תפקיד הנאשם ומעורבותו** — יזם/סוחר לעומת שליח/מחזיק? מעורבות פעילה לעומת פסיבית?
3. **שיטת הביצוע (MO)** — דפוס דומה? אמצעים דומים?
4. **נסיבות הליבה** — כמויות, שימוש בנשק, תכנון, נסיבות מחמירות/מקלות?
5. **ישימות כתקדים** — האם בית משפט יראה את שני התיקים כרלוונטיים זה לזה לצורכי ענישה?

חשוב:
- התמקדי בפרמטרים המשפטיים המהותיים, לא בפרטים טכניים/ביורוקרטיים.
- השתייכות לאותו תחום (סמים/נשק) לבדה **לא מספיקה** לדמיון מהותי.

פורמט תשובה:
1. ניתוח קצר (2-3 משפטים) של כל ממד
2. שורה אחרונה בדיוק בפורמט: SIMILARITY_SCORE: X
   כאשר X הוא מספר שלם בין 0 ל-100.
   0 = שונים לחלוטין, 100 = זהים כמעט. ציון מעל 50 = תקדים רלוונטי."""

USER_TEMPLATE_FACTS = """להלן עובדות מכתב אישום של שני תיקים:

תיק 1:
{fv1}

תיק 2:
{fv2}

מהו ציון הדמיון המהותי (0-100)?"""

# (id, csv_filename, input_kind: "features" | "facts")
REPRESENTATIONS_DEFAULT: list[tuple[str, str, str]] = [
    ("facts", "similarity_database_with_indicment_facts.csv", "facts"),
    ("manual_fe", "similarity_database_fe.csv", "features"),
    ("fe_manual_format", "similarity_database_fe_manual_format.csv", "features"),
    # Structured extraction → deterministic legacy JSON (manual_format_to_legacy_fe.py) for v6 on hand-format vectors.
    (
        "fe_legacy_from_structured",
        "similarity_database_fe_legacy_from_structured.csv",
        "features",
    ),
    # Same structured JSON format as manual_fe, but field names/schema produced via GPT (see thesis data chapter).
    ("fe_gpt_schema", "similarity_database_fe_gpt_schema.csv", "features"),
    # SmartTag-fixed re-extraction (2025-04): same schema, rebuilt from docx with lxml SmartTag parsing.
    ("fe_gpt_schema_v2", "similarity_database_fe_gpt_schema_v2.csv", "features"),
    ("fe_my_gpt_extracted", "similarity_database_fe_my_gpt_extracted.csv", "features"),
    # GPT auto-extracted features in manual-FE format (weapon only; built by build_gpt_manual_features.py)
    ("gpt_manual_features", "gpt_manual_features.csv", "features"),
    ("gpt_law", "similarity_database_with_gpt_law_features.csv", "features"),
    ("gpt_free", "similarity_database_with_gpt_features.csv", "features"),
    ("hybrid_manual_gpt", "similarity_database_hybrid.csv", "features"),
    ("hybrid_gpt", "similarity_database_hybrid_gpt.csv", "features"),
    ("hybrid_full_gpt", "similarity_database_hybrid_full_gpt.csv", "features"),
    # Optional: concept-reduced features (neutral vs full hybrid in prior experiments — use hybrid_full_gpt for main tables)
    ("concepts", "__concepts__", "features"),
]

MAIN_MODELS_DEFAULT = [
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

STATUS_OK = "ok"
STATUS_PARSE_ERROR = "parse_error"
STATUS_API_ERROR = "api_error"
STATUS_RETRY_EXHAUSTED = "retry_exhausted"

# Concurrent model runs: cap in-flight calls per provider to reduce 429 / HF throttling.
_PROVIDER_CONCURRENCY: dict[str, int] = {
    "openai": 5,  # gpt4, gpt5mini, gpt52, gpt51_thinking
    "gemini": 3,  # shared endpoint; call_gemini already rate-limits
    "hf": 4,  # Dicta + Qwen HF + Gemma HF (router.huggingface.co)
    "nim": 5,  # NVIDIA NIM (multiple API keys rotate)
    "anthropic": 10,
    "other": 2,
}


def _provider_bucket(model_backend: str) -> str:
    if model_backend in ("gpt4", "gpt5mini", "gpt52", "gpt51_thinking"):
        return "openai"
    if model_backend == "claude_sonnet_4_6":
        return "anthropic"
    if model_backend in ("gemini_25_pro", "gemini_3_flash"):
        return "gemini"
    if model_backend in ("dicta", "qwen_hf", "qwen3_235b", "gemma3_27b"):
        return "hf"
    if model_backend in (
        "mistral",
        "nemotron3_nano",
        "nemotron3_super",
        "llama3_70b",
        "qwen",
        "llama",
        "nemotron",
        "gpt_oss",
    ):
        return "nim"
    return "other"


def _make_provider_semaphores() -> dict[str, threading.Semaphore]:
    return {k: threading.Semaphore(v) for k, v in _PROVIDER_CONCURRENCY.items()}


def validate_score(s: float | None) -> bool:
    if s is None:
        return False
    try:
        v = float(s)
    except (TypeError, ValueError):
        return False
    if np.isnan(v):
        return False
    return 0.0 <= v <= 100.0


def _domain_paths(domain: str) -> Path:
    d = "weapon" if domain in ("weapon", "wep") else "drugs"
    return BASE_DIR / d


def resolve_csv(domain: str, rep_id: str, csv_name: str) -> Path:
    if csv_name == "__concepts__":
        sub = (
            "similarity_database_hybrid_concepts_drugs.csv"
            if domain == "drugs"
            else "similarity_database_hybrid_concepts_weapon.csv"
        )
        return BASE_DIR / "code" / "post_process_output" / sub
    return _domain_paths(domain) / csv_name


def resolve_csv_maybe_override(
    domain: str,
    rep_id: str,
    csv_name: str,
    drugs_manual_fe_override: Path | None,
) -> Path:
    """Use --drugs-manual-fe-csv for drugs+manual_fe when the corrected matrix lives in a new file."""
    if (
        drugs_manual_fe_override is not None
        and domain == "drugs"
        and rep_id == "manual_fe"
    ):
        return drugs_manual_fe_override
    return resolve_csv(domain, rep_id, csv_name)


def call_model_backend(model_backend: str, system_prompt: str, user_prompt: str) -> str | None:
    """Dispatch to similarity_experiment API helpers; returns raw text or None."""
    import similarity_experiment as se

    try:
        if model_backend == "gpt4":
            return se.call_gpt4_1(system_prompt, user_prompt, log_call=False)
        if model_backend in ("gpt4_mini", "gpt5mini"):
            return se.call_gpt4_mini(system_prompt, user_prompt, log_call=False)
        if model_backend == "gpt52":
            return se.call_gpt52(system_prompt, user_prompt, log_call=False)
        if model_backend == "gpt51_thinking":
            return se.call_gpt51_thinking(system_prompt, user_prompt, log_call=False)
        if model_backend == "dicta":
            return se.call_dicta(system_prompt, user_prompt, log_call=False)
        if model_backend == "qwen_hf":
            return se.call_qwen_hf(system_prompt, user_prompt, log_call=False)
        if model_backend in ("gemini_25_pro", "gemini_3_flash"):
            name = se.MODEL_MAP[model_backend]
            return se.call_gemini(system_prompt, user_prompt, name, log_call=False)
        if model_backend in (
            "qwen",
            "llama",
            "nemotron",
            "gpt_oss",
            "mistral",
            "nemotron3_nano",
            "nemotron3_super",
            "qwen3_235b",
            "llama3_70b",
            "gemma3_27b",
        ):
            return se.call_nim(system_prompt, user_prompt, model_backend, log_call=False)
        if model_backend == "claude_sonnet_4_6":
            return se.call_claude(system_prompt, user_prompt, log_call=False)
        if model_backend == "lm_studio":
            return se.call_lm_studio(system_prompt, user_prompt, log_call=False)
    except Exception as e:
        extra = ""
        if isinstance(e, requests.exceptions.HTTPError):
            r = e.response
            if r is not None:
                try:
                    body = (r.text or "").strip()
                    if body:
                        extra = f"\n    HTTP {r.status_code} response body: {body[:2500]}"
                        if len(body) > 2500:
                            extra += "…"
                    else:
                        extra = f"\n    HTTP {r.status_code} (empty response body)"
                except Exception:
                    extra = f"\n    HTTP {getattr(r, 'status_code', '?')} (could not read body)"
        print(f"  API error ({model_backend}): {e}{extra}")
        return None
    raise ValueError(f"Unknown model backend: {model_backend}")


def _pair_key(row: pd.Series) -> tuple[str, str]:
    return (str(row["verdict_1"]), str(row["verdict_2"]))


def load_row_cache(cache_path: Path, df: pd.DataFrame) -> dict[tuple[str, str], dict]:
    """Load per-pair cache from existing preds CSV (score + response + status)."""
    if not cache_path.exists():
        return {}
    try:
        c = pd.read_csv(cache_path)
    except Exception as e:
        print(f"  ⚠️  Could not read cache {cache_path}: {e}")
        return {}
    if not {"verdict_1", "verdict_2"}.issubset(set(c.columns)):
        return {}
    out: dict[tuple[str, str], dict] = {}
    for _, r in c.iterrows():
        key = (str(r["verdict_1"]), str(r["verdict_2"]))
        raw_sc = r.get("score")
        sc = float(raw_sc) if pd.notna(raw_sc) else None
        resp = r.get("response", "")
        if pd.isna(resp):
            resp = ""
        else:
            resp = str(resp)
        if "status" in c.columns and pd.notna(r.get("status")):
            st = str(r["status"])
        else:
            # Legacy preds CSV: only score/response
            st = STATUS_OK if validate_score(sc) else STATUS_PARSE_ERROR
        out[key] = {"score": sc, "response": resp, "status": st}
    return out


def save_checkpoint(
    out_csv: Path,
    df: pd.DataFrame,
    task: str,
    rows_out: list[dict],
) -> None:
    """Atomic-ish write of partial or full results."""
    extra = pd.DataFrame(rows_out)
    base = df.reset_index(drop=True)
    merged = pd.concat([base, extra], axis=1)
    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
    # QUOTE_NONNUMERIC: quote all text fields so multiline responses / empty scores cannot break rows.
    merged.to_csv(
        tmp,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_NONNUMERIC,
        na_rep="",
    )
    tmp.replace(out_csv)


def _results_subdir(domain: str) -> str:
    return "results_weapon" if domain in ("weapon", "wep") else "results_drugs"


def _v6_build_stats(
    sc: np.ndarray,
    y_true: np.ndarray,
    domain_key: str,
    rep_id: str,
    csv_name: str,
    model_backend: str,
    task: str,
    n_pairs: int,
    n_failed: int,
    preds_csv: str,
    *,
    from_same_scores: bool = False,
) -> dict:
    """Metrics from continuous scores vs one binary label column (same scores, different GT)."""
    best_thr, best_f1 = find_best_threshold(sc, y_true)
    y_pred = (sc >= best_thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    loo = loo_threshold(sc, y_true)
    out = {
        "method": "v6_score_multimodel",
        "complete": n_failed == 0,
        "domain": domain_key,
        "representation_id": rep_id,
        "csv": csv_name,
        "model": model_backend,
        "task": task,
        "n_pairs": n_pairs,
        "n_valid": int(len(sc)),
        "n_failed": int(n_failed),
        "preds_csv": preds_csv,
        "best_threshold": round(float(best_thr), 4),
        "global_best": {
            "f1": round(float(best_f1), 4),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        },
        "loo": loo,
        "score_stats": {
            "mean": round(float(np.nanmean(sc)), 2),
            "std": round(float(np.nanstd(sc)), 2),
            "pos_mean": round(float(np.mean(sc[y_true == 1])), 2) if (y_true == 1).any() else None,
            "neg_mean": round(float(np.mean(sc[y_true == 0])), 2) if (y_true == 0).any() else None,
        },
    }
    if from_same_scores:
        out["eval_note"] = (
            "No extra API calls: same continuous scores as primary task; "
            "threshold/F1 recomputed vs this label column only."
        )
    return out


def _resolve_results_dir(domain: str, output_root: Path | None) -> Path:
    """If output_root is set, write under OUTPUT_ROOT/<drugs|weapon>/results_*/ (mirrors default layout)."""
    sub = _results_subdir(domain)
    if output_root is None:
        p = _domain_paths(domain) / sub
    else:
        dname = "drugs" if domain == "drugs" else "weapon"
        p = output_root / dname / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_one_config(
    domain: str,
    model_backend: str,
    rep_id: str,
    csv_path: Path,
    input_kind: str,
    task: str,
    limit: int | None,
    sleep_sec: float,
    resume: bool,
    max_retries: int,
    flush_every: int,
    output_root: Path | None = None,
) -> list[dict]:
    domain_key = "drugs" if domain == "drugs" else "weapon"
    results_dir = _resolve_results_dir(domain, output_root)

    tag = f"{csv_path.stem}_v6score_{model_backend}_{task}"
    out_csv = results_dir / f"{tag}_preds.csv"
    out_stats = results_dir / f"{tag}_stats.json"

    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.iloc[:limit].copy()

    labels = df[f"similarity_{task}"].values.astype(int)
    if input_kind == "facts":
        system = SYSTEM_PROMPT_V6_FACTS_drugs if domain == "drugs" else SYSTEM_PROMPT_V6_FACTS_wep
    else:
        system = SYSTEM_PROMPT_V6_SCORE_RAW_drugs if domain == "drugs" else SYSTEM_PROMPT_V6_SCORE_RAW_wep

    cache = load_row_cache(out_csv, df) if resume else {}

    rows_out: list[dict] = []
    n_skipped_ok = 0
    n_refetched = 0

    for pos, (_, row) in enumerate(df.iterrows()):
        key = _pair_key(row)
        if input_kind == "facts":
            u = USER_TEMPLATE_FACTS.format(fv1=row["indicment_facts_1"], fv2=row["indicment_facts_2"])
        else:
            u = USER_TEMPLATE_SCORE_RAW.format(fv1=row["feature_vector_1"], fv2=row["feature_vector_2"])

        score: float | None = None
        response = ""
        status = STATUS_API_ERROR
        last_err = ""

        cached = cache.get(key)
        if (
            resume
            and cached
            and cached.get("status") == STATUS_OK
            and validate_score(cached.get("score"))
        ):
            score = float(cached["score"])
            response = str(cached.get("response") or "")
            status = STATUS_OK
            n_skipped_ok += 1
            rows_out.append(
                {"score": score, "response": response, "status": status, "last_error": ""}
            )
            if (pos + 1) % 10 == 0:
                print(f"    {rep_id} {model_backend}: {pos + 1}/{len(df)} (cached ok)")
            continue

        if cached and resume:
            n_refetched += 1

        for attempt in range(max(1, max_retries)):
            raw = call_model_backend(model_backend, system, u)
            response = raw or ""
            if raw is None:
                last_err = "empty_api_response"
                status = STATUS_API_ERROR
                time.sleep(min(2.0 * (attempt + 1), 30.0))
                continue

            parsed = parse_score_v6(raw)
            if validate_score(parsed):
                score = float(parsed)
                status = STATUS_OK
                break

            last_err = "parse_error_or_invalid_score"
            status = STATUS_PARSE_ERROR
            time.sleep(0.5 * (attempt + 1))

        if status != STATUS_OK:
            status = STATUS_RETRY_EXHAUSTED if max_retries > 0 else STATUS_PARSE_ERROR
            score = None  # CSV-friendly; invalid rows excluded from metrics

        rows_out.append(
            {
                "score": score,
                "response": response,
                "status": status,
                "last_error": last_err if status != STATUS_OK else "",
            }
        )

        if sleep_sec > 0:
            time.sleep(sleep_sec)

        if flush_every > 0 and (pos + 1) % flush_every == 0:
            save_checkpoint(out_csv, df, task, rows_out)

        if (pos + 1) % 10 == 0:
            print(f"    {rep_id} {model_backend}: {pos + 1}/{len(df)}")

    # Final write
    save_checkpoint(out_csv, df, task, rows_out)
    if n_skipped_ok:
        print(f"    (row cache: skipped {n_skipped_ok} ok rows; refetched {n_refetched} non-ok/missing)")

    sc_arr = np.full(len(df), np.nan, dtype=float)
    for i, r in enumerate(rows_out):
        if r["status"] == STATUS_OK and validate_score(r.get("score")):
            sc_arr[i] = float(r["score"])
    valid_mask = ~np.isnan(sc_arr)
    n_valid = int(valid_mask.sum())
    n_failed = len(df) - n_valid

    if n_valid == 0:
        print(f"  No valid scores for {rep_id} {model_backend} ({n_failed} failed rows — re-run to retry)")
        partial = {
            "method": "v6_score_multimodel",
            "complete": False,
            "domain": domain_key,
            "representation_id": rep_id,
            "csv": str(csv_path.name),
            "model": model_backend,
            "task": task,
            "n_pairs": len(df),
            "n_valid": 0,
            "n_failed": n_failed,
            "preds_csv": str(out_csv),
        }
        with open(out_stats, "w", encoding="utf-8") as f:
            json.dump(partial, f, indent=2, ensure_ascii=False)
        return [partial]

    y_true = labels[valid_mask]
    sc = sc_arr[valid_mask]

    stats = _v6_build_stats(
        sc,
        y_true,
        domain_key,
        rep_id,
        str(csv_path.name),
        model_backend,
        task,
        len(df),
        n_failed,
        str(out_csv),
        from_same_scores=False,
    )
    best_thr = float(stats["best_threshold"])
    loo = stats["loo"]

    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    out_list: list[dict] = [stats]

    suffix = "" if n_failed == 0 else f" (partial: {n_failed} rows still invalid — delete those rows or re-run)"
    print(
        f"  {rep_id} | {model_backend} | task={task} F1={stats['global_best']['f1']:.3f} "
        f"LOO={loo['f1']:.3f} thr={best_thr:.1f}{suffix}"
    )

    for alt_task in ("binary_0", "binary_1"):
        if alt_task == task:
            continue
        col = f"similarity_{alt_task}"
        if col not in df.columns:
            continue
        y_alt = df[col].values.astype(int)[valid_mask]
        alt_stats = _v6_build_stats(
            sc,
            y_alt,
            domain_key,
            rep_id,
            str(csv_path.name),
            model_backend,
            alt_task,
            len(df),
            n_failed,
            str(out_csv),
            from_same_scores=True,
        )
        alt_path = results_dir / f"{csv_path.stem}_v6score_{model_backend}_{alt_task}_stats.json"
        with open(alt_path, "w", encoding="utf-8") as f:
            json.dump(alt_stats, f, indent=2, ensure_ascii=False)
        out_list.append(alt_stats)
        print(
            f"  {rep_id} | {model_backend} | task={alt_task} (same scores, no extra API) F1={alt_stats['global_best']['f1']:.3f} "
            f"LOO={alt_stats['loo']['f1']:.3f} thr={alt_stats['best_threshold']:.1f} -> {alt_path.name}"
        )

    return out_list


def _run_one_config_capped(
    semaphores: dict[str, threading.Semaphore],
    domain: str,
    model_backend: str,
    rep_id: str,
    csv_path: Path,
    kind: str,
    task: str,
    limit: int | None,
    sleep_sec: float,
    resume: bool,
    max_retries: int,
    flush_every: int,
    output_root: Path | None,
) -> list[dict]:
    b = _provider_bucket(model_backend)
    sem = semaphores.get(b) or semaphores["other"]
    with sem:
        return run_one_config(
            domain,
            model_backend,
            rep_id,
            csv_path,
            kind,
            task,
            limit,
            sleep_sec,
            resume,
            max_retries,
            flush_every,
            output_root,
        )


def _run_drugs_then_weapon_for_one_model(
    semaphores: dict[str, threading.Semaphore],
    *,
    use_cap: bool,
    model_backend: str,
    rep_id: str,
    csv_name: str,
    kind: str,
    task: str,
    limit: int | None,
    sleep_sec: float,
    resume: bool,
    max_retries: int,
    flush_every: int,
    output_root: Path | None,
    drugs_manual_fe_override: Path | None = None,
) -> list[dict]:
    """Run drugs then weapon for the same (rep, model) so slow models on drugs do not block weapon for others."""
    acc: list[dict] = []
    for dom in ("drugs", "weapon"):
        csv_path = resolve_csv_maybe_override(dom, rep_id, csv_name, drugs_manual_fe_override)
        if not csv_path.exists():
            print(f"  SKIP missing CSV: {csv_path}")
            continue
        if use_cap:
            acc.extend(
                _run_one_config_capped(
                    semaphores,
                    dom,
                    model_backend,
                    rep_id,
                    csv_path,
                    kind,
                    task,
                    limit,
                    sleep_sec,
                    resume,
                    max_retries,
                    flush_every,
                    output_root,
                )
            )
        else:
            acc.extend(
                run_one_config(
                    dom,
                    model_backend,
                    rep_id,
                    csv_path,
                    kind,
                    task,
                    limit,
                    sleep_sec,
                    resume,
                    max_retries,
                    flush_every,
                    output_root,
                )
            )
    return acc


def main():
    parser = argparse.ArgumentParser(description="v6 score multimodel similarity experiment")
    parser.add_argument(
        "--domain",
        choices=["drugs", "weapon", "wep", "both"],
        default="drugs",
        help="both = for each (representation, model) run drugs then weapon before the next model, "
        "so weapon does not wait for all models to finish drugs.",
    )
    parser.add_argument("--models", nargs="+", default=MAIN_MODELS_DEFAULT)
    parser.add_argument(
        "--reps",
        nargs="+",
        default=["hybrid_full_gpt"],
        help="Representation ids (see REPRESENTATIONS_DEFAULT) or 'all'. Default: hybrid_full_gpt (flagship hybrid).",
    )
    parser.add_argument(
        "--task",
        default="binary_0",
        help="Primary label for prompts/metrics file names (binary_0 or binary_1). "
        "If the CSV has both similarity_binary_0 and similarity_binary_1, metrics for the "
        "other task are computed from the same SIMILARITY_SCORE without extra API calls.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.15, help="Sleep between API calls")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore row-level cache (*_preds.csv); re-call API for every row",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="No-op: row cache is already on by default (use --fresh to disable). Kept for CLI compatibility.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="API/parse retries per row before marking failed",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help="Write preds CSV every N rows (1 = safest against crashes)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional base folder for all outputs: OUTPUT_ROOT/drugs/results_drugs/ and "
        "OUTPUT_ROOT/weapon/results_weapon/. Default: new_try/drugs|weapon/ (unchanged).",
    )
    parser.add_argument(
        "--drugs-manual-fe-csv",
        type=str,
        default=None,
        help="Optional path to a corrected drugs manual-FE matrix CSV (same columns as "
        "similarity_database_fe.csv). When set, used for domain=drugs and rep=manual_fe only.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Run up to N models in parallel (threads). Provider caps still apply: "
        "OpenAI≤2, HF≤2, NIM≤3, Gemini≤1. Default 1 = sequential.",
    )
    args = parser.parse_args()

    output_root: Path | None = None
    if args.output_root:
        output_root = Path(args.output_root).resolve()
        output_root.mkdir(parents=True, exist_ok=True)

    drugs_manual_fe_override: Path | None = None
    if args.drugs_manual_fe_csv:
        drugs_manual_fe_override = Path(args.drugs_manual_fe_csv).expanduser().resolve()
        if not drugs_manual_fe_override.is_file():
            print(f"--drugs-manual-fe-csv not found: {drugs_manual_fe_override}", file=sys.stderr)
            sys.exit(1)

    rep_list = REPRESENTATIONS_DEFAULT
    if args.reps == ["all"]:
        reps_sel = rep_list
    else:
        id_map = {r[0]: r for r in rep_list}
        reps_sel = []
        for rid in args.reps:
            if rid not in id_map:
                print(f"Unknown rep id: {rid}", file=sys.stderr)
                sys.exit(1)
            reps_sel.append(id_map[rid])

    domains = ["drugs", "weapon"] if args.domain == "both" else [args.domain]
    if domains == ["wep"]:
        domains = ["weapon"]

    all_stats: list[dict] = []
    provider_sems = _make_provider_semaphores()
    interleave_both = len(domains) == 2 and set(domains) == {"drugs", "weapon"}

    if interleave_both:
        for rep_id, csv_name, kind in reps_sel:
            print(f"\n{'='*60}\n  REP: {rep_id}  (drugs → weapon per model)\n{'='*60}")
            if args.parallel <= 1:
                for model_backend in args.models:
                    try:
                        s = _run_drugs_then_weapon_for_one_model(
                            provider_sems,
                            use_cap=False,
                            model_backend=model_backend,
                            rep_id=rep_id,
                            csv_name=csv_name,
                            kind=kind,
                            task=args.task,
                            limit=args.limit,
                            sleep_sec=args.sleep,
                            resume=not args.fresh,
                            max_retries=args.max_retries,
                            flush_every=args.flush_every,
                            output_root=output_root,
                            drugs_manual_fe_override=drugs_manual_fe_override,
                        )
                        if s:
                            all_stats.extend(s)
                    except Exception as e:
                        print(f"  FAIL {rep_id} {model_backend}: {e}")
            else:
                max_workers = max(1, int(args.parallel))
                print(
                    f"  (parallel: up to {max_workers} workers; each runs drugs then weapon; "
                    f"provider caps: OpenAI≤{_PROVIDER_CONCURRENCY['openai']} HF≤{_PROVIDER_CONCURRENCY['hf']} "
                    f"NIM≤{_PROVIDER_CONCURRENCY['nim']} Gemini≤{_PROVIDER_CONCURRENCY['gemini']})"
                )
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures: dict = {}
                    for model_backend in args.models:
                        fut = ex.submit(
                            _run_drugs_then_weapon_for_one_model,
                            provider_sems,
                            use_cap=True,
                            model_backend=model_backend,
                            rep_id=rep_id,
                            csv_name=csv_name,
                            kind=kind,
                            task=args.task,
                            limit=args.limit,
                            sleep_sec=args.sleep,
                            resume=not args.fresh,
                            max_retries=args.max_retries,
                            flush_every=args.flush_every,
                            output_root=output_root,
                            drugs_manual_fe_override=drugs_manual_fe_override,
                        )
                        futures[fut] = model_backend
                    for fut in as_completed(futures):
                        mb = futures[fut]
                        try:
                            s = fut.result()
                            if s:
                                all_stats.extend(s)
                        except Exception as e:
                            print(f"  FAIL {rep_id} {mb}: {e}")
    else:
        for dom in domains:
            print(f"\n{'='*60}\n  DOMAIN: {dom}\n{'='*60}")
            for rep_id, csv_name, kind in reps_sel:
                csv_path = resolve_csv_maybe_override(dom, rep_id, csv_name, drugs_manual_fe_override)
                if not csv_path.exists():
                    print(f"  SKIP missing CSV: {csv_path}")
                    continue
                if args.parallel <= 1:
                    for model_backend in args.models:
                        try:
                            s = run_one_config(
                                dom,
                                model_backend,
                                rep_id,
                                csv_path,
                                kind,
                                args.task,
                                args.limit,
                                args.sleep,
                                resume=not args.fresh,
                                max_retries=args.max_retries,
                                flush_every=args.flush_every,
                                output_root=output_root,
                            )
                            if s:
                                all_stats.extend(s)
                        except Exception as e:
                            print(f"  FAIL {rep_id} {model_backend}: {e}")
                else:
                    max_workers = max(1, int(args.parallel))
                    print(
                        f"  (parallel: up to {max_workers} workers; provider caps: "
                        f"OpenAI≤{_PROVIDER_CONCURRENCY['openai']} HF≤{_PROVIDER_CONCURRENCY['hf']} "
                        f"NIM≤{_PROVIDER_CONCURRENCY['nim']} Gemini≤{_PROVIDER_CONCURRENCY['gemini']})"
                    )
                    with ThreadPoolExecutor(max_workers=max_workers) as ex:
                        futures: dict = {}
                        for model_backend in args.models:
                            fut = ex.submit(
                                _run_one_config_capped,
                                provider_sems,
                                dom,
                                model_backend,
                                rep_id,
                                csv_path,
                                kind,
                                args.task,
                                args.limit,
                                args.sleep,
                                not args.fresh,
                                args.max_retries,
                                args.flush_every,
                                output_root,
                            )
                            futures[fut] = model_backend
                        for fut in as_completed(futures):
                            mb = futures[fut]
                            try:
                                s = fut.result()
                                if s:
                                    all_stats.extend(s)
                            except Exception as e:
                                print(f"  FAIL {rep_id} {mb}: {e}")

    summary_name = f"v6_multimodel_summary_{args.task}.json"
    summary_path = (output_root / summary_name) if output_root else (BASE_DIR / "code" / summary_name)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
