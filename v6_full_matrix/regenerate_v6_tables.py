#!/usr/bin/env python3
"""
Rebuild Excel-style CSVs and CURRENT_RESULTS_AND_PROMPTS.md under v6_full_matrix/
from all complete *_stats.json files (same layout as run_all.sh: 10 reps × 11 models × 2 domains × 2 tasks).
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import average_precision_score

# Keep in sync with run_all.sh
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

MATRIX_REPS = [
    "facts",
    "manual_fe",
    "fe_manual_format",
    "fe_legacy_from_structured",
    "fe_gpt_schema",
    "gpt_law",
    "gpt_free",
    "hybrid_manual_gpt",
    "hybrid_gpt",
    "hybrid_full_gpt",
]

TASKS = ("binary_0", "binary_1")
DOMAINS = ("drugs", "weapon")


def _results_dir(root: Path, domain: str) -> Path:
    return root / domain / f"results_{domain}"


def _resolve_preds_path(root: Path, stats: dict) -> Path | None:
    p = Path(stats.get("preds_csv") or "")
    if not p.name:
        return None
    domain = stats.get("domain")
    if not domain:
        return p if p.exists() else None
    cand = _results_dir(root, domain) / p.name
    if cand.exists():
        return cand
    return p if p.exists() else None


def _ap_from_preds(preds_path: Path, task: str) -> float | None:
    label_col = "similarity_binary_0" if task == "binary_0" else "similarity_binary_1"
    y: list[int] = []
    s: list[float] = []
    with preds_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("status") != "ok":
                continue
            try:
                y.append(int(row[label_col]))
                s.append(float(row["score"]))
            except (ValueError, KeyError):
                continue
    if len(y) < 2 or len(set(y)) < 2:
        return None
    return float(average_precision_score(y, s))


def _load_complete_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for domain in DOMAINS:
        rdir = _results_dir(root, domain)
        if not rdir.is_dir():
            continue
        for path in sorted(rdir.glob("*_stats.json")):
            if not re.match(r".*_v6score_.*_binary_[01]_stats\.json$", path.name):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if not data.get("complete"):
                continue
            task = data.get("task")
            model = data.get("model")
            rep = data.get("representation_id")
            if task not in TASKS or not model or not rep:
                continue
            gb = data.get("global_best") or {}
            preds = _resolve_preds_path(root, data)
            ap_pr = None
            if preds:
                try:
                    ap_pr = _ap_from_preds(preds, task)
                except Exception:
                    ap_pr = None
            rows.append(
                {
                    "domain": domain,
                    "task": task,
                    "model": model,
                    "representation": rep,
                    "f1": gb.get("f1"),
                    "precision": gb.get("precision"),
                    "recall": gb.get("recall"),
                    "ap_pr": ap_pr,
                    "threshold": data.get("best_threshold"),
                    "n_valid": data.get("n_valid"),
                    "n_failed": data.get("n_failed"),
                    "stats_file": path.name,
                    "preds_csv": str(preds) if preds else "",
                }
            )
    return rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _fmt(x: float | None, nd: int = 4) -> str:
    if x is None:
        return ""
    return f"{x:.{nd}f}"


def _md_table(rows: list[dict]) -> str:
    lines = [
        "| model | representation | F1 | precision | recall | AP(PR) | threshold | n_valid | n_failed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: (x["model"], x["representation"])):
        lines.append(
            "| {model} | {rep} | {f1} | {pr} | {rc} | {ap} | {th} | {nv} | {nf} |".format(
                model=r["model"],
                rep=r["representation"],
                f1=_fmt(r.get("f1")),
                pr=_fmt(r.get("precision")),
                rc=_fmt(r.get("recall")),
                ap=_fmt(r.get("ap_pr")),
                th=_fmt(r.get("threshold")),
                nv=r.get("n_valid", ""),
                nf=r.get("n_failed", ""),
            )
        )
    return "\n".join(lines) + "\n"


def _build_prompt_md() -> str:
    code_dir = Path(__file__).resolve().parents[2] / "code"
    v6 = code_dir / "v6_score_multimodel_experiment.py"
    structured = code_dir / "structured_llm_comparison_experiment.py"
    blocks: dict[str, str] = {}

    def grab(name: str, paths: list[Path]) -> None:
        for path in paths:
            text = path.read_text(encoding="utf-8")
            m = re.search(
                rf"^{re.escape(name)}\s*=\s*\"\"\"(.*?)\"\"\"",
                text,
                re.MULTILINE | re.DOTALL,
            )
            if m:
                blocks[name] = m.group(1).strip()
                return

    grab("SYSTEM_PROMPT_V6_FACTS", [v6])
    grab("USER_TEMPLATE_FACTS", [v6])
    grab("SYSTEM_PROMPT_V6_SCORE_RAW", [structured])
    grab("USER_TEMPLATE_SCORE_RAW", [structured])
    out = ["## Representation to Prompt Mapping\n"]
    out.append(
        "- `facts` -> `SYSTEM_PROMPT_V6_FACTS` + `USER_TEMPLATE_FACTS`\n"
        "- `manual_fe`, `fe_manual_format`, `fe_legacy_from_structured`, `fe_gpt_schema`, `gpt_law`, `gpt_free`, `hybrid_manual_gpt`, "
        "`hybrid_gpt`, `hybrid_full_gpt` -> `SYSTEM_PROMPT_V6_SCORE_RAW` + `USER_TEMPLATE_SCORE_RAW`\n"
    )
    out.append("\n## Prompt Texts (Current)\n")
    for title, key in [
        ("SYSTEM_PROMPT_V6_FACTS", "SYSTEM_PROMPT_V6_FACTS"),
        ("USER_TEMPLATE_FACTS", "USER_TEMPLATE_FACTS"),
        ("SYSTEM_PROMPT_V6_SCORE_RAW", "SYSTEM_PROMPT_V6_SCORE_RAW"),
        ("USER_TEMPLATE_SCORE_RAW", "USER_TEMPLATE_SCORE_RAW"),
    ]:
        body = blocks.get(key, "(not found — check v6_score_multimodel_experiment.py)")
        out.append(f"\n### {title}\n```text\n{body}\n```\n")
    return "".join(out)


def main() -> None:
    root = Path(__file__).resolve().parent
    excel = root / "excel_tables"
    rows = _load_complete_rows(root)

    all_fields = [
        "domain",
        "task",
        "model",
        "representation",
        "f1",
        "precision",
        "recall",
        "ap_pr",
        "threshold",
        "n_valid",
        "n_failed",
        "stats_file",
        "preds_csv",
    ]
    _write_csv(excel / "v6_current_completed_all.csv", all_fields, rows)

    for domain in DOMAINS:
        for task in TASKS:
            sub = [r for r in rows if r["domain"] == domain and r["task"] == task]
            _write_csv(
                excel / f"v6_{domain}_{task}_completed.csv",
                all_fields,
                sub,
            )

    # Completion checklist (expected matrix)
    checklist: list[dict] = []
    done = {(r["domain"], r["model"], r["representation"], r["task"]) for r in rows}
    for model in MATRIX_MODELS:
        for rep in MATRIX_REPS:
            for domain in DOMAINS:
                for task in TASKS:
                    key = (domain, model, rep, task)
                    checklist.append(
                        {
                            "model": model,
                            "representation": rep,
                            "domain": domain,
                            "task": task,
                            "status": "completed" if key in done else "missing",
                        }
                    )
    _write_csv(
        excel / "v6_completion_checklist_full.csv",
        ["model", "representation", "domain", "task", "status"],
        checklist,
    )

    # Summaries
    by_rep_dom: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        by_rep_dom[(r["representation"], r["domain"])] += 1
    rep_dom_rows = [
        {"representation": rep, "domain": dom, "completed_cells": by_rep_dom.get((rep, dom), 0)}
        for rep in MATRIX_REPS
        for dom in DOMAINS
    ]
    _write_csv(
        excel / "v6_completion_summary_rep_domain.csv",
        ["representation", "domain", "completed_cells"],
        rep_dom_rows,
    )

    by_model_dom: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        by_model_dom[(r["model"], r["domain"])] += 1
    model_dom_rows = [
        {"model": m, "domain": dom, "completed_cells": by_model_dom.get((m, dom), 0)}
        for m in MATRIX_MODELS
        for dom in DOMAINS
    ]
    _write_csv(
        excel / "v6_completion_summary_model_domain.csv",
        ["model", "domain", "completed_cells"],
        model_dom_rows,
    )

    prompt_rows = [
        {"representation": "facts", "system_prompt": "SYSTEM_PROMPT_V6_FACTS", "user_template": "USER_TEMPLATE_FACTS"},
        {"representation": "manual_fe", "system_prompt": "SYSTEM_PROMPT_V6_SCORE_RAW", "user_template": "USER_TEMPLATE_SCORE_RAW"},
        {"representation": "fe_manual_format", "system_prompt": "SYSTEM_PROMPT_V6_SCORE_RAW", "user_template": "USER_TEMPLATE_SCORE_RAW"},
        {"representation": "fe_legacy_from_structured", "system_prompt": "SYSTEM_PROMPT_V6_SCORE_RAW", "user_template": "USER_TEMPLATE_SCORE_RAW"},
        {"representation": "fe_gpt_schema", "system_prompt": "SYSTEM_PROMPT_V6_SCORE_RAW", "user_template": "USER_TEMPLATE_SCORE_RAW"},
        {"representation": "gpt_law", "system_prompt": "SYSTEM_PROMPT_V6_SCORE_RAW", "user_template": "USER_TEMPLATE_SCORE_RAW"},
        {"representation": "gpt_free", "system_prompt": "SYSTEM_PROMPT_V6_SCORE_RAW", "user_template": "USER_TEMPLATE_SCORE_RAW"},
        {"representation": "hybrid_manual_gpt", "system_prompt": "SYSTEM_PROMPT_V6_SCORE_RAW", "user_template": "USER_TEMPLATE_SCORE_RAW"},
        {"representation": "hybrid_gpt", "system_prompt": "SYSTEM_PROMPT_V6_SCORE_RAW", "user_template": "USER_TEMPLATE_SCORE_RAW"},
        {"representation": "hybrid_full_gpt", "system_prompt": "SYSTEM_PROMPT_V6_SCORE_RAW", "user_template": "USER_TEMPLATE_SCORE_RAW"},
    ]
    _write_csv(
        excel / "v6_representation_prompt_mapping.csv",
        ["representation", "system_prompt", "user_template"],
        prompt_rows,
    )

    # Markdown report
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    snap = []
    for domain in DOMAINS:
        for task in TASKS:
            n = sum(1 for r in rows if r["domain"] == domain and r["task"] == task)
            snap.append(f"- `{domain}` `{task}` completed runs: **{n}**")

    md_parts = [
        "# v6_full_matrix — Current Results + Active Prompts\n\n",
        f"- Updated at: `{ts}`\n",
        f"- Source folder: `experiments/v6_full_matrix`\n",
        "- Matrix: **10** representations × **11** models × **2** domains × **2** binary tasks (440 cells if full).\n",
        "- Includes only files with `complete=true` in `*_stats.json`\n\n",
        "## Current Completion Snapshot\n",
        "\n".join(snap) + "\n\n",
    ]
    for domain in DOMAINS:
        for task in TASKS:
            sub = [r for r in rows if r["domain"] == domain and r["task"] == task]
            title = f"## Results — {domain} / {task}\n\n"
            if not sub:
                md_parts.append(title + "_No complete runs yet._\n\n")
            else:
                md_parts.append(title + _md_table(sub) + "\n")

    md_parts.append(_build_prompt_md())
    (root / "CURRENT_RESULTS_AND_PROMPTS.md").write_text("".join(md_parts), encoding="utf-8")

    print(f"Wrote {len(rows)} rows to excel_tables/ and CURRENT_RESULTS_AND_PROMPTS.md")


if __name__ == "__main__":
    main()
