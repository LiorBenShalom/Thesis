"""
Unified runner: builds prompts for one (model, rep, domain) cell and dispatches
to the right provider — batch (OpenAI/Anthropic/Google) or sync (OpenRouter/HF).

Public entry point: process_cell(model_id, rep_id, domain) — runs end-to-end and
returns path to the per-cell results CSV.

For batch providers, the workflow is:
  1. Build all 100/141 prompts
  2. Submit as a single batch (or part of a larger batch)
  3. Poll until complete
  4. Download outputs and write CSV

For sync providers, simple ThreadPoolExecutor.

State: each (model, rep, domain) gets a dedicated result CSV under results/.
       Resume is automatic: if CSV has all rows, skip; partial → fill missing.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from .config import (
    BATCH_DIR, DOMAINS, MODELS, REPS, RESULTS_DIR,
)
from .data import iter_rep_pairs
from .prompts import system_for, user_for

# ============== HELPERS ==============

def _model_cfg(model_id: str):
    for m, prov, api_id in MODELS:
        if m == model_id:
            return prov, api_id
    raise ValueError(model_id)

def _rep_kind(rep_id: str) -> str:
    return next(k for r, _, k in REPS if r == rep_id)

def _result_path(model_id: str, rep_id: str, domain: str) -> Path:
    return RESULTS_DIR / f"{model_id}__{rep_id}__{domain}.csv"

def parse_score(text: str) -> Optional[int]:
    if not text: return None
    m = re.search(r"SIMILARITY_SCORE:\s*(\d+)", text)
    return int(m.group(1)) if m else None


def _load_existing(path: Path) -> dict:
    """Return {(verdict_1, verdict_2): row_dict}"""
    if not path.exists(): return {}
    df = pd.read_csv(path)
    return {(r["verdict_1"], r["verdict_2"]): dict(r) for _, r in df.iterrows()}


def _write_results(path: Path, rows: list):
    pd.DataFrame(rows).to_csv(path, index=False)


# ============== ENV ==============

def _ensure_env():
    """Load .env files lazily."""
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENROUTER_API_KEY") or not os.getenv("GOOGLE_API_KEY"):
        env_p = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/experiments/.env")
        if env_p.exists():
            for line in env_p.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if not os.getenv("ANTHROPIC_API_KEY"):
        root_env = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/.env")
        if root_env.exists():
            for line in root_env.read_text().splitlines():
                if "=" in line and "ntropic" in line.lower():
                    os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")


# ============== SYNC RUNNER (OpenRouter / HF / fallback) ==============

_OR_CLIENT = _GENAI_CLIENT = None

def _openrouter():
    global _OR_CLIENT
    if _OR_CLIENT is None:
        _ensure_env()
        from openai import OpenAI
        _OR_CLIENT = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"),
                            base_url="https://openrouter.ai/api/v1")
    return _OR_CLIENT


_GENAI_CLIENT = None
def _genai():
    global _GENAI_CLIENT
    if _GENAI_CLIENT is None:
        _ensure_env()
        from google import genai
        _GENAI_CLIENT = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _GENAI_CLIENT


def call_sync(model_id: str, system: str, user: str, max_tokens: int = 200) -> str:
    """Synchronous call. Used for OpenRouter / HF / Google."""
    prov, api_id = _model_cfg(model_id)
    if prov == "openrouter":
        resp = _openrouter().chat.completions.create(
            model=api_id, max_tokens=max_tokens, temperature=0.1,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""
    if prov == "google":
        from google.genai import types
        # Gemini 2.5 Pro is a thinking model — needs lots of tokens.
        # Use 4000 for output to leave room for thoughts (~2k) + actual response.
        cli = _genai()
        try:
            resp = cli.models.generate_content(
                model=api_id,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.1,
                    max_output_tokens=4000,
                ),
            )
            return resp.text or ""
        except Exception as e:
            return f"<<error: {type(e).__name__}: {str(e)[:200]}>>"
    raise NotImplementedError(f"sync not implemented for provider={prov}")


def run_sync_cell(model_id: str, rep_id: str, domain: str, workers: int = 8) -> Path:
    out_path = _result_path(model_id, rep_id, domain)
    pairs = list(iter_rep_pairs(rep_id, _rep_kind(rep_id), domain))
    sys_p = system_for(domain)
    existing = _load_existing(out_path)

    todo = [p for p in pairs if (p["verdict_1"], p["verdict_2"]) not in existing
            or pd.isna(existing[(p["verdict_1"], p["verdict_2"])].get("model_score"))]
    if not todo:
        print(f"  ✓ {out_path.name}: all {len(pairs)} done")
        return out_path
    print(f"  ▶ {out_path.name}: {len(todo)}/{len(pairs)} to call")

    rows = list(existing.values())
    keys = {(r["verdict_1"], r["verdict_2"]) for r in rows}

    def task(p):
        try:
            resp = call_sync(model_id, sys_p, user_for(p["fv1"], p["fv2"]))
            return p, resp, None
        except Exception as e:
            return p, "", f"{type(e).__name__}: {e}"

    pbar = tqdm(total=len(todo), desc=f"{model_id}/{rep_id}/{domain}")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(task, p) for p in todo]
        for fut in as_completed(futs):
            p, resp, err = fut.result()
            row = {
                "verdict_1": p["verdict_1"], "verdict_2": p["verdict_2"],
                "GT": p["gt"], "model_score": parse_score(resp), "raw": resp[:1000] if resp else (err or ""),
            }
            k = (p["verdict_1"], p["verdict_2"])
            if k in keys:
                # replace
                rows = [r for r in rows if (r["verdict_1"], r["verdict_2"]) != k]
            rows.append(row)
            keys.add(k)
            pbar.update(1)
            if pbar.n % 25 == 0:
                _write_results(out_path, rows)
    pbar.close()
    _write_results(out_path, rows)
    return out_path


# ============== OPENAI BATCH ==============

def build_openai_jsonl(cells: list, jsonl_path: Path) -> dict:
    """cells: list of (model_id, rep_id, domain). Returns metadata mapping custom_id → context."""
    _ensure_env()
    custom_id_map = {}
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for model_id, rep_id, domain in cells:
            prov, api_id = _model_cfg(model_id)
            if prov != "openai": continue
            sys_p = system_for(domain)
            existing = _load_existing(_result_path(model_id, rep_id, domain))
            for p in iter_rep_pairs(rep_id, _rep_kind(rep_id), domain):
                k = (p["verdict_1"], p["verdict_2"])
                if k in existing and not pd.isna(existing[k].get("model_score")):
                    continue
                custom_id = f"{model_id}__{rep_id}__{domain}__{p['pair_id']}"
                custom_id_map[custom_id] = {
                    "model_id": model_id, "rep_id": rep_id, "domain": domain,
                    "verdict_1": p["verdict_1"], "verdict_2": p["verdict_2"], "gt": p["gt"],
                }
                # GPT-5 family uses max_completion_tokens instead of max_tokens
                token_key = "max_completion_tokens" if api_id.startswith("gpt-5") else "max_tokens"
                body = {
                    "model": api_id,
                    "messages": [
                        {"role": "system", "content": sys_p},
                        {"role": "user", "content": user_for(p["fv1"], p["fv2"])},
                    ],
                    token_key: 200,
                }
                # GPT-5 family doesn't support custom temperature — only default (1)
                # GPT-4.x supports temperature
                if not api_id.startswith("gpt-5"):
                    body["temperature"] = 0.1
                f.write(json.dumps({"custom_id": custom_id, "method": "POST",
                                    "url": "/v1/chat/completions", "body": body},
                                   ensure_ascii=False) + "\n")
    return custom_id_map


def submit_openai_batch(jsonl_path: Path, description: str) -> str:
    _ensure_env()
    from openai import OpenAI
    cli = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print(f"⬆️  Uploading {jsonl_path.name} ({jsonl_path.stat().st_size/1024/1024:.1f} MB)...")
    with open(jsonl_path, "rb") as f:
        upl = cli.files.create(file=f, purpose="batch")
    b = cli.batches.create(input_file_id=upl.id, endpoint="/v1/chat/completions",
                           completion_window="24h", metadata={"description": description})
    print(f"   batch_id = {b.id}  status = {b.status}")
    return b.id


def fetch_openai_batch(batch_id: str) -> tuple[str, list]:
    _ensure_env()
    from openai import OpenAI
    cli = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    b = cli.batches.retrieve(batch_id)
    if b.status != "completed":
        return b.status, []
    content = cli.files.content(b.output_file_id).content
    lines = content.decode("utf-8").strip().split("\n")
    return "completed", [json.loads(l) for l in lines if l.strip()]


# ============== ANTHROPIC BATCH ==============

def build_anthropic_batch(cells: list) -> tuple[list, dict]:
    """Returns (requests, custom_id_map) for anthropic.messages.batches.create."""
    _ensure_env()
    custom_id_map = {}
    requests = []
    for model_id, rep_id, domain in cells:
        prov, api_id = _model_cfg(model_id)
        if prov != "anthropic": continue
        sys_p = system_for(domain)
        existing = _load_existing(_result_path(model_id, rep_id, domain))
        for p in iter_rep_pairs(rep_id, _rep_kind(rep_id), domain):
            k = (p["verdict_1"], p["verdict_2"])
            if k in existing and not pd.isna(existing[k].get("model_score")):
                continue
            cid = f"{model_id}__{rep_id}__{domain}__{p['pair_id']}"
            custom_id_map[cid] = {
                "model_id": model_id, "rep_id": rep_id, "domain": domain,
                "verdict_1": p["verdict_1"], "verdict_2": p["verdict_2"], "gt": p["gt"],
            }
            requests.append({
                "custom_id": cid,
                "params": {
                    "model": api_id,
                    "max_tokens": 1500,  # claude tends to ignore "score-only"; allow space
                    "temperature": 0.1,
                    "system": sys_p,
                    "messages": [{"role": "user", "content": user_for(p["fv1"], p["fv2"])}],
                },
            })
    return requests, custom_id_map


def submit_anthropic_batch(requests: list) -> str:
    _ensure_env()
    import anthropic
    cli = anthropic.Anthropic()
    b = cli.messages.batches.create(requests=requests)
    print(f"   anthropic batch_id = {b.id}  status = {b.processing_status}")
    return b.id


def fetch_anthropic_batch(batch_id: str) -> tuple[str, list]:
    _ensure_env()
    import anthropic
    cli = anthropic.Anthropic()
    b = cli.messages.batches.retrieve(batch_id)
    if b.processing_status != "ended":
        return b.processing_status, []
    out = []
    for entry in cli.messages.batches.results(batch_id):
        out.append({
            "custom_id": entry.custom_id,
            "result_type": entry.result.type,
            "message": entry.result.message if entry.result.type == "succeeded" else None,
            "error": entry.result.error if entry.result.type == "errored" else None,
        })
    return "ended", out


# ============== GOOGLE BATCH ==============

def build_google_jsonl(cells: list, jsonl_path: Path) -> dict:
    """Google batch via google-genai SDK. Each line is one inlined request."""
    _ensure_env()
    custom_id_map = {}
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for model_id, rep_id, domain in cells:
            prov, api_id = _model_cfg(model_id)
            if prov != "google": continue
            sys_p = system_for(domain)
            existing = _load_existing(_result_path(model_id, rep_id, domain))
            for p in iter_rep_pairs(rep_id, _rep_kind(rep_id), domain):
                k = (p["verdict_1"], p["verdict_2"])
                if k in existing and not pd.isna(existing[k].get("model_score")):
                    continue
                cid = f"{model_id}__{rep_id}__{domain}__{p['pair_id']}"
                custom_id_map[cid] = {
                    "model_id": model_id, "rep_id": rep_id, "domain": domain,
                    "verdict_1": p["verdict_1"], "verdict_2": p["verdict_2"], "gt": p["gt"],
                }
                # Google batch JSONL format: {"key": cid, "request": {"contents": [...], "system_instruction": ...}}
                # Gemini 2.5 Pro (thinking model) consumes ~200-500 tokens for internal "thoughts"
                # before output. Need much higher maxOutputTokens than score-only would suggest.
                req = {
                    "key": cid,
                    "request": {
                        "system_instruction": {"parts": [{"text": sys_p}]},
                        "contents": [{"role": "user", "parts": [{"text": user_for(p["fv1"], p["fv2"])}]}],
                        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2000},
                    },
                }
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
    return custom_id_map


def submit_google_batch(jsonl_path: Path, model_api_id: str) -> str:
    """Submits a Google Gemini batch via the google-genai SDK."""
    _ensure_env()
    from google import genai
    cli = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    print(f"⬆️  Uploading {jsonl_path.name} to Gemini...")
    upl = cli.files.upload(file=str(jsonl_path), config={"display_name": jsonl_path.stem, "mime_type": "application/jsonl"})
    print(f"   file: {upl.name}")
    job = cli.batches.create(model=model_api_id, src=upl.name,
                             config={"display_name": jsonl_path.stem})
    print(f"   batch job: {job.name}  state = {job.state}")
    return job.name


def fetch_google_batch(job_name: str) -> tuple[str, list]:
    _ensure_env()
    from google import genai
    cli = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    job = cli.batches.get(name=job_name)
    state = str(job.state)
    if "JOB_STATE_SUCCEEDED" not in state:
        return state, []
    # Download the output file
    if not job.dest or not job.dest.file_name:
        return "no_output", []
    content = cli.files.download(file=job.dest.file_name)
    lines = content.decode("utf-8").strip().split("\n")
    return "succeeded", [json.loads(l) for l in lines if l.strip()]
