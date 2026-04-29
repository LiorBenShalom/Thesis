#!/usr/bin/env python3
"""
Multi-model multi-rep score-only experiment runner.

Subcommands:
  submit-openai     — build + submit OpenAI batches (one per model)
  submit-anthropic  — build + submit Anthropic batch
  submit-google     — build + submit Google batches (one per model)
  status            — show state of all submitted batches
  process-openai    — download + write per-cell CSVs
  process-anthropic — download + write per-cell CSVs
  process-google    — download + write per-cell CSVs
  run-sync          — run sync (OpenRouter) cells
  run-all-batch     — submit all batch jobs (OpenAI + Anthropic + Google)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import pandas as pd

from .config import BATCH_DIR, MODELS, REPS, RESULTS_DIR
from .runner import (
    build_openai_jsonl, submit_openai_batch, fetch_openai_batch,
    build_anthropic_batch, submit_anthropic_batch, fetch_anthropic_batch,
    build_google_jsonl, submit_google_batch, fetch_google_batch,
    run_sync_cell, _result_path, _load_existing, _write_results,
    parse_score, _ensure_env,
)

STATE_FILE = BATCH_DIR / "state.json"


def load_state() -> dict:
    if not STATE_FILE.exists(): return {}
    return json.loads(STATE_FILE.read_text())

def save_state(s: dict):
    STATE_FILE.write_text(json.dumps(s, indent=2))


def _cells_for(provider: str) -> list:
    out = []
    for m, p, _ in MODELS:
        if p != provider: continue
        for r, _, _ in REPS:
            for d in ["drugs", "weapon"]:
                out.append((m, r, d))
    return out


# ---------------- OPENAI ----------------

def submit_openai_per_model():
    """Submit one batch per OpenAI model (each ~1,687 calls = small)."""
    state = load_state()
    state.setdefault("openai", {})
    for m, prov, _ in MODELS:
        if prov != "openai": continue
        if state["openai"].get(m, {}).get("batch_id"):
            print(f"  skip {m}: already submitted ({state['openai'][m]['batch_id']})")
            continue
        cells = [(m, r, d) for r, _, _ in REPS for d in ["drugs", "weapon"]]
        jsonl = BATCH_DIR / f"openai_{m}.jsonl"
        cid_map = build_openai_jsonl(cells, jsonl)
        if not cid_map:
            print(f"  {m}: nothing to submit")
            continue
        print(f"\n=== submit OpenAI {m}: {len(cid_map)} prompts ===")
        bid = submit_openai_batch(jsonl, f"score_only_{m}")
        state["openai"][m] = {"batch_id": bid, "jsonl": str(jsonl), "n_prompts": len(cid_map)}
        save_state(state)


def process_openai():
    state = load_state()
    for m, info in state.get("openai", {}).items():
        bid = info.get("batch_id")
        if not bid: continue
        try:
            status, results = fetch_openai_batch(bid)
        except Exception as e:
            print(f"  {m}: fetch error: {type(e).__name__}: {str(e)[:100]} (skipping)")
            continue
        if status != "completed":
            print(f"  {m}: status={status} (skipping)")
            continue
        print(f"  {m}: {len(results)} responses ✓")
        # Distribute results by custom_id back to cell CSVs
        by_cell: dict = {}
        for r in results:
            cid = r["custom_id"]
            parts = cid.split("__")
            if len(parts) < 4:
                continue
            model_id, rep_id, domain, pair_id = parts[0], parts[1], parts[2], int(parts[3])
            content = ""
            if not r.get("error"):
                try:
                    content = r["response"]["body"]["choices"][0]["message"]["content"] or ""
                except Exception as e:
                    content = f"<<parse_err: {e}>>"
            by_cell.setdefault((model_id, rep_id, domain), []).append({
                "pair_id": pair_id, "content": content,
            })
        # Merge into CSVs
        write_per_cell(by_cell)


# ---------------- ANTHROPIC ----------------

def submit_anthropic():
    state = load_state()
    if state.get("anthropic", {}).get("batch_id"):
        print(f"  skip anthropic: already submitted ({state['anthropic']['batch_id']})")
        return
    cells = _cells_for("anthropic")
    requests, cid_map = build_anthropic_batch(cells)
    if not requests:
        print("  anthropic: nothing to submit")
        return
    print(f"\n=== submit Anthropic: {len(requests)} prompts ===")
    bid = submit_anthropic_batch(requests)
    state["anthropic"] = {"batch_id": bid, "n_prompts": len(requests)}
    save_state(state)


def process_anthropic():
    state = load_state()
    info = state.get("anthropic", {})
    bid = info.get("batch_id")
    if not bid: return
    status, results = fetch_anthropic_batch(bid)
    if status != "ended":
        print(f"  anthropic: status={status} (skipping)")
        return
    print(f"  anthropic: {len(results)} responses ✓")
    by_cell: dict = {}
    for r in results:
        cid = r["custom_id"]
        parts = cid.split("__")
        if len(parts) < 4: continue
        model_id, rep_id, domain, pair_id = parts[0], parts[1], parts[2], int(parts[3])
        content = ""
        if r["result_type"] == "succeeded" and r["message"] is not None:
            try:
                content = next(b.text for b in r["message"].content if b.type == "text")
            except Exception as e:
                content = f"<<parse_err: {e}>>"
        else:
            content = f"<<error: {r.get('error')}>>"
        by_cell.setdefault((model_id, rep_id, domain), []).append({
            "pair_id": pair_id, "content": content,
        })
    write_per_cell(by_cell)


# ---------------- GOOGLE ----------------

def submit_google_per_model():
    state = load_state()
    state.setdefault("google", {})
    for m, prov, api_id in MODELS:
        if prov != "google": continue
        if state["google"].get(m, {}).get("job_name"):
            print(f"  skip {m}: already submitted")
            continue
        cells = [(m, r, d) for r, _, _ in REPS for d in ["drugs", "weapon"]]
        jsonl = BATCH_DIR / f"google_{m}.jsonl"
        cid_map = build_google_jsonl(cells, jsonl)
        if not cid_map:
            print(f"  {m}: nothing to submit")
            continue
        print(f"\n=== submit Google {m}: {len(cid_map)} prompts ===")
        try:
            jn = submit_google_batch(jsonl, api_id)
            state["google"][m] = {"job_name": jn, "jsonl": str(jsonl), "n_prompts": len(cid_map)}
            save_state(state)
        except Exception as e:
            print(f"  {m}: SUBMIT FAILED: {e}")


def process_google():
    state = load_state()
    for m, info in state.get("google", {}).items():
        jn = info.get("job_name")
        if not jn: continue
        try:
            status, results = fetch_google_batch(jn)
        except Exception as e:
            print(f"  {m}: fetch error: {e}")
            continue
        if status != "succeeded":
            print(f"  {m}: state={status}")
            continue
        print(f"  {m}: {len(results)} responses ✓")
        by_cell: dict = {}
        for r in results:
            cid = r.get("key") or r.get("custom_id")
            if not cid: continue
            parts = cid.split("__")
            if len(parts) < 4: continue
            model_id, rep_id, domain, pair_id = parts[0], parts[1], parts[2], int(parts[3])
            content = ""
            try:
                resp = r.get("response", {})
                cand = resp.get("candidates", [])
                if cand:
                    content = cand[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            except Exception as e:
                content = f"<<parse_err: {e}>>"
            by_cell.setdefault((model_id, rep_id, domain), []).append({
                "pair_id": pair_id, "content": content,
            })
        write_per_cell(by_cell)


# ---------------- WRITE PER-CELL CSVs ----------------

def write_per_cell(by_cell: dict):
    """Given {(model_id, rep_id, domain): [{pair_id, content}]}, merge into per-cell CSVs."""
    from .data import iter_rep_pairs
    from .runner import _rep_kind
    for (model_id, rep_id, domain), entries in by_cell.items():
        path = _result_path(model_id, rep_id, domain)
        existing = _load_existing(path)
        # Build pair_id → pair info
        pair_info = {p["pair_id"]: p for p in iter_rep_pairs(rep_id, _rep_kind(rep_id), domain)}
        rows = list(existing.values())
        keys = {(r["verdict_1"], r["verdict_2"]) for r in rows}
        for e in entries:
            pid = e["pair_id"]
            p = pair_info.get(pid)
            if not p: continue
            k = (p["verdict_1"], p["verdict_2"])
            row = {
                "verdict_1": p["verdict_1"], "verdict_2": p["verdict_2"],
                "GT": p["gt"], "model_score": parse_score(e["content"]),
                "raw": (e["content"] or "")[:1000],
            }
            if k in keys:
                rows = [r for r in rows if (r["verdict_1"], r["verdict_2"]) != k]
            rows.append(row)
            keys.add(k)
        _write_results(path, rows)
        n_ok = sum(1 for r in rows if not pd.isna(r.get("model_score")))
        print(f"    → {path.name}  parsed {n_ok}/{len(rows)}")


# ---------------- SYNC ----------------

def run_sync_all():
    """Run all OpenRouter cells synchronously (with threading)."""
    for m, prov, _ in MODELS:
        if prov != "openrouter": continue
        for r, _, _ in REPS:
            for d in ["drugs", "weapon"]:
                run_sync_cell(m, r, d)


# ---------------- STATUS ----------------

def status():
    state = load_state()
    _ensure_env()
    # OpenAI
    if "openai" in state:
        print("\n## OpenAI batches")
        from openai import OpenAI
        cli = OpenAI()
        for m, info in state["openai"].items():
            bid = info["batch_id"]
            try:
                b = cli.batches.retrieve(bid)
                c = b.request_counts
                print(f"  {m:<22s} {b.status:<14s} {c.completed}/{c.total}")
            except Exception as e:
                print(f"  {m}: ERROR {e}")
    # Anthropic
    if "anthropic" in state:
        print("\n## Anthropic batch")
        import anthropic
        cli = anthropic.Anthropic()
        bid = state["anthropic"]["batch_id"]
        try:
            b = cli.messages.batches.retrieve(bid)
            print(f"  {bid:<35s} {b.processing_status:<14s} {b.request_counts}")
        except Exception as e:
            print(f"  anthropic: ERROR {e}")
    # Google
    if "google" in state:
        print("\n## Google batches")
        from google import genai
        cli = genai.Client()
        for m, info in state["google"].items():
            jn = info["job_name"]
            try:
                j = cli.batches.get(name=jn)
                print(f"  {m:<22s} {str(j.state):<30s}")
            except Exception as e:
                print(f"  {m}: ERROR {e}")


# ---------------- CLI ----------------

CMDS = {
    "submit-openai":     submit_openai_per_model,
    "submit-anthropic":  submit_anthropic,
    "submit-google":     submit_google_per_model,
    "process-openai":    process_openai,
    "process-anthropic": process_anthropic,
    "process-google":    process_google,
    "run-sync":          run_sync_all,
    "status":            status,
    "submit-all":        lambda: (submit_openai_per_model(), submit_anthropic(), submit_google_per_model()),
    "process-all":       lambda: (process_openai(), process_anthropic(), process_google()),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=list(CMDS.keys()))
    args = ap.parse_args()
    _ensure_env()
    CMDS[args.cmd]()


if __name__ == "__main__":
    main()
