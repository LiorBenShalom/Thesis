#!/usr/bin/env python3
"""
Cancel stuck batch chunks and run their prompts synchronously (with concurrency).

For each in-progress batch in state_{name}.json with completed=0:
1. Cancel the OpenAI batch
2. Read the corresponding *.jsonl chunk file
3. Run all prompts via concurrent ThreadPoolExecutor (default 20 workers)
4. Append responses to sync_responses_{name}.json (custom_id → content)

Then process_schema_batch (in batch_hfull.py) merges this with batch results.

Usage:
  drain_stuck_batches.py --name schema_drugs [--workers 20]
  drain_stuck_batches.py --name schema_weapon
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!")
DATA_DIR = ROOT / "new_try" / "experiments" / "data" / "sentencing_range"
BATCH_DIR = DATA_DIR / "hfull_features" / "batch"


def _oai():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def find_stuck_chunks(name: str) -> list[tuple[int, str]]:
    """Returns list of (chunk_index, batch_id) for stuck batches (in_progress, completed=0)."""
    oai = _oai()
    state_path = BATCH_DIR / f"state_{name}.json"
    with open(state_path) as f:
        state = json.load(f)
    stuck = []
    for i, bid in enumerate(state["batch_ids"], 1):
        b = oai.batches.retrieve(bid)
        if b.status == "in_progress" and b.request_counts.completed == 0:
            stuck.append((i, bid))
    return stuck


def find_chunk_jsonl(name: str, chunk_idx: int) -> Path | None:
    """Find the JSONL file corresponding to a chunk index in the state file."""
    domain = name.replace("schema_", "")
    # Most files are *_input.partNN.jsonl or *_input.partNN.partNN.jsonl
    # Map chunk_idx → file by parsing state's order. Easier: just look at file mtimes
    # vs state file order. Actually safest: the state file was built by _upload_and_submit
    # which iterates in order over chunks returned by _split_jsonl. The first chunks were
    # the original .partNN, then resubmits added .partNN.partNN.
    state_path = BATCH_DIR / f"state_{name}.json"
    with open(state_path) as f:
        state = json.load(f)
    # Find batch's input file via the OpenAI API
    bid = state["batch_ids"][chunk_idx - 1]
    oai = _oai()
    b = oai.batches.retrieve(bid)
    file_id = b.input_file_id
    if not file_id:
        return None
    fi = oai.files.retrieve(file_id)
    fname = fi.filename
    candidate = BATCH_DIR / fname
    return candidate if candidate.exists() else None


def call_one(prompt_line: dict) -> tuple[str, str | None, str | None]:
    """Synchronously call OpenAI with one batch JSONL request line.
    Returns (custom_id, content, error)."""
    oai = _oai()
    cid = prompt_line["custom_id"]
    body = prompt_line["body"]
    try:
        resp = oai.chat.completions.create(**body)
        content = resp.choices[0].message.content or ""
        return cid, content, None
    except Exception as e:
        return cid, None, f"{type(e).__name__}: {e}"


def drain_chunk(name: str, chunk_idx: int, batch_id: str, workers: int) -> int:
    """Cancel the batch, read its JSONL, run sync, append to sync_responses."""
    oai = _oai()
    print(f"\n  [{name} chunk {chunk_idx}] cancelling {batch_id}...")
    try:
        oai.batches.cancel(batch_id)
    except Exception as e:
        print(f"    cancel error: {e}")

    jsonl_path = find_chunk_jsonl(name, chunk_idx)
    if not jsonl_path or not jsonl_path.exists():
        print(f"    ⚠️  no JSONL file found for chunk {chunk_idx}")
        return 0
    with open(jsonl_path) as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    print(f"    {len(prompts)} prompts to run sync ({jsonl_path.name})")

    out_path = BATCH_DIR / f"sync_responses_{name}.json"
    sync_resp: dict[str, str] = {}
    if out_path.exists():
        with open(out_path) as f:
            sync_resp = json.load(f)

    pbar = tqdm(total=len(prompts), desc=f"sync chunk {chunk_idx}")
    new_n = 0
    err_n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(call_one, p) for p in prompts if p["custom_id"] not in sync_resp]
        for fut in as_completed(futures):
            cid, content, err = fut.result()
            if err:
                err_n += 1
            else:
                sync_resp[cid] = content
                new_n += 1
            pbar.update(1)
            # Periodic save
            if new_n % 100 == 0 and new_n > 0:
                with open(out_path, "w") as f:
                    json.dump(sync_resp, f, ensure_ascii=False)
    pbar.close()

    with open(out_path, "w") as f:
        json.dump(sync_resp, f, ensure_ascii=False)
    print(f"    ✅ saved {new_n} new responses, {err_n} errors → {out_path.name}")
    return new_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="state name, e.g. schema_drugs")
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    stuck = find_stuck_chunks(args.name)
    if not stuck:
        print(f"No stuck chunks for {args.name}")
        return
    print(f"Found {len(stuck)} stuck chunks for {args.name}: {[i for i,_ in stuck]}")
    total = 0
    for chunk_idx, bid in stuck:
        total += drain_chunk(args.name, chunk_idx, bid, args.workers)
    print(f"\n✅ Drained {total} responses for {args.name}")


if __name__ == "__main__":
    main()
