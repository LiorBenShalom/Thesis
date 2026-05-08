#!/usr/bin/env python3
"""
Batch-mode H-Full extraction (OpenAI Batch API — 50% cheaper).

Works in two steps, each sent as a batch to OpenAI:

  Step 1 (schema, per-domain):
    - Uses the existing DrugFeatureExtractor / FeatureExtractor classes
    - Monkey-patches their internal GPT client with a recording client to capture prompts
    - Writes all prompts to JSONL and submits as a batch
    - When batch completes, replays the extractor with recorded responses to produce features
    - Drugs: 5 calls/verdict, Weapon: 11 calls/verdict

  Step 2 (hybrid enrichment):
    - One GPT call per verdict to enrich schema with case-specific fields
    - Simple batch (no recording harness needed)

Usage:
  # Step 1 — drugs
  batch_hfull.py schema --domain drugs --submit [--limit N]
  batch_hfull.py schema --domain drugs --process    # once batch is complete

  # Step 1 — weapon
  batch_hfull.py schema --domain weapon --submit [--limit N]
  batch_hfull.py schema --domain weapon --process

  # Step 2 — enrichment
  batch_hfull.py enrich --submit [--limit N]
  batch_hfull.py enrich --process

  # Status
  batch_hfull.py status
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

# ---------- paths ----------
ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!")
CODE_DIR = ROOT / "new_try" / "code"
DATA_DIR = ROOT / "new_try" / "experiments" / "data" / "sentencing_range"
OUT_DIR = DATA_DIR / "hfull_features"
BATCH_DIR = OUT_DIR / "batch"
VERDICT_CSV_DIR = DATA_DIR / "verdict_csv"
CLEAN_CSV = DATA_DIR / "verdicts_clean.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA_MODEL = "gpt-4.1-mini"
ENRICH_MODEL = "gpt-4.1"

# OpenAI Batch API: 200MB file limit. Use 100MB chunks to stay safe.
MAX_CHUNK_BYTES = 100 * 1024 * 1024  # 100 MB

# ---------- module loader ----------
_MODULES: Dict[str, Any] = {}

def _load_module(name: str, path: Path):
    if name in _MODULES:
        return _MODULES[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _MODULES[name] = mod
    return mod

def get_domain_module(domain: str):
    if domain == "drugs":
        return _load_module("features_extract_drugs_2", CODE_DIR / "features_extract_drugs 2.py")
    if domain == "weapon":
        return _load_module("features_extract_2", CODE_DIR / "features_extract 2.py")
    raise ValueError(domain)

def get_extractor_class(domain: str):
    m = get_domain_module(domain)
    return m.DrugFeatureExtractor if domain == "drugs" else m.FeatureExtractor


# ---------- recording/replay clients ----------
class _MockResponse:
    def __init__(self, content: str):
        self._content = content
    @property
    def choices(self):
        outer = self
        class _Choice:
            message = type("M", (), {"content": outer._content})()
        return [_Choice()]


class RecordingClient:
    """Drop-in replacement for OpenAI client that RECORDS calls instead of making them."""
    def __init__(self):
        self.requests: List[Dict[str, Any]] = []
        self.chat = self
        self.completions = self
    def create(self, **kwargs):
        self.requests.append(kwargs)
        return _MockResponse("")


class ReplayClient:
    """Drop-in replacement that returns pre-recorded responses in order."""
    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.idx = 0
        self.chat = self
        self.completions = self
    def create(self, **kwargs):
        if self.idx >= len(self.responses):
            raise RuntimeError(f"Replay exhausted after {self.idx} responses")
        r = self.responses[self.idx]
        self.idx += 1
        return _MockResponse(r)


# ---------- batch state ----------
def _state_path(step: str, domain: Optional[str] = None) -> Path:
    suffix = f"_{domain}" if domain else ""
    return BATCH_DIR / f"state_{step}{suffix}.json"


def _save_state(step: str, data: dict, domain: Optional[str] = None):
    p = _state_path(step, domain)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)

def _load_state(step: str, domain: Optional[str] = None) -> Optional[dict]:
    p = _state_path(step, domain)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# ---------- OpenAI helpers ----------
def _oai():
    from openai import OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)


def _split_jsonl(src: Path, max_bytes: int = MAX_CHUNK_BYTES) -> List[Path]:
    """Split a big JSONL into chunks each ≤ max_bytes. Returns chunk paths."""
    total_size = src.stat().st_size
    if total_size <= max_bytes:
        return [src]
    chunks: List[Path] = []
    idx = 1
    out_path = src.with_name(f"{src.stem}.part{idx:02d}.jsonl")
    out_f = open(out_path, "w", encoding="utf-8")
    bytes_written = 0
    chunks.append(out_path)
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            b = len(line.encode("utf-8"))
            if bytes_written + b > max_bytes and bytes_written > 0:
                out_f.close()
                idx += 1
                out_path = src.with_name(f"{src.stem}.part{idx:02d}.jsonl")
                out_f = open(out_path, "w", encoding="utf-8")
                bytes_written = 0
                chunks.append(out_path)
            out_f.write(line)
            bytes_written += b
    out_f.close()
    print(f"   split {src.name} ({total_size/1024/1024:.1f} MB) into {len(chunks)} chunks")
    return chunks


def _upload_and_submit(jsonl_path: Path, description: str) -> List[str]:
    """Upload + submit. Auto-splits files over MAX_CHUNK_BYTES. Returns list of batch_ids."""
    oai = _oai()
    chunks = _split_jsonl(jsonl_path)
    batch_ids: List[str] = []
    for i, chunk in enumerate(chunks, 1):
        size_mb = chunk.stat().st_size / 1024 / 1024
        print(f"⬆️  [{i}/{len(chunks)}] Uploading {chunk.name} ({size_mb:.1f} MB)...")
        with open(chunk, "rb") as f:
            uploaded = oai.files.create(file=f, purpose="batch")
        print(f"   file_id = {uploaded.id}")
        batch = oai.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"{description}_chunk{i}"},
        )
        print(f"   batch_id = {batch.id}  status = {batch.status}")
        batch_ids.append(batch.id)
    return batch_ids


def _poll_batch(batch_id: str) -> dict:
    oai = _oai()
    b = oai.batches.retrieve(batch_id)
    return {
        "id": b.id,
        "status": b.status,
        "counts": dict(b.request_counts) if b.request_counts else {},
        "output_file_id": b.output_file_id,
        "error_file_id": b.error_file_id,
        "created_at": b.created_at,
        "completed_at": getattr(b, "completed_at", None),
    }


def _download_output(output_file_id: str) -> List[dict]:
    oai = _oai()
    content = oai.files.content(output_file_id).content
    lines = content.decode("utf-8").strip().split("\n")
    return [json.loads(line) for line in lines if line.strip()]


def _read_verdict_text(norm_id: str) -> Optional[str]:
    fpath = VERDICT_CSV_DIR / f"{norm_id}.csv"
    if not fpath.exists():
        return None
    try:
        df = pd.read_csv(fpath)
        return "\n".join(df["text"].dropna().astype(str).tolist())
    except Exception:
        return None


# ==================== STEP 1: SCHEMA ====================
def prepare_schema_batch(domain: str, limit: Optional[int] = None) -> Path:
    """Build JSONL of all prompts that the extractor would make for each verdict."""
    df = pd.read_csv(CLEAN_CSV)
    df = df[df["domain"] == domain].copy()
    if limit:
        df = df.head(limit)

    ExtractorCls = get_extractor_class(domain)
    module = get_domain_module(domain)

    jsonl_path = BATCH_DIR / f"schema_{domain}_input.jsonl"
    per_verdict_counts: Dict[str, int] = {}

    print(f"📝 Preparing schema batch for {domain} — {len(df):,} verdicts")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in tqdm(list(df.iterrows()), desc=f"Recording {domain}"):
            vid = str(row["verdict"])
            norm_id = str(row["normalized_id"])
            text = _read_verdict_text(norm_id)
            if not text:
                per_verdict_counts[vid] = 0
                continue
            # Swap client to recorder
            rec = RecordingClient()
            original = module.client
            module.client = rec
            try:
                ex = ExtractorCls(model=SCHEMA_MODEL, debug=False)
                ex.extract_all_features(text)
            except Exception as e:
                tqdm.write(f"  ⚠️  {vid}: {type(e).__name__}: {e}")
            finally:
                module.client = original

            per_verdict_counts[vid] = len(rec.requests)
            for call_idx, req in enumerate(rec.requests):
                # Build batch request
                custom_id = f"{vid}__{call_idx}"
                body = {
                    "model": req.get("model", SCHEMA_MODEL),
                    "messages": req["messages"],
                    "temperature": req.get("temperature", 0.1),
                    "max_tokens": req.get("max_tokens", 500),
                }
                line = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

    total_requests = sum(per_verdict_counts.values())
    print(f"✅ Wrote {total_requests:,} requests to {jsonl_path}")
    print(f"   verdicts with 0 calls (no text): "
          f"{sum(1 for v in per_verdict_counts.values() if v == 0)}")

    # Save count map so process step knows how many responses to replay per verdict
    counts_path = BATCH_DIR / f"schema_{domain}_counts.json"
    with open(counts_path, "w") as cf:
        json.dump(per_verdict_counts, cf)
    return jsonl_path


def process_schema_batch(domain: str) -> Path:
    """After batch completes: download responses, replay through extractor to build features."""
    state = _load_state("schema", domain)
    if not state:
        sys.exit(f"No submitted batch found for schema/{domain}. Run --submit first.")
    batch_ids = state.get("batch_ids") or ([state["batch_id"]] if state.get("batch_id") else [])
    if not batch_ids:
        sys.exit(f"No batch_ids in state for schema/{domain}")

    # Check ALL batches are complete, collect output file IDs
    # Skip cancelled/non-completed batches (they may have been drained synchronously)
    results: list = []
    skipped: list = []
    for bid in batch_ids:
        status = _poll_batch(bid)
        if status["status"] == "completed":
            print(f"  ✅ {bid}: {status['counts']}")
            results.extend(_download_output(status["output_file_id"]))
        elif status["status"] in ("cancelled", "cancelling", "expired", "failed"):
            print(f"  ⏭️  {bid}: {status['status']} — will rely on sync responses if any")
            skipped.append(bid)
        else:
            sys.exit(f"Batch {bid} status={status['status']} (not done). Re-run later.")
    print(f"   received {len(results):,} batch responses, skipped {len(skipped)} cancelled/expired")

    # Build {custom_id -> response_text} map
    response_map: Dict[str, str] = {}
    errors: Dict[str, str] = {}
    for r in results:
        cid = r["custom_id"]
        if r.get("error"):
            errors[cid] = str(r["error"])
            continue
        try:
            content = r["response"]["body"]["choices"][0]["message"]["content"]
            response_map[cid] = content or ""
        except Exception as e:
            errors[cid] = f"parse_error: {e}"
    print(f"   parsed {len(response_map):,} ok, {len(errors)} errors")

    # Merge sync responses (from drain_stuck_batches.py)
    sync_path = BATCH_DIR / f"sync_responses_schema_{domain}.json"
    if sync_path.exists():
        with open(sync_path) as f:
            sync_resp = json.load(f)
        before = len(response_map)
        response_map.update(sync_resp)
        print(f"   merged {len(sync_resp):,} sync responses (added {len(response_map)-before})")

    # Load counts to know how many responses per verdict
    counts_path = BATCH_DIR / f"schema_{domain}_counts.json"
    with open(counts_path) as f:
        counts: Dict[str, int] = json.load(f)

    # Load clean CSV for normalized_id lookup
    df = pd.read_csv(CLEAN_CSV)
    df = df[df["domain"] == domain].copy()
    norm_lookup = dict(zip(df["verdict"].astype(str), df["normalized_id"].astype(str)))

    ExtractorCls = get_extractor_class(domain)
    module = get_domain_module(domain)

    # Replay per verdict
    feature_cache_path = OUT_DIR / f"schema_cache_{domain}.json"
    feature_cache: Dict[str, Any] = {}
    if feature_cache_path.exists():
        with open(feature_cache_path) as f:
            feature_cache = json.load(f)

    for vid, ncalls in tqdm(counts.items(), desc=f"Replay {domain}"):
        if ncalls == 0:
            feature_cache[vid] = {"__error": "no_text"}
            continue
        # Collect the N responses for this verdict in order
        responses = [response_map.get(f"{vid}__{i}", "") for i in range(ncalls)]
        # Check for missing
        missing = sum(1 for r in responses if r == "" and f"{vid}__{i}" in errors for i in range(ncalls))

        # Swap client to replayer
        replayer = ReplayClient(responses)
        original = module.client
        module.client = replayer
        try:
            norm_id = norm_lookup.get(vid)
            text = _read_verdict_text(norm_id) if norm_id else ""
            ex = ExtractorCls(model=SCHEMA_MODEL, debug=False)
            ex.extract_all_features(text)
            feature_cache[vid] = ex.to_dict()
        except Exception as e:
            feature_cache[vid] = {"__error": f"replay_error: {e}"}
        finally:
            module.client = original

    with open(feature_cache_path, "w", encoding="utf-8") as f:
        json.dump(feature_cache, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(feature_cache):,} feature dicts → {feature_cache_path}")
    return feature_cache_path


# ==================== STEP 2: ENRICHMENT ====================
ENRICH_SYS = "You are an expert legal analyst extracting structured facts from Israeli criminal verdicts."

def _enrich_prompt(indictment_facts: str, schema_feats: dict, domain: str) -> str:
    schema_str = json.dumps(schema_feats, ensure_ascii=False, indent=2)
    domain_he = "סמים" if domain == "drugs" else "נשק"
    return f"""להלן עובדות כתב אישום של תיק {domain_he}, יחד עם פיצ'רים שכבר חולצו.
המטרה: להעשיר את הפיצ'רים בשדות נוספים רלוונטיים לתיק הספציפי הזה (שאינם בסכימה הקיימת).

דוגמאות לשדות העשרה אפשריים: שיטת ביצוע, מסלול, שעת העבירה, סכום כספי, אמצעי תקשורת, אופן התפיסה, מניע, נסיבות מיוחדות, וכו'.

הנחיות:
- החזר אך ורק JSON תקין (ללא הסברים).
- כלול את *כל* השדות הקיימים בסכימה + שדות חדשים.
- שדה חדש צריך להיות פרטני, קצר, בעברית, ומבוסס על עובדות הכתב אישום בלבד.

## הסכימה הקיימת:
{schema_str}

## עובדות כתב האישום:
{indictment_facts}

החזר JSON אחד עם כל השדות:"""


def prepare_enrich_batch(limit: Optional[int] = None) -> Path:
    # Load schema caches
    schema_by_verdict: Dict[str, dict] = {}
    for d in ["drugs", "weapon"]:
        cp = OUT_DIR / f"schema_cache_{d}.json"
        if not cp.exists():
            continue
        with open(cp) as f:
            cache = json.load(f)
        for vid, feats in cache.items():
            if "__error" not in feats:
                schema_by_verdict[vid] = feats
    print(f"📥 Schema features for {len(schema_by_verdict):,} verdicts")

    df = pd.read_csv(CLEAN_CSV)
    df = df[df["verdict"].astype(str).isin(schema_by_verdict)].copy()
    if limit:
        df = df.head(limit)

    jsonl_path = BATCH_DIR / "enrich_input.jsonl"
    print(f"📝 Preparing enrich batch — {len(df):,} verdicts")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            vid = str(row["verdict"])
            facts = row.get("indictment_facts") or ""
            schema = schema_by_verdict[vid]
            body = {
                "model": ENRICH_MODEL,
                "messages": [
                    {"role": "system", "content": ENRICH_SYS},
                    {"role": "user", "content": _enrich_prompt(facts, schema, row["domain"])},
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }
            f.write(json.dumps({
                "custom_id": vid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }, ensure_ascii=False) + "\n")
    print(f"✅ Wrote {len(df):,} requests to {jsonl_path}")
    return jsonl_path


def process_enrich_batch() -> Path:
    state = _load_state("enrich")
    if not state:
        sys.exit("No submitted enrich batch. Run --submit first.")
    batch_ids = state.get("batch_ids") or ([state["batch_id"]] if state.get("batch_id") else [])
    if not batch_ids:
        sys.exit("No batch_ids in enrich state")
    results: list = []
    for bid in batch_ids:
        status = _poll_batch(bid)
        print(f"Batch {bid} status: {status['status']}  counts: {status['counts']}")
        if status["status"] != "completed":
            sys.exit(f"Batch {bid} not yet complete.")
        results.extend(_download_output(status["output_file_id"]))
    print(f"⬇️  Received {len(results):,} total responses")

    cache_path = OUT_DIR / "hybrid_full_cache.json"
    cache: Dict[str, Any] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)

    for r in results:
        vid = r["custom_id"]
        if r.get("error"):
            cache[vid] = {"__error": str(r["error"])}
            continue
        try:
            content = r["response"]["body"]["choices"][0]["message"]["content"] or "{}"
            cache[vid] = json.loads(content)
        except Exception as e:
            cache[vid] = {"__error": f"parse: {e}"}

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(cache):,} enriched feature dicts → {cache_path}")
    return cache_path


# ==================== CLI ====================
def cmd_status():
    """Show status of all submitted batches."""
    for state_file in sorted(BATCH_DIR.glob("state_*.json")):
        with open(state_file) as f:
            s = json.load(f)
        name = state_file.stem.replace("state_", "")
        batch_ids = s.get("batch_ids") or ([s["batch_id"]] if s.get("batch_id") else [])
        if not batch_ids:
            print(f"{name:20s}  (no batch id)")
            continue
        agg = {"completed": 0, "failed": 0, "total": 0}
        statuses = []
        for bid in batch_ids:
            st = _poll_batch(bid)
            statuses.append(st["status"])
            for k in ("completed", "failed", "total"):
                agg[k] += st["counts"].get(k, 0)
        unique_status = set(statuses)
        st_label = statuses[0] if len(unique_status) == 1 else f"mixed({','.join(sorted(unique_status))})"
        print(f"{name:20s}  chunks={len(batch_ids)}  status={st_label}  counts={agg}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("schema")
    ps.add_argument("--domain", choices=["drugs", "weapon"], required=True)
    g = ps.add_mutually_exclusive_group(required=True)
    g.add_argument("--submit", action="store_true")
    g.add_argument("--process", action="store_true")
    ps.add_argument("--limit", type=int)
    ps.add_argument("--reuse-jsonl", action="store_true",
                    help="Skip prompt recording; upload existing *_input.jsonl as-is")

    pe = sub.add_parser("enrich")
    g2 = pe.add_mutually_exclusive_group(required=True)
    g2.add_argument("--submit", action="store_true")
    g2.add_argument("--process", action="store_true")
    pe.add_argument("--limit", type=int)

    sub.add_parser("status")

    args = ap.parse_args()

    if args.cmd == "schema":
        if args.submit:
            jsonl = BATCH_DIR / f"schema_{args.domain}_input.jsonl"
            if args.reuse_jsonl and jsonl.exists():
                print(f"♻️  Reusing existing {jsonl.name} ({jsonl.stat().st_size/1024/1024:.1f} MB)")
            else:
                jsonl = prepare_schema_batch(args.domain, args.limit)
            bids = _upload_and_submit(jsonl, f"schema_{args.domain}")
            _save_state("schema", {"batch_ids": bids, "jsonl": str(jsonl),
                                   "domain": args.domain, "limit": args.limit},
                        args.domain)
        elif args.process:
            process_schema_batch(args.domain)

    elif args.cmd == "enrich":
        if args.submit:
            jsonl = prepare_enrich_batch(args.limit)
            bids = _upload_and_submit(jsonl, "enrich")
            _save_state("enrich", {"batch_ids": bids, "jsonl": str(jsonl), "limit": args.limit})
        elif args.process:
            process_enrich_batch()

    elif args.cmd == "status":
        cmd_status()


if __name__ == "__main__":
    main()
