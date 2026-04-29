#!/usr/bin/env python3
"""
Compute Gemini text-embedding cosine similarity for the 85K pairs in
data_per_domain/similarity_scores.csv, using indictment_facts text from
verdicts_clean.csv as the verdict representation.

Output format matches similarity_scores.csv:
  verdict_1, verdict_2, domain, similarity_score
"""
from __future__ import annotations
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!")
EXP = ROOT / "new_try/experiments"
SIM_CSV = EXP / "data_per_domain/similarity_scores_combined.csv"
VERDICTS = ROOT / "new_try/innovation_submission/data_master_final/verdicts_clean.csv"
OUT_CSV = EXP / "data_per_domain/similarity_scores_gemini_combined.csv"
CACHE_DIR = EXP / "data_per_domain/emb_cache_gemini"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "gemini-embedding-001"
BATCH = 50  # Gemini accepts batches; 50 is conservative


def load_env():
    from dotenv import load_dotenv
    load_dotenv(EXP / ".env")
    load_dotenv(ROOT / ".env")


def encode_batch(client, texts: list[str]) -> list[np.ndarray]:
    resp = client.models.embed_content(
        model=MODEL_ID,
        contents=texts,
        config={"task_type": "SEMANTIC_SIMILARITY"},
    )
    out = []
    for e in resp.embeddings:
        v = np.asarray(e.values, dtype=np.float32)
        n = np.linalg.norm(v)
        out.append(v / n if n > 0 else v)
    return out


def encode_verdicts(id_to_text: dict[str, str]) -> dict[str, np.ndarray]:
    """Encode unique verdicts via Gemini, with on-disk cache."""
    from google import genai
    cache: dict[str, np.ndarray] = {}
    todo_ids, todo_texts = [], []
    for vid, txt in id_to_text.items():
        cp = CACHE_DIR / f"{vid}.npy"
        if cp.exists():
            cache[vid] = np.load(cp)
        else:
            todo_ids.append(vid)
            todo_texts.append(txt)
    print(f"Cached: {len(cache):,}, to encode: {len(todo_ids):,}")
    if not todo_ids:
        return cache

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    for i in tqdm(range(0, len(todo_ids), BATCH), desc="Encoding"):
        batch_ids = todo_ids[i:i + BATCH]
        batch_texts = todo_texts[i:i + BATCH]
        # truncate to ~8000 chars to stay under token limits
        batch_texts = [t[:8000] for t in batch_texts]
        try:
            vecs = encode_batch(client, batch_texts)
        except Exception as e:
            print(f"  batch {i} failed: {e}; retrying one-by-one")
            vecs = []
            for t in batch_texts:
                try:
                    vecs.extend(encode_batch(client, [t]))
                except Exception as e2:
                    print(f"    single failed: {e2}; using zero vec")
                    vecs.append(np.zeros(3072, dtype=np.float32))
                time.sleep(0.5)
        for vid, vec in zip(batch_ids, vecs):
            np.save(CACHE_DIR / f"{vid}.npy", vec)
            cache[vid] = vec
    return cache


def main():
    load_env()
    print(f"Loading pairs from {SIM_CSV.name}...")
    pairs = pd.read_csv(SIM_CSV, usecols=["verdict_1", "verdict_2", "domain"])
    print(f"  {len(pairs):,} pairs")

    print(f"Loading verdict texts from {VERDICTS.name}...")
    clean = pd.read_csv(VERDICTS)
    txt_map = {}
    for _, r in clean.iterrows():
        cid = r["canonical_id"]
        if cid in txt_map:
            continue
        txt = r.get("indictment_facts") or r.get("indictment_facts_raw") or ""
        if isinstance(txt, str) and txt.strip():
            txt_map[cid] = txt

    unique_ids = set(pairs["verdict_1"]).union(pairs["verdict_2"])
    have_text = unique_ids & txt_map.keys()
    missing = unique_ids - txt_map.keys()
    print(f"  unique verdicts in pairs: {len(unique_ids):,}")
    print(f"  with text: {len(have_text):,}, missing: {len(missing):,}")

    id_to_text = {vid: txt_map[vid] for vid in have_text}
    embs = encode_verdicts(id_to_text)
    print(f"Got {len(embs):,} embeddings")

    # Compute pair scores
    scores = np.full(len(pairs), np.nan, dtype=np.float32)
    for i, (v1, v2) in enumerate(zip(pairs["verdict_1"], pairs["verdict_2"])):
        if v1 in embs and v2 in embs:
            cos = float(embs[v1] @ embs[v2])
            scores[i] = (max(-1.0, min(1.0, cos)) + 1.0) * 50.0  # [0, 100]

    out = pairs.copy()
    out["similarity_score"] = scores
    n_valid = int((~out["similarity_score"].isna()).sum())
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV}  ({n_valid:,}/{len(out):,} pairs scored)")


if __name__ == "__main__":
    main()
