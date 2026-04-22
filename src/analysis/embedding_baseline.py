"""Embedding baseline: cosine similarity of full verdict facts, via 3 providers.

Pipeline:
  1. Load pair facts (indicment_facts_1/2) from data/{wep,drugs}/facts.csv
  2. Encode each unique verdict text once per embedding model
  3. Score each pair as cosine(emb_1, emb_2), mapped to [0, 100] to match the
     LLM score scale used by paper_results.py / paper_results_qwk.py
  4. Save per-pair predictions and evaluate F1 / AP / QWK against GT

Embedding models (top-3):
  - openai:text-embedding-3-large  (paid, via OPENAI_API_KEY)
  - hf:intfloat/multilingual-e5-large-instruct  (free, local via sentence-transformers)
  - hf:BAAI/bge-m3                              (free, local via sentence-transformers)

Outputs (under experiments/results_paper_baselines/):
  - emb_full.csv            : per-cell metrics (domain x model x metric)
  - emb_preds/*.csv         : per-pair scores (same schema as LLM preds)
  - EMB_REPORT.md           : per-domain metric tables

Usage:
  cd new_try/experiments/src/analysis
  python embedding_baseline.py [--models openai e5 bge] [--skip-openai]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

EXP = Path(__file__).resolve().parents[2]
OUT = EXP / "results_paper_baselines"
OUT.mkdir(exist_ok=True)
(OUT / "emb_preds").mkdir(exist_ok=True)
CACHE = OUT / "emb_cache"
CACHE.mkdir(exist_ok=True)

DOMAINS = {
    "drugs": EXP / "data" / "drugs" / "facts.csv",
    "weapon": EXP / "data" / "wep" / "facts.csv",
}

MODELS = {
    "openai": {
        "provider": "openai",
        "model_id": "text-embedding-3-large",
        "display": "OpenAI 3-large",
    },
    "gemini": {
        "provider": "google",
        "model_id": "gemini-embedding-001",
        "display": "Gemini-embedding-001",
    },
    "e5": {
        "provider": "hf",
        "model_id": "intfloat/multilingual-e5-large-instruct",
        "display": "mE5-large-instruct",
        "query_prefix": "Instruct: Given a legal verdict, retrieve similar verdicts.\nQuery: ",
    },
    "bge": {
        "provider": "hf",
        "model_id": "BAAI/bge-m3",
        "display": "BGE-M3",
    },
}


# ─── Metric helpers (shared with paper_results*.py) ───

def _best_f1(scores: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    best = 0.0
    for thr in np.unique(scores):
        f = f1_score(y, (scores >= thr).astype(int), zero_division=0)
        if f > best:
            best = f
    return best


def _cv_f1(scores: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 42) -> float:
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < k:
        return np.nan
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=int)
    for tr, te in skf.split(scores, y):
        best_f, best_t = 0.0, None
        for t in np.unique(scores[tr]):
            f = f1_score(y[tr], (scores[tr] >= t).astype(int), zero_division=0)
            if f > best_f:
                best_f, best_t = f, t
        if best_t is None:
            best_t = float(np.median(scores[tr]))
        pred[te] = (scores[te] >= best_t).astype(int)
    return f1_score(y, pred, zero_division=0)


def _qwk(y_true: np.ndarray, y_pred: np.ndarray, n_r: int = 3) -> float:
    O = np.zeros((n_r, n_r))
    for t, p in zip(y_true, y_pred):
        O[t - 1, p - 1] += 1
    N = len(y_true)
    ht = np.bincount(y_true - 1, minlength=n_r)
    hp = np.bincount(y_pred - 1, minlength=n_r)
    E = np.outer(ht, hp).astype(float) / N
    W = np.zeros((n_r, n_r))
    for i in range(n_r):
        for j in range(n_r):
            W[i, j] = ((i - j) ** 2) / ((n_r - 1) ** 2)
    denom = np.sum(W * E)
    return 1.0 - (np.sum(W * O) / denom) if denom > 0 else 0.0


def _best_qwk(scores: np.ndarray, gt: np.ndarray) -> float:
    uniq = np.unique(scores)
    if len(uniq) < 3:
        return 0.0
    mids = (uniq[:-1] + uniq[1:]) / 2.0
    best = -1.0
    for i, t1 in enumerate(mids):
        for t2 in mids[i + 1:]:
            preds = np.where(scores < t1, 1, np.where(scores < t2, 2, 3))
            if len(np.unique(preds)) < 2:
                continue
            q = _qwk(gt, preds)
            if q > best:
                best = q
    return max(best, 0.0)


def _cv_qwk(scores: np.ndarray, gt: np.ndarray, k: int = 10, seed: int = 42) -> float:
    if len(np.unique(gt)) < 2 or min(np.bincount(gt - 1)) < k:
        return np.nan
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    preds = np.zeros(len(gt), dtype=int)
    for tr, te in skf.split(scores, gt):
        uniq = np.unique(scores[tr])
        if len(uniq) < 3:
            preds[te] = 2
            continue
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        best_q, bt1, bt2 = -1.0, mids[0], mids[-1]
        for i, t1 in enumerate(mids):
            for t2 in mids[i + 1:]:
                p = np.where(scores[tr] < t1, 1, np.where(scores[tr] < t2, 2, 3))
                if len(np.unique(p)) < 2:
                    continue
                q = _qwk(gt[tr], p)
                if q > best_q:
                    best_q, bt1, bt2 = q, t1, t2
        preds[te] = np.where(scores[te] < bt1, 1, np.where(scores[te] < bt2, 2, 3))
    return _qwk(gt, preds)


# ─── Encoders ───

def _cache_path(model_key: str, verdict_id: str) -> Path:
    return CACHE / f"{model_key}_{verdict_id}.npy"


def _encode_openai(texts: list[str], model_id: str, batch_size: int = 64) -> np.ndarray:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv(EXP.parent.parent / ".env")  # /Thesis!!!/.env
    load_dotenv(EXP / ".env")                 # /Thesis!!!/new_try/experiments/.env
    client = OpenAI()
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=model_id, input=batch)
        vecs.extend([np.asarray(d.embedding, dtype=np.float32) for d in resp.data])
    return np.vstack(vecs)


def _encode_google(texts: list[str], model_id: str,
                   task_type: str = "SEMANTIC_SIMILARITY",
                   batch_size: int = 50) -> np.ndarray:
    from google import genai
    from dotenv import load_dotenv
    load_dotenv(EXP.parent.parent / ".env")
    load_dotenv(EXP / ".env")
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Gemini embed_content accepts a list for batch encoding
        resp = client.models.embed_content(
            model=model_id,
            contents=batch,
            config={"task_type": task_type},
        )
        vecs.extend([np.asarray(e.values, dtype=np.float32) for e in resp.embeddings])
    arr = np.vstack(vecs)
    # Normalize for cosine (Gemini returns unit-length vectors already for
    # SEMANTIC_SIMILARITY task_type, but we re-normalize to be safe).
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.where(norms > 0, norms, 1.0)
    return arr.astype(np.float32)


def _encode_hf(texts: list[str], model_id: str, prefix: str = "") -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    # Silence the tokenizer parallelism warning
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    model = SentenceTransformer(model_id, device=device, trust_remote_code=True)
    inputs = [prefix + t for t in texts] if prefix else texts
    embs = model.encode(inputs, batch_size=8, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype(np.float32)


def encode_verdicts(model_key: str, id_to_text: dict[str, str]) -> dict[str, np.ndarray]:
    """Encode unique verdicts, with per-file cache."""
    cfg = MODELS[model_key]
    ids, texts = [], []
    cached: dict[str, np.ndarray] = {}
    for vid, txt in id_to_text.items():
        cp = _cache_path(model_key, vid)
        if cp.exists():
            cached[vid] = np.load(cp)
        else:
            ids.append(vid)
            texts.append(txt)

    if texts:
        print(f"    encoding {len(texts)} new verdicts via {cfg['display']} "
              f"({len(cached)} cached)...")
        t0 = time.time()
        if cfg["provider"] == "openai":
            vecs = _encode_openai(texts, cfg["model_id"])
        elif cfg["provider"] == "google":
            vecs = _encode_google(texts, cfg["model_id"])
        else:
            vecs = _encode_hf(texts, cfg["model_id"], cfg.get("query_prefix", ""))
        for vid, vec in zip(ids, vecs):
            np.save(_cache_path(model_key, vid), vec)
            cached[vid] = vec
        print(f"    done in {time.time()-t0:.1f}s")
    return cached


# ─── Per-domain evaluation ───

def eval_domain(model_key: str, domain: str, facts_csv: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(facts_csv)
    # Build id->text map (first occurrence wins; verdicts are deduped)
    id_to_text: dict[str, str] = {}
    for _, r in df.iterrows():
        if r["verdict_1"] not in id_to_text:
            id_to_text[r["verdict_1"]] = str(r["indicment_facts_1"])
        if r["verdict_2"] not in id_to_text:
            id_to_text[r["verdict_2"]] = str(r["indicment_facts_2"])

    embs = encode_verdicts(model_key, id_to_text)

    # Cosine scores (embeddings normalized → dot product = cosine); scale to 0..100
    scores = np.array([
        float(embs[v1] @ embs[v2]) for v1, v2 in zip(df["verdict_1"], df["verdict_2"])
    ])
    scores = (scores.clip(-1.0, 1.0) + 1.0) * 50.0  # [0, 100]

    pred_df = df[["verdict_1", "verdict_2", "similarity_scale",
                  "similarity_binary_0", "similarity_binary_1"]].copy()
    pred_df["score"] = scores
    pred_path = OUT / "emb_preds" / f"embedding_{model_key}_{domain}_preds.csv"
    pred_df.to_csv(pred_path, index=False)

    # Metrics
    y0 = df["similarity_binary_0"].astype(int).values
    y1 = df["similarity_binary_1"].astype(int).values
    gt = df["similarity_scale"].astype(int).values

    metrics = {
        "F1_Oracle_b0": _best_f1(scores, y0),
        "F1_Oracle_b1": _best_f1(scores, y1),
        "F1_CV_b0":     _cv_f1(scores, y0),
        "F1_CV_b1":     _cv_f1(scores, y1),
        "AP_b0":        average_precision_score(y0, scores) if y0.sum() > 0 else np.nan,
        "AP_b1":        average_precision_score(y1, scores) if y1.sum() > 0 else np.nan,
        "QWK_Oracle":   _best_qwk(scores, gt),
        "QWK_CV":       _cv_qwk(scores, gt),
    }
    rows = [dict(domain=domain, model_key=model_key,
                 model_display=MODELS[model_key]["display"],
                 metric=k, value=v) for k, v in metrics.items()]
    return pd.DataFrame(rows), metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                    choices=list(MODELS.keys()))
    ap.add_argument("--skip-openai", action="store_true",
                    help="Skip the OpenAI model (useful if no API key)")
    args = ap.parse_args()

    models_to_run = [m for m in args.models if not (args.skip_openai and m == "openai")]
    print(f"Running embedding baseline for models: {models_to_run}")

    all_rows = []
    for model_key in models_to_run:
        print(f"\n=== {MODELS[model_key]['display']} ({MODELS[model_key]['model_id']}) ===")
        for dom, path in DOMAINS.items():
            print(f"  domain={dom}")
            df_rows, metrics = eval_domain(model_key, dom, path)
            all_rows.append(df_rows)
            for k, v in metrics.items():
                print(f"    {k:<14s} = {v:.3f}")

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(OUT / "emb_full.csv", index=False)

    # Pivot into reader-friendly tables
    md = ["# Embedding Baseline Report\n"]
    md.append("_Cosine similarity of full verdict facts (`indicment_facts_1/2`), "
              "rescaled to [0, 100] to match LLM score scale._\n")
    md.append("_Models: "
              + ", ".join(f"`{MODELS[k]['model_id']}`" for k in models_to_run) + "._\n")

    metrics_order = ["F1_Oracle_b0", "F1_Oracle_b1", "F1_CV_b0", "F1_CV_b1",
                     "AP_b0", "AP_b1", "QWK_Oracle", "QWK_CV"]

    for dom in DOMAINS:
        md.append(f"\n## {dom.upper()}\n")
        sub = full[full.domain == dom]
        pivot = sub.pivot_table(index="metric", columns="model_display",
                                values="value", aggfunc="first")
        pivot = pivot.reindex(metrics_order)
        md.append(pivot.round(3).to_markdown())

    (OUT / "EMB_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nDone. Outputs under: {OUT}")


if __name__ == "__main__":
    main()
