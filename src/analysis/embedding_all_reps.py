"""Embedding baseline applied to EVERY representation (ablation).

Extends `embedding_baseline.py` — which embeds only raw verdict facts — to
embed the *structured representations* (Manual, GPT-Schema, GPT-Free, GPT-Law,
Raw-Facts, Hybrid-Manual, Hybrid-Full) and score pair similarity via cosine.

Interpretation:
  - Raw-Facts + embedding  = baseline using only text (no structure).
  - Manual + embedding     = what do you get from a structured manual rep
                             alone, without any LLM reasoning?
  - Manual + LLM (main exp)= full pipeline with LLM reasoning on top.

The three-way gap separates:
  1. Structure (Manual+emb  vs  Facts+emb)
  2. LLM reasoning (Manual+LLM vs  Manual+emb)

Serialization:
  For reps with JSON feature vectors (all except Raw-Facts), we flatten the
  dict into a deterministic key-value string:
      "key1: value1 | key2: value2 | ..."
  Arrays are joined by commas. This is what the embedding model sees.

Outputs (under experiments/results_paper_baselines/):
  - emb_reps_full.csv       : per (domain, rep, emb_model, metric)
  - emb_reps_preds/*.csv    : per-pair scores for each (rep, emb_model, domain)
  - EMB_REPS_REPORT.md      : pivot tables per (domain, metric)

Usage:
  cd new_try/experiments/src/analysis
  python embedding_all_reps.py [--models openai e5 bge] [--reps Manual GPT-Schema ...]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

EXP = Path(__file__).resolve().parents[2]
OUT = EXP / "results_paper_baselines"
OUT.mkdir(exist_ok=True)
(OUT / "emb_reps_preds").mkdir(exist_ok=True)
CACHE = OUT / "emb_cache"
CACHE.mkdir(exist_ok=True)

# Map (rep → source CSV file, feature column) for each domain.
#   feature column is the column containing the feature vector / text for V1
#   (V2 column is the same with suffix _2).
DOMAINS = {
    "drugs":  EXP / "data" / "drugs",
    "weapon": EXP / "data" / "wep",
}
REP_CONFIG = {
    "Manual":        ("manual_fe.csv",         "feature_vector"),
    "GPT-Schema":    ("fe_gpt_schema.csv",     "feature_vector"),
    "GPT-Free":      ("gpt_free.csv",          "feature_vector"),
    "GPT-Law":       ("gpt_law.csv",           "feature_vector"),
    "Raw-Facts":     ("facts.csv",             "indicment_facts"),  # plain text
    "Hybrid-Manual": ("hybrid_manual_gpt.csv", "feature_vector"),
    "Hybrid-Full":   ("hybrid_full_gpt.csv",   "feature_vector"),
}

MODELS = {
    "openai": {"provider": "openai",
               "model_id": "text-embedding-3-large",
               "display": "OpenAI 3-large"},
    "gemini": {"provider": "google",
               "model_id": "gemini-embedding-001",
               "display": "Gemini-embedding-001"},
    "e5": {"provider": "hf",
           "model_id": "intfloat/multilingual-e5-large-instruct",
           "display": "mE5-large-instruct",
           "query_prefix": "Instruct: Given a legal verdict, retrieve similar verdicts.\nQuery: "},
    "bge": {"provider": "hf",
            "model_id": "BAAI/bge-m3",
            "display": "BGE-M3"},
}


# ─── Metric helpers ───

def _best_f1(scores, y):
    if len(np.unique(y)) < 2:
        return np.nan
    best = 0.0
    for thr in np.unique(scores):
        f = f1_score(y, (scores >= thr).astype(int), zero_division=0)
        if f > best:
            best = f
    return best


def _cv_f1(scores, y, k=5, seed=42):
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


def _qwk(y_true, y_pred, n_r=3):
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


def _best_qwk(scores, gt):
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


def _cv_qwk(scores, gt, k=10, seed=42):
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


# ─── Feature serialization ───

def _stringify_value(v: Any) -> str:
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    if isinstance(v, dict):
        return "; ".join(f"{k}={_stringify_value(val)}" for k, val in v.items())
    return str(v) if v is not None else ""


def serialize_feature(raw: Any) -> str:
    """Turn a rep's feature value (JSON string / dict / plain text) into a
    deterministic key-value string for embedding."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    return " | ".join(f"{k}: {_stringify_value(v)}"
                                      for k, v in parsed.items()
                                      if _stringify_value(v) != "")
                return stripped
            except json.JSONDecodeError:
                return stripped
        return stripped
    if isinstance(raw, dict):
        return " | ".join(f"{k}: {_stringify_value(v)}" for k, v in raw.items())
    return str(raw)


# ─── Encoders (shared with embedding_baseline.py) ───

def _cache_path(model_key: str, rep: str, verdict_id: str) -> Path:
    # Include rep in cache key so feature_vector differs across reps per verdict.
    safe_rep = rep.replace("-", "_").replace(" ", "_").lower()
    return CACHE / f"{model_key}_{safe_rep}_{verdict_id}.npy"


def _encode_openai(texts, model_id, batch_size=64):
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv(EXP.parent.parent / ".env")
    load_dotenv(EXP / ".env")
    client = OpenAI()
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=model_id, input=batch)
        vecs.extend([np.asarray(d.embedding, dtype=np.float32) for d in resp.data])
    return np.vstack(vecs)


def _encode_google(texts, model_id, task_type="SEMANTIC_SIMILARITY",
                   batch_size=50, initial_delay=15.0):
    """Gemini embedding with retry on 429 rate limits.

    Free tier quotas are tight (~5 RPM for gemini-embedding-001). We sleep
    between batches and back off exponentially on 429.
    """
    from google import genai
    from google.genai import errors as genai_errors
    from dotenv import load_dotenv
    import time as _time
    load_dotenv(EXP.parent.parent / ".env")
    load_dotenv(EXP / ".env")
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        delay = initial_delay
        attempt = 0
        while True:
            try:
                resp = client.models.embed_content(
                    model=model_id,
                    contents=batch,
                    config={"task_type": task_type},
                )
                break
            except genai_errors.ClientError as e:
                # 429 = rate limit. Back off and retry. The genai ClientError
                # exposes the HTTP status on the `code` attribute.
                err_code = getattr(e, "code", None)
                if err_code == 429 and attempt < 8:
                    attempt += 1
                    print(f"        429 rate limit, sleeping {delay:.0f}s (attempt {attempt})...")
                    _time.sleep(delay)
                    delay = min(delay * 2, 180.0)
                else:
                    raise
        vecs.extend([np.asarray(e.values, dtype=np.float32) for e in resp.embeddings])
        # Gentle pause between successful batches to stay under RPM limit.
        if i + batch_size < len(texts):
            _time.sleep(15.0)
    # Final sleep so the NEXT (rep, domain) call has spacing from this one.
    # 15s => 4 RPM effective, below the 5 RPM free-tier cap.
    _time.sleep(15.0)
    arr = np.vstack(vecs)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.where(norms > 0, norms, 1.0)
    return arr.astype(np.float32)


_HF_MODELS: dict[str, "SentenceTransformer"] = {}


def _get_hf_model(model_id: str):
    """Load a HF SentenceTransformer once per process (cached)."""
    from sentence_transformers import SentenceTransformer
    import torch
    if model_id in _HF_MODELS:
        return _HF_MODELS[model_id]
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(f"      (loading {model_id} on {device} — one-time cost)")
    t0 = time.time()
    _HF_MODELS[model_id] = SentenceTransformer(model_id, device=device, trust_remote_code=True)
    print(f"      (loaded in {time.time()-t0:.1f}s)")
    return _HF_MODELS[model_id]


def _encode_hf(texts, model_id, prefix=""):
    model = _get_hf_model(model_id)
    inputs = [prefix + t if t else "empty" for t in texts]
    embs = model.encode(inputs, batch_size=8, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype(np.float32)


def encode_features(model_key: str, rep: str,
                    id_to_text: dict[str, str]) -> dict[str, np.ndarray]:
    cfg = MODELS[model_key]
    ids, texts = [], []
    cached: dict[str, np.ndarray] = {}
    for vid, txt in id_to_text.items():
        cp = _cache_path(model_key, rep, vid)
        if cp.exists():
            cached[vid] = np.load(cp)
        else:
            ids.append(vid)
            texts.append(txt)

    if texts:
        print(f"      encoding {len(texts)} new (rep={rep}) via {cfg['display']} "
              f"({len(cached)} cached)...")
        t0 = time.time()
        if cfg["provider"] == "openai":
            vecs = _encode_openai(texts, cfg["model_id"])
        elif cfg["provider"] == "google":
            vecs = _encode_google(texts, cfg["model_id"])
        else:
            vecs = _encode_hf(texts, cfg["model_id"], cfg.get("query_prefix", ""))
        for vid, vec in zip(ids, vecs):
            np.save(_cache_path(model_key, rep, vid), vec)
            cached[vid] = vec
        print(f"      done in {time.time()-t0:.1f}s")
    return cached


# ─── Per (domain, rep, emb_model) evaluation ───

def eval_combo(domain: str, rep: str, model_key: str,
               data_dir: Path) -> tuple[pd.DataFrame, dict]:
    fn, feat_col_base = REP_CONFIG[rep]
    fpath = data_dir / fn
    df = pd.read_csv(fpath)

    col_v1 = f"{feat_col_base}_1"
    col_v2 = f"{feat_col_base}_2"

    # Build id → serialized-text map (dedupe verdicts).
    id_to_text: dict[str, str] = {}
    for _, r in df.iterrows():
        if r["verdict_1"] not in id_to_text:
            id_to_text[r["verdict_1"]] = serialize_feature(r.get(col_v1, ""))
        if r["verdict_2"] not in id_to_text:
            id_to_text[r["verdict_2"]] = serialize_feature(r.get(col_v2, ""))

    embs = encode_features(model_key, rep, id_to_text)

    scores = np.array([
        float(embs[v1] @ embs[v2]) for v1, v2 in zip(df["verdict_1"], df["verdict_2"])
    ])
    scores = (scores.clip(-1.0, 1.0) + 1.0) * 50.0   # [0, 100]

    pred_df = df[["verdict_1", "verdict_2", "similarity_scale",
                  "similarity_binary_0", "similarity_binary_1"]].copy()
    pred_df["score"] = scores
    pred_df.to_csv(OUT / "emb_reps_preds" /
                   f"embrep_{rep.replace(' ','_')}_{model_key}_{domain}_preds.csv",
                   index=False)

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
    rows = [dict(domain=domain, rep=rep,
                 emb_model_key=model_key,
                 emb_model_display=MODELS[model_key]["display"],
                 metric=k, value=v) for k, v in metrics.items()]
    return pd.DataFrame(rows), metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                    choices=list(MODELS.keys()))
    ap.add_argument("--reps", nargs="+", default=list(REP_CONFIG.keys()),
                    choices=list(REP_CONFIG.keys()))
    args = ap.parse_args()

    print(f"Running embedding-on-reps for models={args.models} reps={args.reps}")

    all_rows = []
    for model_key in args.models:
        print(f"\n=== {MODELS[model_key]['display']} ===")
        for rep in args.reps:
            print(f"  rep={rep}")
            for dom, data_dir in DOMAINS.items():
                print(f"    domain={dom}")
                df_rows, metrics = eval_combo(dom, rep, model_key, data_dir)
                all_rows.append(df_rows)
                print(f"      F1_CV_b0={metrics['F1_CV_b0']:.3f} | "
                      f"F1_CV_b1={metrics['F1_CV_b1']:.3f} | "
                      f"AP_b0={metrics['AP_b0']:.3f} | "
                      f"QWK_CV={metrics['QWK_CV']:.3f}")

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(OUT / "emb_reps_full.csv", index=False)

    # Markdown report: for each metric, a (domain, rep) × emb_model table.
    md = ["# Embedding-on-Representations Baseline — Ablation Report\n"]
    md.append("_Cosine similarity of embedded feature vectors (structured reps) / raw text "
              "(Raw-Facts). Scores rescaled to [0, 100]. All 7 representations x 3 embedding "
              "models x 2 domains._\n")
    md.append("_Purpose: separates 'structure helps' (rep+emb vs facts+emb) from "
              "'LLM reasoning helps' (rep+LLM vs rep+emb)._\n")

    metrics_order = ["F1_Oracle_b0", "F1_Oracle_b1", "F1_CV_b0", "F1_CV_b1",
                     "AP_b0", "AP_b1", "QWK_Oracle", "QWK_CV"]

    for dom in DOMAINS:
        md.append(f"\n## {dom.upper()}\n")
        sub = full[full.domain == dom]
        for metric in metrics_order:
            m_sub = sub[sub.metric == metric]
            if m_sub.empty:
                continue
            pivot = m_sub.pivot_table(index="rep", columns="emb_model_display",
                                      values="value", aggfunc="first")
            pivot = pivot.reindex(list(REP_CONFIG))
            pivot["Row mean"] = pivot.mean(axis=1)
            md.append(f"\n### {metric}\n")
            md.append(pivot.round(3).to_markdown())

    (OUT / "EMB_REPS_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nDone. Outputs under: {OUT}")


if __name__ == "__main__":
    main()
