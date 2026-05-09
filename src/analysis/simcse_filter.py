#!/usr/bin/env python3
"""
SimCSE — unsupervised sentence-encoder for filtering candidate verdict pairs.

Trains DictaBERT-base with SimCSE (Gao et al. 2021) on indictment-facts text
from all 8,446 drugs+weapon verdicts. Output: a sentence encoder + per-verdict
embeddings (.npy + ids.txt) that can be used as an alternative to the citation
filter — without ever seeing LLM-panel scores or sentencing labels.

USAGE
─────
    # train + encode (default ~1h on a single CUDA GPU, ~2-3h on MPS)
    python src/analysis/simcse_filter.py

    # smaller test run
    python src/analysis/simcse_filter.py --epochs 1 --batch-size 32 --limit 500

    # encode only (skip training, reuse saved model)
    python src/analysis/simcse_filter.py --encode-only

PURE UNSUPERVISED
─────────────────
We do NOT use:
  • LLM-panel similarity scores (similarity_scores_combined.csv)
  • citation network (citation_pair_types.csv)
  • sentencing range labels (sentencing_range_low/high)
We use ONLY: the raw indictment-facts text. SimCSE positives = same text passed
twice through the encoder with different dropout masks. In-batch negatives.
This guarantees a clean comparison vs the citation-based filter downstream.
"""
from __future__ import annotations
import argparse, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
SRC  = ROOT / "innovation_submission/data_master/verdicts_master.csv"
OUT  = ROOT / "experiments/results/0_preprocessing/embedding_filter"
MODEL_DIR = OUT / "model"


def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def load_facts(limit=None):
    df = pd.read_csv(SRC, usecols=["verdict", "domain", "indictment_facts"])
    df = df[df.domain.isin(["drugs", "weapon"]) & df.indictment_facts.notna()]
    df = df.drop_duplicates("verdict").reset_index(drop=True)
    if limit: df = df.head(limit).reset_index(drop=True)
    return df


def build_model(base_model: str, max_seq_len: int) -> SentenceTransformer:
    word = models.Transformer(base_model, max_seq_length=max_seq_len)
    # SimCSE original uses CLS pooling; we mirror that.
    pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="cls")
    return SentenceTransformer(modules=[word, pool])


def train(model, df, batch_size, epochs, lr, device):
    examples = [InputExample(texts=[t, t]) for t in df.indictment_facts]
    random.shuffle(examples)
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size, drop_last=True)
    loss_fn = losses.MultipleNegativesRankingLoss(model)   # InfoNCE
    warmup = max(1, int(len(loader) * epochs * 0.1))

    print(f"\n=== Training SimCSE ===")
    print(f"  examples       : {len(examples):,}")
    print(f"  steps/epoch    : {len(loader):,}")
    print(f"  total steps    : {len(loader) * epochs:,}")
    print(f"  warmup steps   : {warmup:,}")
    print(f"  device         : {device}")
    print(f"  AMP (fp16/bf16): {device == 'cuda'}")

    t0 = time.time()
    model.fit(
        train_objectives=[(loader, loss_fn)],
        epochs=epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": lr},
        output_path=str(MODEL_DIR),
        use_amp=(device == "cuda"),  # AMP on CUDA only; flaky on MPS
        show_progress_bar=True,
    )
    print(f"  trained in {(time.time()-t0)/60:.1f} min → {MODEL_DIR}")


def encode_all(model, df, batch_size):
    print(f"\n=== Encoding {len(df):,} verdicts ===")
    t0 = time.time()
    emb = model.encode(
        df.indictment_facts.tolist(),
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,    # so cosine = dot product
    )
    print(f"  encoded in {(time.time()-t0)/60:.1f} min → shape {emb.shape}")

    np.save(OUT / "verdict_embeddings.npy", emb)
    pd.DataFrame({"verdict": df.verdict, "domain": df.domain}).to_csv(
        OUT / "verdict_index.csv", index=False)
    print(f"  saved → {OUT/'verdict_embeddings.npy'}")
    print(f"          {OUT/'verdict_index.csv'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="dicta-il/dictabert",
                   help="HF model id. Alternatives: avichr/heBERT, onlplab/alephbert-base")
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--epochs",      type=int,   default=1)
    p.add_argument("--lr",          type=float, default=3e-5)
    p.add_argument("--max-seq-len", type=int,   default=512)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--limit",       type=int,   default=None,
                   help="If set, train+encode on first N verdicts only (smoke test).")
    p.add_argument("--encode-only", action="store_true",
                   help="Skip training; load saved model from MODEL_DIR and only encode.")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)

    df = load_facts(limit=args.limit)
    print(f"Loaded {len(df):,} verdicts ({df.domain.value_counts().to_dict()})")
    print(f"  facts mean len: {df.indictment_facts.str.len().mean():.0f} chars "
          f"(P95: {df.indictment_facts.str.len().quantile(0.95):.0f})")

    device = pick_device()

    if args.encode_only:
        if not MODEL_DIR.exists():
            raise FileNotFoundError(f"--encode-only requires saved model at {MODEL_DIR}")
        print(f"Loading saved model from {MODEL_DIR}")
        model = SentenceTransformer(str(MODEL_DIR), device=device)
    else:
        model = build_model(args.base_model, args.max_seq_len)
        model.to(device)
        train(model, df, args.batch_size, args.epochs, args.lr, device)

    encode_all(model, df, args.batch_size)
    print("\nDone.")


if __name__ == "__main__":
    main()
