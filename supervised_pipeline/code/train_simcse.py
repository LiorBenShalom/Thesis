#!/usr/bin/env python3
"""
SimCSE — unsupervised sentence-encoder training on indictment-facts.

Trains DictaBERT-base with SimCSE (Gao et al. 2021) on 8,446 Israeli criminal
verdicts (drugs + weapon). Output: a sentence encoder + per-verdict embeddings
for use as an alternative filter to the citation-based candidate selection.

PURE UNSUPERVISED — no LLM-panel scores, no citation network, no sentencing
labels. Only the raw indictment-facts text. Positives = same text passed
twice through the encoder with different dropout masks. In-batch negatives.

PORTABLE — all paths relative to this script's location.

GRADIENT ACCUMULATION — physical batch is small (low VRAM); effective batch
(via accumulation) drives SimCSE quality. Default: physical=8, accum=8 →
effective=64. Uses ~3-4 GB VRAM, suitable for shared GPUs.

USAGE
─────
    # Smoke test (~5 min on A10)
    python train_simcse.py --limit 500

    # Full training (~45-90 min on A10 with shared GPU)
    python train_simcse.py

    # Larger effective batch if VRAM allows
    python train_simcse.py --batch-size 16 --grad-accum 8   # effective 128

    # Encode only (after training is done)
    python train_simcse.py --encode-only
"""
from __future__ import annotations
import argparse, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer import losses, modules

HERE      = Path(__file__).resolve().parent
DATA_CSV  = HERE / "data" / "indictment_facts.csv"
OUT_DIR   = HERE / "outputs"
MODEL_DIR = OUT_DIR / "model"


def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def load_facts(limit=None):
    df = pd.read_csv(DATA_CSV)
    df = df.dropna(subset=["indictment_facts"]).reset_index(drop=True)
    if limit: df = df.head(limit).reset_index(drop=True)
    return df


def build_model(base_model: str, max_seq_len: int) -> SentenceTransformer:
    word = modules.Transformer(base_model, max_seq_length=max_seq_len)
    pool = modules.Pooling(word.get_embedding_dimension(), pooling_mode="cls")
    return SentenceTransformer(modules=[word, pool])


def train(model, df, batch_size, grad_accum, epochs, lr, device, precision):
    # Two columns of identical text → SimCSE positive pair (different dropout masks).
    ds = Dataset.from_dict({
        "anchor":   df.indictment_facts.tolist(),
        "positive": df.indictment_facts.tolist(),
    })

    loss_fn = losses.MultipleNegativesRankingLoss(model)   # InfoNCE
    effective_batch = batch_size * grad_accum

    args = SentenceTransformerTrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.1,
        bf16=(precision == "bf16"),
        fp16=(precision == "fp16"),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        dataloader_drop_last=True,
        remove_unused_columns=False,
    )

    print(f"\n=== Training SimCSE ===")
    print(f"  examples           : {len(ds):,}")
    print(f"  physical batch     : {batch_size}")
    print(f"  grad accum steps   : {grad_accum}")
    print(f"  effective batch    : {effective_batch}")
    print(f"  epochs             : {epochs}")
    print(f"  lr                 : {lr}")
    print(f"  device             : {device}")
    print(f"  precision          : {precision}")

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        loss=loss_fn,
    )
    t0 = time.time()
    trainer.train()
    print(f"  ✓ trained in {(time.time()-t0)/60:.1f} min")

    # Save final model (Trainer also saves checkpoints under output_dir).
    model.save(str(MODEL_DIR))
    print(f"  saved final model → {MODEL_DIR}")


def encode_all(model, df, batch_size):
    print(f"\n=== Encoding {len(df):,} verdicts ===")
    t0 = time.time()
    emb = model.encode(
        df.indictment_facts.tolist(),
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"  ✓ encoded in {(time.time()-t0)/60:.1f} min → shape {emb.shape}")

    np.save(OUT_DIR / "verdict_embeddings.npy", emb)
    pd.DataFrame({"verdict": df.verdict, "domain": df.domain}).to_csv(
        OUT_DIR / "verdict_index.csv", index=False)
    print(f"  saved → {OUT_DIR/'verdict_embeddings.npy'}  ({emb.nbytes/1e6:.1f} MB)")
    print(f"          {OUT_DIR/'verdict_index.csv'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="dicta-il/dictabert",
                   help="HF model id. Alternatives: avichr/heBERT, onlplab/alephbert-base")
    p.add_argument("--batch-size",  type=int,   default=8,
                   help="Physical batch (per GPU). Low default for shared GPUs.")
    p.add_argument("--grad-accum",  type=int,   default=8,
                   help="Gradient accumulation steps. Effective batch = batch * accum.")
    p.add_argument("--epochs",      type=int,   default=1)
    p.add_argument("--lr",          type=float, default=3e-5)
    p.add_argument("--max-seq-len", type=int,   default=512)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--limit",       type=int,   default=None,
                   help="Smoke test on first N verdicts only.")
    p.add_argument("--encode-only", action="store_true",
                   help="Skip training; reuse saved model.")
    p.add_argument("--precision",   choices=["bf16", "fp16", "fp32"], default="bf16",
                   help="bf16 (A10/A100/H100), fp16 (V100/T4), or fp32 fallback.")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)

    df = load_facts(limit=args.limit)
    print(f"Loaded {len(df):,} verdicts ({df.domain.value_counts().to_dict()})")
    print(f"  facts mean len: {df.indictment_facts.str.len().mean():.0f} chars "
          f"(P95: {df.indictment_facts.str.len().quantile(0.95):.0f})")

    device = pick_device()
    precision = args.precision if device == "cuda" else "fp32"

    if args.encode_only:
        if not MODEL_DIR.exists():
            raise FileNotFoundError(f"--encode-only requires saved model at {MODEL_DIR}")
        print(f"Loading saved model from {MODEL_DIR}")
        model = SentenceTransformer(str(MODEL_DIR), device=device)
    else:
        model = build_model(args.base_model, args.max_seq_len)
        model.to(device)
        train(model, df, args.batch_size, args.grad_accum,
              args.epochs, args.lr, device, precision)

    encode_all(model, df, max(args.batch_size, 16))   # encoding is cheaper, can use bigger batch
    print("\n✓ Done.")
    print(f"\nNext: copy {OUT_DIR}/ back to your Mac for downstream analysis.")


if __name__ == "__main__":
    main()
