#!/usr/bin/env python3
"""
SUPERVISED contrastive — sentence encoder for sentencing-range similarity.

Trains DictaBERT-base with MultipleNegativesRankingLoss using sentencing-range
similarity as the supervision signal:
  positive pair:  |Δ_low| ≤ THR_POS  AND  |Δ_high| ≤ THR_POS  (default THR_POS=6 mo)
In-batch negatives are everything else (automatic with MultipleNegativesRanking).

Per-domain training (drugs / weapon separately) — sentencing scales differ.

LEAKAGE-PREVENTION SPLIT:
  - Random 80/20 split AT THE VERDICT LEVEL (seed=42)
  - Training pairs = pairs where BOTH verdicts are in the train set
  - Test verdicts NEVER appear in any training pair
  - Embeddings produced for all 3,898 verdicts (test verdicts encoded via the
    model trained on train-only)
  - Downstream eval (separate script) computes top-K neighbors for test
    queries, restricting candidate pool to train verdicts.

USAGE
─────
    # Smoke (1 domain, small subset)
    python train_supervised.py --domain drugs --limit 500

    # Full training, both domains (~2-4h on A10)
    python train_supervised.py --domain drugs
    python train_supervised.py --domain weapon

    # Encode only (after training)
    python train_supervised.py --domain drugs --encode-only
"""
from __future__ import annotations
import argparse, random, time, json
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
DATA_CSV  = HERE / "data" / "supervised_data.csv"
OUT_DIR   = HERE / "outputs_supervised"


def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def load_data(domain, limit=None, seed=42):
    df = pd.read_csv(DATA_CSV)
    df = df[df.domain == domain].dropna(subset=["indictment_facts"]).reset_index(drop=True)
    if limit: df = df.head(limit).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    n_train = int(0.8 * len(df))
    train_idx = sorted(perm[:n_train].tolist())
    test_idx  = sorted(perm[n_train:].tolist())
    df["split"] = "test"
    df.loc[train_idx, "split"] = "train"
    print(f"  {domain}: total={len(df):,}, train={len(train_idx):,}, test={len(test_idx):,}")
    return df


def build_positive_pairs(df, thr_pos, max_pairs, mode="threshold", topk_per_anchor=20, seed=42):
    """Build positive pairs in train set.

    mode='threshold': pairs with |Δlow|≤thr AND |Δhigh|≤thr (random sample if >max_pairs)
    mode='topk':      for each anchor, take top-K closest by sqrt(Δlow² + Δhigh²)
                      → balanced coverage, scale-adaptive, no threshold
    """
    train = df[df.split == "train"].reset_index(drop=True)
    n = len(train)
    los = train.sentencing_range_low.values.astype(float)
    his = train.sentencing_range_high.values.astype(float)
    rng = np.random.default_rng(seed)

    if mode == "threshold":
        pos_pairs = []
        for i in range(n):
            d_lo = np.abs(los[i+1:] - los[i])
            d_hi = np.abs(his[i+1:] - his[i])
            keep = (d_lo <= thr_pos) & (d_hi <= thr_pos)
            for j_offset in np.where(keep)[0]:
                pos_pairs.append((i, i + 1 + int(j_offset)))
        print(f"  [threshold] positive pairs (|Δlow|≤{thr_pos} AND |Δhigh|≤{thr_pos}): {len(pos_pairs):,}")
        if len(pos_pairs) > max_pairs:
            idx = rng.choice(len(pos_pairs), size=max_pairs, replace=False)
            pos_pairs = [pos_pairs[k] for k in idx]
            print(f"  sampled to: {len(pos_pairs):,}")

    elif mode == "topk":
        # For each anchor, take top-K closest by Euclidean distance in (low, high) space
        pos_pairs = []
        for i in range(n):
            d_lo = los - los[i]
            d_hi = his - his[i]
            dist = np.sqrt(d_lo * d_lo + d_hi * d_hi)
            dist[i] = np.inf  # exclude self
            top = np.argpartition(dist, topk_per_anchor)[:topk_per_anchor]
            for j in top:
                if i < j:    pos_pairs.append((i, int(j)))
                elif i > j:  pos_pairs.append((int(j), i))
        # Dedupe (i,j) — same pair may appear from both anchors
        pos_pairs = list(set(pos_pairs))
        print(f"  [topk] positive pairs (each anchor's top-{topk_per_anchor} closest): {len(pos_pairs):,}")
        if len(pos_pairs) > max_pairs:
            idx = rng.choice(len(pos_pairs), size=max_pairs, replace=False)
            pos_pairs = [pos_pairs[k] for k in idx]
            print(f"  sampled to: {len(pos_pairs):,}")

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return train, pos_pairs


def build_model(base_model, max_seq_len):
    word = modules.Transformer(base_model, max_seq_length=max_seq_len)
    pool = modules.Pooling(word.get_embedding_dimension(), pooling_mode="cls")
    return SentenceTransformer(modules=[word, pool])


def train(model, train_df, pos_pairs, batch_size, grad_accum, epochs, lr, device, precision, model_dir):
    anchors   = [train_df.indictment_facts.iloc[i] for i, _ in pos_pairs]
    positives = [train_df.indictment_facts.iloc[j] for _, j in pos_pairs]
    ds = Dataset.from_dict({"anchor": anchors, "positive": positives})

    loss_fn = losses.MultipleNegativesRankingLoss(model)
    eff_batch = batch_size * grad_accum
    args = SentenceTransformerTrainingArguments(
        output_dir=str(model_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.1,
        bf16=(precision == "bf16"),
        fp16=(precision == "fp16"),
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        dataloader_drop_last=True,
        remove_unused_columns=False,
    )
    print(f"\n=== Supervised training ===")
    print(f"  examples         : {len(ds):,}")
    print(f"  physical batch   : {batch_size}")
    print(f"  grad accum       : {grad_accum}")
    print(f"  effective batch  : {eff_batch}")
    print(f"  epochs           : {epochs}")
    print(f"  lr               : {lr}")
    print(f"  precision        : {precision}")
    print(f"  device           : {device}")

    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss_fn)
    t0 = time.time()
    trainer.train()
    print(f"  ✓ trained in {(time.time()-t0)/60:.1f} min")
    model.save(str(model_dir))
    print(f"  saved → {model_dir}")


def encode_all(model, df, batch_size, out_dir, domain):
    print(f"\n=== Encoding {len(df):,} verdicts ({domain}) ===")
    t0 = time.time()
    emb = model.encode(df.indictment_facts.tolist(), batch_size=max(batch_size, 16),
                       convert_to_numpy=True, show_progress_bar=True,
                       normalize_embeddings=True)
    print(f"  ✓ encoded in {(time.time()-t0)/60:.1f} min  shape {emb.shape}")
    np.save(out_dir / f"verdict_embeddings_{domain}.npy", emb)
    df[["verdict","domain","split"]].to_csv(out_dir / f"verdict_index_{domain}.csv", index=False)
    print(f"  saved → {out_dir}/verdict_embeddings_{domain}.npy ({emb.nbytes/1e6:.1f} MB)")
    print(f"          {out_dir}/verdict_index_{domain}.csv  (with train/test split)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain",      required=True, choices=["drugs", "weapon"])
    p.add_argument("--base-model",  default="dicta-il/dictabert")
    p.add_argument("--mode",        choices=["threshold","topk"], default="threshold",
                   help="threshold: |Δlow|≤thr AND |Δhigh|≤thr; topk: each anchor's K closest by Euclidean distance")
    p.add_argument("--thr-pos",     type=int,   default=6,
                   help="(threshold mode only) Months. Positive pair iff |Δlow|≤thr AND |Δhigh|≤thr")
    p.add_argument("--topk-per-anchor", type=int, default=20,
                   help="(topk mode only) Each anchor gets its top-K closest neighbors as positives")
    p.add_argument("--max-pairs",   type=int,   default=200_000,
                   help="Sample positive pairs to this many if exceeded")
    p.add_argument("--batch-size",  type=int,   default=8)
    p.add_argument("--grad-accum",  type=int,   default=8)
    p.add_argument("--epochs",      type=int,   default=2)
    p.add_argument("--lr",          type=float, default=3e-5)
    p.add_argument("--max-seq-len", type=int,   default=256)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--limit",       type=int,   default=None,
                   help="Smoke test on first N verdicts")
    p.add_argument("--encode-only", action="store_true")
    p.add_argument("--precision",   choices=["bf16","fp16","fp32"], default="bf16")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Mode-specific output dir to keep model variants separate
    suffix = f"_{args.mode}" if args.mode == "topk" else ""
    model_dir = OUT_DIR / f"model_{args.domain}{suffix}"
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)

    print(f"=== {args.domain.upper()} supervised contrastive ===")
    df = load_data(args.domain, limit=args.limit, seed=args.seed)

    device = pick_device()
    precision = args.precision if device == "cuda" else "fp32"

    if args.encode_only:
        if not model_dir.exists():
            raise FileNotFoundError(f"--encode-only needs saved model at {model_dir}")
        model = SentenceTransformer(str(model_dir), device=device)
    else:
        train_df, pos_pairs = build_positive_pairs(
            df, args.thr_pos, args.max_pairs,
            mode=args.mode, topk_per_anchor=args.topk_per_anchor, seed=args.seed)
        if len(pos_pairs) < 100:
            raise ValueError(f"Too few positive pairs ({len(pos_pairs)}) — relax thr-pos")
        model = build_model(args.base_model, args.max_seq_len)
        model.to(device)
        train(model, train_df, pos_pairs, args.batch_size, args.grad_accum,
              args.epochs, args.lr, device, precision, model_dir)

    encode_all(model, df, args.batch_size, OUT_DIR, args.domain + suffix)
    print("\n✓ Done.")
    print(f"\nNext: python train_supervised.py --domain "
          f"{'weapon' if args.domain=='drugs' else 'drugs'}")


if __name__ == "__main__":
    main()
