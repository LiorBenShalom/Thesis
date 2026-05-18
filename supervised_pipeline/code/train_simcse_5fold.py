#!/usr/bin/env python3
"""
SimCSE — 5-fold, holdout-correct, on the 4,432 canonical corpus.

Mirrors train_supervised_filtered.py's fold contract so SimCSE is rigor-
consistent and directly comparable to sup_only / sup+LLM:
  - train/test split is READ from the EXACT filtered fold index files
    (outputs_supervised_filtered/verdict_index_{dom}_topk_fold{f}_offenseFiltered.csv)
    => SimCSE test set == rigor test set, no representation leakage
    (SimCSE trained on fold-TRAIN text only; test encoded by train-only model).
  - PURE unsupervised: positives = same indictment_facts twice (dropout),
    in-batch negatives (MultipleNegativesRankingLoss / InfoNCE).

Output (outputs_simcse_5fold/):
  verdict_embeddings_simcse_{dom}_fold{f}.npy
  verdict_index_simcse_{dom}_fold{f}.csv   (verdict, domain, split)

Usage:
  python train_simcse_5fold.py --domain drugs  --fold 1
  (runner run_5fold_cv_simcse.sh does all 10)
"""
from __future__ import annotations
import argparse, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer, SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments)
from sentence_transformers.sentence_transformer import losses, modules

HERE = Path(__file__).resolve().parent
DATA_CSV = HERE / "data" / "supervised_data.csv"          # 4,432 canonical
FOLD_DIR = HERE / "outputs_supervised_filtered"            # read splits from here
OUT_DIR  = HERE / "outputs_simcse_5fold"
N_FOLDS  = 5


def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def build_model(base_model, max_seq_len):
    word = modules.Transformer(base_model, max_seq_length=max_seq_len)
    pool = modules.Pooling(word.get_embedding_dimension(), pooling_mode="cls")
    return SentenceTransformer(modules=[word, pool])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True, choices=["drugs", "weapon"])
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    p.add_argument("--base-model", default="dicta-il/dictabert")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-seq-len", type=int, default=256)   # match filtered
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", choices=["bf16","fp16","fp32"], default="bf16")
    a = p.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed); np.random.seed(a.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = pick_device()

    sup = pd.read_csv(DATA_CSV)
    sup["verdict"] = sup.verdict.astype(str)
    txt = dict(zip(sup.verdict, sup.indictment_facts))

    fi = FOLD_DIR / f"verdict_index_{a.domain}_topk_fold{a.fold}_offenseFiltered.csv"
    if not fi.exists():
        raise FileNotFoundError(f"need filtered fold index: {fi}")
    idx = pd.read_csv(fi); idx["verdict"] = idx.verdict.astype(str)
    train_ids = idx[idx.split == "train"].verdict.tolist()
    train_txt = [txt[v] for v in train_ids
                 if v in txt and isinstance(txt[v], str) and txt[v].strip()]
    print(f"=== SimCSE {a.domain} fold {a.fold}/{a.n_folds} ===")
    print(f"  train texts: {len(train_txt):,}  | encode all: {len(idx):,}")

    model = build_model(a.base_model, a.max_seq_len)
    if dev == "cuda": model = model.to("cuda")
    ds = Dataset.from_dict({"anchor": train_txt, "positive": train_txt})
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(OUT_DIR / f"_ckpt_{a.domain}_fold{a.fold}"),
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, warmup_ratio=0.1,
        bf16=(a.precision == "bf16"), fp16=(a.precision == "fp16"),
        logging_steps=10, save_strategy="no", report_to="none",
        dataloader_drop_last=True, remove_unused_columns=False,
    )
    t0 = time.time()
    SentenceTransformerTrainer(model=model, args=args,
                               train_dataset=ds, loss=loss_fn).train()
    print(f"  ✓ trained in {(time.time()-t0)/60:.1f} min")

    enc_df = idx[idx.verdict.isin(txt)].copy()
    enc_df["indictment_facts"] = enc_df.verdict.map(txt)
    emb = model.encode(enc_df.indictment_facts.tolist(),
                       batch_size=max(a.batch_size, 16),
                       show_progress_bar=True, normalize_embeddings=False)
    np.save(OUT_DIR / f"verdict_embeddings_simcse_{a.domain}_fold{a.fold}.npy", emb)
    enc_df[["verdict","domain","split"]].to_csv(
        OUT_DIR / f"verdict_index_simcse_{a.domain}_fold{a.fold}.csv", index=False)
    print(f"  saved → verdict_embeddings_simcse_{a.domain}_fold{a.fold}.npy "
          f"{emb.shape}  + index (with split)")


if __name__ == "__main__":
    main()
