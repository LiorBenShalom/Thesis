#!/usr/bin/env python3
"""
SUPERVISED contrastive with OFFENSE-OVERLAP FILTER.

Same as train_supervised.py, but adds a filter so every positive pair shares
at least one offense label. Pipeline:
  (1) Top-K=20 nearest neighbors per anchor in Euclidean (low, high) space
  (2) FILTER: drop pairs where offense_sets(anchor) ∩ offense_sets(neighbor) is empty
  (3) Drop anchors that end up with zero positives (they're excluded from training)

Offense labels are derived from the H-Full cache:
  - drugs:  section_6/7/13/14/19 / other_drug_offense (drugs-schema flags)
            fallback: parse offense_number for drug-section refs if drugs-schema
            is absent
  - weapon: parse offense_number / offense_type / additional_offenses for
            144 subsections + 145 + 146

USAGE
─────
    python train_supervised_filtered.py --domain drugs  --mode topk
    python train_supervised_filtered.py --domain weapon --mode topk
    python train_supervised_filtered.py --domain drugs  --mode topk --fold 1
"""
from __future__ import annotations
import argparse, random, time, json, re
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
HFULL_JSON = HERE / "data" / "hybrid_full_cache.json"   # bundled with the data folder
OUT_DIR   = HERE / "outputs_supervised_filtered"


# ============================================================
# Offense-set extraction (drugs / weapon)
# ============================================================
def yesno(v):
    if v is None: return False
    s = str(v).strip()
    return s not in ("", "לא", "nan", "None", "0", "0.0")


def drugs_offense_set(feats):
    """Drug-offense labels from H-Full. Handles both drugs-schema and weapon-schema."""
    if not feats: return set()
    s = set()
    # Primary: drugs-schema flags
    for sec in ("6", "7", "13", "14", "19"):
        if yesno(feats.get(f"section_{sec}")): s.add(f"sec_{sec}")
    if yesno(feats.get("other_drug_offense")): s.add("other")
    if s: return s
    # Fallback: drug sections embedded in weapon-schema fields
    blob = " ".join(str(feats.get(k, "")) for k in ("offense_number", "offense_type", "additional_offenses"))
    if not re.search(r"סם מסוכן|פקודת הסמים|חוק הסמים|קנבי?ס|קוקאין|הרואין|אקסטזי|MDMA|מתאמפטמין|סחר בסם|החזקת סם|ייבוא סם", blob):
        return s
    for sec in ("6", "7", "13", "14", "19"):
        if re.search(rf"\b{sec}\b[^\d]*?(?:לפקודה|לפקודת הסמים|פקודת הסמים)", blob):
            s.add(f"sec_{sec}")
    on = str(feats.get("offense_number", ""))
    ot = str(feats.get("offense_type", ""))
    if re.search(r"סם מסוכן|סחר בסם|החזקת סם|ייבוא סם|גידול סם|פקודת הסמים", ot):
        for tok in re.findall(r"\b(\d+)\b", on):
            if tok in ("6", "7", "13", "14", "19"):
                s.add(f"sec_{tok}")
        if re.search(r"\b7\s*\(?[אג]\)?", on): s.add("sec_7")
    return s


WPAT = [
    (r"144\s*\(\s*א\s*\)",       "144a"),
    (r"144\s*\(\s*ב\s*2\s*\)",   "144b2"),
    (r"144\s*\(\s*ב\s*\)",       "144b"),
    (r"144\s*\(\s*ג\s*\)",       "144c"),
    (r"144\s*\(\s*ז\s*\)",       "144g"),
    (r"\b145\b",                  "145"),
    (r"\b146\b",                  "146"),
]
def weapon_offense_set(feats):
    if not feats: return set()
    blob = " ".join(str(feats.get(k, "")) for k in ("offense_number", "offense_type", "additional_offenses"))
    return {label for pat, label in WPAT if re.search(pat, blob)}


def get_offense_set(verdict, hf_cache, domain):
    feats = hf_cache.get(verdict, {})
    return drugs_offense_set(feats) if domain == "drugs" else weapon_offense_set(feats)


# ============================================================
# Setup
# ============================================================
def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def load_data(domain, limit=None, seed=42, fold=None, n_folds=5):
    df = pd.read_csv(DATA_CSV)
    df = df[df.domain == domain].dropna(subset=["indictment_facts"]).reset_index(drop=True)
    if limit: df = df.head(limit).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    if fold is None:
        n_train = int(0.8 * len(df))
        train_idx = sorted(perm[:n_train].tolist())
        test_idx  = sorted(perm[n_train:].tolist())
        print(f"  {domain}: 80/20 split → train={len(train_idx):,}, test={len(test_idx):,}")
    else:
        assert 1 <= fold <= n_folds, f"fold must be 1..{n_folds}"
        fold_size = len(df) // n_folds
        start = (fold - 1) * fold_size
        end   = start + fold_size if fold < n_folds else len(df)
        test_idx  = sorted(perm[start:end].tolist())
        train_idx = sorted([i for i in range(len(df)) if i not in set(test_idx)])
        print(f"  {domain} fold {fold}/{n_folds}: train={len(train_idx):,}, test={len(test_idx):,}")
    df["split"] = "test"
    df.loc[train_idx, "split"] = "train"
    return df


def build_positive_pairs(df, thr_pos, max_pairs, mode="topk", topk_per_anchor=20,
                         seed=42, hf_cache=None, domain=None, max_distance=12.0):
    """Build positive pairs with offense-overlap filter + backfill.

    For each anchor with non-empty offense-set, walk through neighbors in
    ascending Euclidean (low, high) distance and collect up to K=20 neighbors
    that share at least one offense label AND are within `max_distance` months.
    If fewer than K=20 qualify within the cap, take what's available.
    """
    train = df[df.split == "train"].reset_index(drop=True)
    n = len(train)
    los = train.sentencing_range_low.values.astype(float)
    his = train.sentencing_range_high.values.astype(float)
    rng = np.random.default_rng(seed)

    offense_sets = [get_offense_set(v, hf_cache, domain) for v in train.verdict]
    n_empty = sum(1 for s in offense_sets if not s)
    print(f"  train verdicts with empty offense-set: {n_empty:,} / {n:,}  "
          f"({n_empty/n*100:.1f}%)  ← excluded as anchors (still encoded at end)")

    if mode == "topk":
        # ===== BACKFILL: per-anchor, collect K offense-sharing positives within distance cap =====
        pair_set = set()
        K_per_anchor = np.zeros(n, dtype=int)
        max_dist_taken_per_anchor = np.full(n, -1.0)
        for i in range(n):
            if not offense_sets[i]:
                continue
            d_lo = los - los[i]; d_hi = his - his[i]
            dist = np.sqrt(d_lo * d_lo + d_hi * d_hi); dist[i] = np.inf
            order = np.argsort(dist)
            collected = 0
            for j in order:
                if collected >= topk_per_anchor: break
                if dist[j] > max_distance: break  # ← cap
                sj = offense_sets[int(j)]
                if not sj: continue
                if not (offense_sets[i] & sj): continue
                a, b = (i, int(j)) if i < int(j) else (int(j), i)
                pair_set.add((a, b))
                K_per_anchor[i] = K_per_anchor[i] + 1
                collected += 1
                max_dist_taken_per_anchor[i] = dist[j]
        filtered_pairs = list(pair_set)

        # Stats
        non_empty = (np.array([len(s) for s in offense_sets]) > 0)
        eligible_anchors = int(non_empty.sum())
        reached_K = int((K_per_anchor[non_empty] >= topk_per_anchor).sum())
        partial   = int(((K_per_anchor[non_empty] > 0) & (K_per_anchor[non_empty] < topk_per_anchor)).sum())
        zero      = int((K_per_anchor[non_empty] == 0).sum())

        print(f"  [topk+backfill, cap={max_distance:.0f} months]")
        print(f"  eligible anchors (non-empty offense-set): {eligible_anchors:,}")
        print(f"    ↳ reached K={topk_per_anchor}: {reached_K:,}  ({reached_K/eligible_anchors*100:.1f}%)")
        print(f"    ↳ got 1..{topk_per_anchor-1} (partial): {partial:,}  ({partial/eligible_anchors*100:.1f}%)")
        print(f"    ↳ got 0 (no offense match within cap):  {zero:,}  ({zero/eligible_anchors*100:.1f}%)")
        active_K = K_per_anchor[non_empty]
        if eligible_anchors > 0:
            print(f"  K per active anchor: median={int(np.median(active_K))}  "
                  f"mean={active_K.mean():.1f}  min={int(active_K.min())}  max={int(active_K.max())}")
        print(f"  unique positive pairs after backfill: {len(filtered_pairs):,}")

        # Distance stats
        valid_dists = max_dist_taken_per_anchor[max_dist_taken_per_anchor >= 0]
        if len(valid_dists):
            print(f"  max-distance-to-K-th-positive per anchor:")
            print(f"    median={np.median(valid_dists):.1f}  p75={np.percentile(valid_dists,75):.1f}  "
                  f"p90={np.percentile(valid_dists,90):.1f}  p95={np.percentile(valid_dists,95):.1f}  "
                  f"max={valid_dists.max():.1f}")

        pos_pairs = filtered_pairs
        if len(pos_pairs) > max_pairs:
            idx = rng.choice(len(pos_pairs), size=max_pairs, replace=False)
            pos_pairs = [pos_pairs[k] for k in idx]
            print(f"  sampled to: {len(pos_pairs):,}")

    elif mode == "threshold":
        pos_pairs = []
        n_raw = 0
        n_dropped_empty = 0
        n_dropped_disjoint = 0
        for i in range(n):
            d_lo = np.abs(los[i+1:] - los[i])
            d_hi = np.abs(his[i+1:] - his[i])
            keep = (d_lo <= thr_pos) & (d_hi <= thr_pos)
            for j_offset in np.where(keep)[0]:
                j = i + 1 + int(j_offset)
                n_raw += 1
                sa, sb = offense_sets[i], offense_sets[j]
                if not sa or not sb:
                    n_dropped_empty += 1; continue
                if not (sa & sb):
                    n_dropped_disjoint += 1; continue
                pos_pairs.append((i, j))
        print(f"  [threshold] raw pairs (|Δlow|≤{thr_pos} AND |Δhigh|≤{thr_pos}):  {n_raw:,}")
        print(f"  [filter] pairs dropped (one side empty offenses):   {n_dropped_empty:,}")
        print(f"  [filter] pairs dropped (disjoint offense sets):     {n_dropped_disjoint:,}")
        print(f"  [filter] pairs RETAINED:                             {len(pos_pairs):,}  "
              f"({len(pos_pairs)/max(n_raw,1)*100:.1f}% of raw)")
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
    print(f"\n=== Supervised training (offense-filtered) ===")
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
    p.add_argument("--mode",        choices=["threshold","topk"], default="topk")
    p.add_argument("--thr-pos",     type=int,   default=6)
    p.add_argument("--topk-per-anchor", type=int, default=20)
    p.add_argument("--max-pairs",   type=int,   default=200_000)
    p.add_argument("--batch-size",  type=int,   default=8)
    p.add_argument("--grad-accum",  type=int,   default=8)
    p.add_argument("--epochs",      type=int,   default=2)
    p.add_argument("--lr",          type=float, default=3e-5)
    p.add_argument("--max-seq-len", type=int,   default=256)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--limit",       type=int,   default=None)
    p.add_argument("--encode-only", action="store_true")
    p.add_argument("--precision",   choices=["bf16","fp16","fp32"], default="bf16")
    p.add_argument("--fold",        type=int, default=None)
    p.add_argument("--n-folds",     type=int, default=5)
    p.add_argument("--hfull-json",  type=str, default=str(HFULL_JSON),
                   help="Path to hybrid_full_cache.json")
    p.add_argument("--max-distance", type=float, default=12.0,
                   help="Max Euclidean distance (months) to accept a positive (cap for backfill). Default 12.")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    if args.mode == "topk": parts.append("topk")
    if args.fold is not None: parts.append(f"fold{args.fold}")
    parts.append("offenseFiltered")
    suffix = "_" + "_".join(parts) if parts else ""
    model_dir = OUT_DIR / f"model_{args.domain}{suffix}"
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)

    print(f"=== {args.domain.upper()} supervised contrastive (OFFENSE-FILTERED) ===")
    print(f"H-Full cache: {args.hfull_json}")
    with open(args.hfull_json) as f:
        hf_cache = json.load(f)
    print(f"  loaded {len(hf_cache):,} verdicts with H-Full features\n")

    df = load_data(args.domain, limit=args.limit, seed=args.seed,
                   fold=args.fold, n_folds=args.n_folds)

    device = pick_device()
    precision = args.precision if device == "cuda" else "fp32"

    if args.encode_only:
        if not model_dir.exists():
            raise FileNotFoundError(f"--encode-only needs saved model at {model_dir}")
        model = SentenceTransformer(str(model_dir), device=device)
    else:
        train_df, pos_pairs = build_positive_pairs(
            df, args.thr_pos, args.max_pairs,
            mode=args.mode, topk_per_anchor=args.topk_per_anchor, seed=args.seed,
            hf_cache=hf_cache, domain=args.domain, max_distance=args.max_distance)
        if len(pos_pairs) < 100:
            raise ValueError(f"Too few positive pairs after filter ({len(pos_pairs)})")
        model = build_model(args.base_model, args.max_seq_len)
        model.to(device)
        train(model, train_df, pos_pairs, args.batch_size, args.grad_accum,
              args.epochs, args.lr, device, precision, model_dir)

    encode_all(model, df, args.batch_size, OUT_DIR, args.domain + suffix)
    print("\n✓ Done.")
    print(f"\nNext: python train_supervised_filtered.py --domain "
          f"{'weapon' if args.domain=='drugs' else 'drugs'} --mode topk")


if __name__ == "__main__":
    main()
