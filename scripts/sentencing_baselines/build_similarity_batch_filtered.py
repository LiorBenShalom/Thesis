#!/usr/bin/env python3
"""
LLM batch scoring for the OFFENSE-FILTERED supervised model.

For each fold (1..5), each domain (drugs, weapon):
  - test_query × top-K=20 train neighbors  (cosine over filtered embeddings)
Dedupes against:
  - ALL existing LLM-scored pairs (combined / simcse / supervised / 5fold v1)
  - All pairs currently IN-FLIGHT in similarity_batch_5fold_v2 (so we don't
    double-score what those batches will return)

Output:
  similarity_batch_filtered/similarity_input.jsonl
  similarity_batch_filtered/state.json
  similarity_batch_filtered/results/similarity_scores_filtered.csv

Usage:
  python3 build_similarity_batch_filtered.py prepare
  python3 build_similarity_batch_filtered.py submit
  python3 build_similarity_batch_filtered.py status
  python3 build_similarity_batch_filtered.py process
"""
from __future__ import annotations
import argparse, json, os, re, sys, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
EXP  = ROOT / "experiments"
OUT  = EXP / "data_per_domain"
WORK = OUT / "similarity_batch_filtered"
WORK.mkdir(exist_ok=True)
RESULTS = WORK / "results"
RESULTS.mkdir(exist_ok=True)

# Filtered model embeddings live in the bundle (not in experiments/simcse_outputs)
FILTERED_DIR = ROOT / "simcse_cuda_bundle/outputs_supervised_filtered"
INFLIGHT_V2  = OUT / "similarity_batch_5fold_v2" / "similarity_input.jsonl"

MODEL = "gpt-4.1"
MAX_BYTES_PER_BATCH = 100 * 1024 * 1024
N_FOLDS = 5
K_TOP = 20

# V6 prompts — IDENTICAL to all other batch scripts (do not change)
SYSTEM_DRUGS = """את/ה מומחית לדין הפלילי בישראל. מוצגים בפנייך פיצ'רים מובנים של שני תיקים פליליים.

המשימה: הערכת דמיון מהותי — עד כמה תיק אחד יכול לשמש כתקדים ענייני לשני?

נתחי כל ממד בנפרד:
1. **סוג העבירה וחומרתה** — האם מדובר באותו סוג עבירה? באותה רמת חומרה?
2. סוג הסם וכמותו — האם מדובר בסוג דומה בחומרתו? בכמויות דומות בסדר גודל?
3. **שיטת הביצוע (MO)** — דפוס דומה? אמצעים דומים?
4. **נסיבות הליבה** — מעבדה, תכנון, נסיבות מחמירות/מקלות?
5. **ישימות כתקדים** — האם בית משפט יראה את שני התיקים כרלוונטיים?

חשוב:
- התמקדי בפרמטרים המשפטיים המהותיים, לא בפרטים טכניים/ביורוקרטיים.
- השתייכות לאותו תחום לבדה לא מספיקה לדמיון מהותי.

פורמט תשובה:שורה אחת בדיוק בפורמט: SIMILARITY_SCORE: X
   כאשר X הוא מספר שלם בין 0 ל-100.
   0 = שונים לחלוטין, 100 = זהים כמעט. ציון מעל 50 = תקדים רלוונטי."""

SYSTEM_WEAPON = """את/ה מומחית לדין הפלילי בישראל. מוצגים בפנייך **פיצ'רים מובנים** של שני תיקים פליליים.

המשימה: הערכת דמיון מהותי — עד כמה תיק אחד יכול לשמש כתקדים ענייני לשני?

נתחי כל ממד בנפרד:
1. **סוג העבירה וחומרתה** — האם מדובר באותו סוג עבירה? באותה רמת חומרה?
2. **תפקיד הנאשם ומעורבותו** — יזם/סוחר לעומת שליח/מחזיק? מעורבות פעילה לעומת פסיבית?
3. **שיטת הביצוע (MO)** — דפוס דומה? אמצעים דומים?
4. **נסיבות הליבה** — כמות תחמושת (סדר גודל), שימוש בנשק, תכנון, נסיבות מחמירות/מקלות?
5. **ישימות כתקדים** — האם בית משפט יראה את שני התיקים כרלוונטיים זה לזה לצורכי ענישה?

חשוב:
- התמקדי בפרמטרים המשפטיים המהותיים, לא בפרטים טכניים/ביורוקרטיים.
- השתייכות לאותו תחום (סמים/נשק) לבדה **לא מספיקה** לדמיון מהותי.
- הבחנה בין 144(א) (החזקה בלבד) ל-144(ב) (נשיאה/הובלה) היא מהותית — תיק שהוא 144(א) לבדו כמעט אף פעם לא מהווה תקדים לתיק 144(ב).

פורמט תשובה:שורה אחת בדיוק בפורמט: SIMILARITY_SCORE: X
   כאשר X הוא מספר שלם בין 0 ל-100.
   0 = שונים לחלוטין, 100 = זהים כמעט. ציון מעל 50 = תקדים רלוונטי."""


def canonical(s):
    if not s or pd.isna(s): return ""
    s = unicodedata.normalize("NFKC", str(s).strip())
    s = re.sub(r'["\'״׳`]', '', s)
    s = re.sub(r'\(.*?\)', '', s)
    s = re.sub(r'[\s/∕\\.]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_- ')
    return s


def build_filtered_pairs(K):
    """test×top-K train neighbors across all folds, both domains, on FILTERED embeddings."""
    domain_of = {}
    all_pairs = set()
    for dom in ["drugs", "weapon"]:
        for fold in range(1, N_FOLDS + 1):
            ep = FILTERED_DIR / f"verdict_embeddings_{dom}_topk_fold{fold}_offenseFiltered.npy"
            ip = FILTERED_DIR / f"verdict_index_{dom}_topk_fold{fold}_offenseFiltered.csv"
            if not ep.exists() or not ip.exists():
                print(f"  ⚠ missing fold {fold} for {dom}: {ep.name}"); continue
            emb = np.load(ep).astype(np.float32)
            idx = pd.read_csv(ip)
            v2i = dict(zip(idx.verdict, range(len(idx))))
            train_ids = idx[idx.split == "train"].verdict.tolist()
            test_ids  = idx[idx.split == "test"].verdict.tolist()
            train_idx = np.array([v2i[v] for v in train_ids])
            for v in idx.verdict: domain_of[v] = dom
            for q in test_ids:
                qi = v2i[q]
                sims = emb[qi] @ emb[train_idx].T
                sims = np.nan_to_num(sims, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
                order = np.argsort(-sims)[:K]
                for j in order:
                    all_pairs.add(tuple(sorted([q, train_ids[j]])))
    print(f"  filtered top-{K} pairs (deduped across folds): {len(all_pairs):,}")

    # Dedup against all already-scored pairs
    existing = set()
    sources = [
        OUT / "similarity_scores_combined.csv",
        OUT / "similarity_batch_simcse/results/similarity_scores_simcse.csv",
        OUT / "similarity_batch_supervised/results/similarity_scores_supervised.csv",
        OUT / "similarity_batch_5fold/results/similarity_scores_5fold.csv",
        OUT / "similarity_batch_5fold_v2/results/similarity_scores_5fold_v2.csv",  # if processed
    ]
    for f in sources:
        if not f.exists():
            print(f"  ⚠ missing source: {f}"); continue
        df = pd.read_csv(f, usecols=["verdict_1", "verdict_2"])
        for r in df.itertuples(index=False):
            a, b = sorted([r.verdict_1, r.verdict_2])
            existing.add((a, b))
    print(f"  existing LLM-scored pairs: {len(existing):,}")

    # ALSO dedup against in-flight v2 batch pairs (they will be processed soon)
    inflight = set()
    if INFLIGHT_V2.exists():
        with open(INFLIGHT_V2) as f:
            for line in f:
                r = json.loads(line)
                cid = r["custom_id"]
                # custom_id format: "v1__v2__domain"
                parts = cid.rsplit("__", 1)
                v12 = parts[0].split("__", 1)
                if len(v12) == 2:
                    a, b = sorted(v12)
                    inflight.add((a, b))
        print(f"  in-flight v2 pairs (will be scored soon, dedup): {len(inflight):,}")

    skip = existing | inflight
    new_pairs = all_pairs - skip
    print(f"  NEW pairs to score (truly missing):   {len(new_pairs):,}")
    print(f"  already scored (skip):                {len(all_pairs & existing):,}")
    print(f"  will be scored by v2 (skip):          {len(all_pairs & inflight):,}")
    return new_pairs, domain_of


def build_features_lookup():
    hf_path = EXP / "data/sentencing_range-old/hfull_features/hybrid_full_cache.json"
    hfull = json.load(open(hf_path))

    alias = pd.read_csv(ROOT / "innovation_submission/data_master_final/verdict_alias.csv")
    orig_to_canon = dict(zip(alias.original_id.astype(str), alias.canonical_id.astype(str)))

    canon_to_features = {}
    for vid, feats in hfull.items():
        canon = orig_to_canon.get(vid) or canonical(vid)
        if isinstance(feats, dict) and "__error" not in feats:
            canon_to_features[canon] = json.dumps(feats, ensure_ascii=False)
    print(f"  H-Full features: {len(canon_to_features):,}")

    clean = pd.read_csv(ROOT / "innovation_submission/data_master_final/verdicts_clean.csv",
                        usecols=["canonical_id","indictment_facts"])
    canon_to_facts = dict(zip(clean.canonical_id.astype(str),
                              clean.indictment_facts.astype(str)))
    print(f"  indictment_facts fallback: {len(canon_to_facts):,}")
    return canon_to_features, canon_to_facts


def cmd_prepare(args):
    print(f"[1] Building filtered-model top-{K_TOP} test×train pairs...")
    pairs, domain_of = build_filtered_pairs(K_TOP)

    print(f"\n[2] Loading features...")
    feats_hfull, feats_facts = build_features_lookup()

    print(f"\n[3] Building JSONL...")
    jsonl_path = WORK / "similarity_input.jsonl"
    n_written = 0; n_missing = 0
    total_input_chars = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for a, b in pairs:
            dom = domain_of.get(a) or domain_of.get(b)
            if not dom: continue
            fa = feats_hfull.get(a) or feats_facts.get(a)
            fb = feats_hfull.get(b) or feats_facts.get(b)
            if not fa or not fb:
                n_missing += 1; continue
            sys_p = SYSTEM_DRUGS if dom == "drugs" else SYSTEM_WEAPON
            user = f"תיק 1:\n{fa}\n\nתיק 2:\n{fb}\n\nמהו ציון הדמיון המהותי (0-100)?"
            cid = f"{a}__{b}__{dom}"
            line = {
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role":"system","content":sys_p},
                        {"role":"user","content":user},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 80,
                },
            }
            total_input_chars += len(sys_p) + len(user)
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            n_written += 1
    sz_mb = jsonl_path.stat().st_size / 1024 / 1024
    avg_chars = total_input_chars / max(1, n_written)
    est_input_tokens = total_input_chars * 0.8
    est_output_tokens = n_written * 10

    print(f"\n  wrote {n_written:,} pairs ({sz_mb:.1f} MB) → {jsonl_path}")
    print(f"  skipped {n_missing:,} (missing features)")
    print(f"  avg chars per pair: {avg_chars:.0f}")
    print(f"  est input tokens:   {est_input_tokens:,.0f}")
    print(f"  est output tokens:  {est_output_tokens:,.0f}")
    print(f"\n  COST ESTIMATE (GPT-4.1 Batch API):")
    print(f"    input  @ $1.00 / 1M → ~${est_input_tokens/1e6 * 1.0:.2f}")
    print(f"    output @ $4.00 / 1M → ~${est_output_tokens/1e6 * 4.0:.2f}")
    print(f"    TOTAL: ~${est_input_tokens/1e6 * 1.0 + est_output_tokens/1e6 * 4.0:.2f}")
    print(f"\n  → To submit, run:  python3 build_similarity_batch_filtered.py submit")


def _load_env():
    if not os.getenv("OPENAI_API_KEY"):
        env_p = EXP / ".env"
        if env_p.exists():
            for line in env_p.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def cmd_submit(args):
    from openai import OpenAI
    _load_env()
    cli = OpenAI()
    jsonl = WORK / "similarity_input.jsonl"
    if not jsonl.exists(): sys.exit("Run 'prepare' first")
    sz = jsonl.stat().st_size

    if sz > MAX_BYTES_PER_BATCH:
        print(f"  Splitting {sz/1024/1024:.1f} MB...")
        chunks = []; idx = 1
        out = WORK / f"similarity_input.part{idx:02d}.jsonl"
        f_out = open(out, "w", encoding="utf-8"); chunks.append(out); bytes_w = 0
        with open(jsonl) as fi:
            for line in fi:
                b = len(line.encode("utf-8"))
                if bytes_w + b > MAX_BYTES_PER_BATCH and bytes_w > 0:
                    f_out.close(); idx += 1
                    out = WORK / f"similarity_input.part{idx:02d}.jsonl"
                    f_out = open(out, "w", encoding="utf-8"); chunks.append(out); bytes_w = 0
                f_out.write(line); bytes_w += b
        f_out.close()
    else:
        chunks = [jsonl]

    state = {"batch_ids": []}
    for i, chunk in enumerate(chunks, 1):
        sz_mb = chunk.stat().st_size / 1024 / 1024
        print(f"  [{i}/{len(chunks)}] uploading {chunk.name} ({sz_mb:.1f} MB)...")
        with open(chunk, "rb") as f:
            up = cli.files.create(file=f, purpose="batch")
        b = cli.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions",
                               completion_window="24h",
                               metadata={"description": f"filtered_supervised_topk20_chunk{i}"})
        print(f"     batch_id = {b.id}  status = {b.status}")
        state["batch_ids"].append(b.id)
    json.dump(state, open(WORK / "state.json", "w"), indent=2)


def cmd_status(args):
    from openai import OpenAI
    _load_env()
    cli = OpenAI()
    state = json.loads((WORK / "state.json").read_text()) if (WORK / "state.json").exists() else {}
    if "batch_ids" not in state: print("No state."); return
    for bid in state["batch_ids"]:
        b = cli.batches.retrieve(bid)
        print(f"  {bid}: {b.status}  {b.request_counts}")


def cmd_process(args):
    from openai import OpenAI
    _load_env()
    cli = OpenAI()
    state = json.loads((WORK / "state.json").read_text())
    rows = []
    for bid in state["batch_ids"]:
        b = cli.batches.retrieve(bid)
        if b.status != "completed":
            print(f"  {bid}: still {b.status}"); continue
        content = cli.files.content(b.output_file_id).content.decode("utf-8")
        for line in content.strip().split("\n"):
            r = json.loads(line)
            cid = r["custom_id"]; score = None
            if not r.get("error"):
                try:
                    txt = r["response"]["body"]["choices"][0]["message"]["content"] or ""
                    m = re.search(r"SIMILARITY_SCORE:\s*(\d+)", txt)
                    if m: score = int(m.group(1))
                except: pass
            parts = cid.split("__")
            if len(parts) >= 3:
                rows.append({"verdict_1": parts[0], "verdict_2": parts[1],
                             "domain": parts[2], "similarity_score": score})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "similarity_scores_filtered.csv", index=False)
    print(f"\n✅ saved {len(df):,} → {RESULTS/'similarity_scores_filtered.csv'}")
    print(f"   parsed: {df['similarity_score'].notna().sum():,}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("prepare")
    sub.add_parser("submit")
    sub.add_parser("status")
    sub.add_parser("process")
    a = p.parse_args()
    {
        "prepare": cmd_prepare,
        "submit":  cmd_submit,
        "status":  cmd_status,
        "process": cmd_process,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
