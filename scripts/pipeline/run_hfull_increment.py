#!/usr/bin/env python3
"""
Incremental H-Full runner — re-uses batch_hfull.py UNCHANGED, only re-points
its path constants and feeds it a targeted CLEAN_CSV.

Why a wrapper (not editing batch_hfull):
  - batch_hfull's hardcoded DATA_DIR = .../experiments/data/sentencing_range
    was renamed to sentencing_range-old since the original run.
  - We must MERGE into the existing schema_cache_drugs.json / hybrid_full_cache.json
    (keyed by Hebrew canonical id), not overwrite them.
  - We must process ONLY the targeted increment, not the whole corpus.

Usage:
  run_hfull_increment.py build-target            # build the 192-drugs CSV
  run_hfull_increment.py schema --submit
  run_hfull_increment.py schema --process
  run_hfull_increment.py enrich --submit
  run_hfull_increment.py enrich --process
  run_hfull_increment.py status
"""
import sys, json, shutil
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!")
LIVE = ROOT / "new_try/experiments/data/sentencing_range-old"
HF_DIR = LIVE / "hfull_features"
TARGET_CSV = HF_DIR / "verdicts_clean_hfull_192.csv"
PIPE = ROOT / "new_try/experiments/scripts/pipeline"
sys.path.insert(0, str(PIPE))


def build_target():
    clean = pd.read_csv(ROOT / "new_try/innovation_submission/data_master_final/verdicts_clean.csv")
    sup = set(pd.read_csv(ROOT / "new_try/simcse_cuda_bundle/data/supervised_data.csv")
              .verdict.astype(str))
    hf = set(map(str, json.load(open(
        ROOT / "new_try/simcse_cuda_bundle/data/hybrid_full_cache.json")).keys()))
    clean["canonical_id"] = clean.canonical_id.astype(str)
    tgt = clean[(~clean.canonical_id.isin(sup)) & (~clean.canonical_id.isin(hf))].copy()
    assert (tgt.domain == "drugs").all(), "expected all-drugs target"
    assert (tgt.sentencing_confidence == "גבוהה").all(), "expected all high-conf"
    # batch_hfull keys the cache by row['verdict']; the existing 3,942 cache is
    # Hebrew-canonical-keyed, so align verdict := canonical_id (input scoping,
    # not a method change).
    tgt["verdict"] = tgt["canonical_id"]
    HF_DIR.mkdir(parents=True, exist_ok=True)
    tgt.to_csv(TARGET_CSV, index=False)
    print(f"✅ {len(tgt)} rows -> {TARGET_CSV}")
    print(tgt.domain.value_counts().to_string())


def _patch_and_run(argv):
    import batch_hfull as bh
    # Re-point every path constant at the live -old location + targeted CSV.
    bh.DATA_DIR = LIVE
    bh.OUT_DIR = HF_DIR
    bh.BATCH_DIR = HF_DIR / "batch"
    bh.VERDICT_CSV_DIR = LIVE / "verdict_csv"
    bh.CLEAN_CSV = TARGET_CSV
    bh.BATCH_DIR.mkdir(parents=True, exist_ok=True)

    # One-time provenance backup of the original-run batch artifacts.
    bk = bh.BATCH_DIR / "_orig_backup"
    if not bk.exists():
        bk.mkdir()
        for pat in ("state_schema_drugs.json", "schema_drugs_counts.json",
                    "schema_drugs_input.jsonl", "state_enrich.json"):
            src = bh.BATCH_DIR / pat
            if src.exists():
                shutil.copy2(src, bk / pat)
        print(f"🗄  backed up original batch artifacts -> {bk}")

    # Clean up the empty stale dir batch_hfull created at import time.
    stale = ROOT / "new_try/experiments/data/sentencing_range"
    if stale.exists():
        try:
            shutil.rmtree(stale)
        except OSError:
            pass

    sys.argv = ["batch_hfull.py"] + argv
    bh.main()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "build-target":
        build_target()
    else:
        _patch_and_run(sys.argv[1:])
