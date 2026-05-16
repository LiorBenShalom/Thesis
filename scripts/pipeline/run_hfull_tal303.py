#!/usr/bin/env python3
"""
H-Full runner for the 303 NEW tal-data survivors.

Same as run_hfull_increment.py but:
  - CLEAN_CSV       -> tal_303_hfull_target.csv (215 drugs + 88 weapon)
  - VERDICT_CSV_DIR -> the tal verdict_csv dir (tal_new_verdict_csv)
batch_hfull.py itself is UNCHANGED — only its path constants are re-pointed.
Merges into the SAME schema_cache_{domain}.json + hybrid_full_cache.json.

Usage:
  run_hfull_tal303.py schema --domain drugs  --submit | --process
  run_hfull_tal303.py schema --domain weapon --submit | --process
  run_hfull_tal303.py enrich --submit | --process
  run_hfull_tal303.py status
"""
import sys, shutil
from pathlib import Path

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!")
LIVE = ROOT / "new_try/experiments/data/sentencing_range-old"
HF_DIR = LIVE / "hfull_features"
TARGET_CSV = HF_DIR / "tal_303_hfull_target.csv"
TAL_VERDICT_CSV = ROOT / "new_try/innovation_submission/output/tal_new_verdict_csv"
PIPE = ROOT / "new_try/experiments/scripts/pipeline"
sys.path.insert(0, str(PIPE))


def _patch_and_run(argv):
    import batch_hfull as bh
    bh.DATA_DIR = LIVE
    bh.OUT_DIR = HF_DIR
    bh.BATCH_DIR = HF_DIR / "batch"
    bh.VERDICT_CSV_DIR = TAL_VERDICT_CSV
    bh.CLEAN_CSV = TARGET_CSV
    bh.BATCH_DIR.mkdir(parents=True, exist_ok=True)

    # remove the empty stale dir batch_hfull creates at import time
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
    _patch_and_run(sys.argv[1:])
