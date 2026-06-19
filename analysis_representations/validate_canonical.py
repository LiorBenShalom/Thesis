"""Prove the robustness scorer reproduces the CANONICAL pipeline: re-score pairs in
run-A orientation (fv1=feature_vector_1) and compare to the STORED canonical `score`
in the v6_final prediction file. If |Δ| is on the order of the temp-0 self-jitter
(A5) and not systematic, the pipeline is identical and the ONLY manipulation in A1
is the order swap. Uses the exact feature vectors FROM the canonical preds file."""
from __future__ import annotations
import os, sys, csv, argparse, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent
for line in (EXP / ".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))
sys.path.insert(0, str(EXP / "src" / "scoring"))
sys.path.insert(0, str(ROOT / "code"))
from structured_llm_comparison_experiment import (   # noqa: E402
    USER_TEMPLATE_SCORE_RAW, parse_score_v6,
    SYSTEM_PROMPT_V6_SCORE_RAW_drugs, SYSTEM_PROMPT_V6_SCORE_RAW_wep,
)
import similarity_experiment as se                    # noqa: E402
csv.field_size_limit(10**9)
CALL = {"gpt4": se.call_gpt4_1, "gpt52": se.call_gpt52}
SYS = {"drugs": SYSTEM_PROMPT_V6_SCORE_RAW_drugs, "weapon": SYSTEM_PROMPT_V6_SCORE_RAW_wep}


def validate_score(s):  # identical to canonical v6 validate_score
    try:
        v = float(s)
    except (TypeError, ValueError):
        return False
    return v == v and 0.0 <= v <= 100.0


def score(model, domain, fv1, fv2):
    up = USER_TEMPLATE_SCORE_RAW.format(fv1=fv1, fv2=fv2)
    for _ in range(3):
        try:
            raw = CALL[model](SYS[domain], up, log_call=False)
            v = parse_score_v6(raw)
            if validate_score(v):
                return float(v)
        except Exception:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="canonical *_preds.csv path")
    ap.add_argument("--model", default="gpt52")
    ap.add_argument("--domain", default="drugs")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.preds, encoding="utf-8-sig")))[: a.n]
    print(f"validate {a.model}/{a.domain} vs canonical {Path(a.preds).name} (N={len(rows)})")
    diffs = []
    for r in rows:
        canon = float(r["score"]) if r.get("score") not in (None, "", "nan") else None
        mine = score(a.model, a.domain, r["feature_vector_1"], r["feature_vector_2"])
        if canon is None or mine is None:
            print(f"  {r['verdict_1'][:18]} x {r['verdict_2'][:18]}: canon={canon} mine={mine} (skip)")
            continue
        d = abs(mine - canon); diffs.append(d)
        print(f"  {r['verdict_1'][:18]} x {r['verdict_2'][:18]}: canon={canon:5.1f}  mine={mine:5.1f}  |Δ|={d:.0f}")
    if diffs:
        print(f"\n  mean|Δ| run-A vs canonical = {st.mean(diffs):.2f}  max={max(diffs):.0f}  "
              f"identical={sum(1 for d in diffs if d==0)}/{len(diffs)}")
        print(f"  (compare to A5 temp-0 self-jitter for this model/domain; "
              f"similar magnitude ⇒ identical pipeline)")


if __name__ == "__main__":
    main()
