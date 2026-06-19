"""A1 (pair-order independence) + A5 (temp-0 run-to-run determinism) robustness
checks, using the CANONICAL v6 scoring (same system prompt, USER_TEMPLATE_SCORE_RAW,
parse_score_v6) so the result is valid for the paper's method.

For each sampled pair and model:
  - run A: score(fv1, fv2)
  - run B: score(fv1, fv2)        -> A5: |A-B| at temperature 0
  - run R: score(fv2, fv1)        -> A1: |A-R| under order swap
OpenAI models only (gpt4=gpt-4.1, gpt52=gpt-5.2) — the keys we have.
"""
from __future__ import annotations
import os, sys, csv, json, argparse, statistics as st
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
EXP = HERE.parent                      # .../experiments
ROOT = EXP.parent                      # .../new_try

# --- load OPENAI_API_KEY from experiments/.env into the environment ---
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
REP_FILE = "similarity_database_hybrid_full_gpt.csv"


def _valid(s):  # identical gate to canonical v6 validate_score: float, non-NaN, in [0,100]
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
            if _valid(v):
                return float(v)
        except Exception:
            pass
    return None


def load_pairs(domain, n):
    path = ROOT / domain / REP_FILE
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    # stratify by similarity_binary_0
    pos = [r for r in rows if str(r.get("similarity_binary_0", "0")).strip() == "1"]
    neg = [r for r in rows if str(r.get("similarity_binary_0", "0")).strip() != "1"]
    half = n // 2
    sample = pos[:half] + neg[:n - half]
    return sample[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="drugs")
    ap.add_argument("--model", default="gpt4", choices=list(CALL))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--workers", type=int, default=40)
    a = ap.parse_args()
    pairs = load_pairs(a.domain, a.n)
    print(f"model={a.model} domain={a.domain} pairs={len(pairs)} (canonical v6 scoring)")

    def work(idx_row):
        i, r = idx_row
        fv1, fv2 = r["feature_vector_1"], r["feature_vector_2"]
        a_ = score(a.model, a.domain, fv1, fv2)
        b_ = score(a.model, a.domain, fv1, fv2)   # A5: same order, 2nd run
        rev = score(a.model, a.domain, fv2, fv1)  # A1: swapped order
        return {"i": i, "gold": r.get("similarity_binary_0"),
                "A": a_, "B": b_, "R": rev}

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(work, enumerate(pairs)))

    out = HERE / "out" / f"robustness_{a.model}_{a.domain}.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["i", "gold", "A", "B", "R"]); w.writeheader(); w.writerows(res)

    def deltas(k1, k2):
        return [abs(r[k1] - r[k2]) for r in res if r[k1] is not None and r[k2] is not None]

    def flips(k1, k2, thr=50.0):
        ok = [(r[k1], r[k2]) for r in res if r[k1] is not None and r[k2] is not None]
        return sum(1 for x, y in ok if (x >= thr) != (y >= thr)), len(ok)

    a5 = deltas("A", "B"); a1 = deltas("A", "R")
    a5f = flips("A", "B"); a1f = flips("A", "R")
    nfail = sum(1 for r in res if None in (r["A"], r["B"], r["R"]))
    print(f"\n  parsed OK pairs: {len(res)-nfail}/{len(res)} (failed={nfail})")
    print(f"  A5 (run-to-run, temp 0): mean|Δ|={st.mean(a5):.2f}  max|Δ|={max(a5):.0f}  "
          f"identical={sum(1 for d in a5 if d==0)}/{len(a5)}  binary-flips={a5f[0]}/{a5f[1]}")
    print(f"  A1 (order swap):         mean|Δ|={st.mean(a1):.2f}  max|Δ|={max(a1):.0f}  "
          f"identical={sum(1 for d in a1 if d==0)}/{len(a1)}  binary-flips={a1f[0]}/{a1f[1]}")
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
