"""Score a list of verdict pairs with the LOCAL model, using the canonical v6
prompts. Checkpoint + resume + (small) concurrency. Safe to Ctrl-C and re-run.

Input  : a pairs CSV with columns verdict_1, verdict_2, domain
         feature lookup from data/features_hfull.json  (verdict_id -> H-Full dict)
Output : <out>.csv with verdict_1, verdict_2, domain, similarity_score, status, n_out_tokens
         (status: ok | parse_fail | no_features | error)

Examples
  # THE experiment: one domain, ALL pairs, no filter, number-only (exact sentencing prompt)
  python3 score_pairs.py --all-domain weapon --score-only \
          --out out/gemma_weapon_allpairs.csv --workers 3

  # resume: just run the same command again — finished pairs are skipped
  python3 score_pairs.py --all-domain weapon --score-only \
          --out out/gemma_weapon_allpairs.csv --workers 3

  # (alternative) a pre-built candidate set instead of all-pairs
  python3 score_pairs.py --pairs data/pairs_topk10.csv --score-only --out out/gemma_topk10.csv
"""
import argparse
import csv
import itertools
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from prompts import USER_TEMPLATE_SCORE_RAW, select, validate_score
import local_client as lc

csv.field_size_limit(10 ** 9)
HERE = os.path.dirname(os.path.abspath(__file__))

OUT_COLS = ["verdict_1", "verdict_2", "domain", "similarity_score", "status", "n_out_tokens"]


def load_features(path):
    cache = json.load(open(path, encoding="utf-8"))
    out = {}
    for vid, feats in cache.items():
        if isinstance(feats, dict) and "__error" not in feats:
            out[vid] = json.dumps(feats, ensure_ascii=False)
    return out


def load_done(out_path):
    """Return set of frozenset({v1,v2}) already scored OK, so resume can skip
    them regardless of pair order."""
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status") == "ok" and validate_score(r.get("similarity_score")):
                done.add(frozenset((r["verdict_1"], r["verdict_2"])))
    return done


def all_pairs_for_domain(domain):
    """Enumerate every unordered pair of usable verdicts in one domain, on the
    fly (no giant file). Reads data/verdict_domain.csv."""
    path = os.path.join(HERE, "data", "verdict_domain.csv")
    vids = sorted(r["verdict"] for r in csv.DictReader(open(path, encoding="utf-8-sig"))
                  if r["domain"] == domain)
    for a, b in itertools.combinations(vids, 2):
        yield {"verdict_1": a, "verdict_2": b, "domain": domain}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", help="CSV: verdict_1, verdict_2, domain")
    ap.add_argument("--all-domain", choices=["drugs", "weapon"], default=None,
                    help="instead of --pairs: score ALL unordered pairs of this domain")
    ap.add_argument("--out", required=True, help="output CSV (also the resume file)")
    ap.add_argument("--features", default=os.path.join(HERE, "data", "features_hfull.json"))
    ap.add_argument("--score-only", action="store_true",
                    help="use the EXACT sentencing prompt (one-line score, no analysis)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="cap output tokens (default 16 in --score-only, else server default)")
    ap.add_argument("--workers", type=int, default=2,
                    help="concurrent requests to the local server (keep small: 1-4)")
    ap.add_argument("--limit", type=int, default=0, help="score at most N pairs (0=all)")
    ap.add_argument("--shard", default=None, help="contiguous index range 'lo:hi' into the enumerated pairs")
    args = ap.parse_args()
    if not args.pairs and not args.all_domain:
        ap.error("give --pairs FILE or --all-domain {drugs,weapon}")

    max_tok = args.max_tokens if args.max_tokens is not None else (16 if args.score_only else None)

    feats = load_features(args.features)
    if args.all_domain:
        pairs = list(all_pairs_for_domain(args.all_domain))
        src_desc = f"ALL-PAIRS({args.all_domain})"
    else:
        pairs = list(csv.DictReader(open(args.pairs, encoding="utf-8-sig")))
        src_desc = os.path.basename(args.pairs)
    if args.shard:
        keep = []
        for rng in args.shard.split(","):
            lo, hi = rng.split(":")
            keep += pairs[int(lo):int(hi)]
        pairs = keep
        src_desc += f" [shard {args.shard}]"
    done = load_done(args.out)
    todo = [r for r in pairs if frozenset((r["verdict_1"], r["verdict_2"])) not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"server : {lc.info()}")
    print(f"prompt : {'SCORE-ONLY (sentencing, number-only)' if args.score_only else 'v6 (with analysis)'}"
          f" | max_tokens={max_tok}")
    print(f"source : {src_desc}")
    print(f"pairs  : {len(pairs)} total | {len(done)} already done | {len(todo)} to score")
    if not todo:
        print("nothing to do — all pairs already scored. ✅")
        return

    # open output in append mode; write header if new file
    new_file = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    fout = open(args.out, "a", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(fout, fieldnames=OUT_COLS)
    if new_file:
        writer.writeheader()
    wlock = threading.Lock()
    counters = {"ok": 0, "parse_fail": 0, "no_features": 0, "error": 0}
    clock = threading.Lock()
    t0 = time.time()

    def work(r):
        v1, v2, dom = r["verdict_1"], r["verdict_2"], r.get("domain", "")
        fv1, fv2 = feats.get(v1), feats.get(v2)
        if not fv1 or not fv2:
            return dict(verdict_1=v1, verdict_2=v2, domain=dom, similarity_score="",
                        status="no_features", n_out_tokens="")
        sys_p, template, parser = select(dom, args.score_only)
        user = template.format(fv1=fv1, fv2=fv2)
        try:
            text, usage = lc.call_local(sys_p, user, return_usage=True, max_tokens=max_tok)
            score = parser(text)
            ntok = usage.get("completion_tokens", "")
            if validate_score(score):
                return dict(verdict_1=v1, verdict_2=v2, domain=dom,
                            similarity_score=score, status="ok", n_out_tokens=ntok)
            return dict(verdict_1=v1, verdict_2=v2, domain=dom, similarity_score="",
                        status="parse_fail", n_out_tokens=ntok)
        except Exception:  # noqa: BLE001
            return dict(verdict_1=v1, verdict_2=v2, domain=dom, similarity_score="",
                        status="error", n_out_tokens="")

    def flush(rec):
        with wlock:
            writer.writerow(rec)
            fout.flush()
        with clock:
            counters[rec["status"]] = counters.get(rec["status"], 0) + 1
            n = sum(counters.values())
            if n % 25 == 0 or n == len(todo):
                el = time.time() - t0
                rate = n / el if el else 0
                eta = (len(todo) - n) / rate if rate else 0
                print(f"  {n}/{len(todo)} | ok={counters['ok']} "
                      f"parse_fail={counters['parse_fail']} err={counters['error']} "
                      f"nofeat={counters['no_features']} | "
                      f"{rate*3600:.0f}/h | ETA {eta/3600:.1f}h", flush=True)

    try:
        if args.workers <= 1:
            for r in todo:
                flush(work(r))
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for fut in as_completed([ex.submit(work, r) for r in todo]):
                    flush(fut.result())
    except KeyboardInterrupt:
        print("\ninterrupted — partial results saved; re-run the same command to resume.")
    finally:
        fout.close()

    el = time.time() - t0
    print(f"\ndone: {sum(counters.values())} scored in {el/3600:.2f}h | {counters}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
