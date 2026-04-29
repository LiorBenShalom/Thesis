#!/usr/bin/env python3
"""
Paper-style (Table 9) sentencing-range prediction pipeline.

Faithfully reproduces predict_sentencing_range.py setup:
  - filter to citation-linked pairs only (1-hop or 2-hop in the citation graph)
  - aggregation = weighted_mean with weights = sim/100
  - sim >= THR, k >= 3 (at least 3 such neighbors)
  - +σ-filter: keep only verdicts with σ_combined ≤ Q50

Usage:
  predict_paper_style.py --sim-csv <csv> --rep <label> --out-dir <dir>
                         [--thr 60] [--use-corrected-canonical]

If --use-corrected-canonical is set, citation graph uses best_lookup() (canonical
+ alias fallback) — this is the FIXED version that catches the ת"פ_/תפ_ bug.
Otherwise uses the original to_canon() logic (which had the bug).
"""
from __future__ import annotations
import argparse, json, re, unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
DATA_MASTER = ROOT / "innovation_submission/data_master_final/verdicts_clean.csv"
ALIAS = ROOT / "innovation_submission/data_master_final/verdict_alias.csv"
UNI = ROOT / "innovation_submission/output/all_domains_unified.csv"


def canonical(s):
    if not s or pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKC", str(s).strip())
    s = re.sub(r'["\'״׳`]', '', s)
    s = re.sub(r'\(.*?\)', '', s)
    s = re.sub(r'[\s/∕\\.]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_- ')
    return s


def build_citation_graph(use_corrected: bool):
    """Same-domain 1-hop + 2-hop citation graph.
    use_corrected: best_lookup (alias + canonical fallback) — fixes the bug.
                   else: original to_canon (alias OR canonical, but NOT both).
    """
    clean = pd.read_csv(DATA_MASTER)
    canon_set = set(clean.canonical_id)
    canon_to_dom = dict(zip(clean.canonical_id, clean.domain))
    alias = pd.read_csv(ALIAS)
    orig_to_canon = dict(zip(alias.original_id.astype(str), alias.canonical_id.astype(str)))

    def to_canon_orig(s):  # original buggy logic from predict_sentencing_range.py
        return orig_to_canon.get(str(s)) or canonical(s)

    def best_lookup(t):  # corrected
        if t in canon_set: return t
        c = canonical(t)
        if c in canon_set: return c
        a = orig_to_canon.get(t)
        if a:
            if a in canon_set: return a
            ac = canonical(a)
            if ac in canon_set: return ac
        return None

    out_e = defaultdict(set)
    for _, row in clean.iterrows():
        src = row.canonical_id
        raw = row.citations_json
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            arr = json.loads(raw)
        except Exception:
            continue
        for c in arr:
            tgt_raw = c.get("cited_case", "")
            if use_corrected:
                tgt = best_lookup(tgt_raw)
            else:
                tgt = to_canon_orig(tgt_raw)
                if tgt not in canon_set:
                    tgt = None
            if tgt and tgt != src and canon_to_dom.get(src) == canon_to_dom.get(tgt):
                out_e[src].add(tgt)

    def has_1hop(a, b):
        return a in out_e.get(b, set()) or b in out_e.get(a, set())

    def has_2hop(a, b):
        if any(b in out_e.get(X, set()) for X in out_e.get(a, set())):
            return True
        return any(a in out_e.get(X, set()) for X in out_e.get(b, set()))

    return canon_to_dom, has_1hop, has_2hop


def load_targets():
    clean = pd.read_csv(DATA_MASTER)
    df = clean[["canonical_id", "domain", "sentencing_range_low", "sentencing_range_high"]].copy()
    df = df.dropna(subset=["sentencing_range_low", "sentencing_range_high"])
    df = df.drop_duplicates(subset=["canonical_id"])
    return df.set_index("canonical_id")


def predict_one(sim_csv: Path, rep_label: str, thr: float,
                citation_only: bool, use_corrected: bool, targets: pd.DataFrame,
                thr_per_dom: dict | None = None):
    sims = pd.read_csv(sim_csv, usecols=["verdict_1", "verdict_2", "domain", "similarity_score"])
    sims = sims.dropna(subset=["similarity_score"])

    if citation_only:
        canon_to_dom, has_1hop, has_2hop = build_citation_graph(use_corrected)
        # Keep only citation-linked pairs
        keep = sims.apply(lambda r: has_1hop(r.verdict_1, r.verdict_2) or has_2hop(r.verdict_1, r.verdict_2),
                          axis=1)
        sims = sims[keep]
        print(f"  citation-linked pairs after filter: {len(sims):,}")
    else:
        canon_to_dom = dict(zip(targets.index, targets.domain))

    # Build neighbors per query
    ngh = defaultdict(list)
    for v1, v2, dom, s in sims.itertuples(index=False):
        ngh[v1].append((v2, float(s)))
        ngh[v2].append((v1, float(s)))

    rows = []
    for q, cands in ngh.items():
        if q not in targets.index:
            continue
        q_dom = targets.at[q, "domain"]
        local_thr = (thr_per_dom or {}).get(q_dom, thr)
        good = [(n, s) for n, s in cands if n in targets.index and s >= local_thr
                and targets.at[n, "domain"] == q_dom]
        if len(good) < 3:
            continue
        ws = np.array([s / 100 for _, s in good])
        if ws.sum() == 0:
            continue
        lo = np.array([targets.at[n, "sentencing_range_low"] for n, _ in good])
        hi = np.array([targets.at[n, "sentencing_range_high"] for n, _ in good])
        pl = float(np.average(lo, weights=ws))
        ph = float(np.average(hi, weights=ws))
        rows.append({
            "verdict": q,
            "domain": targets.at[q, "domain"],
            "rep": rep_label,
            "n_neighbors": len(good),
            "actual_low":  float(targets.at[q, "sentencing_range_low"]),
            "actual_high": float(targets.at[q, "sentencing_range_high"]),
            "pred_low": pl,
            "pred_high": ph,
            "sigma_low":  float(lo.std()),
            "sigma_high": float(hi.std()),
            "mean_sim": float(np.mean([s for _, s in good])),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame()

    df["err_low"]  = (df["pred_low"]  - df["actual_low"]).abs()
    df["err_high"] = (df["pred_high"] - df["actual_high"]).abs()
    df["sig_combined"] = df["sigma_low"] + df["sigma_high"]
    inter = np.maximum(0, np.minimum(df.pred_high, df.actual_high) - np.maximum(df.pred_low, df.actual_low))
    union = np.maximum(df.pred_high, df.actual_high) - np.minimum(df.pred_low, df.actual_low)
    df["iou"] = (inter / np.maximum(union, 1)).astype(float)

    # Per-domain metrics with and without σ-filter (Q50 of sig_combined)
    metric_rows = []
    for dom, sub in df.groupby("domain"):
        for sig in ["no_sigma", "with_sigma"]:
            ev = sub
            if sig == "with_sigma":
                cut = sub["sig_combined"].quantile(0.5)
                ev = sub[sub["sig_combined"] <= cut]
            if len(ev) == 0:
                continue
            metric_rows.append({
                "rep": rep_label, "domain": dom, "thr": thr, "sigma": sig,
                "n": len(ev), "avg_n_neighbors": float(ev["n_neighbors"].mean()),
                "MAE_low":  float(ev["err_low"].mean()),
                "MAE_high": float(ev["err_high"].mean()),
                "MedAE_low":  float(ev["err_low"].median()),
                "MedAE_high": float(ev["err_high"].median()),
                "IoU": float(ev["iou"].mean()),
            })
    return df, pd.DataFrame(metric_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-csv", required=True, type=Path)
    ap.add_argument("--rep", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--thr", type=float, default=60.0)
    ap.add_argument("--thr-drugs", type=float, default=None,
                    help="Override threshold for drugs (e.g., per-rep percentile-equivalent)")
    ap.add_argument("--thr-weapon", type=float, default=None,
                    help="Override threshold for weapon")
    ap.add_argument("--no-citation-filter", action="store_true",
                    help="Skip citation-linked filter (use ALL pairs in sim_csv)")
    ap.add_argument("--use-corrected-canonical", action="store_true",
                    help="Use bug-fixed best_lookup for citation graph (default: original to_canon)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = load_targets()
    thr_per_dom = None
    if args.thr_drugs is not None or args.thr_weapon is not None:
        thr_per_dom = {"drugs": args.thr_drugs if args.thr_drugs is not None else args.thr,
                       "weapon": args.thr_weapon if args.thr_weapon is not None else args.thr}
    preds, metrics = predict_one(
        args.sim_csv, args.rep, args.thr,
        citation_only=not args.no_citation_filter,
        use_corrected=args.use_corrected_canonical,
        targets=targets,
        thr_per_dom=thr_per_dom,
    )
    safe = args.rep.replace(" ", "_").replace("/", "_")
    suffix = f"_thr{int(args.thr)}"
    if args.no_citation_filter:
        suffix += "_allpairs"
    if args.use_corrected_canonical:
        suffix += "_corrected"
    p_path = args.out_dir / f"preds_{safe}{suffix}.csv"
    m_path = args.out_dir / f"metrics_{safe}{suffix}.csv"
    preds.to_csv(p_path, index=False)
    metrics.to_csv(m_path, index=False)
    print(f"Saved {len(preds)} predictions -> {p_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
