#!/usr/bin/env python3
"""
Annotator lists + Cohen's kappa (linear/quadratic) for similarity, once you copy
per-tagger CSVs (e.g. Guy.csv, Itay.csv) from Google Drive into --similarity-dir.

manual_fe.csv alone has no annotator column — it cannot answer "who tagged whom".

Example:
  python agreement_analysis.py \\
    --weapon-manual "../data/wep/manual_fe.csv" \\
    --drugs-manual "../data/drugs/manual_fe.csv" \\
    --similarity-dir ./tagger_csvs_drugs

Optional: --feature-csv with columns case_id, annotator (from Forms export).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    cohen_kappa_score = None


def _norm_vid(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    t = str(s).strip()
    if len(t) >= 2 and t[0:2].lower() in ("me", "sh"):
        t = t[0:2].upper() + t[2:]
    return t


def pair_key(a: str, b: str) -> Tuple[str, str]:
    x, y = _norm_vid(a), _norm_vid(b)
    return (x, y) if x <= y else (y, x)


def detect_columns(df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
    if "verdict_1" in df.columns and "verdict_2" in df.columns:
        v1, v2 = "verdict_1", "verdict_2"
    else:
        raise ValueError(f"Need verdict_1, verdict_2. Got: {list(df.columns)}")

    sim_col = None
    for name in ("similarity_scale", "similarity", "דמיון", "scale"):
        if name in df.columns:
            sim_col = name
            break
    return v1, v2, sim_col


def load_tagger_folder(folder: Path) -> Dict[str, pd.DataFrame]:
    out = {}
    for p in sorted(folder.glob("*.csv")):
        if p.stem.lower().endswith("_assignments"):
            continue
        try:
            out[p.stem] = pd.read_csv(p, encoding="utf-8")
        except UnicodeDecodeError:
            out[p.stem] = pd.read_csv(p, encoding="utf-8-sig")
    return out


def build_pair_maps(
    tagger_dfs: Dict[str, pd.DataFrame],
) -> Tuple[Dict[Tuple[str, str], Set[str]], Dict[str, Dict[Tuple[str, str], float]]]:
    pair_to_ann: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    ann_scores: Dict[str, Dict[Tuple[str, str], float]] = defaultdict(dict)

    for ann, df in tagger_dfs.items():
        v1c, v2c, simc = detect_columns(df)
        for _, r in df.iterrows():
            k = pair_key(r[v1c], r[v2c])
            if not k[0]:
                continue
            pair_to_ann[k].add(ann)
            if simc and pd.notna(r.get(simc)):
                try:
                    ann_scores[ann][k] = float(r[simc])
                except (TypeError, ValueError):
                    pass
    return dict(pair_to_ann), dict(ann_scores)


def kappa_table(ann_scores: Dict[str, Dict[Tuple[str, str], float]], names: List[str]) -> List[dict]:
    if cohen_kappa_score is None:
        return []
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            keys = set(ann_scores.get(a, {})) & set(ann_scores.get(b, {}))
            if len(keys) < 2:
                rows.append(
                    {
                        "a": a,
                        "b": b,
                        "n": len(keys),
                        "kappa_lin": None,
                        "kappa_q": None,
                    }
                )
                continue
            sk = sorted(keys)
            ya = [ann_scores[a][k] for k in sk]
            yb = [ann_scores[b][k] for k in sk]
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "n": len(keys),
                    "kappa_lin": float(cohen_kappa_score(ya, yb, weights="linear")),
                    "kappa_q": float(cohen_kappa_score(ya, yb, weights="quadratic")),
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weapon-manual", type=Path)
    ap.add_argument("--drugs-manual", type=Path)
    ap.add_argument("--similarity-dir", type=Path, required=True)
    ap.add_argument("--feature-csv", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    out = args.out_dir or (base / "out")
    out.mkdir(parents=True, exist_ok=True)

    tagger_dfs = load_tagger_folder(args.similarity_dir)
    if not tagger_dfs:
        raise SystemExit(f"No CSVs in {args.similarity_dir}")

    pair_to_ann, ann_scores = build_pair_maps(tagger_dfs)
    pd.DataFrame(
        [
            {"verdict_1": a, "verdict_2": b, "annotators": ",".join(sorted(v))}
            for (a, b), v in sorted(pair_to_ann.items())
        ]
    ).to_csv(out / "pairs_who_tagged.csv", index=False)

    names = sorted(tagger_dfs.keys())
    pd.DataFrame(kappa_table(ann_scores, names)).to_csv(out / "cohen_kappa_similarity.csv", index=False)

    for label, path in ("weapon", args.weapon_manual), ("drugs", args.drugs_manual):
        if path is None or not path.exists():
            continue
        manual = pd.read_csv(path, encoding="utf-8")
        pairs = {
            pair_key(r["verdict_1"], r["verdict_2"])
            for _, r in manual.iterrows()
            if "verdict_1" in manual.columns
        }
        rows = []
        for pk in sorted(pairs):
            anns = sorted(pair_to_ann.get(pk, set()))
            rows.append(
                {
                    "verdict_1": pk[0],
                    "verdict_2": pk[1],
                    "n_taggers": len(anns),
                    "taggers": ",".join(anns),
                }
            )
        pd.DataFrame(rows).to_csv(out / f"manual_pairs_x_taggers_{label}.csv", index=False)
        n2 = sum(1 for pk in pairs if len(pair_to_ann.get(pk, set())) >= 2)
        (out / f"summary_{label}.json").write_text(
            json.dumps(
                {
                    "domain": label,
                    "pairs_in_manual": len(pairs),
                    "pairs_with_2plus_taggers": n2,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    if args.feature_csv and args.feature_csv.exists():
        fe = pd.read_csv(args.feature_csv, encoding="utf-8")
        cid = "case_id" if "case_id" in fe.columns else fe.columns[0]
        an = "annotator" if "annotator" in fe.columns else fe.columns[1]
        g = fe.groupby(cid)[an].apply(lambda s: ",".join(sorted(set(map(str, s)))))
        g.reset_index().to_csv(out / "cases_who_extracted_features.csv", index=False)

    print("Done:", out)


if __name__ == "__main__":
    main()
