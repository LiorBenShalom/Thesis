"""Q3: did the hybrid ENRICHMENT add real depth or trivial/idiosyncratic detail?

For each hybrid rep we split fields into:
  - CORE     : the manual-schema backbone (present in ~all verdicts)
  - ENRICH   : everything the GPT added on top
and characterise the enrichment by:
  - recurrence (how many verdicts a concept spans; singletons = one-off)
  - value informativeness (fill-rate, distinct values, triviality)
  - whether it duplicates a legal-reasoning concept (overlap with GPT-Law)
Also compares Hybrid-Manual vs Hybrid-Full enrichment.
"""
from __future__ import annotations
import json, re, csv
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from common import load_rep, FOCUS, DOMAINS
from inventory import normalize

OUT = Path(__file__).parent / "out"
CACHE = json.loads((OUT / "emb_cache.json").read_text())
THRESHOLD = 0.18

TRIVIAL_VAL = re.compile(
    r"^\s*(לא|כן|אין|לא צוין|לא ידוע|לא רלוונטי|לא ברור|לא מפורט|none|n/?a|null|-|—)\s*$",
    re.IGNORECASE)


def cluster(domain):
    keysets = {}
    all_keys = set()
    for rep in FOCUS:
        keysets[rep] = {v: {normalize(k) for k in d} for v, d in load_rep(domain, rep).items()}
        for s in keysets[rep].values():
            all_keys |= s
    keys = [k for k in sorted(all_keys) if k in CACHE]
    X = np.array([CACHE[k] for k in keys], np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=THRESHOLD,
                                  metric="cosine", linkage="average").fit_predict(X)
    return {k: int(c) for k, c in zip(keys, lab)}, keysets


def analyze(domain):
    k2c, keysets = cluster(domain)
    print("=" * 80); print("DOMAIN:", domain)

    # GPT-Law concept set (legal-reasoning reference)
    law_concepts = set()
    for s in keysets["GPT-Law"].values():
        law_concepts |= {k2c[k] for k in s if k in k2c}

    rows = []
    for rep in ("Hybrid-Manual", "Hybrid-Full"):
        raw = load_rep(domain, rep)
        nv = len(raw)
        # concept coverage (verdicts per concept) and value stats per concept
        cov = Counter()
        vals = defaultdict(list)
        for vid, d in raw.items():
            seen = set()
            for k, v in d.items():
                c = k2c.get(normalize(k))
                if c is None:
                    continue
                if c not in seen:
                    cov[c] += 1; seen.add(c)
                vals[c].append("" if v is None else str(v).strip())

        core = {c for c, n in cov.items() if n >= 0.9 * nv}
        enrich = {c for c in cov if c not in core}
        singl = {c for c in enrich if cov[c] == 1}
        recur = {c for c in enrich if cov[c] >= 3}

        # value triviality among enrichment fields
        def trivial_rate(cset):
            tot = triv = 0
            for c in cset:
                for v in vals[c]:
                    tot += 1
                    if v == "" or TRIVIAL_VAL.match(v):
                        triv += 1
            return triv / tot if tot else 0

        # recurring enrichment that duplicates a legal concept already in GPT-Law
        recur_in_law = sum(1 for c in recur if c in law_concepts)

        print(f"\n  {rep}: {nv} verdicts")
        print(f"    CORE concepts (>=90% verdicts): {len(core)}")
        print(f"    ENRICHMENT concepts: {len(enrich)}")
        print(f"      - appear in exactly 1 verdict (one-off): {len(singl)} "
              f"({100*len(singl)/len(enrich):.0f}% of enrichment)")
        print(f"      - recurring (>=3 verdicts): {len(recur)} "
              f"({100*len(recur)/len(enrich):.0f}%)")
        print(f"    trivial/empty value-rate among enrichment: {100*trivial_rate(enrich):.0f}%")
        print(f"    recurring-enrichment concepts also present in GPT-Law: "
              f"{recur_in_law}/{len(recur)}")
        # show recurring enrichment (the part that can actually create cross-case matches)
        top_recur = sorted(recur, key=lambda c: -cov[c])[:20]
        # readable label = most common original name
        def label(c):
            names = Counter()
            for rr in FOCUS:
                for d in load_rep(domain, rr).values():
                    for k in d:
                        if k2c.get(normalize(k)) == c:
                            names[k.strip()] += 1
            return names.most_common(1)[0][0] if names else f"c{c}"
        print("    recurring enrichment concepts (coverage | sample values):")
        for c in top_recur:
            sample = [v for v in vals[c] if v and not TRIVIAL_VAL.match(v)][:3]
            print(f"        {cov[c]:3d}  {label(c):28s}  e.g. {sample}")

        rows.append({"domain": domain, "rep": rep, "verdicts": nv,
                     "core": len(core), "enrichment": len(enrich),
                     "oneoff": len(singl), "oneoff_pct": round(100*len(singl)/len(enrich)),
                     "recurring": len(recur), "trivial_val_pct": round(100*trivial_rate(enrich)),
                     "recur_in_law": recur_in_law})
    return rows


if __name__ == "__main__":
    allrows = []
    for d in DOMAINS:
        allrows += analyze(d)
    with open(OUT / "hybrid_depth_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
    print("\nsaved ->", OUT / "hybrid_depth_summary.csv")
