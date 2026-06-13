"""Inventory + stability analysis of feature KEYS per (domain, representation).

Answers: how many distinct field names? how many appear in only 1 verdict
(unstable, GPT invented a one-off name)? how stable is the de-facto schema?
Also collapses near-duplicate names by normalisation to estimate true vocabulary.
"""
from __future__ import annotations
import csv, re, statistics as st
from collections import Counter
from pathlib import Path
from common import load_rep, FOCUS, DOMAINS

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

# Strip ONLY the yes/no question marker "האם" — a pure grammatical prefix with no
# content. We deliberately do NOT strip content words like מספר/סוג/כמות/מקום/אופן,
# because they distinguish DIFFERENT schema fields (e.g. "מספר עבירה" = legal-section
# number vs "סוג עבירה" = offence type; "כמות תחמושת" vs "סוג תחמושת"). Stripping them
# collapsed distinct fields into one concept. True synonyms are merged by the embedding
# clustering, not by prefix stripping.
_PREFIX = re.compile(r"^(האם)[_\s]")


def normalize(name: str) -> str:
    n = name.strip().strip('"').replace("״", "").replace('"', "")
    n = re.sub(r"[\s\-]+", "_", n)
    n = re.sub(r"_+", "_", n).strip("_")
    n2 = _PREFIX.sub("", n)
    return n2 if len(n2) >= 3 else n


def run():
    rows = []
    for dom in DOMAINS:
        print("=" * 78)
        print(f"DOMAIN = {dom}")
        for rep in FOCUS:
            data = load_rep(dom, rep)
            nv = len(data)
            freq = Counter()
            per_verdict_sizes = []
            for d in data.values():
                per_verdict_sizes.append(len(d))
                for k in d.keys():
                    freq[k] += 1
            distinct = len(freq)
            singletons = sum(1 for k, c in freq.items() if c == 1)
            core_half = sum(1 for k, c in freq.items() if c >= nv * 0.5)
            core_90 = sum(1 for k, c in freq.items() if c >= nv * 0.9)
            norm = Counter()
            for k, c in freq.items():
                norm[normalize(k)] += c
            distinct_norm = len(norm)
            med_size = st.median(per_verdict_sizes) if per_verdict_sizes else 0
            print(f"\n  {rep}")
            print(f"    verdicts={nv}  median_keys/verdict={med_size:.0f}  "
                  f"total_key_instances={sum(freq.values())}")
            print(f"    distinct_keys={distinct}  (after name-normalisation={distinct_norm})")
            print(f"    singletons(appear in 1 verdict)={singletons} "
                  f"({100*singletons/distinct:.0f}%)")
            print(f"    core>=50% verdicts={core_half}   core>=90% verdicts={core_90}")
            for k, c in freq.most_common(15):
                print(f"        {c:4d}/{nv}  {k}")
            rows.append({
                "domain": dom, "rep": rep, "verdicts": nv,
                "median_keys_per_verdict": med_size,
                "distinct_keys": distinct, "distinct_keys_normalized": distinct_norm,
                "singletons": singletons, "singleton_pct": round(100*singletons/distinct, 1),
                "core_50pct": core_half, "core_90pct": core_90,
            })
    with open(OUT / "inventory_summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nsaved ->", OUT / "inventory_summary.csv")


if __name__ == "__main__":
    run()
