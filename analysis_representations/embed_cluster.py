"""Semantic clustering of feature names across representations (per domain).

Collapses synonymous field names (e.g. סעיף_חוק ~ סעיפי חיקוק ~ סעיפי_חוק) into
CONCEPTS via OpenAI embeddings + agglomerative clustering. Then measures, per
domain: concept coverage per representation, cross-rep overlap, unique concepts.

Embeddings are cached to out/emb_cache.json so re-runs are free.
"""
from __future__ import annotations
import json, os, csv
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from common import load_rep, FOCUS, DOMAINS
from inventory import normalize

HERE = Path(__file__).parent
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
CACHE = OUT / "emb_cache.json"

# distance threshold: cosine distance. 0.18 chosen after a sweep (see tune_threshold.py):
# groups genuine synonyms while avoiding merging distinct predicates that share
# boilerplate Hebrew words (e.g. the "הנאשם פעל X" family).
THRESHOLD = float(os.environ.get("CLUSTER_THRESH", "0.18"))
EMB_MODEL = "text-embedding-3-small"


def _load_key():
    for p in [HERE.parents[1] / "experiments" / ".env",
              HERE.parents[2] / ".env"]:
        if p.exists():
            for line in p.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"')
    return os.environ.get("OPENAI_API_KEY", "")


def embed_all(strings: list[str]) -> dict[str, list[float]]:
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [s for s in strings if s not in cache]
    if todo:
        from openai import OpenAI
        client = OpenAI(api_key=_load_key())
        B = 256
        for i in range(0, len(todo), B):
            batch = todo[i:i + B]
            resp = client.embeddings.create(model=EMB_MODEL, input=batch)
            for s, d in zip(batch, resp.data):
                cache[s] = d.embedding
            print(f"  embedded {min(i+B,len(todo))}/{len(todo)}")
        CACHE.write_text(json.dumps(cache))
    return {s: cache[s] for s in strings}


def cluster_domain(domain: str):
    # per rep: verdict -> set(normalized keys); and global key->freq
    rep_data = {}
    all_keys = set()
    orig_freq = Counter()          # normalized_key -> Counter of ORIGINAL names (for readable labels)
    norm2orig = defaultdict(Counter)
    for rep in FOCUS:
        raw = load_rep(domain, rep)
        norm = {}
        for vid, d in raw.items():
            nk = set()
            for k in d.keys():
                nn = normalize(k)
                nk.add(nn)
                norm2orig[nn][k.strip()] += 1
            norm[vid] = nk
            all_keys.update(nk)
        rep_data[rep] = norm
    keys = sorted(all_keys)
    emb = embed_all(keys)
    X = np.array([emb[k] for k in keys], dtype=np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    cl = AgglomerativeClustering(
        n_clusters=None, distance_threshold=THRESHOLD,
        metric="cosine", linkage="average")
    labels = cl.fit_predict(X)
    key2cluster = {k: int(c) for k, c in zip(keys, labels)}

    # cluster -> member keys, and a representative label = highest total verdict-freq
    cluster_members = defaultdict(list)
    for k, c in key2cluster.items():
        cluster_members[c].append(k)

    # concept coverage per rep: # verdicts in rep that contain >=1 key of cluster
    rep_concept_cov = {rep: Counter() for rep in FOCUS}
    rep_nverdicts = {rep: len(rep_data[rep]) for rep in FOCUS}
    key_total_freq = Counter()
    for rep in FOCUS:
        for vid, kset in rep_data[rep].items():
            seen = set()
            for k in kset:
                key_total_freq[k] += 1
                c = key2cluster[k]
                if c not in seen:
                    rep_concept_cov[rep][c] += 1
                    seen.add(c)

    def rep_label(c):
        # pick the most frequent ORIGINAL (un-normalised) name across the cluster
        best = max(cluster_members[c], key=lambda k: key_total_freq[k])
        if norm2orig[best]:
            return norm2orig[best].most_common(1)[0][0]
        return best

    concepts = []
    for c, members in cluster_members.items():
        cov = {rep: rep_concept_cov[rep][c] for rep in FOCUS}
        reps_present = [rep for rep in FOCUS if cov[rep] > 0]
        concepts.append({
            "cluster": c, "label": rep_label(c), "n_member_names": len(members),
            "n_reps": len(reps_present), "reps": "|".join(reps_present),
            **{f"cov_{rep}": cov[rep] for rep in FOCUS},
            "members": " ; ".join(sorted(members, key=lambda k: -key_total_freq[k])[:12]),
        })
    concepts.sort(key=lambda r: (-r["n_reps"], -sum(r[f"cov_{rep}"] for rep in FOCUS)))
    return concepts, rep_nverdicts, rep_data, key2cluster


def main():
    for domain in DOMAINS:
        print("=" * 80); print("DOMAIN:", domain)
        concepts, nv, rep_data, k2c = cluster_domain(domain)
        # save concept table + key->cluster map for reuse
        cols = ["cluster", "label", "n_member_names", "n_reps", "reps"] + \
               [f"cov_{r}" for r in FOCUS] + ["members"]
        with open(OUT / f"concepts_{domain}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(concepts)
        (OUT / f"key2cluster_{domain}.json").write_text(
            json.dumps({"key2cluster": k2c,
                        "labels": {str(c["cluster"]): c["label"] for c in concepts}},
                       ensure_ascii=False))

        n_concepts = len(concepts)
        print(f"  total CONCEPTS (after semantic merge) = {n_concepts}")
        # synonym fragmentation: names per concept, per rep
        print("  name fragmentation (distinct NAMES per CONCEPT, within rep):")
        for r in FOCUS:
            names = sum(c["n_member_names"] for c in concepts if c[f"cov_{r}"] > 0)
            cons = sum(1 for c in concepts if c[f"cov_{r}"] > 0)
            print(f"      {r:13s}: {names} names / {cons} concepts "
                  f"= {names/cons:.2f} names per concept")
        # per-rep TOP concepts (what GPT actually chose, by coverage)
        for r in FOCUS:
            top = sorted([c for c in concepts if c[f"cov_{r}"] > 0],
                         key=lambda c: -c[f"cov_{r}"])[:15]
            print(f"  TOP concepts — {r} (coverage / {nv[r]} verdicts):")
            for c in top:
                print(f"      {c[f'cov_{r}']:3d}  {c['label']}")
        # per-rep concept count, unique concepts
        per_rep_concepts = {r: set() for r in FOCUS}
        for con in concepts:
            for r in FOCUS:
                if con[f"cov_{r}"] > 0:
                    per_rep_concepts[r].add(con["cluster"])
        print(f"  {'rep':14s} {'#concepts':>9s} {'unique':>7s} {'core>=50%':>9s}")
        for r in FOCUS:
            cset = per_rep_concepts[r]
            uniq = sum(1 for con in concepts
                       if con[f"cov_{r}"] > 0 and con["n_reps"] == 1)
            core = sum(1 for con in concepts if con[f"cov_{r}"] >= 0.5 * nv[r])
            print(f"  {r:14s} {len(cset):9d} {uniq:7d} {core:9d}")

        # pairwise concept overlap (Jaccard) among the 4 reps
        print("  pairwise concept overlap (shared concepts / Jaccard%):")
        for i, a in enumerate(FOCUS):
            for b in FOCUS[i+1:]:
                A, B = per_rep_concepts[a], per_rep_concepts[b]
                inter = len(A & B); uni = len(A | B)
                print(f"      {a:13s} ∩ {b:13s}: shared={inter:3d}  "
                      f"Jaccard={100*inter/uni:.0f}%")
        print("  saved ->", OUT / f"concepts_{domain}.csv")


if __name__ == "__main__":
    main()
