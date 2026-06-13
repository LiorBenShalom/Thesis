"""Probe several clustering thresholds using cached embeddings; show how a couple
of known families split/merge so we can pick a threshold that groups true
synonyms without merging distinct predicates that share boilerplate words."""
import json
from pathlib import Path
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from common import load_rep, FOCUS
from inventory import normalize

CACHE = json.loads((Path(__file__).parent / "out" / "emb_cache.json").read_text())

def keys_for(domain):
    s = set()
    for rep in FOCUS:
        for d in load_rep(domain, rep).values():
            s.update(normalize(k) for k in d.keys())
    return sorted(s)

PROBES = {
    "drugs": ["הנאשם_פעל_לבד", "משקל_הסם", "שווי_הסם", "תמורה_כספית"],
    "weapon": ["שימוש_בנשק", "החזקת_הנשק", "היה_רישיון_לנשק", "הייתה_כוונה_להשתמש_בנשק"],
}

for domain in ["drugs", "weapon"]:
    keys = [k for k in keys_for(domain) if k in CACHE]
    X = np.array([CACHE[k] for k in keys], np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    print("=" * 80, "\nDOMAIN", domain, " (", len(keys), "names )")
    for thr in [0.14, 0.18, 0.22, 0.26, 0.30]:
        lab = AgglomerativeClustering(n_clusters=None, distance_threshold=thr,
                                      metric="cosine", linkage="average").fit_predict(X)
        nclust = len(set(lab))
        k2c = {k: c for k, c in zip(keys, lab)}
        # size of the cluster that each probe lands in
        from collections import Counter
        sizes = Counter(lab)
        info = []
        for p in PROBES[domain]:
            if p in k2c:
                info.append(f"{p}->{sizes[k2c[p]]}")
        print(f"  thr={thr:.2f}  #clusters={nclust:4d}  probe_cluster_sizes: " + " | ".join(info))
PY = None
