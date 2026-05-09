#!/usr/bin/env python3
"""
Cross-task evaluation: can the SUPERVISED-on-sentencing model serve as a
similarity scorer too? (i.e., replace the gpt-4.1 LLM panel for similarity
prediction.)

Compares scorers against the manual GT (241 pairs labeled 1/2/3 by humans):
  - supervised cosine (per domain)
  - SimCSE cosine (unsupervised)
  - gpt-4.1 LLM score (the canonical paper baseline, where available)
  - citation strength (1hop=3 / 2hop=2 / cocite=1 / none=0)

Metrics: Spearman ρ, Pearson r, and a coarse alignment % when binarizing the
3-class GT to {high (=3), low (=1)} vs the scorer's median split.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
EXP  = ROOT / "experiments"

# Manual GT
GT_FILES = {
    "drugs":  EXP / "data/final/drugs/facts.csv",
    "weapon": EXP / "data/final/wep/facts.csv",
}

# Scorers
EMB_SIM       = EXP / "simcse_outputs/verdict_embeddings.npy"
IDX_SIM       = EXP / "simcse_outputs/verdict_index.csv"
SUP = {
    "drugs":  (EXP / "simcse_outputs/supervised/verdict_embeddings_drugs.npy",
               EXP / "simcse_outputs/supervised/verdict_index_drugs.csv"),
    "weapon": (EXP / "simcse_outputs/supervised/verdict_embeddings_weapon.npy",
               EXP / "simcse_outputs/supervised/verdict_index_weapon.csv"),
}
LLM_OLD = EXP / "data_per_domain/similarity_scores_combined.csv"
LLM_NEW = EXP / "data_per_domain/similarity_batch_simcse/results/similarity_scores_simcse.csv"
LLM_SUP = EXP / "data_per_domain/similarity_batch_supervised/results/similarity_scores_supervised.csv"

CIT     = EXP / "data_per_domain/network_analysis/citation_pair_types.csv"

ALIAS   = ROOT / "innovation_submission/data_master_final/verdict_alias.csv"

OUT     = EXP / "results/0_preprocessing/embedding_filter/eval_4_vs_gt_similarity.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

CIT_W = {"1hop": 3, "2hop": 2, "cocite": 1, "none": 0}


def canonical(s):
    import re, unicodedata
    if not s or pd.isna(s): return ""
    s = unicodedata.normalize("NFKC", str(s).strip())
    s = re.sub(r'["\'״׳`]', '', s)
    s = re.sub(r'\(.*?\)', '', s)
    s = re.sub(r'[\s/∕\\.]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_- ')
    return s


def main():
    # ---- alias for ID resolution ----
    alias = pd.read_csv(ALIAS)
    orig_to_canon = dict(zip(alias.original_id.astype(str), alias.canonical_id.astype(str)))
    canon_set = set(alias.canonical_id.astype(str))

    def best_lookup(t):
        if not t: return None
        if t in canon_set: return t
        c = canonical(t)
        if c in canon_set: return c
        a = orig_to_canon.get(t)
        if a:
            if a in canon_set: return a
            ac = canonical(a)
            if ac in canon_set: return ac
        return None

    # ---- load scorers ----
    emb_sim = np.load(EMB_SIM)
    idx_sim = pd.read_csv(IDX_SIM)
    sim_v2i = {v: i for i, v in enumerate(idx_sim.verdict)}

    sup = {}
    for d, (ep, ip) in SUP.items():
        emb = np.load(ep)
        idx = pd.read_csv(ip)
        sup[d] = {"emb": emb, "v2i": dict(zip(idx.verdict, range(len(idx))))}

    llm_pairs = {}
    for f in [LLM_OLD, LLM_NEW, LLM_SUP]:
        df = pd.read_csv(f, usecols=["verdict_1","verdict_2","similarity_score"])
        for r in df.itertuples(index=False):
            a, b = sorted([r.verdict_1, r.verdict_2])
            if pd.notna(r.similarity_score):
                llm_pairs[(a, b)] = float(r.similarity_score)

    cit = pd.read_csv(CIT, usecols=["verdict_1","verdict_2","citation_type"])
    def cit_score(t):
        if not isinstance(t, str): return 0
        return max((CIT_W.get(p, 0) for p in t.split(",")), default=0)
    cit["s"] = cit.citation_type.map(cit_score)
    cit_pairs = {}
    for r in cit.itertuples(index=False):
        a, b = sorted([r.verdict_1, r.verdict_2])
        cit_pairs[(a, b)] = r.s

    # ---- evaluate per domain ----
    all_rows = []
    for dom, gt_path in GT_FILES.items():
        gt = pd.read_csv(gt_path, engine="python", quoting=1, on_bad_lines="skip")
        gt = gt.dropna(subset=["similarity_scale"]).copy()

        # canonicalize
        gt["v1c"] = gt.verdict_1.map(best_lookup)
        gt["v2c"] = gt.verdict_2.map(best_lookup)
        before = len(gt)
        gt = gt[gt.v1c.notna() & gt.v2c.notna()].reset_index(drop=True)
        print(f"\n=== {dom} GT: {len(gt)}/{before} pairs (after canonicalize) ===")
        print(f"  GT scale dist: {gt.similarity_scale.value_counts().to_dict()}")
        if len(gt) == 0: continue

        # compute scorers
        scorers = {"sup_cos": [], "sim_cos": [], "llm_score": [], "cit_strength": []}
        gt_y = []
        for r in gt.itertuples(index=False):
            a, b = r.v1c, r.v2c
            # supervised cosine — must use this domain's model and both verdicts present
            sd = sup[dom]
            if a in sd["v2i"] and b in sd["v2i"]:
                ea, eb = sd["emb"][sd["v2i"][a]], sd["emb"][sd["v2i"][b]]
                # supervised emb is already normalized
                scorers["sup_cos"].append(float(np.dot(ea, eb) / (np.linalg.norm(ea)*np.linalg.norm(eb)+1e-9)))
            else:
                scorers["sup_cos"].append(np.nan)
            # SimCSE cosine
            if a in sim_v2i and b in sim_v2i:
                ea, eb = emb_sim[sim_v2i[a]], emb_sim[sim_v2i[b]]
                scorers["sim_cos"].append(float(np.dot(ea, eb) / (np.linalg.norm(ea)*np.linalg.norm(eb)+1e-9)))
            else:
                scorers["sim_cos"].append(np.nan)
            # LLM
            scorers["llm_score"].append(llm_pairs.get(tuple(sorted([a, b])), np.nan))
            # citation
            scorers["cit_strength"].append(cit_pairs.get(tuple(sorted([a, b])), 0))
            gt_y.append(int(r.similarity_scale))

        gt_y = np.array(gt_y)
        for sname, vals in scorers.items():
            arr = np.array(vals, dtype=float)
            mask = ~np.isnan(arr)
            n = mask.sum()
            if n < 5:
                all_rows.append({"domain": dom, "scorer": sname, "n": n,
                                 "spearman": None, "pearson": None})
                continue
            s_rho, _ = spearmanr(arr[mask], gt_y[mask])
            p_r, _ = pearsonr(arr[mask], gt_y[mask])
            all_rows.append({"domain": dom, "scorer": sname, "n": int(n),
                             "spearman": round(float(s_rho), 3),
                             "pearson":  round(float(p_r), 3)})

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT, index=False)
    print("\n\n=== CORRELATION WITH HUMAN GT (similarity_scale 1/2/3) ===")
    print(df.to_string(index=False))

    print("\nNote: similarity_scale: 1=least similar, 3=most similar (verify in your data!)")
    print("      → positive correlation = scorer agrees with human")
    print(f"\n💾 saved → {OUT}")


if __name__ == "__main__":
    main()
