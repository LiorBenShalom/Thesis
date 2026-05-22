#!/usr/bin/env python3
"""
citation_only ablation: is the LLM-rerank worth it over raw citation-neighbor averaging?
Plus the 1-hop-only vs full(1hop+2hop+cocite) candidate-set trade-off, and 2 concrete
LLM-rescue examples. All apples-to-apples with rigor (same folds, same rng_lo, same K).

Outputs (-> data/):
  citation_only_vs_llm.csv          matched MAE + paired tests, both candidate sets
  citation_llm_rescue_examples.csv  per-neighbor breakdown for 2 example queries
"""
import pandas as pd, numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[5]
FILT = ROOT / "simcse_cuda_bundle/outputs_supervised_filtered"
EXP  = ROOT / "experiments"
DATA = Path(__file__).resolve().parents[1] / "data"
K = 10

sup = pd.read_csv(ROOT / "simcse_cuda_bundle/data/supervised_data.csv")
sup["verdict"] = sup.verdict.astype(str)
LO  = dict(zip(sup.verdict, sup.sentencing_range_low))
HI  = dict(zip(sup.verdict, sup.sentencing_range_high))

cit = pd.read_csv(EXP / "data_per_domain/network_analysis/citation_pair_types.csv")
def build_adj(pred):
    a = defaultdict(set)
    for r in cit.itertuples(index=False):
        if pred(str(r.citation_type)):
            x, y = str(r.verdict_1), str(r.verdict_2)
            a[x].add(y); a[y].add(x)
    return a
adj_full = build_adj(lambda t: t in ("1hop", "2hop", "cocite"))   # what citation_llm uses
adj_1hop = build_adj(lambda t: "1hop" in t)                       # direct citations only

# fold -> train set, and test queries
foldmap = {}
for dom in ("drugs", "weapon"):
    for f in range(1, 6):
        ip = FILT / f"verdict_index_{dom}_topk_fold{f}_offenseFiltered.csv"
        if not ip.exists(): continue
        idx = pd.read_csv(ip); idx["verdict"] = idx.verdict.astype(str)
        tr = set(idx.loc[idx.split == "train", "verdict"])
        for q in idx.loc[idx.split == "test", "verdict"]:
            foldmap[(dom, q)] = tr

def med(vs, d):
    v = [d[x] for x in vs if x in d and pd.notna(d[x])]
    return float(np.median(v)) if v else None

e2 = pd.read_csv(DATA / "rigor_per_query_errors.csv")
tot = {"drugs": 2713, "weapon": 1719}

def citation_only_errors(adj):
    rows = []
    for (dom, q), tr in foldmap.items():
        if q not in LO or pd.isna(LO[q]): continue
        nb = {n for n in adj.get(q, set()) & tr if n in LO}
        if not nb: continue
        rows.append(dict(query=q, domain=dom, n_nb=len(nb),
                         co_lo=abs(med(nb, LO) - LO[q]), co_hi=abs(med(nb, HI) - HI[q])))
    return pd.DataFrame(rows)

def paired(co, dom, meth, fld="err_lo", co_fld="co_lo"):
    o = e2[(e2.method == meth) & (e2.domain == dom)].drop_duplicates("query").set_index("query")
    ci = co[co.domain == dom].drop_duplicates("query").set_index("query")
    common = ci.index.intersection(o.index)
    diff = (ci.loc[common, co_fld] - o.loc[common, fld]).to_numpy()
    rng = np.random.default_rng(0)
    bs = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(2000)])
    return diff.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), wilcoxon(diff).pvalue

out = []
for setname, adj in [("1hop_only", adj_1hop), ("full_1hop+2hop+cocite", adj_full)]:
    co = citation_only_errors(adj)
    for dom in ("drugs", "weapon"):
        cm = co[co.domain == dom]
        qs = set(cm["query"])
        row = dict(candidate_set=setname, domain=dom, coverage=round(len(cm) / tot[dom], 3),
                   n=len(cm), median_neighbors=int(cm.n_nb.median()),
                   citation_only_lo=round(cm.co_lo.mean(), 2), citation_only_hi=round(cm.co_hi.mean(), 2))
        for meth in ("llm_best", "citation_llm"):
            d, lo, hi, p = paired(co, dom, meth)
            mm = e2[(e2.method == meth) & (e2.domain == dom) & (e2["query"].isin(qs))]
            row[f"{meth}_lo_matched"] = round(mm.err_lo.mean(), 2)
            row[f"delta_vs_{meth}"] = round(d, 2)
            row[f"ci_vs_{meth}"] = f"[{lo:+.2f},{hi:+.2f}]"
            row[f"p_vs_{meth}"] = f"{p:.1e}"
            row[f"sig_vs_{meth}"] = "yes" if (lo > 0 or hi < 0) else "no"
        out.append(row)
res = pd.DataFrame(out)
res.to_csv(DATA / "citation_only_vs_llm.csv", index=False)
print("wrote citation_only_vs_llm.csv\n", res.to_string(index=False))

# ---- 2 concrete LLM-rescue examples ----
llm = {}
d = pd.read_csv(EXP / "data_per_domain/similarity_scores_combined.csv")
for a, b, s in zip(d.verdict_1.astype(str), d.verdict_2.astype(str), d.similarity_score):
    if pd.notna(s): llm[tuple(sorted([a, b]))] = s
ex_rows = []
for q, dom in [("תפח_20623-11-12", "weapon"), ("תפ_4760-10-18", "drugs")]:
    tr = foldmap.get((dom, q), set())
    nb = [n for n in adj_full.get(q, set()) & tr if n in LO]
    ranked = sorted([(n, llm.get(tuple(sorted([q, n])))) for n in nb],
                    key=lambda x: -(x[1] if x[1] is not None else -1))
    scored = [(n, s) for n, s in ranked if s is not None]
    co_pred = np.median([LO[n] for n in nb])
    cl_pred = np.median([LO[n] for n, _ in scored[:K]]) if scored else None
    for rank, (n, s) in enumerate(scored, 1):
        ex_rows.append(dict(query=q, true_low=LO[q], true_high=HI[q],
                            n_neighbors=len(nb), citation_only_pred_low=co_pred,
                            citation_llm_pred_low=cl_pred, neighbor=n, llm_score=s,
                            neighbor_low=LO[n], neighbor_high=HI[n],
                            in_llm_top10=(rank <= K)))
pd.DataFrame(ex_rows).to_csv(DATA / "citation_llm_rescue_examples.csv", index=False)
print("\nwrote citation_llm_rescue_examples.csv")
