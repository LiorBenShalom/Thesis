"""
Offense-matched random — sweep the minimum overlap threshold.
M2 uses |offense(q) ∩ offense(t)| >= 1. Test >= 1, 2, 3, and exact-match.

For each threshold:
  - candidates = train verdicts sharing >= T offense labels with q
  - K=10 random (seed=hash(q)+1), predict median
  - report: coverage (queries with >=1 candidate), n participants,
            MAE-lo/hi with bootstrap 95% CI
"""
from pathlib import Path
import json, re, hashlib
import numpy as np
import pandas as pd

def stable_seed(s: str) -> int:
    """Deterministic across processes (Python's built-in hash() is salted)."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
EXP  = ROOT / "experiments"
FILTERED_DIR = ROOT / "simcse_cuda_bundle/outputs_supervised_filtered"
N_FOLDS = 5
K = 10

sup = pd.read_csv(ROOT / "simcse_cuda_bundle/data/supervised_data.csv")
v_to_dom = dict(zip(sup.verdict, sup.domain))

m = pd.read_csv(EXP / "data_per_domain/master_inventory.csv",
                usecols=["canonical_id","domain","sentencing_range_low",
                         "sentencing_range_high","sentencing_confidence"])
m = m[m.domain.isin(["drugs","weapon"]) & m.sentencing_range_low.notna()
      & (m.sentencing_confidence == "גבוהה")].drop_duplicates("canonical_id")
rng_lo = dict(zip(m.canonical_id, m.sentencing_range_low))
rng_hi = dict(zip(m.canonical_id, m.sentencing_range_high))

with open(EXP / "data/sentencing_range-old/hfull_features/hybrid_full_cache.json") as f:
    hf = json.load(f)

def yesno(v):
    if v is None: return False
    return str(v).strip() not in ("", "לא", "nan", "None", "0", "0.0")

def drugs_offense_set(feats):
    if not feats: return set()
    s = set()
    for sec in ("6","7","13","14","19"):
        if yesno(feats.get(f"section_{sec}")): s.add(f"sec_{sec}")
    if yesno(feats.get("other_drug_offense")): s.add("other")
    return s

WPAT = [(r"144\s*\(\s*א\s*\)","144a"),(r"144\s*\(\s*ב\s*2\s*\)","144b2"),
        (r"144\s*\(\s*ב\s*\)","144b"),(r"144\s*\(\s*ג\s*\)","144c"),
        (r"144\s*\(\s*ז\s*\)","144g"),(r"\b145\b","145"),(r"\b146\b","146")]
def weapon_offense_set(feats):
    if not feats: return set()
    blob = " ".join(str(feats.get(k,"")) for k in ("offense_number","offense_type","additional_offenses"))
    return {label for pat,label in WPAT if re.search(pat, blob)}

verdict_offenses = {v: (drugs_offense_set(hf.get(v,{})) if v_to_dom[v]=="drugs"
                       else weapon_offense_set(hf.get(v,{}))) for v in v_to_dom}

# folds (only need the splits)
folds = {}
for dom in ("drugs","weapon"):
    for f in range(1, N_FOLDS+1):
        ip = FILTERED_DIR / f"verdict_index_{dom}_topk_fold{f}_offenseFiltered.csv"
        if not ip.exists(): continue
        idx = pd.read_csv(ip)
        folds[(dom,f)] = {"train_ids": idx[idx.split=="train"].verdict.tolist(),
                          "test_ids":  idx[idx.split=="test"].verdict.tolist()}

def run(threshold, exact=False):
    """threshold = min |intersection|. exact=True → identical offense sets."""
    rows = []
    for dom in ("drugs","weapon"):
        errs_lo=[]; errs_hi=[]; n_total=0; n_pred=0
        offset_size_dist=[]
        for (d,fid),ff in folds.items():
            if d!=dom: continue
            for q in ff["test_ids"]:
                if q not in rng_lo: continue
                q_off = verdict_offenses.get(q,set())
                if not q_off: continue
                n_total+=1
                cands=[]
                for t in ff["train_ids"]:
                    if t==q: continue
                    t_off=verdict_offenses.get(t,set())
                    if exact:
                        if t_off==q_off: cands.append(t)
                    else:
                        if len(q_off & t_off) >= threshold: cands.append(t)
                if not cands: continue        # min_k = 1 (consistent with all other methods M1-M9)
                rng2=np.random.default_rng(stable_seed(q))   # deterministic across runs
                if len(cands)>=K:
                    sel=rng2.permutation(len(cands))[:K]
                    picked=[cands[i] for i in sel]
                else:
                    picked=cands
                picked=[p for p in picked if p in rng_lo]
                if not picked: continue
                plo=float(np.median([rng_lo[p] for p in picked]))
                phi=float(np.median([rng_hi[p] for p in picked]))
                errs_lo.append(abs(plo-rng_lo[q]))
                errs_hi.append(abs(phi-rng_hi[q]))
                n_pred+=1
        a_lo=np.array(errs_lo); a_hi=np.array(errs_hi)
        rng=np.random.default_rng(42); B=2000
        if len(a_lo):
            blo=[a_lo[rng.integers(0,len(a_lo),len(a_lo))].mean() for _ in range(B)]
            bhi=[a_hi[rng.integers(0,len(a_hi),len(a_hi))].mean() for _ in range(B)]
            ci_lo=np.percentile(blo,[2.5,97.5]); ci_hi=np.percentile(bhi,[2.5,97.5])
        else:
            ci_lo=ci_hi=[None,None]
        rows.append({"domain":dom,"threshold":("exact" if exact else threshold),
                     "n_eligible_q":n_total,"n_pred":n_pred,
                     "coverage":n_pred/n_total if n_total else 0,
                     "mae_lo":a_lo.mean() if len(a_lo) else None,
                     "mae_lo_ci":f"[{ci_lo[0]:.2f},{ci_lo[1]:.2f}]" if len(a_lo) else "—",
                     "mae_hi":a_hi.mean() if len(a_hi) else None,
                     "mae_hi_ci":f"[{ci_hi[0]:.2f},{ci_hi[1]:.2f}]" if len(a_hi) else "—"})
    return rows

print(f"{'thr':>6s} {'dom':6s} {'cov':>6s} {'n_pred':>7s} {'MAE-lo [95%CI]':>22s} {'MAE-hi [95%CI]':>22s}")
print("-"*75)
all_rows=[]
for thr in [1,2,3]:
    for r in run(thr):
        all_rows.append(r)
        print(f"{thr:>6} {r['domain']:6s} {r['coverage']*100:>5.0f}% {r['n_pred']:>7d} "
              f"{r['mae_lo']:>6.2f} {r['mae_lo_ci']:>15s} {r['mae_hi']:>6.2f} {r['mae_hi_ci']:>15s}")
print()
for r in run(None, exact=True):
    all_rows.append(r)
    print(f"{'exact':>6} {r['domain']:6s} {r['coverage']*100:>5.0f}% {r['n_pred']:>7d} "
          f"{r['mae_lo']:>6.2f} {r['mae_lo_ci']:>15s} {r['mae_hi']:>6.2f} {r['mae_hi_ci']:>15s}")

pd.DataFrame(all_rows).to_csv("/tmp/offense_matched_overlap_sweep.csv", index=False)
print("\n✅ /tmp/offense_matched_overlap_sweep.csv")
print("\n(Reference [4,432, 2026-05-16]: global_median drugs 8.48/14.07, weapon 17.49/26.21 | sup+LLM drugs 5.83/9.48, weapon 12.95/19.19)")
