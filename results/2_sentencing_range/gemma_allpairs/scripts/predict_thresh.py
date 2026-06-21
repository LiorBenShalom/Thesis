import csv, collections
import numpy as np
import pandas as pd
csv.field_size_limit(10**9)
BASE = "/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try"
GEM = BASE + "/gemma_local_similarity/out/gemma_weapon_schema_FINAL.csv"
SUP = BASE + "/simcse_cuda_bundle/data/supervised_data.csv"

sup = pd.read_csv(SUP); w = sup[sup.domain == "weapon"]
lo = {r.verdict: float(r.sentencing_range_low) for r in w.itertuples(index=False)}
hi = {r.verdict: float(r.sentencing_range_high) for r in w.itertuples(index=False)}
weapon = set(lo)

best = collections.defaultdict(dict)
for r in csv.DictReader(open(GEM, encoding="utf-8-sig")):
    s = r.get("similarity_score")
    if not r.get("verdict_1") or s in (None, ""): continue
    try: s = float(s)
    except ValueError: continue
    a, b = r["verdict_1"], r["verdict_2"]
    if a in weapon and b in weapon:
        if s > best[a].get(b, -1): best[a][b] = s
        if s > best[b].get(a, -1): best[b][a] = s
pre = {q: sorted(d.items(), key=lambda x: -x[1]) for q, d in best.items()}

# context: how many high-sim twins do queries have?
for thr in (60, 80, 90, 95):
    cnts = [sum(1 for _, s in pre.get(q, []) if s >= thr) for q in weapon]
    print(f"thr>={thr}: queries with >=3 twins = {sum(c>=3 for c in cnts)}/{len(weapon)}  "
          f"(median twins/query = {int(np.median(cnts))}, mean={np.mean(cnts):.0f})")

def evaluate(thr, maxK=10, minK=3, strict=False):
    elo, ehi, ncov, nfloor = [], [], 0, 0
    for q in weapon:
        items = pre.get(q, [])
        above = [c for c, s in items if s >= thr][:maxK]
        if len(above) < minK:
            if strict:
                continue                       # only predict where >=minK confident twins
            above = [c for c, _ in items[:minK]]  # floor to top-minK
            nfloor += 1
        if not above: continue
        ncov += 1
        plo = float(np.median([lo[c] for c in above]))
        phi = float(np.median([hi[c] for c in above]))
        elo.append(abs(plo - lo[q])); ehi.append(abs(phi - hi[q]))
    return np.mean(elo), np.mean(ehi), ncov, nfloor

print("\nReference (plain top-10, no threshold):  MAE_lo=13.05  MAE_hi=18.71  (all 1718)")
print("\n=== threshold + floor-to-3 (always predict) ===")
for thr in (50, 60, 70):
    mlo, mhi, ncov, nfl = evaluate(thr, 10, 3, strict=False)
    print(f"thr>={thr}, K<=10, min3: MAE_lo={mlo:.2f}  MAE_hi={mhi:.2f}  (cov {ncov}, floored {nfl})")

print("\n=== STRICT (only queries with >=3 twins above threshold; tests Gemma's MOST confident twins) ===")
for thr in (60, 80, 85, 90, 95):
    mlo, mhi, ncov, nfl = evaluate(thr, 10, 3, strict=True)
    print(f"thr>={thr}, K<=10, >=3:  MAE_lo={mlo:.2f}  MAE_hi={mhi:.2f}  (cov {ncov}/{len(weapon)})")

# ====== CONTROLLED test: is the high-threshold degradation just small-K, on a harder query subset? ======
print("\n\n=== CONTROLLED: same query subset, threshold-only vs top-K overall (isolates K from quality) ===")
def mae_on(qset, pick):
    elo, ehi, eff = [], [], []
    for q in qset:
        c = pick(q)
        if not c: continue
        eff.append(len(c))
        elo.append(abs(np.median([lo[x] for x in c]) - lo[q]))
        ehi.append(abs(np.median([hi[x] for x in c]) - hi[q]))
    return np.mean(elo), np.mean(ehi), np.mean(eff), len(elo)

for thr in (80, 90, 95):
    S = [q for q in weapon if sum(1 for _, s in pre.get(q, []) if s >= thr) >= 3]
    a = mae_on(S, lambda q, t=thr: [c for c, s in pre.get(q, []) if s >= t][:10])
    b = mae_on(S, lambda q: [c for c, _ in pre.get(q, [])[:10]])
    cc = mae_on(S, lambda q: [c for c, _ in pre.get(q, [])[:3]])
    print(f"\nthr>={thr}  subset n={len(S)}:")
    print(f"   >= {thr}-only (<=10) : MAE_lo={a[0]:.2f}  MAE_hi={a[1]:.2f}  effK={a[2]:.1f}")
    print(f"   top-10 overall      : MAE_lo={b[0]:.2f}  MAE_hi={b[1]:.2f}  effK={b[2]:.1f}")
    print(f"   top-3  overall      : MAE_lo={cc[0]:.2f}  MAE_hi={cc[1]:.2f}  effK={cc[2]:.1f}")

# ====== USER's two-threshold scheme: pool>=80 for prediction, GATE = predict only if >=3 twins >=90 ======
print("\n\n=== TWO-THRESHOLD: predict from pool>=POOL (<=10); ABSTAIN unless >=GMIN twins >=GATE ===")
def two_thr(pool_thr, gate_thr, gate_min=3, maxK=10):
    elo, ehi, eff, ncov = [], [], [], 0
    for q in weapon:
        items = pre.get(q, [])
        if sum(1 for _, s in items if s >= gate_thr) < gate_min:   # confidence gate
            continue                                                # abstain
        pool = [c for c, s in items if s >= pool_thr][:maxK]
        if not pool: continue
        ncov += 1; eff.append(len(pool))
        elo.append(abs(np.median([lo[c] for c in pool]) - lo[q]))
        ehi.append(abs(np.median([hi[c] for c in pool]) - hi[q]))
    return np.mean(elo), np.mean(ehi), np.mean(eff), ncov

for pool_thr, gate_thr in [(80, 90), (80, 95), (60, 90), (80, 85)]:
    mlo, mhi, ek, nc = two_thr(pool_thr, gate_thr, 3, 10)
    print(f"pool>={pool_thr}, gate>=3@{gate_thr}: MAE_lo={mlo:.2f} MAE_hi={mhi:.2f} effK={ek:.1f} covered={nc}/{len(weapon)} ({100*nc/len(weapon):.0f}%)")

# for reference: same covered subset, but predict from plain top-10 (no pool restriction)
print("\n--- reference: on the SAME covered subset (gate>=3@90), plain top-10 ---")
S = [q for q in weapon if sum(1 for _, s in pre.get(q, []) if s >= 90) >= 3]
elo = [abs(np.median([lo[c] for c, _ in pre[q][:10]]) - lo[q]) for q in S]
ehi = [abs(np.median([hi[c] for c, _ in pre[q][:10]]) - hi[q]) for q in S]
print(f"top-10 overall on gated subset (n={len(S)}): MAE_lo={np.mean(elo):.2f} MAE_hi={np.mean(ehi):.2f}")
