import csv, collections
csv.field_size_limit(10**9)
DEST = "/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/gemma_local_similarity/out"
SRC = DEST + "/gemma_weapon_schema_COMPLETE.csv"
OUT = DEST + "/gemma_weapon_schema_FINAL.csv"
cols = ["verdict_1", "verdict_2", "domain", "similarity_score", "status", "n_out_tokens"]
best = {}
nrow = 0
for r in csv.DictReader(open(SRC, encoding="utf-8-sig")):
    if not r.get("verdict_1"):
        continue
    nrow += 1
    k = frozenset((r["verdict_1"], r["verdict_2"]))
    cur = best.get(k)
    if cur is None or (r.get("status") == "ok" and cur.get("status") != "ok"):
        best[k] = r
status_ct = collections.Counter(r.get("status") for r in best.values())
w = csv.DictWriter(open(OUT, "w", encoding="utf-8-sig", newline=""), fieldnames=cols)
w.writeheader()
for r in best.values():
    w.writerow({c: r.get(c, "") for c in cols})
print(f"rows read       = {nrow}")
print(f"unique pairs    = {len(best)}")
print(f"status breakdown= {dict(status_ct)}")
miss = 1476621 - len(best)
print(f"vs expected 1,476,621 -> {'COMPLETE ✅' if miss <= 0 else 'MISSING ' + str(miss)}")
print(f"-> {OUT}")
