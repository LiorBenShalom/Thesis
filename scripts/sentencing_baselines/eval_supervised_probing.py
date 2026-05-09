#!/usr/bin/env python3
"""
Linear probing — what features did the supervised embedding actually learn?

For each interpretable HFull feature (drug type, role, weapon type, planning,
sentencing range itself, ...), train a small linear classifier/regressor on
the embedding to predict the feature. High score = "the embedding encodes
this feature well".

Comparison: supervised embedding vs SimCSE embedding vs majority/mean baseline.

Train/test split = the SAME 80/20 used for supervised model training.

Output: results/0_preprocessing/embedding_filter/eval_5_probing.csv
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
EXP  = ROOT / "experiments"
HFULL = json.load(open(EXP / "data/sentencing_range-old/hfull_features/hybrid_full_cache.json"))

EMB_SUP = {
    "drugs":  EXP / "simcse_outputs/supervised/verdict_embeddings_drugs.npy",
    "weapon": EXP / "simcse_outputs/supervised/verdict_embeddings_weapon.npy",
}
IDX_SUP = {
    "drugs":  EXP / "simcse_outputs/supervised/verdict_index_drugs.csv",
    "weapon": EXP / "simcse_outputs/supervised/verdict_index_weapon.csv",
}
EMB_SIM = EXP / "simcse_outputs/verdict_embeddings.npy"
IDX_SIM = EXP / "simcse_outputs/verdict_index.csv"
MST     = EXP / "data_per_domain/master_inventory.csv"

OUT = EXP / "results/0_preprocessing/embedding_filter/eval_5_probing.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)


# ========== Feature extractors per domain ==========

def has_drug(feats, drug_name):
    """Binary — verdict mentions any quantity of <drug_name>."""
    v = feats.get(drug_name, "")
    return 1 if (v and str(v).strip() not in ("", "[]", "None", "0")) else 0

def total_drug_grams(feats, drug_name):
    """Numeric — sum of grams of <drug_name>. Returns NaN if missing."""
    v = feats.get(drug_name, "")
    if not v or str(v).strip() in ("", "[]", "None"): return np.nan
    s = str(v)
    nums = re.findall(r"(\d+(?:\.\d+)?)\s*-\s*גרם", s)
    if not nums: return np.nan
    return float(sum(float(n) for n in nums))

def role_simple(feats):
    """Categorical — simplify role to a few buckets."""
    v = str(feats.get("role", "")).strip()
    if not v: return None
    if "שליח" in v or "מעביר" in v: return "messenger"
    if "סוחר" in v or "מוכר" in v: return "dealer"
    if "מגדל" in v or "גידול" in v: return "grower"
    if "מסייע" in v: return "accomplice"
    if "בעל הסמים" in v: return "owner"
    if "משתמש" in v: return "user"
    return "other"

def has_weapon(feats, weapon_name):
    v = str(feats.get(weapon_name, "")).strip()
    if not v or v in ("None", "0", ""): return 0
    try: return 1 if int(float(v)) > 0 else 0
    except: return 0

def planning_yn(feats):
    v = str(feats.get("planning", "")).strip()
    if "כן" in v: return 1
    if "לא" in v: return 0
    return None

def laboratory_yn(feats):
    v = str(feats.get("laboratory", "")).strip()
    if v == "כן": return 1
    if v == "לא": return 0
    return None

def sold_to_agent_yn(feats):
    v = str(feats.get("sold_to_agent", "")).strip()
    if "כן" in v: return 1
    if "לא" in v: return 0
    return None


DRUGS_FEATURES = [
    ("has_cocaine",     "binary",  lambda f: has_drug(f, "cocaine")),
    ("has_cannabis",    "binary",  lambda f: has_drug(f, "cannabis")),
    ("has_mdma",        "binary",  lambda f: has_drug(f, "mdma")),
    ("has_hashish",     "binary",  lambda f: has_drug(f, "hashish")),
    ("has_methamph",    "binary",  lambda f: has_drug(f, "methamphetamine")),
    ("laboratory",      "binary",  laboratory_yn),
    ("sold_to_agent",   "binary",  sold_to_agent_yn),
    ("role",            "multi",   role_simple),
    ("cocaine_grams",   "numeric", lambda f: total_drug_grams(f, "cocaine")),
    ("cannabis_grams",  "numeric", lambda f: total_drug_grams(f, "cannabis")),
]

WEAPON_FEATURES = [
    ("has_pistol",       "binary",  lambda f: has_weapon(f, "pistol")),
    ("has_submachine",   "binary",  lambda f: has_weapon(f, "submachine_gun")),
    ("has_explosive",    "binary",  lambda f: has_weapon(f, "explosive")),
    ("has_assault_rifle","binary",  lambda f: has_weapon(f, "assault_rifle")),
    ("planning",         "binary",  planning_yn),
]

# Always include sentencing range itself — the actual signal we trained on.
SENTENCING_FEATS = [
    ("sentencing_low",   "numeric", None),
    ("sentencing_high",  "numeric", None),
]


# ========== Probing ==========

def probe(X_train, y_train, X_test, y_test, kind):
    if kind == "binary":
        # Skip if degenerate
        if len(set(y_train)) < 2 or len(set(y_test)) < 2:
            return None, None, None
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc = accuracy_score(y_test, pred)
        # majority baseline
        baseline = max(np.mean(y_test == 1), np.mean(y_test == 0))
        return acc, baseline, len(y_test)
    elif kind == "multi":
        if len(set(y_train)) < 2: return None, None, None
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc = accuracy_score(y_test, pred)
        from collections import Counter
        baseline = Counter(y_test).most_common(1)[0][1] / len(y_test)
        return acc, baseline, len(y_test)
    elif kind == "numeric":
        clf = Ridge(alpha=1.0)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        r2 = r2_score(y_test, pred)
        # baseline = mean predictor; r2 against mean = 0
        return r2, 0.0, len(y_test)
    raise ValueError(kind)


def main():
    # Master inventory for sentencing ranges
    m = pd.read_csv(MST, usecols=["canonical_id","domain","sentencing_range_low","sentencing_range_high"])
    range_low  = dict(zip(m.canonical_id, m.sentencing_range_low))
    range_high = dict(zip(m.canonical_id, m.sentencing_range_high))

    # SimCSE embedding once
    emb_sim = np.load(EMB_SIM)
    idx_sim = pd.read_csv(IDX_SIM)
    sim_v2i = {v: i for i, v in enumerate(idx_sim.verdict)}

    rows = []
    for dom, feature_list in [("drugs", DRUGS_FEATURES), ("weapon", WEAPON_FEATURES)]:
        emb_sup = np.load(EMB_SUP[dom])
        idx_sup = pd.read_csv(IDX_SUP[dom])
        sup_v2i = {v: i for i, v in enumerate(idx_sup.verdict)}
        train_set = set(idx_sup[idx_sup.split == "train"].verdict)
        test_set  = set(idx_sup[idx_sup.split == "test"].verdict)

        # Build (verdict, supervised_emb, simcse_emb, feature_value, split) per feature
        for feat_name, kind, extractor in feature_list + SENTENCING_FEATS:
            X_sup_tr, X_sup_te, y_tr, y_te = [], [], [], []
            X_sim_tr, X_sim_te = [], []
            for v in idx_sup.verdict:
                if v not in HFULL: continue
                f = HFULL[v]
                if feat_name == "sentencing_low":
                    val = range_low.get(v)
                elif feat_name == "sentencing_high":
                    val = range_high.get(v)
                else:
                    val = extractor(f) if isinstance(f, dict) else None
                if val is None or (isinstance(val, float) and np.isnan(val)): continue
                if v not in sim_v2i: continue   # need both embeddings
                e_sup = emb_sup[sup_v2i[v]]
                e_sim = emb_sim[sim_v2i[v]]
                if v in train_set:
                    X_sup_tr.append(e_sup); X_sim_tr.append(e_sim); y_tr.append(val)
                elif v in test_set:
                    X_sup_te.append(e_sup); X_sim_te.append(e_sim); y_te.append(val)

            n_tr, n_te = len(y_tr), len(y_te)
            if n_tr < 20 or n_te < 10:
                rows.append({"domain": dom, "feature": feat_name, "kind": kind,
                             "n_train": n_tr, "n_test": n_te, "note": "too few"})
                continue

            X_sup_tr = np.array(X_sup_tr); X_sup_te = np.array(X_sup_te)
            X_sim_tr = np.array(X_sim_tr); X_sim_te = np.array(X_sim_te)
            y_tr_arr = np.array(y_tr); y_te_arr = np.array(y_te)

            sup_score, baseline, _ = probe(X_sup_tr, y_tr_arr, X_sup_te, y_te_arr, kind)
            sim_score, _, _        = probe(X_sim_tr, y_tr_arr, X_sim_te, y_te_arr, kind)
            rows.append({
                "domain": dom, "feature": feat_name, "kind": kind,
                "n_train": n_tr, "n_test": n_te,
                "supervised_score": round(sup_score, 3) if sup_score is not None else None,
                "simcse_score":     round(sim_score, 3) if sim_score is not None else None,
                "baseline":         round(baseline, 3) if baseline is not None else None,
                "metric": "accuracy" if kind in ("binary","multi") else "R²",
                "sup_lift_vs_baseline": round(sup_score - baseline, 3) if (sup_score is not None and baseline is not None) else None,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("=" * 110)
    print("LINEAR PROBING RESULTS (supervised vs SimCSE embeddings)")
    print("=" * 110)
    print(df.to_string(index=False))
    print(f"\n💾 Saved → {OUT}")

    # Summary: which features did supervised learn DRAMATICALLY better than SimCSE?
    print("\n\n=== KEY INSIGHTS ===")
    valid = df.dropna(subset=["supervised_score","simcse_score"]).copy()
    valid["delta"] = valid.supervised_score - valid.simcse_score
    valid = valid.sort_values("delta", ascending=False)
    print("\n  Top features supervised does BETTER than SimCSE:")
    print(valid.head(10)[["domain","feature","kind","supervised_score","simcse_score","delta"]].to_string(index=False))
    print("\n  Top features SimCSE does BETTER than supervised:")
    print(valid.tail(10)[["domain","feature","kind","supervised_score","simcse_score","delta"]].to_string(index=False))


if __name__ == "__main__":
    main()
