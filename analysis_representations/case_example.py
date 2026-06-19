"""Show ONE real verdict per domain across ALL representations, side by side."""
from __future__ import annotations
import csv, json, re, argparse
from pathlib import Path

csv.field_size_limit(10**9)
ROOT = Path(__file__).resolve().parents[2]   # .../new_try

# rep -> (filename, value-column-base). Raw-Facts uses indicment_facts_{1,2}.
REPS = [
    ("Manual",        "similarity_database_fe.csv",                    "feature_vector"),
    ("GPT-Schema",    "similarity_database_fe_gpt_schema.csv",         "feature_vector"),
    ("Hybrid-Manual", "similarity_database_hybrid.csv",                "feature_vector"),
    ("Hybrid-Full",   "similarity_database_hybrid_full_gpt.csv",       "feature_vector"),
    ("GPT-Free",      "similarity_database_with_gpt_features.csv",     "feature_vector"),
    ("GPT-Law",       "similarity_database_with_gpt_law_features.csv", "feature_vector"),
    ("Raw-Facts",     "similarity_database_with_indicment_facts.csv",  "indicment_facts"),
]


def _clean(s):
    return s.strip().lstrip("﻿").strip('"') if s else s


def parse_json(cell):
    if not cell:
        return None
    t = cell.strip()
    if not t.startswith("{"):
        i, j = t.find("{"), t.rfind("}")
        if i == -1:
            return None
        t = t[i:j + 1]
    for a in (t, re.sub(r",\s*}", "}", t)):
        try:
            return json.loads(a, strict=False)
        except Exception:
            pass
    return None


def load_rep(domain, fname, base):
    """verdict_id -> raw cell (dict for features, text for facts)."""
    path = ROOT / domain / fname
    out = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        hdr = [_clean(h) for h in next(r)]
        idx = {h: i for i, h in enumerate(hdr)}
        for row in r:
            if not row:
                continue
            for s in ("1", "2"):
                vid = row[idx[f"verdict_{s}"]].strip()
                cell = row[idx[f"{base}_{s}"]] if f"{base}_{s}" in idx else ""
                if vid and vid not in out:
                    out[vid] = cell
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--verdict", default=None, help="verdict id; if omitted, list candidates")
    ap.add_argument("--top", type=int, default=15, help="how many candidates to list")
    a = ap.parse_args()
    data = {name: load_rep(a.domain, fn, base) for name, fn, base in REPS}

    # verdicts present in every representation
    common = set.intersection(*[set(d) for d in data.values() if d])
    if not a.verdict:
        # rank candidates by Hybrid-Full field count (richness), middle range preferred
        hf = data["Hybrid-Full"]
        cand = []
        for v in common:
            d = parse_json(hf.get(v, ""))
            cand.append((v, len(d) if d else 0))
        cand.sort(key=lambda x: -x[1])
        print(f"domain={a.domain}: {len(common)} verdicts present in all {len(data)} reps")
        print("candidates (verdict | #Hybrid-Full fields):")
        for v, n in cand[:a.top]:
            print(f"  {v:24s}  {n}")
        return

    v = a.verdict
    print("=" * 78); print(f"DOMAIN={a.domain}  VERDICT={v}"); print("=" * 78)
    for name, _fn, base in REPS:
        cell = data[name].get(v)
        print(f"\n### {name}")
        if cell is None:
            print("  (absent)"); continue
        if base == "indicment_facts":
            txt = cell.strip()
            print(f"  [free text, {len(txt)} chars]\n  {txt[:700]}{'...' if len(txt)>700 else ''}")
        else:
            d = parse_json(cell)
            if not d:
                print("  (unparseable)"); continue
            print(f"  [{len(d)} fields]")
            for k, val in d.items():
                vs = json.dumps(val, ensure_ascii=False) if not isinstance(val, str) else val
                print(f"    {k}: {vs[:120]}")


if __name__ == "__main__":
    main()
