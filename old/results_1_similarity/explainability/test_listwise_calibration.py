#!/usr/bin/env python3
"""
Test: does listwise score MATCH pairwise score per (target, candidate)?

Design:
  - Pick targets that appear in many GT pairs
  - For each target: candidates = ALL its GT partners (we have pairwise scores for these)
  - Run listwise: target + all candidates → get a score per candidate
  - Compare per-candidate: listwise score vs pairwise score
  - Compute: correlation, mean absolute diff, agreement
"""
import os, re, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
PAIRWISE_DIR = ROOT / "experiments/explainability_annotation/multimodel_score_only/results"
OUT = ROOT / "experiments/explainability_annotation/listwise_validation"
OUT.mkdir(parents=True, exist_ok=True)

SYSTEM_DRUGS = """את/ה מומחית לדין הפלילי בישראל. מוצגים בפנייך תיק יעד ורשימה של תיקים מתחרים — כולם בעבירות סמים.

המשימה: לתת ציון 0-100 לכל תיק מתחרה לפי דמיון מהותי לתיק היעד — עד כמה הוא יכול לשמש כתקדים ענייני.

נתחי כל ממד בקצרה:
1. סוג העבירה וחומרתה
2. סוג הסם וכמותו
3. שיטת הביצוע (MO)
4. נסיבות הליבה
5. ישימות כתקדים

חשוב: תני לכל תיק את הציון העצמאי שלו (ללא תלות באחרים). 0 = שונים לחלוטין, 100 = זהים כמעט. ציון מעל 50 = תקדים רלוונטי.

פורמט תשובה: שורה אחת לכל מתחרה, ללא הסברים:
CANDIDATE_<מספר>: <ציון 0-100>"""

SYSTEM_WEAPON = """את/ה מומחית לדין הפלילי בישראל. מוצגים בפנייך תיק יעד ורשימה של תיקים מתחרים — כולם בעבירות נשק.

המשימה: לתת ציון 0-100 לכל תיק מתחרה לפי דמיון מהותי לתיק היעד.

נתחי כל ממד בקצרה:
1. סוג העבירה וחומרתה
2. תפקיד הנאשם ומעורבותו
3. שיטת הביצוע (MO)
4. נסיבות הליבה (כמות תחמושת, שימוש, תכנון)
5. ישימות כתקדים

חשוב: תני לכל תיק את הציון העצמאי שלו (ללא תלות באחרים). 0 = שונים לחלוטין, 100 = זהים כמעט. הבחנה בין 144(א) ל-144(ב) מהותית.

פורמט תשובה: שורה אחת לכל מתחרה, ללא הסברים:
CANDIDATE_<מספר>: <ציון 0-100>"""


def build_user(target_fv, candidates_fv):
    parts = [f"תיק היעד:\n{target_fv}\n", f"\nתיקים מתחרים ({len(candidates_fv)}):"]
    for i, fv in enumerate(candidates_fv, 1):
        parts.append(f"\n--- CANDIDATE_{i} ---\n{fv}")
    parts.append(f"\n\nתני ציון 0-100 לכל אחד מהמתחרים מול תיק היעד.")
    return "\n".join(parts)


def parse_listwise(text, n):
    out = {}
    for m in re.finditer(r"CANDIDATE_(\d+):\s*(\d+)", text):
        idx = int(m.group(1)); s = int(m.group(2))
        if 1 <= idx <= n and 0 <= s <= 100:
            out[idx] = s
    return out


def main():
    cli = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    all_results = []

    for dom in ["drugs", "weapon"]:
        print(f"\n{'='*70}\n{dom.upper()}\n{'='*70}")
        gt = pd.read_csv(ROOT / dom / "similarity_database_hybrid_full_gpt.csv")
        v2fv = {}
        for _, r in gt.iterrows():
            v2fv.setdefault(r["verdict_1"], r["feature_vector_1"])
            v2fv.setdefault(r["verdict_2"], r["feature_vector_2"])

        # Pairwise scores (gpt4 score-only)
        pw = pd.read_csv(PAIRWISE_DIR / f"gpt4__hybrid_full_gpt__{dom}.csv")
        # build (v1, v2) → score (symmetric: also v2,v1)
        pw_sym = {}
        for _, r in pw.iterrows():
            pw_sym[(r["verdict_1"], r["verdict_2"])] = r["model_score"]
            pw_sym[(r["verdict_2"], r["verdict_1"])] = r["model_score"]

        # For each verdict, find all its GT partners
        partners = defaultdict(list)
        for _, r in gt.iterrows():
            partners[r["verdict_1"]].append(r["verdict_2"])
            partners[r["verdict_2"]].append(r["verdict_1"])

        # Pick targets with most partners (top 5)
        target_scores = sorted(partners.items(), key=lambda kv: -len(kv[1]))[:5]
        for target, parts_list in target_scores:
            parts_list = list(set(parts_list))[:15]  # cap at 15 candidates
            if len(parts_list) < 4:  # need enough candidates
                continue
            print(f"\n  target={target}  candidates={len(parts_list)}")

            cand_fvs = [v2fv[p] for p in parts_list]
            sys_p = SYSTEM_DRUGS if dom == "drugs" else SYSTEM_WEAPON
            user = build_user(v2fv[target], cand_fvs)
            try:
                resp = cli.chat.completions.create(
                    model="gpt-4.1", messages=[
                        {"role": "system", "content": sys_p},
                        {"role": "user", "content": user},
                    ], temperature=0.1, max_tokens=300,
                )
                txt = resp.choices[0].message.content or ""
                lw_scores = parse_listwise(txt, len(parts_list))
            except Exception as e:
                print(f"    err: {e}")
                continue

            for i, partner in enumerate(parts_list, 1):
                lw = lw_scores.get(i)
                pw_score = pw_sym.get((target, partner))
                all_results.append({
                    "domain": dom, "target": target, "candidate": partner,
                    "pairwise": pw_score, "listwise": lw,
                })

    df = pd.DataFrame(all_results)
    df.to_csv(OUT / "calibration.csv", index=False)
    valid = df.dropna()
    print(f"\n✅ saved calibration.csv  ({len(valid)} valid pairs)")

    print("\n" + "="*70)
    print("PAIRWISE vs LISTWISE — per-pair score comparison")
    print("="*70)
    if len(valid) > 0:
        diff = valid["listwise"] - valid["pairwise"]
        print(f"\n  Mean abs diff:    {diff.abs().mean():.1f}")
        print(f"  Mean signed diff: {diff.mean():+.1f}  (positive = listwise > pairwise)")
        rho_s, _ = spearmanr(valid["pairwise"], valid["listwise"])
        rho_p, _ = pearsonr(valid["pairwise"], valid["listwise"])
        print(f"  Spearman rho:     {rho_s:.3f}")
        print(f"  Pearson r:        {rho_p:.3f}")

        # By domain
        for dom in ["drugs", "weapon"]:
            sub = valid[valid["domain"]==dom]
            if len(sub) < 3: continue
            d = sub["listwise"] - sub["pairwise"]
            print(f"\n  {dom}:  n={len(sub)}  abs_diff={d.abs().mean():.1f}  mean_diff={d.mean():+.1f}  rho={spearmanr(sub['pairwise'], sub['listwise'])[0]:.3f}")

    # Show all per-target tables
    print("\n" + "="*70)
    print("PER-PAIR SCORES")
    print("="*70)
    for (dom, tgt), grp in valid.groupby(["domain","target"]):
        print(f"\n{dom} | target={tgt}")
        sub = grp[["candidate","pairwise","listwise"]].copy()
        sub["diff"] = sub["listwise"] - sub["pairwise"]
        print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
