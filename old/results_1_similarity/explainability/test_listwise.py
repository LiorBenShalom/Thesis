#!/usr/bin/env python3
"""
Quick validation: listwise vs pairwise on Hybrid-Full GT pairs.

Design:
  - For each GT pair (target, partner, scale), build list of 10 candidates:
    partner + 9 random distractors from same domain GT
  - Run GPT-4.1 listwise: ask to score all 10 candidates 0-100 vs target
  - Compare listwise score for partner to its pairwise score (already available)
  - Compute correlation, rank agreement, and metric impact
"""
import json, os, re, sys, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
OUT = ROOT / "experiments/explainability_annotation/listwise_validation"
OUT.mkdir(parents=True, exist_ok=True)

random.seed(42)

# Load existing pairwise scores from multimodel experiment
PAIRWISE_DIR = ROOT / "experiments/explainability_annotation/multimodel_score_only/results"

SYSTEM_DRUGS = """את/ה מומחית לדין הפלילי בישראל. מוצגים בפנייך תיק יעד ורשימה של תיקים מתחרים — כולם בעבירות סמים.

המשימה: לדרג כל אחד מהתיקים המתחרים מול תיק היעד לפי דמיון מהותי — עד כמה כל תיק יכול לשמש כתקדים ענייני לתיק היעד.

נתחי כל ממד בקצרה:
1. סוג העבירה וחומרתה
2. סוג הסם וכמותו
3. שיטת הביצוע (MO)
4. נסיבות הליבה
5. ישימות כתקדים

חשוב:
- התמקדי בפרמטרים המשפטיים המהותיים, לא בפרטים טכניים/ביורוקרטיים.
- השתייכות לאותו תחום לבדה לא מספיקה לדמיון מהותי.

פורמט תשובה: תני ציון נפרד לכל מתחרה. שורה אחת לכל מתחרה בפורמט:
CANDIDATE_<מספר>: <ציון 0-100>

ללא הסבר, רק שורות בפורמט הזה. 0 = שונים לחלוטין, 100 = זהים כמעט. ציון מעל 50 = תקדים רלוונטי."""

SYSTEM_WEAPON = """את/ה מומחית לדין הפלילי בישראל. מוצגים בפנייך תיק יעד ורשימה של תיקים מתחרים — כולם בעבירות נשק.

המשימה: לדרג כל אחד מהתיקים המתחרים מול תיק היעד לפי דמיון מהותי — עד כמה כל תיק יכול לשמש כתקדים ענייני לתיק היעד.

נתחי כל ממד בקצרה:
1. סוג העבירה וחומרתה
2. תפקיד הנאשם ומעורבותו
3. שיטת הביצוע (MO)
4. נסיבות הליבה (כמות תחמושת, שימוש, תכנון)
5. ישימות כתקדים

חשוב:
- התמקדי בפרמטרים המשפטיים המהותיים, לא בפרטים טכניים/ביורוקרטיים.
- השתייכות לאותו תחום לבדה לא מספיקה לדמיון מהותי.
- הבחנה בין 144(א) (החזקה) ל-144(ב) (נשיאה) היא מהותית.

פורמט תשובה: תני ציון נפרד לכל מתחרה. שורה אחת לכל מתחרה בפורמט:
CANDIDATE_<מספר>: <ציון 0-100>

ללא הסבר, רק שורות בפורמט הזה. 0 = שונים לחלוטין, 100 = זהים כמעט. ציון מעל 50 = תקדים רלוונטי."""


def build_user(target_fv: str, candidates_fv: list) -> str:
    parts = [f"תיק היעד:\n{target_fv}\n"]
    parts.append(f"\nתיקים מתחרים ({len(candidates_fv)}):\n")
    for i, fv in enumerate(candidates_fv, 1):
        parts.append(f"\n--- CANDIDATE_{i} ---\n{fv}")
    parts.append("\n\nתני ציון 0-100 לכל אחד מהמתחרים מול תיק היעד.")
    return "\n".join(parts)


def parse_listwise(text: str, n_candidates: int) -> dict[int, int]:
    """Returns {candidate_idx: score}."""
    out = {}
    for m in re.finditer(r"CANDIDATE_(\d+):\s*(\d+)", text):
        idx = int(m.group(1))
        score = int(m.group(2))
        if 1 <= idx <= n_candidates and 0 <= score <= 100:
            out[idx] = score
    return out


def main():
    cli = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    all_results = []
    for dom in ["drugs", "weapon"]:
        print(f"\n{'='*70}\n{dom.upper()}\n{'='*70}")
        gt = pd.read_csv(ROOT / dom / "similarity_database_hybrid_full_gpt.csv")
        # Build pool of all unique verdicts in GT
        all_verdicts = list(set(gt["verdict_1"]) | set(gt["verdict_2"]))
        # verdict → feature_vector (use first occurrence)
        v2fv = {}
        for _, r in gt.iterrows():
            v2fv.setdefault(r["verdict_1"], r["feature_vector_1"])
            v2fv.setdefault(r["verdict_2"], r["feature_vector_2"])

        # Load pairwise GPT-4 scores (from multimodel experiment)
        pw = pd.read_csv(PAIRWISE_DIR / f"gpt4__hybrid_full_gpt__{dom}.csv")
        pw_scores = {(r["verdict_1"], r["verdict_2"]): r["model_score"] for _, r in pw.iterrows()}

        # Sample 10 GT pairs (mix of all scales)
        sample = gt.sample(min(10, len(gt)), random_state=42).reset_index(drop=True)
        print(f"Sampling {len(sample)} pairs")

        for i, row in tqdm(list(sample.iterrows()), desc=dom):
            target = row["verdict_1"]
            partner = row["verdict_2"]
            true_scale = int(row["similarity_scale"])
            target_fv = row["feature_vector_1"]
            partner_fv = row["feature_vector_2"]

            # Build 10 candidates: partner + 9 random distractors
            distractor_pool = [v for v in all_verdicts if v not in (target, partner) and v in v2fv]
            distractors = random.sample(distractor_pool, min(9, len(distractor_pool)))
            candidates = [partner] + distractors
            random.shuffle(candidates)
            partner_idx = candidates.index(partner) + 1  # 1-indexed
            cand_fvs = [v2fv[v] for v in candidates]

            # Listwise call
            sys_p = SYSTEM_DRUGS if dom == "drugs" else SYSTEM_WEAPON
            user = build_user(target_fv, cand_fvs)
            try:
                resp = cli.chat.completions.create(
                    model="gpt-4.1", messages=[
                        {"role": "system", "content": sys_p},
                        {"role": "user", "content": user},
                    ], temperature=0.1, max_tokens=300,
                )
                txt = resp.choices[0].message.content or ""
                scores = parse_listwise(txt, len(candidates))
            except Exception as e:
                tqdm.write(f"err: {e}")
                scores = {}

            partner_listwise = scores.get(partner_idx)
            partner_pairwise = pw_scores.get((target, partner))

            # Rank of partner among candidates
            if scores:
                ordered = sorted(scores.items(), key=lambda kv: -kv[1])
                rank = next((i for i, (k, _) in enumerate(ordered, 1) if k == partner_idx), None)
            else:
                rank = None

            all_results.append({
                "domain": dom, "target": target, "partner": partner, "true_scale": true_scale,
                "pairwise_score": partner_pairwise, "listwise_score": partner_listwise,
                "listwise_rank": rank, "n_candidates": len(candidates),
                "raw": txt[:500] if txt else "",
            })

    df = pd.DataFrame(all_results)
    df.to_csv(OUT / "validation.csv", index=False)
    print(f"\n✅ saved {OUT / 'validation.csv'}  ({len(df)} pairs)")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    valid = df.dropna(subset=["listwise_score","pairwise_score"])
    print(f"\nValid comparisons: {len(valid)}/{len(df)}")
    if len(valid) > 0:
        from scipy.stats import spearmanr, pearsonr
        rho_s, _ = spearmanr(valid["pairwise_score"], valid["listwise_score"])
        rho_p, _ = pearsonr(valid["pairwise_score"], valid["listwise_score"])
        print(f"\nPairwise vs Listwise correlation (over all pairs):")
        print(f"  Spearman: {rho_s:.3f}")
        print(f"  Pearson:  {rho_p:.3f}")

    print("\n## By true similarity scale:")
    for dom in ["drugs", "weapon"]:
        d = valid[valid["domain"]==dom]
        print(f"\n{dom}:")
        for sc in sorted(d["true_scale"].unique()):
            sub = d[d["true_scale"]==sc]
            print(f"  scale={sc} (n={len(sub)}):")
            print(f"    pairwise mean: {sub['pairwise_score'].mean():.1f}")
            print(f"    listwise mean: {sub['listwise_score'].mean():.1f}")
            print(f"    median rank of partner (out of 10): {sub['listwise_rank'].median():.0f}")


if __name__ == "__main__":
    main()
