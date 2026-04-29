#!/usr/bin/env python3
"""
Quick experiment: re-run GPT-4.1 on GT 241 pairs WITHOUT requesting explanation.
Compare metrics (QWK, C-index, AP-PR) to the with-explanation baseline.
"""
from __future__ import annotations
import json, os, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!")
DRUGS_DB = ROOT / "new_try/drugs/similarity_database_hybrid_full_gpt.csv"
WEAPON_DB = ROOT / "new_try/weapon/similarity_database_hybrid_full_gpt.csv"
OUT = ROOT / "new_try/experiments/explainability_annotation/hybrid_full"

MODEL = "gpt-4.1"

# Score-only prompts: same essence but request ONLY the score, no analysis
SYSTEM_DRUGS_SCORE_ONLY = """את/ה מומחית לדין הפלילי בישראל. מוצגים בפנייך פיצ'רים מובנים של שני תיקים פליליים.

המשימה: הערכת דמיון מהותי — עד כמה תיק אחד יכול לשמש כתקדים ענייני לשני?

נתחי כל ממד בנפרד:
1. **סוג העבירה וחומרתה** — האם מדובר באותו סוג עבירה? באותה רמת חומרה?
2. סוג הסם וכמותו — האם מדובר בסוג דומה בחומרתו? בכמויות דומות בסדר גודל?
3. **שיטת הביצוע (MO)** — דפוס דומה? אמצעים דומים?
4. **נסיבות הליבה** — מעבדה, תכנון, נסיבות מחמירות/מקלות?
5. **ישימות כתקדים** — האם בית משפט יראה את שני התיקים כרלוונטיים?

חשוב:
- התמקדי בפרמטרים המשפטיים המהותיים, לא בפרטים טכניים/ביורוקרטיים.
- השתייכות לאותו תחום לבדה לא מספיקה לדמיון מהותי.

פורמט תשובה:שורה אחת בדיוק בפורמט: SIMILARITY_SCORE: X
   כאשר X הוא מספר שלם בין 0 ל-100.
   0 = שונים לחלוטין, 100 = זהים כמעט. ציון מעל 50 = תקדים רלוונטי."""

SYSTEM_WEAPON_SCORE_ONLY = """את/ה מומחית לדין הפלילי בישראל. מוצגים בפנייך **פיצ'רים מובנים** של שני תיקים פליליים.

המשימה: הערכת דמיון מהותי — עד כמה תיק אחד יכול לשמש כתקדים ענייני לשני?

נתחי כל ממד בנפרד:
1. **סוג העבירה וחומרתה** — האם מדובר באותו סוג עבירה? באותה רמת חומרה?
2. **תפקיד הנאשם ומעורבותו** — יזם/סוחר לעומת שליח/מחזיק? מעורבות פעילה לעומת פסיבית?
3. **שיטת הביצוע (MO)** — דפוס דומה? אמצעים דומים?
4. **נסיבות הליבה** — כמות תחמושת (סדר גודל), שימוש בנשק, תכנון, נסיבות מחמירות/מקלות?
5. **ישימות כתקדים** — האם בית משפט יראה את שני התיקים כרלוונטיים זה לזה לצורכי ענישה?

חשוב:
- התמקדי בפרמטרים המשפטיים המהותיים, לא בפרטים טכניים/ביורוקרטיים.
- השתייכות לאותו תחום (סמים/נשק) לבדה **לא מספיקה** לדמיון מהותי.
- הבחנה בין 144(א) (החזקה בלבד) ל-144(ב) (נשיאה/הובלה) היא מהותית — תיק שהוא 144(א) לבדו כמעט אף פעם לא מהווה תקדים לתיק 144(ב).

פורמט תשובה:שורה אחת בדיוק בפורמט: SIMILARITY_SCORE: X
   כאשר X הוא מספר שלם בין 0 ל-100.
   0 = שונים לחלוטין, 100 = זהים כמעט. ציון מעל 50 = תקדים רלוונטי."""

USER_TPL = """תיק 1:
{fv1}

תיק 2:
{fv2}

מהו ציון הדמיון המהותי (0-100)?"""


def parse_score(text):
    m = re.search(r"SIMILARITY_SCORE:\s*(\d+)", text or "")
    return int(m.group(1)) if m else None


def call_one(client, system, user):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.1,
        max_tokens=50,  # keep it tiny — score-only
    )
    return resp.choices[0].message.content or ""


def run_domain(domain, db_path, system_prompt, workers=10):
    df = pd.read_csv(db_path)
    print(f"\n=== {domain}: {len(df)} pairs ===")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    out_path = OUT / f"score_only_{domain}_gpt4.csv"

    done = {}
    if out_path.exists():
        ex = pd.read_csv(out_path)
        for _, r in ex.iterrows():
            done[(r["verdict_1"], r["verdict_2"])] = dict(r)

    results = {}
    pbar = tqdm(total=len(df), desc=domain)

    def task(idx, row):
        key = (row["verdict_1"], row["verdict_2"])
        if key in done:
            return idx, done[key]
        user = USER_TPL.format(fv1=row["feature_vector_1"], fv2=row["feature_vector_2"])
        try:
            resp = call_one(client, system_prompt, user)
            return idx, {
                "verdict_1": row["verdict_1"], "verdict_2": row["verdict_2"],
                "GT": row["similarity_scale"],
                "model_score": parse_score(resp),
                "raw": resp,
            }
        except Exception as e:
            return idx, {"verdict_1": row["verdict_1"], "verdict_2": row["verdict_2"],
                         "GT": row["similarity_scale"], "model_score": None, "raw": str(e)}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(task, i, row) for i, row in df.iterrows()]
        for fut in as_completed(futs):
            idx, data = fut.result()
            results[idx] = data
            pbar.update(1)
            if len(results) % 50 == 0:
                pd.DataFrame([results[i] for i in sorted(results)]).to_csv(out_path, index=False)
    pbar.close()

    out_df = pd.DataFrame([results[i] for i in sorted(results)])
    out_df.to_csv(out_path, index=False)
    n_ok = out_df["model_score"].notna().sum()
    print(f"  ✅ saved {len(out_df)} → {out_path.name}  parsed={n_ok}")
    return out_path


def main():
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    run_domain("drugs", DRUGS_DB, SYSTEM_DRUGS_SCORE_ONLY)
    run_domain("weapon", WEAPON_DB, SYSTEM_WEAPON_SCORE_ONLY)


if __name__ == "__main__":
    main()
