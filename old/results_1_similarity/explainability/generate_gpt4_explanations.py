#!/usr/bin/env python3
"""
Generate GPT-4.1 explanations on the same 100 drugs + 141 weapon H-Full GT pairs
that Claude Sonnet and Gemma were evaluated on.

Mirrors the format of `explainability_{drugs,weapon}_{claude_sonnet_4_6,gemma4_31b_or}.csv`.
Uses the same V6 score-raw system prompts.
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!")
DRUGS_DB = ROOT / "new_try/drugs/similarity_database_hybrid_full_gpt.csv"
WEAPON_DB = ROOT / "new_try/weapon/similarity_database_hybrid_full_gpt.csv"
OUT_DIR = ROOT / "new_try/experiments/explainability_annotation/hybrid_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-4.1"

# ---- V6 prompts (verbatim from src/scoring/structured_llm_comparison_experiment.py) ----
SYSTEM_DRUGS = """את/ה מומחית לדין הפלילי בישראל. מוצגים בפנייך **פיצ'רים מובנים** של שני תיקים פליליים.

המשימה: הערכת דמיון מהותי — עד כמה תיק אחד יכול לשמש כתקדים ענייני לשני?

נתחי כל ממד בנפרד:
1. **סוג העבירה וחומרתה** — האם מדובר באותו סוג עבירה? באותה רמת חומרה?
2.  סוג הסם וכמותו — האם מדובר בסוג דומה בחומרתו? בכמויות דומות בסדר גודל?
3. **שיטת הביצוע (MO)** — דפוס דומה? אמצעים דומים?
4. **נסיבות הליבה** — מעבדה, תכנון, נסיבות מחמירות/מקלות?
5. **ישימות כתקדים** — האם בית משפט יראה את שני התיקים כרלוונטיים זה לזה לצורכי ענישה?

חשוב:
- התמקדי בפרמטרים המשפטיים המהותיים, לא בפרטים טכניים/ביורוקרטיים.
- השתייכות לאותו תחום (סמים/נשק) לבדה **לא מספיקה** לדמיון מהותי.

פורמט תשובה:
1. ניתוח קצר (2-3 משפטים) של כל ממד
2. שורה אחרונה בדיוק בפורמט: SIMILARITY_SCORE: X
   כאשר X הוא מספר שלם בין 0 ל-100.
   0 = שונים לחלוטין, 100 = זהים כמעט. ציון מעל 50 = תקדים רלוונטי."""

SYSTEM_WEAPON = """את/ה מומחית לדין הפלילי בישראל. מוצגים בפנייך **פיצ'רים מובנים** של שני תיקים פליליים.

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


פורמט תשובה:
1. ניתוח קצר (2-3 משפטים) של כל ממד
2. שורה אחרונה בדיוק בפורמט: SIMILARITY_SCORE: X
   כאשר X הוא מספר שלם בין 0 ל-100.
   0 = שונים לחלוטין, 100 = זהים כמעט. ציון מעל 50 = תקדים רלוונטי."""

USER_TEMPLATE = """להלן שני תיקים פליליים עם פיצ'רים מובנים:

תיק 1:
{fv1}

תיק 2:
{fv2}

מהו ציון הדמיון המהותי (0-100)?"""


def parse_score(text: str) -> int | None:
    m = re.search(r"SIMILARITY_SCORE:\s*(\d+)", text)
    return int(m.group(1)) if m else None


def score_to_scale(score: int) -> int:
    if score < 25: return 0
    if score < 50: return 1
    if score < 75: return 2
    return 3


def call_one(client: OpenAI, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.1,
        max_tokens=2000,
    )
    return resp.choices[0].message.content or ""


def run_domain(domain: str, db_path: Path, system_prompt: str, workers: int = 8):
    df = pd.read_csv(db_path)
    print(f"\n=== {domain}: {len(df)} pairs ===")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    out_path = OUT_DIR / f"explainability_{domain}_gpt4.csv"

    # Resume support
    done: dict = {}
    if out_path.exists():
        existing = pd.read_csv(out_path)
        for _, r in existing.iterrows():
            done[(r["verdict_1"], r["verdict_2"])] = dict(r)
        print(f"  resuming — {len(done)} already done")

    rows = []
    pbar = tqdm(total=len(df), desc=domain)

    def task(idx, row):
        key = (row["verdict_1"], row["verdict_2"])
        if key in done:
            return idx, done[key], None
        user = USER_TEMPLATE.format(fv1=row["feature_vector_1"], fv2=row["feature_vector_2"])
        try:
            resp = call_one(client, system_prompt, user)
            score = parse_score(resp)
            scale = score_to_scale(score) if score is not None else None
            return idx, {
                "pair_id": idx + 1,
                "verdict_1": row["verdict_1"],
                "verdict_2": row["verdict_2"],
                "GT": row["similarity_scale"],
                "feature_vector_1": row["feature_vector_1"],
                "feature_vector_2": row["feature_vector_2"],
                "model_score": score if score is not None else "",
                "model_pred_scale": scale if scale is not None else "",
                "explanation": resp,
            }, None
        except Exception as e:
            return idx, None, f"{type(e).__name__}: {e}"

    results: dict = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(task, i, row) for i, row in df.iterrows()]
        for fut in as_completed(futs):
            idx, data, err = fut.result()
            if err:
                tqdm.write(f"  ❌ {idx}: {err}")
            else:
                results[idx] = data
            pbar.update(1)
            # Periodic save
            if len(results) % 25 == 0:
                pd.DataFrame([results[i] for i in sorted(results)]).to_csv(out_path, index=False)
    pbar.close()

    out_df = pd.DataFrame([results[i] for i in sorted(results)])
    out_df.to_csv(out_path, index=False)
    n_score = sum(1 for r in results.values() if r["model_score"] != "")
    print(f"  ✅ saved {len(out_df)} → {out_path.name}  (parsed score: {n_score}/{len(out_df)})")
    return out_path


def main():
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    run_domain("drugs", DRUGS_DB, SYSTEM_DRUGS)
    run_domain("weapon", WEAPON_DB, SYSTEM_WEAPON)


if __name__ == "__main__":
    main()
