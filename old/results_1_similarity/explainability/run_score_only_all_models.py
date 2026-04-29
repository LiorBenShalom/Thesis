#!/usr/bin/env python3
"""
Run the score-only V6 prompt on 241 GT pairs across 3 models:
  - GPT-4.1            (OpenAI)
  - Claude Sonnet 4.6  (Anthropic)
  - Gemma 4 31B        (OpenRouter)

Saves per (domain, model) CSV to:
  experiments/explainability_annotation/hybrid_full/score_only_{domain}_{model}.csv
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!")
DRUGS_DB = ROOT / "new_try/drugs/similarity_database_hybrid_full_gpt.csv"
WEAPON_DB = ROOT / "new_try/weapon/similarity_database_hybrid_full_gpt.csv"
OUT = ROOT / "new_try/experiments/explainability_annotation/hybrid_full"

# ---- Model config ----
MODELS = {
    "gpt4":              {"provider": "openai",     "id": "gpt-4.1"},
    "claude_sonnet_4_6": {"provider": "anthropic",  "id": os.getenv("CLAUDE_SONNET_4_6_MODEL", "claude-sonnet-4-5")},
    "gemma4_31b_or":     {"provider": "openrouter", "id": "google/gemma-4-31b-it"},
}

# ---- V6 score-only prompts (same as original V6 except removed line "1. ניתוח קצר ...") ----
SYSTEM_DRUGS = """את/ה מומחית לדין הפלילי בישראל. מוצגים בפנייך פיצ'רים מובנים של שני תיקים פליליים.

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

פורמט תשובה:שורה אחת בדיוק בפורמט: SIMILARITY_SCORE: X
   כאשר X הוא מספר שלם בין 0 ל-100.
   0 = שונים לחלוטין, 100 = זהים כמעט. ציון מעל 50 = תקדים רלוונטי."""

USER_TPL = """תיק 1:
{fv1}

תיק 2:
{fv2}

מהו ציון הדמיון המהותי (0-100)?"""


# ---- Provider clients (lazy) ----
_OPENAI = _ANTHROPIC = _OPENROUTER = None

def get_openai():
    global _OPENAI
    if _OPENAI is None:
        from openai import OpenAI
        _OPENAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _OPENAI

def get_anthropic():
    global _ANTHROPIC
    if _ANTHROPIC is None:
        import anthropic
        # Read from root .env (same logic as claude_annotate.py)
        if not os.getenv("ANTHROPIC_API_KEY"):
            for line in (ROOT / ".env").read_text().splitlines():
                if "=" in line and "ntropic" in line.lower():
                    os.environ["ANTHROPIC_API_KEY"] = line.split("=",1)[1].strip().strip('"').strip("'")
                    break
        _ANTHROPIC = anthropic.Anthropic()
    return _ANTHROPIC

def get_openrouter():
    global _OPENROUTER
    if _OPENROUTER is None:
        from openai import OpenAI
        _OPENROUTER = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
    return _OPENROUTER


def call_model(model_key: str, system: str, user: str) -> str:
    cfg = MODELS[model_key]
    if cfg["provider"] == "openai":
        resp = get_openai().chat.completions.create(
            model=cfg["id"],
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.1, max_tokens=80,
        )
        return resp.choices[0].message.content or ""
    if cfg["provider"] == "anthropic":
        # Claude tends to ignore "score-only" instruction and analyze anyway → need bigger budget
        resp = get_anthropic().messages.create(
            model=cfg["id"],
            max_tokens=1500, temperature=0.1,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return next(b.text for b in resp.content if b.type == "text")
    if cfg["provider"] == "openrouter":
        resp = get_openrouter().chat.completions.create(
            model=cfg["id"],
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.1, max_tokens=80,
        )
        return resp.choices[0].message.content or ""
    raise ValueError(cfg["provider"])


def parse_score(text: str):
    m = re.search(r"SIMILARITY_SCORE:\s*(\d+)", text or "")
    return int(m.group(1)) if m else None


def run_one(model_key: str, domain: str, db_path: Path, system: str, workers: int = 10):
    df = pd.read_csv(db_path)
    out_path = OUT / f"score_only_{domain}_{model_key}.csv"
    print(f"\n=== {model_key} × {domain}  ({len(df)} pairs)  →  {out_path.name}")

    done = {}
    if out_path.exists():
        ex = pd.read_csv(out_path)
        for _, r in ex.iterrows():
            done[(r["verdict_1"], r["verdict_2"])] = dict(r)

    results = {}
    pbar = tqdm(total=len(df), desc=f"{model_key} {domain}")

    def task(idx, row):
        key = (row["verdict_1"], row["verdict_2"])
        if key in done:
            return idx, done[key]
        user = USER_TPL.format(fv1=row["feature_vector_1"], fv2=row["feature_vector_2"])
        try:
            resp = call_model(model_key, system, user)
            return idx, {
                "verdict_1": row["verdict_1"], "verdict_2": row["verdict_2"],
                "GT": row["similarity_scale"],
                "model_score": parse_score(resp),
                "raw": resp,
            }
        except Exception as e:
            return idx, {"verdict_1": row["verdict_1"], "verdict_2": row["verdict_2"],
                         "GT": row["similarity_scale"], "model_score": None,
                         "raw": f"{type(e).__name__}: {e}"}

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
    print(f"  ✅ saved {len(out_df)} ({n_ok} parsed) → {out_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--domains", nargs="+", default=["drugs", "weapon"])
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    print(f"Models: {args.models}  Domains: {args.domains}  Workers: {args.workers}")
    for model_key in args.models:
        for dom in args.domains:
            db = DRUGS_DB if dom == "drugs" else WEAPON_DB
            sys_p = SYSTEM_DRUGS if dom == "drugs" else SYSTEM_WEAPON
            run_one(model_key, dom, db, sys_p, args.workers)


if __name__ == "__main__":
    main()
