# Paste-ready blocks for the paper — A4 (model versions), A5 (determinism), A1 (order)

Produced by `robustness_a1_a5.py` (A1/A5) and preds-file mtimes (A4). Canonical v6 scoring
(`SYSTEM_PROMPT_V6_SCORE_RAW_*` + `USER_TEMPLATE_SCORE_RAW` + `parse_score_v6`), OpenAI models.

---

## A4 — Model versions & access dates

| key | display | backend | API model string | access (≈, preds mtime) |
|---|---|---|---|---|
| gpt4 | GPT-4.1 | openai | `gpt-4.1` | 2026-04-16 |
| gpt5mini | GPT-5-mini | openai | `gpt-5-mini-2025-08-07` | 2026-04-16 |
| gpt52 | GPT-5.2 | openai | `gpt-5.2` | 2026-04-12 |
| gpt51_thinking | GPT-5.1 (reasoning) | openai | `gpt-5.1` | 2026-04-12 |
| claude_sonnet_4_6 | Claude Sonnet 4.6 | anthropic | `claude-sonnet-4-6` | 2026-04-16 |
| claude_haiku_4_5 | Claude Haiku 4.5 | anthropic | `claude-haiku-4-5` | 2026-04-28 |
| gemini_25_pro | Gemini 2.5 Pro | google | `gemini-2.5-pro` | 2026-04-16 |
| gemini_3_flash | Gemini 3 Flash | google | `gemini-3-flash-preview` | 2026-04-16 |
| gemma4_31b_or | Gemma-4-31B | openrouter | `google/gemma-4-31b-it` | 2026-04-16 |
| qwen3_vl_235b_or | Qwen3-VL-235B | openrouter | `qwen/qwen3-vl-235b-a22b-instruct` | 2026-04-16 |
| mistral_large_or | Mistral-Large | openrouter | `mistralai/mistral-large-2411` | 2026-04-27 |
| deepseek_r1_or | DeepSeek-R1 | openrouter | `deepseek/deepseek-r1` | 2026-04-27 |
| kimi_k26_or | Kimi-K2.6 | openrouter | `moonshotai/kimi-k2.6` | 2026-04-28 |

Sources: `experiments/src/common/model_config.py`, `code/similarity_experiment.py`
(`OPENROUTER_MODEL_MAP`, `GEMINI_*_MODEL`, `CLAUDE_SONNET_4_6_MODEL`), and
`v6_score_multimodel_experiment.py` (`claude-haiku-4-5`). Access dates are the
generation timestamps of the canonical per-model prediction files under
`experiments/results/1_similarity/v6_final_predictions/` (multi-stage run, 12–28 Apr 2026).

> ⚠️ Two file-level discrepancies resolved to the engine that produced the predictions:
> `claude_sonnet_4_6` → `claude-sonnet-4-6` (an older config listed `claude-sonnet-4-5`);
> `gemini_3_flash` → `gemini-3-flash-preview` (an older config listed `gemini-2.5-flash`).

---

## A5 — Determinism at temperature 0 (DONE, was "needs API")

Each pair re-scored twice in the **same** order; continuous 0–100 score + binary decision (≥50).

| model | domain | N | mean·\|Δ\| | max·\|Δ\| | exact-identical | **binary flips** |
|---|---|---|---|---|---|---|
| GPT-4.1 | drugs | 40 | 2.88 | 20 | 18/40 | 0/40 |
| GPT-5.2 | drugs | 20 | 2.70 | 6 | 7/20 | 0/20 |
| GPT-4.1 | weapon | 40 | 2.27 | 13 | 24/40 | 0/40 |
| GPT-5.2 | weapon | 20 | 1.15 | 6 | 12/20 | 1/20 |
| **pooled** | both | **120** | **≈2.3** | 20 | 61/120 | **1/120 (0.8%)** |

**Paper wording (EN):**
> *We empirically assessed run-to-run determinism by re-scoring 120 pairs (GPT-4.1 and GPT-5.2,
> both domains) twice at temperature 0. The continuous 0–100 similarity score shows only minor
> API-level jitter (mean |Δ|≈2.3, max 20; 51% of pairs identical), and the binary decision used
> for all reported metrics is essentially fully reproducible (1/120 = 0.8% label flips).*

**ניסוח (עברית):**
> *בדקנו אמפירית יציבות בין-ריצות ע"י ניקוד חוזר של 120 זוגות (GPT-4.1 ו-GPT-5.2, שני התחומים)
> פעמיים ב-temperature 0. הציון הרציף (0–100) מגלה רעש קל ברמת ה-API (mean |Δ|≈2.3), אך ההחלטה
> הבינארית — שעליה מבוססות כל המטריקות — יציבה כמעט לחלוטין (היפוך תווית ב-0.8% מהזוגות).*

---

## A1 — Pair-order independence (DONE, was "needs API")

Each pair also scored in the **reversed** order (fv2, fv1) and compared to the original.

| model | domain | N | mean·\|Δ\| | max·\|Δ\| | **binary flips** |
|---|---|---|---|---|---|
| GPT-4.1 | drugs | 40 | 4.20 | 15 | 1/40 (2.5%) |
| GPT-5.2 | drugs | 20 | 3.85 | 20 | 2/20 (10%) |
| GPT-4.1 | weapon | 40 | 6.03 | 20 | 6/40 (15%) |
| GPT-5.2 | weapon | 20 | 4.20 | 14 | 2/20 (10%) |
| **pooled** | both | **120** | **≈4.5** | 20 | **11/120 (9.2%)** |

**Honest finding**: order has a **small but real** effect — larger than the temp-0 jitter, and
stronger in the weapon domain (~13%). This is a limitation to disclose, not a clean
order-invariance claim.

**Paper wording (EN):**
> *To probe order sensitivity we re-scored each of 120 pairs with the two cases swapped. The
> continuous score shifts by mean |Δ|≈4.5/100, and the binary label flips in 9.2% of pairs
> (11/120; higher in the weapon domain, ~13%). All reported results use a single fixed
> presentation order; symmetrizing over both orders is a natural robustness extension and would
> mainly affect borderline pairs.*

**ניסוח (עברית):**
> *לבחינת רגישות-סדר ניקדנו מחדש 120 זוגות עם החלפת מקום שני התיקים. הציון הרציף משתנה ב-mean
> |Δ|≈4.5/100, וההחלטה הבינארית מתהפכת ב-9.2% מהזוגות (11/120; גבוה יותר בנשק, ~13%). כל
> התוצאות המדווחות משתמשות בסדר-הצגה קבוע אחד; סימטריזציה על שני הסדרים היא הרחבת-robustness
> טבעית, שתשפיע בעיקר על זוגות גבוליים.*

---

### Reproduce
```bash
cd experiments/analysis_representations
python3 robustness_a1_a5.py --domain drugs  --model gpt4  --n 40
python3 robustness_a1_a5.py --domain drugs  --model gpt52 --n 20
python3 robustness_a1_a5.py --domain weapon --model gpt4  --n 40
python3 robustness_a1_a5.py --domain weapon --model gpt52 --n 20
# raw per-pair scores -> out/robustness_{model}_{domain}.csv
```
