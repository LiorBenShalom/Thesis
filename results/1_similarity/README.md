# Task 1 — Similarity prediction (ordinal scale)

Detecting whether two criminal verdicts are similar. Pairs annotated on
similarity_scale ∈ {1, 2, 3} (3 = most similar).

**Paper uses ordinal metrics only**: QWK, C-index, Spearman ρ.
No binary task (F1) in the current paper.

## Setup

- **GT**: 100 drugs pairs + 141 weapon pairs.
- **Model panel (N=13)**: claude_haiku_4_5, claude_sonnet_4_6, deepseek_r1_or,
  gemini_25_pro, gemini_3_flash, gemma4_31b_or, gpt4 (gpt-4.1), gpt51_thinking,
  gpt52, gpt5mini, kimi_k26_or, mistral_large_or, qwen3_vl_235b_or.
- **Representations (7)**: Manual, Hybrid-Manual, Hybrid-Full, Raw-Facts,
  GPT-Schema, GPT-Free, GPT-Law.
- **Prompt**: V6 score-only (returns `SIMILARITY_SCORE: X`).

## Layout

| Dir | What |
|---|---|
| `v6_final_predictions/` | Per-model × per-representation prediction CSVs (raw scores). |
| `qwk/` | **The canonical N=13 result.** QWK-Oracle, QWK-CV, Spearman ρ, C-index summary + CLD figures. `summary_qwk_n13.csv` has the headline table. |
| `baselines/` | Comparison vs Random null + 4 embedding baselines (OpenAI 3-large, Gemini-embedding-001, mE5-large-instruct, BGE-M3). Includes QWK / Spearman significance (Wilcoxon, BH-FDR, Bonferroni). Note: reports also contain F1/AP columns — ignore those, they're not in the paper. |

## Headline (QWK-CV, mean across 13 models)

| Domain | Manual | Hybrid-Manual | Hybrid-Full | GPT-Schema | Raw-Facts | GPT-Free | GPT-Law |
|---|---:|---:|---:|---:|---:|---:|---:|
| drugs | **.831** | .765 | .767 | .775 | .765 | .743 | .733 |
| weapon | **.757** | .707 | .699 | .720 | .655 | .674 | .671 |

Source: `qwk/summary_qwk_n13.csv`
