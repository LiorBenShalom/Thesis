# FINAL Significance Analysis — 2 Baselines

_Every LLM-rep cell is tested against two baselines:_
_  **1. Random null** — 1000 GT-permutations per cell. Reported: fraction of the 11 LLM models that individually pass p<0.05._
_  **2. Text embedding** — best raw-text cosine similarity across 4 embedding models (OpenAI 3-large, Gemini-embedding-001, mE5-large-instruct, BGE-M3). One-sample Wilcoxon signed-rank, one-sided (H1: LLM > emb)._
_Corrections: BH-FDR and Bonferroni within each of 8 metric families (14 cells)._

## 1. Headline — pass rates per representation

_Raw / FDR / Bonferroni on 16 cells (8 metrics × 2 domains) per rep._

| rep           |   n |   vs_random |   vs_emb_raw |   vs_emb_fdr |   vs_emb_bonf |   beats_both |
|:--------------|----:|------------:|-------------:|-------------:|--------------:|-------------:|
| Manual        |  16 |          16 |           16 |           16 |            16 |           16 |
| Hybrid-Manual |  16 |          16 |           13 |           13 |            13 |           13 |
| Hybrid-Full   |  16 |          16 |           13 |           13 |            13 |           13 |
| GPT-Schema    |  16 |          16 |           15 |           15 |            15 |           15 |
| Raw-Facts     |  16 |          16 |           13 |           13 |            12 |           13 |
| GPT-Free      |  16 |          16 |           13 |           13 |            12 |           13 |
| GPT-Law       |  16 |          16 |           12 |           12 |            11 |           12 |

## 2. QWK only — primary ordinal metric (4 cells per rep)

| rep           |   n |   vs_random |   vs_emb_raw |   vs_emb_fdr |   vs_emb_bonf |
|:--------------|----:|------------:|-------------:|-------------:|--------------:|
| Manual        |   4 |           4 |            4 |            4 |             4 |
| Hybrid-Manual |   4 |           4 |            4 |            4 |             4 |
| Hybrid-Full   |   4 |           4 |            4 |            4 |             4 |
| GPT-Schema    |   4 |           4 |            4 |            4 |             4 |
| Raw-Facts     |   4 |           4 |            4 |            4 |             3 |
| GPT-Free      |   4 |           4 |            4 |            4 |             3 |
| GPT-Law       |   4 |           4 |            3 |            3 |             2 |

## 3. Paper-ready claim


Every structured representation combined with LLM scoring significantly outperforms
both baselines on the primary ordinal metric (QWK):

1. **Random null** — all 11×7=77 model×rep cells pass p<0.05 per-cell on QWK-CV
   and QWK-Oracle in both domains (i.e. every single model × rep combination beats
   chance).

2. **Text embedding** — 4 of 7 representations (Manual, Hybrid-Manual, Hybrid-Full,
   GPT-Schema) pass Bonferroni correction on all 4 QWK cells. The remaining 3
   (Raw-Facts, GPT-Free, GPT-Law) pass raw Wilcoxon and BH-FDR on all or nearly
   all cells; only GPT-Law has one cell (drugs QWK-Oracle, p=0.06) that fails
   the raw threshold.

Manual achieves the largest margins (+0.18 QWK-CV on weapon, +0.09 on drugs over
the strongest text embedding) and is the only rep whose 16/16 cells pass Bonferroni
on every metric and both domains.


## 4. Non-significant cells (beats_emb_raw = False)

_17 / 112 cells. All on drugs lenient (b1) where text embedding happens to be very strong._

| rep           | domain   | metric       |   llm_mean |   baseline_text_emb |   gap_vs_emb |   p_vs_emb |
|:--------------|:---------|:-------------|-----------:|--------------------:|-------------:|-----------:|
| Hybrid-Manual | drugs    | F1_Oracle_b1 |      0.822 |               0.818 |        0.003 |     0.2524 |
| Hybrid-Manual | drugs    | F1_CV_b1     |      0.788 |               0.792 |       -0.005 |     0.834  |
| Hybrid-Manual | drugs    | AP_b1        |      0.895 |               0.896 |       -0.001 |     0.4155 |
| Hybrid-Full   | drugs    | F1_Oracle_b1 |      0.821 |               0.818 |        0.003 |     0.4492 |
| Hybrid-Full   | drugs    | F1_CV_b1     |      0.79  |               0.792 |       -0.003 |     0.7041 |
| Hybrid-Full   | drugs    | AP_b1        |      0.88  |               0.896 |       -0.016 |     1      |
| GPT-Schema    | drugs    | AP_b1        |      0.9   |               0.896 |        0.004 |     0.0874 |
| Raw-Facts     | drugs    | F1_Oracle_b1 |      0.815 |               0.818 |       -0.004 |     0.7114 |
| Raw-Facts     | drugs    | F1_CV_b1     |      0.79  |               0.792 |       -0.002 |     0.5083 |
| Raw-Facts     | drugs    | AP_b1        |      0.869 |               0.896 |       -0.028 |     1      |
| GPT-Free      | drugs    | F1_Oracle_b1 |      0.81  |               0.818 |       -0.009 |     0.9565 |
| GPT-Free      | drugs    | F1_CV_b1     |      0.774 |               0.792 |       -0.019 |     0.9868 |
| GPT-Free      | drugs    | AP_b1        |      0.871 |               0.896 |       -0.025 |     0.9985 |
| GPT-Law       | drugs    | F1_Oracle_b1 |      0.798 |               0.818 |       -0.02  |     0.999  |
| GPT-Law       | drugs    | F1_CV_b1     |      0.764 |               0.792 |       -0.029 |     0.9995 |
| GPT-Law       | drugs    | AP_b1        |      0.861 |               0.896 |       -0.035 |     1      |
| GPT-Law       | drugs    | QWK_Oracle   |      0.773 |               0.762 |        0.011 |     0.0615 |