# Significance vs. Text-Only Embedding Baseline

_Natural paper baseline: best raw-text embedding (no structured features)._
_Per (domain, metric), the baseline is max across 4 text-embedding models (OpenAI text-embedding-3-large, Gemini-embedding-001, mE5-large-instruct, BGE-M3)._
_Test: one-sample Wilcoxon signed-rank, one-sided (H1: LLM-rep mean > best text emb)._
_Corrections: BH-FDR and Bonferroni applied within each metric family (14 cells each)._

## 1. Pass rates per representation (16 cells each: 2 domains × 8 metrics)

| rep           |   total |   raw |   fdr |   bonf |
|:--------------|--------:|------:|------:|-------:|
| Manual        |      16 |    16 |    16 |     16 |
| Hybrid-Manual |      16 |    13 |    13 |     13 |
| Hybrid-Full   |      16 |    13 |    13 |     13 |
| GPT-Schema    |      16 |    15 |    15 |     15 |
| Raw-Facts     |      16 |    13 |    13 |     12 |
| GPT-Free      |      16 |    13 |    13 |     12 |
| GPT-Law       |      16 |    12 |    12 |     11 |

## 2. QWK-only pass rates per rep (4 cells each: 2 domains × 2 QWK variants)

| rep           |   total |   raw |   fdr |   bonf |
|:--------------|--------:|------:|------:|-------:|
| Manual        |       4 |     4 |     4 |      4 |
| Hybrid-Manual |       4 |     4 |     4 |      4 |
| Hybrid-Full   |       4 |     4 |     4 |      4 |
| GPT-Schema    |       4 |     4 |     4 |      4 |
| Raw-Facts     |       4 |     4 |     4 |      3 |
| GPT-Free      |       4 |     4 |     4 |      3 |
| GPT-Law       |       4 |     3 |     3 |      2 |

## 3. Detailed QWK tables


### QWK_CV

| Domain   | Rep           |   LLM mean |   Best text emb |   Gap |   p(raw) |   p(FDR) |   p(Bonf) | R   | F   | B   |
|:---------|:--------------|-----------:|----------------:|------:|---------:|---------:|----------:|:----|:----|:----|
| drugs    | Manual        |      0.843 |           0.667 | 0.176 |   0.0005 |   0.0005 |    0.0068 | ✓   | ✓   | ✓   |
| drugs    | Hybrid-Manual |      0.75  |           0.667 | 0.083 |   0.0005 |   0.0005 |    0.0068 | ✓   | ✓   | ✓   |
| drugs    | Hybrid-Full   |      0.767 |           0.667 | 0.1   |   0.0005 |   0.0005 |    0.0068 | ✓   | ✓   | ✓   |
| drugs    | GPT-Schema    |      0.783 |           0.667 | 0.116 |   0.0005 |   0.0005 |    0.0068 | ✓   | ✓   | ✓   |
| drugs    | Raw-Facts     |      0.761 |           0.667 | 0.094 |   0.0005 |   0.0005 |    0.0068 | ✓   | ✓   | ✓   |
| drugs    | GPT-Free      |      0.744 |           0.667 | 0.077 |   0.0005 |   0.0005 |    0.0068 | ✓   | ✓   | ✓   |
| drugs    | GPT-Law       |      0.741 |           0.667 | 0.074 |   0.0068 |   0.0068 |    0.0957 | ✓   | ✓   |     |
| weapon   | Manual        |      0.743 |           0.415 | 0.328 |   0.0005 |   0.0005 |    0.0068 | ✓   | ✓   | ✓   |
| weapon   | Hybrid-Manual |      0.704 |           0.415 | 0.289 |   0.0005 |   0.0005 |    0.0068 | ✓   | ✓   | ✓   |
| weapon   | Hybrid-Full   |      0.707 |           0.415 | 0.292 |   0.0005 |   0.0005 |    0.0068 | ✓   | ✓   | ✓   |
| weapon   | GPT-Schema    |      0.688 |           0.415 | 0.273 |   0.0005 |   0.0005 |    0.0068 | ✓   | ✓   | ✓   |
| weapon   | Raw-Facts     |      0.661 |           0.415 | 0.246 |   0.0005 |   0.0005 |    0.0068 | ✓   | ✓   | ✓   |
| weapon   | GPT-Free      |      0.67  |           0.415 | 0.255 |   0.0005 |   0.0005 |    0.0068 | ✓   | ✓   | ✓   |
| weapon   | GPT-Law       |      0.66  |           0.415 | 0.245 |   0.0005 |   0.0005 |    0.0068 | ✓   | ✓   | ✓   |

### QWK_Oracle

| Domain   | Rep           |   LLM mean |   Best text emb |   Gap |   p(raw) |   p(FDR) |   p(Bonf) | R   | F   | B   |
|:---------|:--------------|-----------:|----------------:|------:|---------:|---------:|----------:|:----|:----|:----|
| drugs    | Manual        |      0.859 |           0.762 | 0.096 |   0.0005 |   0.0006 |    0.0068 | ✓   | ✓   | ✓   |
| drugs    | Hybrid-Manual |      0.804 |           0.762 | 0.042 |   0.0005 |   0.0006 |    0.0068 | ✓   | ✓   | ✓   |
| drugs    | Hybrid-Full   |      0.808 |           0.762 | 0.046 |   0.0005 |   0.0006 |    0.0068 | ✓   | ✓   | ✓   |
| drugs    | GPT-Schema    |      0.823 |           0.762 | 0.061 |   0.0005 |   0.0006 |    0.0068 | ✓   | ✓   | ✓   |
| drugs    | Raw-Facts     |      0.795 |           0.762 | 0.033 |   0.0049 |   0.0053 |    0.0684 | ✓   | ✓   |     |
| drugs    | GPT-Free      |      0.787 |           0.762 | 0.025 |   0.0049 |   0.0053 |    0.0684 | ✓   | ✓   |     |
| drugs    | GPT-Law       |      0.773 |           0.762 | 0.011 |   0.0615 |   0.0615 |    0.8613 |     |     |     |
| weapon   | Manual        |      0.763 |           0.486 | 0.277 |   0.0005 |   0.0006 |    0.0068 | ✓   | ✓   | ✓   |
| weapon   | Hybrid-Manual |      0.734 |           0.486 | 0.248 |   0.0005 |   0.0006 |    0.0068 | ✓   | ✓   | ✓   |
| weapon   | Hybrid-Full   |      0.74  |           0.486 | 0.254 |   0.0005 |   0.0006 |    0.0068 | ✓   | ✓   | ✓   |
| weapon   | GPT-Schema    |      0.726 |           0.486 | 0.24  |   0.0005 |   0.0006 |    0.0068 | ✓   | ✓   | ✓   |
| weapon   | Raw-Facts     |      0.69  |           0.486 | 0.204 |   0.0005 |   0.0006 |    0.0068 | ✓   | ✓   | ✓   |
| weapon   | GPT-Free      |      0.695 |           0.486 | 0.209 |   0.0005 |   0.0006 |    0.0068 | ✓   | ✓   | ✓   |
| weapon   | GPT-Law       |      0.692 |           0.486 | 0.206 |   0.0005 |   0.0006 |    0.0068 | ✓   | ✓   | ✓   |

## 4. Non-significant cells (raw p>=0.05)

_17 / 112 cells non-significant. All on drugs lenient (b1) where Gemini raw text happens to be very strong._

| rep           | domain   | metric       |   llm_mean |   best_text_emb |    gap |   p_raw |
|:--------------|:---------|:-------------|-----------:|----------------:|-------:|--------:|
| Hybrid-Manual | drugs    | F1_Oracle_b1 |      0.822 |           0.818 |  0.003 |  0.2524 |
| Hybrid-Manual | drugs    | F1_CV_b1     |      0.788 |           0.792 | -0.005 |  0.834  |
| Hybrid-Manual | drugs    | AP_b1        |      0.895 |           0.896 | -0.001 |  0.4155 |
| Hybrid-Full   | drugs    | F1_Oracle_b1 |      0.821 |           0.818 |  0.003 |  0.4492 |
| Hybrid-Full   | drugs    | F1_CV_b1     |      0.79  |           0.792 | -0.003 |  0.7041 |
| Hybrid-Full   | drugs    | AP_b1        |      0.88  |           0.896 | -0.016 |  1      |
| GPT-Schema    | drugs    | AP_b1        |      0.9   |           0.896 |  0.004 |  0.0874 |
| Raw-Facts     | drugs    | F1_Oracle_b1 |      0.815 |           0.818 | -0.004 |  0.7114 |
| Raw-Facts     | drugs    | F1_CV_b1     |      0.79  |           0.792 | -0.002 |  0.5083 |
| Raw-Facts     | drugs    | AP_b1        |      0.869 |           0.896 | -0.028 |  1      |
| GPT-Free      | drugs    | F1_Oracle_b1 |      0.81  |           0.818 | -0.009 |  0.9565 |
| GPT-Free      | drugs    | F1_CV_b1     |      0.774 |           0.792 | -0.019 |  0.9868 |
| GPT-Free      | drugs    | AP_b1        |      0.871 |           0.896 | -0.025 |  0.9985 |
| GPT-Law       | drugs    | F1_Oracle_b1 |      0.798 |           0.818 | -0.02  |  0.999  |
| GPT-Law       | drugs    | F1_CV_b1     |      0.764 |           0.792 | -0.029 |  0.9995 |
| GPT-Law       | drugs    | AP_b1        |      0.861 |           0.896 | -0.035 |  1      |
| GPT-Law       | drugs    | QWK_Oracle   |      0.773 |           0.762 |  0.011 |  0.0615 |