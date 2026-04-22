# Statistical Significance — LLM-rep vs. Embedding Baselines

_Test: one-sample Wilcoxon signed-rank, one-sided (H1: LLM-rep median > best embedding)._
_N=11 LLM models per cell. Best embedding = max across 24 embedding cells (4 text-embedding models + 7 reps × 4 = 28)._
_FDR / Bonferroni applied within each of 8 metric families (14 cells each)._

## 1. Overall pass-rates by metric (14 cells = 7 reps × 2 domains)

| metric       |   cells |   pass_raw |   pass_fdr |   pass_bonf |   pass_vs_random |
|:-------------|--------:|-----------:|-----------:|------------:|-----------------:|
| QWK_Oracle   |      14 |         13 |         13 |          10 |               14 |
| QWK_CV       |      14 |          9 |          9 |           7 |               14 |
| F1_Oracle_b0 |      14 |          9 |          8 |           5 |               14 |
| F1_Oracle_b1 |      14 |          8 |          8 |           6 |               14 |
| F1_CV_b0     |      14 |          5 |          3 |           1 |               14 |
| F1_CV_b1     |      14 |          8 |          6 |           5 |               14 |
| AP_b0        |      14 |         12 |         12 |          11 |               14 |
| AP_b1        |      14 |         14 |         14 |          14 |               14 |

## 2. Detailed QWK significance (per cell)

_One row per (domain, rep), ordered by gap vs. best embedding._


### QWK_CV

| Domain   | Rep           |   LLM mean |   Best emb | Source                               |    Gap |   p(raw) |   p(FDR) |   p(Bonf) | Raw<.05   | FDR<.05   | Bonf<.05   |
|:---------|:--------------|-----------:|-----------:|:-------------------------------------|-------:|---------:|---------:|----------:|:----------|:----------|:-----------|
| drugs    | GPT-Law       |      0.741 |      0.756 | Emb-on-Manual+Gemini-embedding-001   | -0.015 |   0.6812 |   0.7662 |    1      |           |           |            |
| drugs    | GPT-Free      |      0.744 |      0.756 | Emb-on-Manual+Gemini-embedding-001   | -0.013 |   0.9126 |   0.9126 |    1      |           |           |            |
| drugs    | Hybrid-Manual |      0.75  |      0.756 | Emb-on-Manual+Gemini-embedding-001   | -0.006 |   0.7114 |   0.7662 |    1      |           |           |            |
| drugs    | Raw-Facts     |      0.761 |      0.756 | Emb-on-Manual+Gemini-embedding-001   |  0.005 |   0.3501 |   0.4456 |    1      |           |           |            |
| drugs    | Hybrid-Full   |      0.767 |      0.756 | Emb-on-Manual+Gemini-embedding-001   |  0.011 |   0.2065 |   0.2892 |    1      |           |           |            |
| drugs    | GPT-Schema    |      0.783 |      0.756 | Emb-on-Manual+Gemini-embedding-001   |  0.027 |   0.0122 |   0.019  |    0.1709 | ✓         | ✓         |            |
| drugs    | Manual        |      0.843 |      0.756 | Emb-on-Manual+Gemini-embedding-001   |  0.086 |   0.0005 |   0.0023 |    0.0068 | ✓         | ✓         | ✓          |
| weapon   | GPT-Law       |      0.66  |      0.568 | Emb-on-GPT-Free+Gemini-embedding-001 |  0.093 |   0.0068 |   0.012  |    0.0957 | ✓         | ✓         |            |
| weapon   | Raw-Facts     |      0.661 |      0.568 | Emb-on-GPT-Free+Gemini-embedding-001 |  0.094 |   0.001  |   0.0034 |    0.0137 | ✓         | ✓         | ✓          |
| weapon   | GPT-Free      |      0.67  |      0.568 | Emb-on-GPT-Free+Gemini-embedding-001 |  0.103 |   0.0015 |   0.0034 |    0.0205 | ✓         | ✓         | ✓          |
| weapon   | GPT-Schema    |      0.688 |      0.568 | Emb-on-GPT-Free+Gemini-embedding-001 |  0.121 |   0.0024 |   0.0049 |    0.0342 | ✓         | ✓         | ✓          |
| weapon   | Hybrid-Manual |      0.704 |      0.568 | Emb-on-GPT-Free+Gemini-embedding-001 |  0.137 |   0.0015 |   0.0034 |    0.0205 | ✓         | ✓         | ✓          |
| weapon   | Hybrid-Full   |      0.707 |      0.568 | Emb-on-GPT-Free+Gemini-embedding-001 |  0.139 |   0.0005 |   0.0023 |    0.0068 | ✓         | ✓         | ✓          |
| weapon   | Manual        |      0.743 |      0.568 | Emb-on-GPT-Free+Gemini-embedding-001 |  0.176 |   0.0005 |   0.0023 |    0.0068 | ✓         | ✓         | ✓          |

### QWK_Oracle

| Domain   | Rep           |   LLM mean |   Best emb | Source                              |   Gap |   p(raw) |   p(FDR) |   p(Bonf) | Raw<.05   | FDR<.05   | Bonf<.05   |
|:---------|:--------------|-----------:|-----------:|:------------------------------------|------:|---------:|---------:|----------:|:----------|:----------|:-----------|
| drugs    | GPT-Law       |      0.773 |      0.767 | Emb-on-Manual+Gemini-embedding-001  | 0.006 |   0.1826 |   0.1826 |    1      |           |           |            |
| drugs    | GPT-Free      |      0.787 |      0.767 | Emb-on-Manual+Gemini-embedding-001  | 0.019 |   0.0122 |   0.0131 |    0.1709 | ✓         | ✓         |            |
| drugs    | Raw-Facts     |      0.795 |      0.767 | Emb-on-Manual+Gemini-embedding-001  | 0.028 |   0.0049 |   0.0062 |    0.0684 | ✓         | ✓         |            |
| drugs    | Hybrid-Manual |      0.804 |      0.767 | Emb-on-Manual+Gemini-embedding-001  | 0.037 |   0.0005 |   0.0014 |    0.0068 | ✓         | ✓         | ✓          |
| drugs    | Hybrid-Full   |      0.808 |      0.767 | Emb-on-Manual+Gemini-embedding-001  | 0.041 |   0.0005 |   0.0014 |    0.0068 | ✓         | ✓         | ✓          |
| drugs    | GPT-Schema    |      0.823 |      0.767 | Emb-on-Manual+Gemini-embedding-001  | 0.055 |   0.0005 |   0.0014 |    0.0068 | ✓         | ✓         | ✓          |
| drugs    | Manual        |      0.859 |      0.767 | Emb-on-Manual+Gemini-embedding-001  | 0.091 |   0.0005 |   0.0014 |    0.0068 | ✓         | ✓         | ✓          |
| weapon   | Raw-Facts     |      0.69  |      0.628 | Emb-on-GPT-Law+Gemini-embedding-001 | 0.061 |   0.001  |   0.002  |    0.0137 | ✓         | ✓         | ✓          |
| weapon   | GPT-Law       |      0.692 |      0.628 | Emb-on-GPT-Law+Gemini-embedding-001 | 0.063 |   0.0122 |   0.0131 |    0.1709 | ✓         | ✓         |            |
| weapon   | GPT-Free      |      0.695 |      0.628 | Emb-on-GPT-Law+Gemini-embedding-001 | 0.066 |   0.0034 |   0.0048 |    0.0479 | ✓         | ✓         | ✓          |
| weapon   | GPT-Schema    |      0.726 |      0.628 | Emb-on-GPT-Law+Gemini-embedding-001 | 0.098 |   0.0034 |   0.0048 |    0.0479 | ✓         | ✓         | ✓          |
| weapon   | Hybrid-Manual |      0.734 |      0.628 | Emb-on-GPT-Law+Gemini-embedding-001 | 0.105 |   0.0024 |   0.0043 |    0.0342 | ✓         | ✓         | ✓          |
| weapon   | Hybrid-Full   |      0.74  |      0.628 | Emb-on-GPT-Law+Gemini-embedding-001 | 0.112 |   0.0005 |   0.0014 |    0.0068 | ✓         | ✓         | ✓          |
| weapon   | Manual        |      0.763 |      0.628 | Emb-on-GPT-Law+Gemini-embedding-001 | 0.134 |   0.001  |   0.002  |    0.0137 | ✓         | ✓         | ✓          |

## 3. Non-significant cells (LLM NOT significantly > best embedding, raw p≥0.05)

_34 / 112 cells._

| Domain   | Rep           | Metric       |   LLM mean |   Best emb |    Gap |   p(raw) |
|:---------|:--------------|:-------------|-----------:|-----------:|-------:|---------:|
| drugs    | Raw-Facts     | F1_CV_b0     |      0.841 |      0.841 | -0     |   0.5078 |
| drugs    | GPT-Schema    | F1_CV_b0     |      0.842 |      0.841 |  0.002 |   0.4829 |
| drugs    | GPT-Law       | F1_CV_b0     |      0.853 |      0.841 |  0.012 |   0.1826 |
| drugs    | GPT-Free      | F1_CV_b0     |      0.855 |      0.841 |  0.014 |   0.0835 |
| drugs    | Hybrid-Manual | F1_CV_b0     |      0.857 |      0.841 |  0.016 |   0.0693 |
| drugs    | GPT-Law       | F1_CV_b1     |      0.764 |      0.815 | -0.051 |   1      |
| drugs    | GPT-Free      | F1_CV_b1     |      0.774 |      0.815 | -0.041 |   0.9995 |
| drugs    | Hybrid-Manual | F1_CV_b1     |      0.788 |      0.815 | -0.027 |   0.9966 |
| drugs    | Hybrid-Full   | F1_CV_b1     |      0.79  |      0.815 | -0.025 |   0.9946 |
| drugs    | Raw-Facts     | F1_CV_b1     |      0.79  |      0.815 | -0.024 |   0.9976 |
| drugs    | GPT-Schema    | F1_CV_b1     |      0.818 |      0.815 |  0.003 |   0.3105 |
| drugs    | GPT-Law       | F1_Oracle_b0 |      0.867 |      0.857 |  0.009 |   0.1548 |
| drugs    | GPT-Schema    | F1_Oracle_b0 |      0.868 |      0.857 |  0.011 |   0.1826 |
| drugs    | GPT-Law       | F1_Oracle_b1 |      0.798 |      0.836 | -0.038 |   1      |
| drugs    | GPT-Free      | F1_Oracle_b1 |      0.81  |      0.836 | -0.027 |   0.9995 |
| drugs    | Raw-Facts     | F1_Oracle_b1 |      0.815 |      0.836 | -0.022 |   0.999  |
| drugs    | Hybrid-Full   | F1_Oracle_b1 |      0.821 |      0.836 | -0.015 |   0.9966 |
| drugs    | Hybrid-Manual | F1_Oracle_b1 |      0.822 |      0.836 | -0.015 |   0.9717 |
| drugs    | GPT-Schema    | F1_Oracle_b1 |      0.835 |      0.836 | -0.001 |   0.6089 |
| drugs    | GPT-Law       | QWK_CV       |      0.741 |      0.756 | -0.015 |   0.6812 |
| drugs    | GPT-Free      | QWK_CV       |      0.744 |      0.756 | -0.013 |   0.9126 |
| drugs    | Hybrid-Manual | QWK_CV       |      0.75  |      0.756 | -0.006 |   0.7114 |
| drugs    | Raw-Facts     | QWK_CV       |      0.761 |      0.756 |  0.005 |   0.3501 |
| drugs    | Hybrid-Full   | QWK_CV       |      0.767 |      0.756 |  0.011 |   0.2065 |
| drugs    | GPT-Law       | QWK_Oracle   |      0.773 |      0.767 |  0.006 |   0.1826 |
| weapon   | GPT-Free      | AP_b0        |      0.733 |      0.707 |  0.026 |   0.1392 |
| weapon   | GPT-Law       | AP_b0        |      0.733 |      0.707 |  0.026 |   0.0737 |
| weapon   | GPT-Free      | F1_CV_b0     |      0.702 |      0.705 | -0.002 |   0.7114 |
| weapon   | GPT-Law       | F1_CV_b0     |      0.712 |      0.705 |  0.008 |   0.2598 |
| weapon   | Raw-Facts     | F1_CV_b0     |      0.721 |      0.705 |  0.016 |   0.103  |
| weapon   | Hybrid-Manual | F1_CV_b0     |      0.729 |      0.705 |  0.024 |   0.1392 |
| weapon   | GPT-Law       | F1_Oracle_b0 |      0.736 |      0.729 |  0.007 |   0.2065 |
| weapon   | GPT-Free      | F1_Oracle_b0 |      0.736 |      0.729 |  0.007 |   0.2324 |
| weapon   | Raw-Facts     | F1_Oracle_b0 |      0.744 |      0.729 |  0.015 |   0.0591 |


## 4. Vs. Random Baseline (per cell Wilcoxon from permutation test)

_Permutation-test p-value from random_baseline.py (shuffle GT 1000x)._

| metric       |   total |   sig_vs_random |
|:-------------|--------:|----------------:|
| AP_b0        |      14 |              14 |
| AP_b1        |      14 |              14 |
| F1_CV_b0     |      14 |              14 |
| F1_CV_b1     |      14 |              14 |
| F1_Oracle_b0 |      14 |              14 |
| F1_Oracle_b1 |      14 |              14 |
| QWK_CV       |      14 |              14 |
| QWK_Oracle   |      14 |              14 |