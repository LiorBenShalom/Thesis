# Pairwise representation significance — continuous task (QWK 10-fold CV)

_The task uses ordinal labels {1,2,3} with continuous LLM scores (0–100)._
_Reported metric: **QWK 10-fold CV** (one paired score per (rep × model) cell across 11 models)._

**Methodology (matches the existing `significance_qwk_cv.csv`):**
- One-sided paired Wilcoxon signed-rank test (alternative: A > B), n=11 models
- BH-FDR correction within each focus rep (6 comparisons per focus)
- For each (rep, domain) cell: count significant 'beats' against the other 6 reps

## QWK_CV

### Mean QWK-CV across 11 models
| rep           |   drugs |   weapon |
|:--------------|--------:|---------:|
| Manual        |   0.843 |    0.743 |
| Hybrid-Manual |   0.75  |    0.704 |
| Hybrid-Full   |   0.767 |    0.707 |
| GPT-Schema    |   0.783 |    0.688 |
| GPT-Free      |   0.744 |    0.67  |
| GPT-Law       |   0.741 |    0.66  |
| Raw-Facts     |   0.761 |    0.661 |

### Significant beats (A > B, FDR p<0.05) — per focus rep × domain
| rep           | drugs wins   | weapon wins   |
|:--------------|:-------------|:--------------|
| Manual        | 6/6          | 6/6           |
| Hybrid-Manual | 0/6          | 0/6           |
| Hybrid-Full   | 0/6          | 1/6           |
| GPT-Schema    | 1/6          | 0/6           |
| GPT-Free      | 0/6          | 0/6           |
| GPT-Law       | 0/6          | 0/6           |
| Raw-Facts     | 0/6          | 0/6           |
