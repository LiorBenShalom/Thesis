# Pairwise representation significance — continuous / ordinal task

_NOT a binary task — the labels are ordinal (1,2,3) and the LLM scores are continuous (0–100). Metrics:_
_- **QWK-Oracle**: Quadratic Weighted Kappa with best threshold per fold (upper bound)._
_- **QWK-CV**: 5-fold cross-validated threshold (generalisation estimate)._
_- **C-index**: ranking quality — probability that a random higher-label pair gets a higher predicted score._

Two-sided paired Wilcoxon (n=11 models) on each unordered rep-pair (21 pairs × 2 domains = 42 comparisons per metric). BH-FDR within each (metric, domain) cell.

## QWK_CV

| rep           |   drugs |   weapon |
|:--------------|--------:|---------:|
| Manual        |   0.843 |    0.743 |
| Hybrid-Manual |   0.75  |    0.704 |
| Hybrid-Full   |   0.767 |    0.707 |
| GPT-Schema    |   0.783 |    0.688 |
| GPT-Free      |   0.744 |    0.67  |
| GPT-Law       |   0.741 |    0.66  |
| Raw-Facts     |   0.761 |    0.661 |

### Significance summary (FDR p<0.05) across 2 domains:

| rep           |   wins |   losses |   ties |   net |
|:--------------|-------:|---------:|-------:|------:|
| Manual        |      9 |        0 |      3 |     9 |
| Hybrid-Full   |      1 |        1 |     10 |     0 |
| Hybrid-Manual |      0 |        1 |     11 |    -1 |
| GPT-Schema    |      1 |        2 |      9 |    -1 |
| Raw-Facts     |      0 |        1 |     11 |    -1 |
| GPT-Law       |      0 |        2 |     10 |    -2 |
| GPT-Free      |      0 |        4 |      8 |    -4 |
