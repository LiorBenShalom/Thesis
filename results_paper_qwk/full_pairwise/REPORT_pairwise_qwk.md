# Pairwise representation significance — continuous / ordinal task

_NOT a binary task — the labels are ordinal (1,2,3) and the LLM scores are continuous (0–100). Metrics:_
_- **QWK-Oracle**: Quadratic Weighted Kappa with best threshold per fold (upper bound)._
_- **QWK-CV**: 5-fold cross-validated threshold (generalisation estimate)._
_- **C-index**: ranking quality — probability that a random higher-label pair gets a higher predicted score._

Two-sided paired Wilcoxon (n=11 models) on each unordered rep-pair (21 pairs × 2 domains = 42 comparisons per metric). BH-FDR within each (metric, domain) cell.

## QWK_Oracle

| rep           |   drugs |   weapon |
|:--------------|--------:|---------:|
| Manual        |   0.859 |    0.763 |
| Hybrid-Manual |   0.804 |    0.734 |
| Hybrid-Full   |   0.808 |    0.74  |
| GPT-Schema    |   0.823 |    0.726 |
| GPT-Free      |   0.787 |    0.695 |
| GPT-Law       |   0.773 |    0.692 |
| Raw-Facts     |   0.795 |    0.69  |

### Significance summary (FDR p<0.05) across 2 domains:

| rep           |   wins |   losses |   ties |   net |
|:--------------|-------:|---------:|-------:|------:|
| Manual        |     11 |        0 |      1 |    11 |
| Hybrid-Full   |      4 |        1 |      7 |     3 |
| Hybrid-Manual |      4 |        2 |      6 |     2 |
| GPT-Schema    |      4 |        2 |      6 |     2 |
| Raw-Facts     |      1 |        4 |      7 |    -3 |
| GPT-Free      |      1 |        6 |      5 |    -5 |
| GPT-Law       |      0 |       10 |      2 |   -10 |

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

## C_index

| rep           |   drugs |   weapon |
|:--------------|--------:|---------:|
| Manual        |   0.91  |    0.901 |
| Hybrid-Manual |   0.901 |    0.888 |
| Hybrid-Full   |   0.894 |    0.885 |
| GPT-Schema    |   0.899 |    0.881 |
| GPT-Free      |   0.882 |    0.85  |
| GPT-Law       |   0.864 |    0.854 |
| Raw-Facts     |   0.877 |    0.854 |

### Significance summary (FDR p<0.05) across 2 domains:

| rep           |   wins |   losses |   ties |   net |
|:--------------|-------:|---------:|-------:|------:|
| Manual        |     12 |        0 |      0 |    12 |
| Hybrid-Manual |      6 |        2 |      4 |     4 |
| Hybrid-Full   |      6 |        2 |      4 |     4 |
| GPT-Schema    |      6 |        2 |      4 |     4 |
| GPT-Free      |      1 |        8 |      3 |    -7 |
| Raw-Facts     |      1 |        8 |      3 |    -7 |
| GPT-Law       |      0 |       10 |      2 |   -10 |
