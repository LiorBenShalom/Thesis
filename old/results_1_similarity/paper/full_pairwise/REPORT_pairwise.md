# Full pairwise representation significance

7 representations × 11 models. Two-sided paired Wilcoxon (n=11) for each unordered pair (21 pairs per cell). Benjamini-Hochberg FDR applied across the **21 pairs within each (metric, domain, task) cell**.

Cells: 3 metrics (F1-Oracle, F1-CV, AP-PR) × 2 domains × 2 tasks = 12 cells. Total comparisons: 21 × 12 = **252 pairs**.

## F1

| rep           |   drugs/t0 |   drugs/t1 |   weapon/t0 |   weapon/t1 |
|:--------------|-----------:|-----------:|------------:|------------:|
| Manual        |      0.912 |      0.849 |       0.787 |       0.846 |
| Hybrid-Manual |      0.88  |      0.822 |       0.763 |       0.828 |
| Hybrid-Full   |      0.892 |      0.821 |       0.763 |       0.835 |
| GPT-Schema    |      0.868 |      0.835 |       0.759 |       0.82  |
| GPT-Free      |      0.878 |      0.81  |       0.736 |       0.806 |
| GPT-Law       |      0.867 |      0.798 |       0.736 |       0.813 |
| Raw-Facts     |      0.874 |      0.815 |       0.744 |       0.805 |

### Wins / losses across 4 cells (24 pairs per rep, FDR p<0.05):

| rep           |   wins |   losses |   ties |   net |
|:--------------|-------:|---------:|-------:|------:|
| Manual        |     18 |        0 |      6 |    18 |
| Hybrid-Full   |      4 |        2 |     18 |     2 |
| GPT-Schema    |      4 |        2 |     18 |     2 |
| Hybrid-Manual |      2 |        3 |     19 |    -1 |
| Raw-Facts     |      1 |        5 |     18 |    -4 |
| GPT-Free      |      1 |        8 |     15 |    -7 |
| GPT-Law       |      0 |       10 |     14 |   -10 |

## F1_CV

| rep           |   drugs/t0 |   drugs/t1 |   weapon/t0 |   weapon/t1 |
|:--------------|-----------:|-----------:|------------:|------------:|
| Manual        |      0.894 |      0.835 |       0.766 |       0.828 |
| Hybrid-Manual |      0.857 |      0.788 |       0.729 |       0.805 |
| Hybrid-Full   |      0.874 |      0.79  |       0.742 |       0.821 |
| GPT-Schema    |      0.842 |      0.818 |       0.737 |       0.801 |
| GPT-Free      |      0.855 |      0.774 |       0.702 |       0.785 |
| GPT-Law       |      0.853 |      0.764 |       0.712 |       0.793 |
| Raw-Facts     |      0.841 |      0.79  |       0.721 |       0.785 |

### Wins / losses across 4 cells (24 pairs per rep, FDR p<0.05):

| rep           |   wins |   losses |   ties |   net |
|:--------------|-------:|---------:|-------:|------:|
| Manual        |      7 |        0 |     17 |     7 |
| GPT-Schema    |      4 |        0 |     20 |     4 |
| Hybrid-Full   |      2 |        2 |     20 |     0 |
| Hybrid-Manual |      0 |        1 |     23 |    -1 |
| Raw-Facts     |      1 |        3 |     20 |    -2 |
| GPT-Free      |      0 |        3 |     21 |    -3 |
| GPT-Law       |      0 |        5 |     19 |    -5 |

## AP_PR

| rep           |   drugs/t0 |   drugs/t1 |   weapon/t0 |   weapon/t1 |
|:--------------|-----------:|-----------:|------------:|------------:|
| Manual        |      0.956 |      0.917 |       0.827 |       0.898 |
| Hybrid-Manual |      0.93  |      0.895 |       0.803 |       0.876 |
| Hybrid-Full   |      0.92  |      0.88  |       0.783 |       0.885 |
| GPT-Schema    |      0.936 |      0.9   |       0.807 |       0.879 |
| GPT-Free      |      0.904 |      0.871 |       0.733 |       0.864 |
| GPT-Law       |      0.89  |      0.861 |       0.733 |       0.847 |
| Raw-Facts     |      0.903 |      0.869 |       0.741 |       0.846 |

### Wins / losses across 4 cells (24 pairs per rep, FDR p<0.05):

| rep           |   wins |   losses |   ties |   net |
|:--------------|-------:|---------:|-------:|------:|
| Manual        |     23 |        0 |      1 |    23 |
| GPT-Schema    |     11 |        4 |      9 |     7 |
| Hybrid-Manual |      9 |        4 |     11 |     5 |
| Hybrid-Full   |      8 |        4 |     12 |     4 |
| GPT-Free      |      1 |       11 |     12 |   -10 |
| Raw-Facts     |      0 |       12 |     12 |   -12 |
| GPT-Law       |      0 |       17 |      7 |   -17 |
