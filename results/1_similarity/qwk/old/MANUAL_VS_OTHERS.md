# QWK Statistical Significance — Manual vs each alternative

_One-sided Wilcoxon signed-rank paired across 11 models, BH-FDR corrected (per focus)._


## Headline

**48 / 48** comparisons where Manual is significantly better at α=0.05.

_2 domains × 4 ordinal metrics × 6 alternatives = 48 tests._


## Table

| Config | vs Hybrid-Manual | vs Hybrid-Full | vs GPT-Schema | vs Raw-Facts | vs GPT-Free | vs GPT-Law |
|---|---|---|---|---|---|---|
| Drugs — QWK Oracle | *** | *** | *** | *** | *** | *** |
| Weapon — QWK Oracle | * | * | ** | ** | ** | ** |
| Drugs — QWK 10-fold | *** | *** | *** | *** | *** | *** |
| Weapon — QWK 10-fold | * | * | ** | * | ** | ** |
| Drugs — C-index | ** | *** | ** | *** | *** | *** |
| Weapon — C-index | ** | ** | *** | ** | *** | *** |
| Drugs — Spearman ρ | ** | ** | ** | *** | *** | *** |
| Weapon — Spearman ρ | ** | ** | *** | ** | *** | *** |

_* p<0.05  ** p<0.01  *** p<0.001  (FDR-corrected)._
