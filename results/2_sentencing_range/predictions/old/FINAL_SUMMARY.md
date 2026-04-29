# Final summary — sentencing-range prediction

## Per-source ablation (paper-style pipeline)

All Hybrid-Full, citation-linked filter (with corrected canonical normalization), THR=60, k≥3, weighted_mean aggregation, +σ-filter at Q50 of σ_combined.


| config                     |   ('IoU', 'drugs') |   ('IoU', 'weapon') |   ('MAE_high', 'drugs') |   ('MAE_high', 'weapon') |   ('MAE_low', 'drugs') |   ('MAE_low', 'weapon') |   ('n', 'drugs') |   ('n', 'weapon') |
|:---------------------------|-------------------:|--------------------:|------------------------:|-------------------------:|-----------------------:|------------------------:|-----------------:|------------------:|
| HF +both (combined 144K)   |              0.583 |               0.606 |                   4.753 |                    6.677 |                  3.009 |                   4.776 |              396 |               236 |
| HF +external_cocite        |              0.606 |               0.627 |                   5.262 |                    6.059 |                  3.385 |                   4.445 |              233 |               196 |
| HF +internal_corrected     |              0.583 |               0.606 |                   4.753 |                    6.677 |                  3.009 |                   4.776 |              396 |               236 |
| HF orig (85K, buggy graph) |              0.581 |               0.622 |                   5.676 |                    6.32  |                  3.631 |                   4.606 |              198 |               190 |



## 4-way comparison: paper-style on combined 144K

Same pipeline as ablation. Each rep uses percentile-equivalent THR (so same fraction of pairs kept).


### no_sigma

| config                 |   ('IoU', 'drugs') |   ('IoU', 'weapon') |   ('MAE_high', 'drugs') |   ('MAE_high', 'weapon') |   ('MAE_low', 'drugs') |   ('MAE_low', 'weapon') |   ('n', 'drugs') |   ('n', 'weapon') |
|:-----------------------|-------------------:|--------------------:|------------------------:|-------------------------:|-----------------------:|------------------------:|-----------------:|------------------:|
| Gemini (paper-style)   |              0.529 |               0.499 |                   7.281 |                   11.757 |                  4.579 |                   8.799 |              765 |               483 |
| HF (paper-style)       |              0.542 |               0.499 |                   7.108 |                   13.089 |                  4.682 |                   9.929 |              791 |               472 |
| Random-K (paper-style) |              0.493 |               0.451 |                   8.632 |                   15.098 |                  5.606 |                  11.234 |              970 |               594 |
| TF-IDF (paper-style)   |              0.528 |               0.471 |                   7.431 |                   14.032 |                  4.672 |                  10.946 |              801 |               493 |


### with_sigma

| config                 |   ('IoU', 'drugs') |   ('IoU', 'weapon') |   ('MAE_high', 'drugs') |   ('MAE_high', 'weapon') |   ('MAE_low', 'drugs') |   ('MAE_low', 'weapon') |   ('n', 'drugs') |   ('n', 'weapon') |
|:-----------------------|-------------------:|--------------------:|------------------------:|-------------------------:|-----------------------:|------------------------:|-----------------:|------------------:|
| Gemini (paper-style)   |              0.558 |               0.584 |                   5.361 |                    7.928 |                  3.372 |                   5.8   |              383 |               242 |
| HF (paper-style)       |              0.583 |               0.606 |                   4.753 |                    6.677 |                  3.009 |                   4.776 |              396 |               236 |
| Random-K (paper-style) |              0.521 |               0.557 |                   6.391 |                    8.968 |                  4.092 |                   6.411 |              485 |               297 |
| TF-IDF (paper-style)   |              0.558 |               0.569 |                   5.702 |                    8.218 |                  3.549 |                   5.597 |              401 |               247 |



## 4-way comparison: top-3 mode on original 85K

Top-3 nearest neighbors per query, no sim threshold; agg=median (drugs) / softmax (weapon).


### no_sigma

| config                |   ('IoU', 'drugs') |   ('IoU', 'weapon') |   ('MAE_high', 'drugs') |   ('MAE_high', 'weapon') |   ('MAE_low', 'drugs') |   ('MAE_low', 'weapon') |   ('n', 'drugs') |   ('n', 'weapon') |
|:----------------------|-------------------:|--------------------:|------------------------:|-------------------------:|-----------------------:|------------------------:|-----------------:|------------------:|
| Gemini (top-3, 85K)   |              0.484 |               0.428 |                  10.214 |                   18.957 |                  6.461 |                  12.388 |              944 |               963 |
| HF (top-3, 85K)       |              0.499 |               0.471 |                   9.371 |                   17.421 |                  6.042 |                  11.146 |              944 |               963 |
| Random-K (top-3, 85K) |              0.419 |               0.34  |                  12.431 |                   24.944 |                  7.676 |                  17.152 |              944 |               963 |
| TF-IDF (top-3, 85K)   |              0.489 |               0.432 |                  10.104 |                   18.74  |                  6.341 |                  12.628 |              944 |               963 |


### with_sigma

| config                |   ('IoU', 'drugs') |   ('IoU', 'weapon') |   ('MAE_high', 'drugs') |   ('MAE_high', 'weapon') |   ('MAE_low', 'drugs') |   ('MAE_low', 'weapon') |   ('n', 'drugs') |   ('n', 'weapon') |
|:----------------------|-------------------:|--------------------:|------------------------:|-------------------------:|-----------------------:|------------------------:|-----------------:|------------------:|
| Gemini (top-3, 85K)   |              0.564 |               0.53  |                   6.886 |                    9.428 |                  4.343 |                   6.071 |              402 |               408 |
| HF (top-3, 85K)       |              0.573 |               0.586 |                   6.01  |                    8.325 |                  3.932 |                   5.171 |              399 |               405 |
| Random-K (top-3, 85K) |              0.479 |               0.433 |                   8.904 |                   15.851 |                  5.478 |                  10.339 |              416 |               430 |
| TF-IDF (top-3, 85K)   |              0.57  |               0.546 |                   6.409 |                    9.517 |                  4.102 |                   6.048 |              413 |               431 |

