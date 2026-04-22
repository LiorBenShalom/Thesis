# Random Permutation Baseline — Advisor's Method

_Method: shuffle GROUND TRUTH 1000 times per (domain, task) with fixed seed. One shared shuffle set per (domain, task), used across all 77 rep x model cells. Shuffling preserves class proportions exactly._

_Equivalent under the null to shuffling predictions; the GT-shuffle framing matches the canonical permutation-test formulation (see `new_try/code/calculate_baseline_CORRECT.py`)._


## 1. Shared baseline per (domain, metric)

_One number per (domain, metric) — the null distribution mean, averaged across rep x model cells that share the same shuffled GT. `frac_sig` = fraction of cells whose p-value < 0.05._


### DRUGS

| metric       |   Random baseline (mean) |   CI-lo |   CI-hi |   Observed (avg across 77 cells) |   Observed (best cell) |   Frac cells p<0.05 |
|:-------------|-------------------------:|--------:|--------:|---------------------------------:|-----------------------:|--------------------:|
| F1_Oracle_b0 |                    0.358 |   0.243 |   0.48  |                            0.881 |                  0.955 |                   1 |
| F1_Oracle_b1 |                    0.48  |   0.379 |   0.579 |                            0.821 |                  0.872 |                   1 |
| F1_CV_b0     |                    0.357 |   0.243 |   0.479 |                            0.859 |                  0.939 |                   1 |
| F1_CV_b1     |                    0.484 |   0.383 |   0.582 |                            0.794 |                  0.872 |                   1 |
| AP_b0        |                    0.368 |   0.299 |   0.464 |                            0.92  |                  0.975 |                   1 |
| AP_b1        |                    0.515 |   0.438 |   0.606 |                            0.885 |                  0.937 |                   1 |
| QWK_Oracle   |                   -0.002 |  -0.197 |   0.2   |                            0.807 |                  0.897 |                   1 |
| QWK_CV       |                   -0.001 |  -0.199 |   0.199 |                            0.77  |                  0.891 |                   1 |

### WEAPON

| metric       |   Random baseline (mean) |   CI-lo |   CI-hi |   Observed (avg across 77 cells) |   Observed (best cell) |   Frac cells p<0.05 |
|:-------------|-------------------------:|--------:|--------:|---------------------------------:|-----------------------:|--------------------:|
| F1_Oracle_b0 |                    0.354 |   0.252 |   0.464 |                            0.755 |                  0.863 |                   1 |
| F1_Oracle_b1 |                    0.481 |   0.395 |   0.567 |                            0.822 |                  0.887 |                   1 |
| F1_CV_b0     |                    0.355 |   0.25  |   0.464 |                            0.73  |                  0.863 |                   1 |
| F1_CV_b1     |                    0.478 |   0.391 |   0.565 |                            0.802 |                  0.887 |                   1 |
| AP_b0        |                    0.344 |   0.283 |   0.429 |                            0.775 |                  0.898 |                   1 |
| AP_b1        |                    0.474 |   0.409 |   0.554 |                            0.871 |                  0.939 |                   1 |
| QWK_Oracle   |                    0.001 |  -0.165 |   0.164 |                            0.72  |                  0.832 |                   1 |
| QWK_CV       |                    0.001 |  -0.165 |   0.164 |                            0.691 |                  0.815 |                   1 |


## 2. Per-rep summary — mean across 11 models


### DRUGS


**F1_Oracle_b0**

| rep           |   Observed |   Random null |   Δ (obs - null) |   Frac cells p<0.05 |
|:--------------|-----------:|--------------:|-----------------:|--------------------:|
| Manual        |      0.912 |         0.343 |            0.568 |                   1 |
| GPT-Schema    |      0.868 |         0.367 |            0.501 |                   1 |
| GPT-Free      |      0.878 |         0.359 |            0.52  |                   1 |
| GPT-Law       |      0.867 |         0.361 |            0.506 |                   1 |
| Raw-Facts     |      0.874 |         0.357 |            0.517 |                   1 |
| Hybrid-Manual |      0.88  |         0.362 |            0.517 |                   1 |
| Hybrid-Full   |      0.892 |         0.358 |            0.534 |                   1 |

**F1_Oracle_b1**

| rep           |   Observed |   Random null |   Δ (obs - null) |   Frac cells p<0.05 |
|:--------------|-----------:|--------------:|-----------------:|--------------------:|
| Manual        |      0.849 |         0.471 |            0.379 |                   1 |
| GPT-Schema    |      0.835 |         0.485 |            0.35  |                   1 |
| GPT-Free      |      0.81  |         0.493 |            0.316 |                   1 |
| GPT-Law       |      0.798 |         0.471 |            0.328 |                   1 |
| Raw-Facts     |      0.815 |         0.47  |            0.345 |                   1 |
| Hybrid-Manual |      0.822 |         0.48  |            0.342 |                   1 |
| Hybrid-Full   |      0.821 |         0.491 |            0.33  |                   1 |

**F1_CV_b0**

| rep           |   Observed |   Random null |   Δ (obs - null) |   Frac cells p<0.05 |
|:--------------|-----------:|--------------:|-----------------:|--------------------:|
| Manual        |      0.894 |         0.341 |            0.553 |                   1 |
| GPT-Schema    |      0.842 |         0.363 |            0.479 |                   1 |
| GPT-Free      |      0.855 |         0.357 |            0.498 |                   1 |
| GPT-Law       |      0.853 |         0.357 |            0.496 |                   1 |
| Raw-Facts     |      0.841 |         0.359 |            0.481 |                   1 |
| Hybrid-Manual |      0.857 |         0.361 |            0.496 |                   1 |
| Hybrid-Full   |      0.874 |         0.362 |            0.511 |                   1 |

**F1_CV_b1**

| rep           |   Observed |   Random null |   Δ (obs - null) |   Frac cells p<0.05 |
|:--------------|-----------:|--------------:|-----------------:|--------------------:|
| Manual        |      0.835 |         0.477 |            0.358 |                   1 |
| GPT-Schema    |      0.818 |         0.487 |            0.33  |                   1 |
| GPT-Free      |      0.774 |         0.498 |            0.275 |                   1 |
| GPT-Law       |      0.764 |         0.483 |            0.28  |                   1 |
| Raw-Facts     |      0.79  |         0.467 |            0.323 |                   1 |
| Hybrid-Manual |      0.788 |         0.481 |            0.306 |                   1 |
| Hybrid-Full   |      0.79  |         0.491 |            0.299 |                   1 |

**AP_b0**

| rep           |   Observed |   Random null |   Δ (obs - null) |   Frac cells p<0.05 |
|:--------------|-----------:|--------------:|-----------------:|--------------------:|
| Manual        |      0.956 |         0.369 |            0.587 |                   1 |
| GPT-Schema    |      0.936 |         0.371 |            0.565 |                   1 |
| GPT-Free      |      0.904 |         0.368 |            0.536 |                   1 |
| GPT-Law       |      0.89  |         0.368 |            0.522 |                   1 |
| Raw-Facts     |      0.903 |         0.366 |            0.537 |                   1 |
| Hybrid-Manual |      0.93  |         0.369 |            0.561 |                   1 |
| Hybrid-Full   |      0.92  |         0.368 |            0.553 |                   1 |

**AP_b1**

| rep           |   Observed |   Random null |   Δ (obs - null) |   Frac cells p<0.05 |
|:--------------|-----------:|--------------:|-----------------:|--------------------:|
| Manual        |      0.917 |         0.516 |            0.401 |                   1 |
| GPT-Schema    |      0.9   |         0.517 |            0.384 |                   1 |
| GPT-Free      |      0.871 |         0.515 |            0.356 |                   1 |
| GPT-Law       |      0.861 |         0.515 |            0.347 |                   1 |
| Raw-Facts     |      0.869 |         0.513 |            0.356 |                   1 |
| Hybrid-Manual |      0.895 |         0.516 |            0.379 |                   1 |
| Hybrid-Full   |      0.88  |         0.514 |            0.365 |                   1 |

**QWK_Oracle**

| rep           |   Observed |   Random null |   Δ (obs - null) |   Frac cells p<0.05 |
|:--------------|-----------:|--------------:|-----------------:|--------------------:|
| Manual        |      0.859 |        -0.001 |            0.86  |                   1 |
| GPT-Schema    |      0.823 |        -0.002 |            0.825 |                   1 |
| GPT-Free      |      0.787 |        -0.001 |            0.788 |                   1 |
| GPT-Law       |      0.773 |        -0.001 |            0.775 |                   1 |
| Raw-Facts     |      0.795 |        -0.002 |            0.797 |                   1 |
| Hybrid-Manual |      0.804 |        -0.002 |            0.806 |                   1 |
| Hybrid-Full   |      0.808 |        -0.002 |            0.81  |                   1 |

**QWK_CV**

| rep           |   Observed |   Random null |   Δ (obs - null) |   Frac cells p<0.05 |
|:--------------|-----------:|--------------:|-----------------:|--------------------:|
| Manual        |      0.843 |        -0.001 |            0.844 |                   1 |
| GPT-Schema    |      0.783 |        -0.002 |            0.785 |                   1 |
| GPT-Free      |      0.744 |        -0.001 |            0.745 |                   1 |
| GPT-Law       |      0.741 |        -0.001 |            0.742 |                   1 |
| Raw-Facts     |      0.761 |        -0.002 |            0.762 |                   1 |
| Hybrid-Manual |      0.75  |        -0.001 |            0.752 |                   1 |
| Hybrid-Full   |      0.767 |        -0.001 |            0.768 |                   1 |

### WEAPON


**F1_Oracle_b0**

| rep           |   Observed |   Random null |   Δ (obs - null) |   Frac cells p<0.05 |
|:--------------|-----------:|--------------:|-----------------:|--------------------:|
| Manual        |      0.787 |         0.354 |            0.433 |                   1 |
| GPT-Schema    |      0.759 |         0.35  |            0.409 |                   1 |
| GPT-Free      |      0.736 |         0.334 |            0.402 |                   1 |
| GPT-Law       |      0.736 |         0.372 |            0.364 |                   1 |
| Raw-Facts     |      0.744 |         0.354 |            0.391 |                   1 |
| Hybrid-Manual |      0.763 |         0.355 |            0.408 |                   1 |
| Hybrid-Full   |      0.763 |         0.36  |            0.403 |                   1 |

**F1_Oracle_b1**

| rep           |   Observed |   Random null |   Δ (obs - null) |   Frac cells p<0.05 |
|:--------------|-----------:|--------------:|-----------------:|--------------------:|
| Manual        |      0.846 |         0.472 |            0.374 |                   1 |
| GPT-Schema    |      0.82  |         0.475 |            0.344 |                   1 |
| GPT-Free      |      0.806 |         0.472 |            0.334 |                   1 |
| GPT-Law       |      0.813 |         0.479 |            0.334 |                   1 |
| Raw-Facts     |      0.805 |         0.484 |            0.32  |                   1 |
| Hybrid-Manual |      0.828 |         0.51  |            0.318 |                   1 |
| Hybrid-Full   |      0.835 |         0.474 |            0.36  |                   1 |

**F1_CV_b0**

| rep           |   Observed |   Random null |   Δ (obs - null) |   Frac cells p<0.05 |
|:--------------|-----------:|--------------:|-----------------:|--------------------:|
| Manual        |      0.766 |         0.352 |            0.414 |                   1 |
| GPT-Schema    |      0.737 |         0.35  |            0.387 |                   1 |
| GPT-Free      |      0.702 |         0.347 |            0.356 |                   1 |
| GPT-Law       |      0.712 |         0.368 |            0.345 |                   1 |
| Raw-Facts     |      0.721 |         0.354 |            0.367 |                   1 |
| Hybrid-Manual |      0.729 |         0.355 |            0.374 |                   1 |
| Hybrid-Full   |      0.742 |         0.359 |            0.383 |                   1 |

**F1_CV_b1**

| rep           |   Observed |   Random null |   Δ (obs - null) |   Frac cells p<0.05 |
|:--------------|-----------:|--------------:|-----------------:|--------------------:|
| Manual        |      0.828 |         0.471 |            0.357 |                   1 |
| GPT-Schema    |      0.801 |         0.475 |            0.326 |                   1 |
| GPT-Free      |      0.785 |         0.47  |            0.315 |                   1 |
| GPT-Law       |      0.793 |         0.48  |            0.313 |                   1 |
| Raw-Facts     |      0.785 |         0.484 |            0.302 |                   1 |
| Hybrid-Manual |      0.805 |         0.497 |            0.308 |                   1 |
| Hybrid-Full   |      0.821 |         0.472 |            0.348 |                   1 |

**AP_b0**

| rep           |   Observed |   Random null |   Δ (obs - null) |   Frac cells p<0.05 |
|:--------------|-----------:|--------------:|-----------------:|--------------------:|
| Manual        |      0.827 |         0.346 |            0.481 |                   1 |
| GPT-Schema    |      0.807 |         0.344 |            0.463 |                   1 |
| GPT-Free      |      0.733 |         0.344 |            0.389 |                   1 |
| GPT-Law       |      0.733 |         0.343 |            0.39  |                   1 |
| Raw-Facts     |      0.741 |         0.342 |            0.399 |                   1 |
| Hybrid-Manual |      0.803 |         0.344 |            0.459 |                   1 |
| Hybrid-Full   |      0.783 |         0.343 |            0.44  |                   1 |

**AP_b1**

| rep           |   Observed |   Random null |   Δ (obs - null) |   Frac cells p<0.05 |
|:--------------|-----------:|--------------:|-----------------:|--------------------:|
| Manual        |      0.898 |         0.475 |            0.423 |                   1 |
| GPT-Schema    |      0.879 |         0.475 |            0.404 |                   1 |
| GPT-Free      |      0.864 |         0.475 |            0.389 |                   1 |
| GPT-Law       |      0.847 |         0.474 |            0.373 |                   1 |
| Raw-Facts     |      0.846 |         0.474 |            0.372 |                   1 |
| Hybrid-Manual |      0.876 |         0.474 |            0.401 |                   1 |
| Hybrid-Full   |      0.885 |         0.474 |            0.411 |                   1 |

**QWK_Oracle**

| rep           |   Observed |   Random null |   Δ (obs - null) |   Frac cells p<0.05 |
|:--------------|-----------:|--------------:|-----------------:|--------------------:|
| Manual        |      0.763 |         0.001 |            0.762 |                   1 |
| GPT-Schema    |      0.726 |         0.001 |            0.725 |                   1 |
| GPT-Free      |      0.695 |         0.002 |            0.693 |                   1 |
| GPT-Law       |      0.692 |         0.002 |            0.69  |                   1 |
| Raw-Facts     |      0.69  |         0.001 |            0.688 |                   1 |
| Hybrid-Manual |      0.734 |         0.001 |            0.733 |                   1 |
| Hybrid-Full   |      0.74  |         0.001 |            0.739 |                   1 |

**QWK_CV**

| rep           |   Observed |   Random null |   Δ (obs - null) |   Frac cells p<0.05 |
|:--------------|-----------:|--------------:|-----------------:|--------------------:|
| Manual        |      0.743 |         0.001 |            0.743 |                   1 |
| GPT-Schema    |      0.688 |         0.001 |            0.687 |                   1 |
| GPT-Free      |      0.67  |         0.002 |            0.668 |                   1 |
| GPT-Law       |      0.66  |         0.002 |            0.658 |                   1 |
| Raw-Facts     |      0.661 |         0.002 |            0.66  |                   1 |
| Hybrid-Manual |      0.704 |         0     |            0.704 |                   1 |
| Hybrid-Full   |      0.707 |         0.002 |            0.705 |                   1 |