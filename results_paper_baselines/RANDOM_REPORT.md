# Random (Permutation) Baseline Report

_N permutations: fast metrics (AP, F1-Oracle) = 1000; slow metrics (CV, QWK-Oracle) = 100._
_Per-cell null: shuffle the model's own predicted scores and recompute the metric against fixed ground truth. Preserves the score marginal._

_Reported cell = mean across 11 LLM models. Δ = observed - null mean._


## DRUGS


### F1_Oracle_b0

| rep           |   Observed (mean) |   Random null (mean) |   Δ (obs - null) |
|:--------------|------------------:|---------------------:|-----------------:|
| Manual        |             0.912 |                0.533 |            0.379 |
| GPT-Schema    |             0.868 |                0.533 |            0.335 |
| GPT-Free      |             0.878 |                0.533 |            0.346 |
| GPT-Law       |             0.867 |                0.533 |            0.334 |
| Raw-Facts     |             0.874 |                0.533 |            0.341 |
| Hybrid-Manual |             0.88  |                0.533 |            0.347 |
| Hybrid-Full   |             0.892 |                0.533 |            0.359 |

### F1_Oracle_b1

| rep           |   Observed (mean) |   Random null (mean) |   Δ (obs - null) |
|:--------------|------------------:|---------------------:|-----------------:|
| Manual        |             0.849 |                0.674 |            0.176 |
| GPT-Schema    |             0.835 |                0.674 |            0.162 |
| GPT-Free      |             0.81  |                0.673 |            0.136 |
| GPT-Law       |             0.798 |                0.674 |            0.125 |
| Raw-Facts     |             0.815 |                0.673 |            0.141 |
| Hybrid-Manual |             0.822 |                0.674 |            0.148 |
| Hybrid-Full   |             0.821 |                0.674 |            0.148 |

### F1_CV_b0

| rep           |   Observed (mean) |   Random null (mean) |   Δ (obs - null) |
|:--------------|------------------:|---------------------:|-----------------:|
| Manual        |             0.894 |                0.506 |            0.388 |
| GPT-Schema    |             0.842 |                0.506 |            0.336 |
| GPT-Free      |             0.855 |                0.505 |            0.35  |
| GPT-Law       |             0.853 |                0.505 |            0.347 |
| Raw-Facts     |             0.841 |                0.505 |            0.336 |
| Hybrid-Manual |             0.857 |                0.504 |            0.352 |
| Hybrid-Full   |             0.874 |                0.506 |            0.367 |

### F1_CV_b1

| rep           |   Observed (mean) |   Random null (mean) |   Δ (obs - null) |
|:--------------|------------------:|---------------------:|-----------------:|
| Manual        |             0.835 |                0.66  |            0.175 |
| GPT-Schema    |             0.818 |                0.659 |            0.158 |
| GPT-Free      |             0.774 |                0.659 |            0.115 |
| GPT-Law       |             0.764 |                0.659 |            0.104 |
| Raw-Facts     |             0.79  |                0.659 |            0.131 |
| Hybrid-Manual |             0.788 |                0.659 |            0.128 |
| Hybrid-Full   |             0.79  |                0.66  |            0.13  |

### AP_b0

| rep           |   Observed (mean) |   Random null (mean) |   Δ (obs - null) |
|:--------------|------------------:|---------------------:|-----------------:|
| Manual        |             0.956 |                0.378 |            0.579 |
| GPT-Schema    |             0.936 |                0.379 |            0.557 |
| GPT-Free      |             0.904 |                0.377 |            0.527 |
| GPT-Law       |             0.89  |                0.376 |            0.514 |
| Raw-Facts     |             0.903 |                0.376 |            0.527 |
| Hybrid-Manual |             0.93  |                0.377 |            0.553 |
| Hybrid-Full   |             0.92  |                0.377 |            0.543 |

### AP_b1

| rep           |   Observed (mean) |   Random null (mean) |   Δ (obs - null) |
|:--------------|------------------:|---------------------:|-----------------:|
| Manual        |             0.917 |                0.522 |            0.394 |
| GPT-Schema    |             0.9   |                0.522 |            0.379 |
| GPT-Free      |             0.871 |                0.523 |            0.349 |
| GPT-Law       |             0.861 |                0.523 |            0.338 |
| Raw-Facts     |             0.869 |                0.523 |            0.346 |
| Hybrid-Manual |             0.895 |                0.522 |            0.372 |
| Hybrid-Full   |             0.88  |                0.522 |            0.358 |

### QWK_Oracle

| rep           |   Observed (mean) |   Random null (mean) |   Δ (obs - null) |
|:--------------|------------------:|---------------------:|-----------------:|
| Manual        |             0.859 |                0.101 |            0.757 |
| GPT-Schema    |             0.823 |                0.1   |            0.723 |
| GPT-Free      |             0.787 |                0.1   |            0.687 |
| GPT-Law       |             0.773 |                0.099 |            0.675 |
| Raw-Facts     |             0.795 |                0.097 |            0.698 |
| Hybrid-Manual |             0.804 |                0.096 |            0.708 |
| Hybrid-Full   |             0.808 |                0.095 |            0.713 |

### QWK_CV

| rep           |   Observed (mean) |   Random null (mean) |   Δ (obs - null) |
|:--------------|------------------:|---------------------:|-----------------:|
| Manual        |             0.843 |               -0.016 |            0.858 |
| GPT-Schema    |             0.783 |               -0.013 |            0.796 |
| GPT-Free      |             0.744 |               -0.009 |            0.753 |
| GPT-Law       |             0.741 |               -0.013 |            0.754 |
| Raw-Facts     |             0.761 |               -0.011 |            0.772 |
| Hybrid-Manual |             0.75  |               -0.011 |            0.761 |
| Hybrid-Full   |             0.767 |               -0.011 |            0.778 |

## WEAPON


### F1_Oracle_b0

| rep           |   Observed (mean) |   Random null (mean) |   Δ (obs - null) |
|:--------------|------------------:|---------------------:|-----------------:|
| Manual        |             0.787 |                0.504 |            0.283 |
| GPT-Schema    |             0.759 |                0.503 |            0.256 |
| GPT-Free      |             0.736 |                0.504 |            0.232 |
| GPT-Law       |             0.736 |                0.503 |            0.232 |
| Raw-Facts     |             0.744 |                0.504 |            0.241 |
| Hybrid-Manual |             0.763 |                0.504 |            0.259 |
| Hybrid-Full   |             0.763 |                0.504 |            0.26  |

### F1_Oracle_b1

| rep           |   Observed (mean) |   Random null (mean) |   Δ (obs - null) |
|:--------------|------------------:|---------------------:|-----------------:|
| Manual        |             0.846 |                0.637 |            0.208 |
| GPT-Schema    |             0.82  |                0.637 |            0.182 |
| GPT-Free      |             0.806 |                0.637 |            0.169 |
| GPT-Law       |             0.813 |                0.637 |            0.176 |
| Raw-Facts     |             0.805 |                0.637 |            0.168 |
| Hybrid-Manual |             0.828 |                0.637 |            0.191 |
| Hybrid-Full   |             0.835 |                0.637 |            0.198 |

### F1_CV_b0

| rep           |   Observed (mean) |   Random null (mean) |   Δ (obs - null) |
|:--------------|------------------:|---------------------:|-----------------:|
| Manual        |             0.766 |                0.482 |            0.283 |
| GPT-Schema    |             0.737 |                0.481 |            0.256 |
| GPT-Free      |             0.702 |                0.482 |            0.22  |
| GPT-Law       |             0.712 |                0.482 |            0.23  |
| Raw-Facts     |             0.721 |                0.484 |            0.238 |
| Hybrid-Manual |             0.729 |                0.482 |            0.247 |
| Hybrid-Full   |             0.742 |                0.482 |            0.26  |

### F1_CV_b1

| rep           |   Observed (mean) |   Random null (mean) |   Δ (obs - null) |
|:--------------|------------------:|---------------------:|-----------------:|
| Manual        |             0.828 |                0.625 |            0.203 |
| GPT-Schema    |             0.801 |                0.624 |            0.177 |
| GPT-Free      |             0.785 |                0.625 |            0.159 |
| GPT-Law       |             0.793 |                0.626 |            0.167 |
| Raw-Facts     |             0.785 |                0.625 |            0.161 |
| Hybrid-Manual |             0.805 |                0.625 |            0.18  |
| Hybrid-Full   |             0.821 |                0.625 |            0.195 |

### AP_b0

| rep           |   Observed (mean) |   Random null (mean) |   Δ (obs - null) |
|:--------------|------------------:|---------------------:|-----------------:|
| Manual        |             0.827 |                0.349 |            0.478 |
| GPT-Schema    |             0.807 |                0.349 |            0.458 |
| GPT-Free      |             0.733 |                0.35  |            0.383 |
| GPT-Law       |             0.733 |                0.35  |            0.383 |
| Raw-Facts     |             0.741 |                0.351 |            0.391 |
| Hybrid-Manual |             0.803 |                0.349 |            0.454 |
| Hybrid-Full   |             0.783 |                0.35  |            0.434 |

### AP_b1

| rep           |   Observed (mean) |   Random null (mean) |   Δ (obs - null) |
|:--------------|------------------:|---------------------:|-----------------:|
| Manual        |             0.898 |                0.478 |            0.42  |
| GPT-Schema    |             0.879 |                0.477 |            0.401 |
| GPT-Free      |             0.864 |                0.479 |            0.385 |
| GPT-Law       |             0.847 |                0.478 |            0.368 |
| Raw-Facts     |             0.846 |                0.479 |            0.366 |
| Hybrid-Manual |             0.876 |                0.478 |            0.398 |
| Hybrid-Full   |             0.885 |                0.478 |            0.407 |

### QWK_Oracle

| rep           |   Observed (mean) |   Random null (mean) |   Δ (obs - null) |
|:--------------|------------------:|---------------------:|-----------------:|
| Manual        |             0.763 |                0.084 |            0.679 |
| GPT-Schema    |             0.726 |                0.083 |            0.643 |
| GPT-Free      |             0.695 |                0.083 |            0.611 |
| GPT-Law       |             0.692 |                0.083 |            0.609 |
| Raw-Facts     |             0.69  |                0.084 |            0.605 |
| Hybrid-Manual |             0.734 |                0.084 |            0.65  |
| Hybrid-Full   |             0.74  |                0.083 |            0.657 |

### QWK_CV

| rep           |   Observed (mean) |   Random null (mean) |   Δ (obs - null) |
|:--------------|------------------:|---------------------:|-----------------:|
| Manual        |             0.743 |               -0.004 |            0.747 |
| GPT-Schema    |             0.688 |               -0.006 |            0.694 |
| GPT-Free      |             0.67  |               -0.011 |            0.681 |
| GPT-Law       |             0.66  |               -0.01  |            0.67  |
| Raw-Facts     |             0.661 |               -0.008 |            0.669 |
| Hybrid-Manual |             0.704 |               -0.005 |            0.709 |
| Hybrid-Full   |             0.707 |               -0.004 |            0.711 |