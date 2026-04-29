# Score-Only vs With-Explanation — Multi-Model Comparison

GT pairs: 100 drugs + 141 weapon. Same V6 prompt; only line `1. ניתוח קצר ...` removed.


## DRUGS


### QWK_scale

| model | with_expl | score_only | Δ |
|---|---|---|---|
| gpt4 | 0.700 | 0.745 | +0.045 🟢 |
| claude_sonnet_4_6 | 0.617 | 0.811 | +0.194 🟢 |
| gemma4_31b_or | 0.662 | 0.709 | +0.047 🟢 |

### C_index

| model | with_expl | score_only | Δ |
|---|---|---|---|
| gpt4 | 0.906 | 0.925 | +0.020 🟢 |
| claude_sonnet_4_6 | 0.944 | 0.964 | +0.020 🟢 |
| gemma4_31b_or | 0.924 | 0.929 | +0.005 🟢 |

### AP_strict

| model | with_expl | score_only | Δ |
|---|---|---|---|
| gpt4 | 0.921 | 0.913 | -0.007 🔴 |
| claude_sonnet_4_6 | 0.941 | 0.925 | -0.015 🔴 |
| gemma4_31b_or | 0.898 | 0.904 | +0.006 🟢 |

### AP_lenient

| model | with_expl | score_only | Δ |
|---|---|---|---|
| gpt4 | 0.886 | 0.879 | -0.006 🔴 |
| claude_sonnet_4_6 | 0.906 | 0.903 | -0.003 🔴 |
| gemma4_31b_or | 0.864 | 0.867 | +0.004 🟢 |

### Spearman

| model | with_expl | score_only | Δ |
|---|---|---|---|
| gpt4 | 0.757 | 0.778 | +0.021 🟢 |
| claude_sonnet_4_6 | 0.790 | 0.811 | +0.021 🟢 |
| gemma4_31b_or | 0.775 | 0.777 | +0.002 🟢 |

### Wilcoxon (per-pair score paired test)

| model | n | W | p | median Δ | mean Δ |
|---|---|---|---|---|---|
| gpt4 | 100 | 1412 | 0.0001 ⚠️ p<0.05 | +5.0 | +3.18 |
| claude_sonnet_4_6 | 63 | 30 | 0.0000 ⚠️ p<0.05 | +13.0 | +12.97 |
| gemma4_31b_or | 100 | 1188 | 0.0000 ⚠️ p<0.05 | +5.0 | +3.46 |

## WEAPON


### QWK_scale

| model | with_expl | score_only | Δ |
|---|---|---|---|
| gpt4 | 0.696 | 0.706 | +0.010 🟢 |
| claude_sonnet_4_6 | 0.532 | 0.476 | -0.056 🔴 |
| gemma4_31b_or | 0.666 | 0.638 | -0.028 🔴 |

### C_index

| model | with_expl | score_only | Δ |
|---|---|---|---|
| gpt4 | 0.921 | 0.936 | +0.015 🟢 |
| claude_sonnet_4_6 | 0.921 | 0.924 | +0.003 🟢 |
| gemma4_31b_or | 0.920 | 0.925 | +0.005 🟢 |

### AP_strict

| model | with_expl | score_only | Δ |
|---|---|---|---|
| gpt4 | 0.854 | 0.883 | +0.028 🟢 |
| claude_sonnet_4_6 | 0.761 | 0.723 | -0.038 🔴 |
| gemma4_31b_or | 0.808 | 0.806 | -0.002 🔴 |

### AP_lenient

| model | with_expl | score_only | Δ |
|---|---|---|---|
| gpt4 | 0.918 | 0.914 | -0.003 🔴 |
| claude_sonnet_4_6 | 0.933 | 0.827 | -0.107 🔴 |
| gemma4_31b_or | 0.907 | 0.878 | -0.028 🔴 |

### Spearman

| model | with_expl | score_only | Δ |
|---|---|---|---|
| gpt4 | 0.770 | 0.781 | +0.012 🟢 |
| claude_sonnet_4_6 | 0.776 | 0.723 | -0.053 🔴 |
| gemma4_31b_or | 0.768 | 0.753 | -0.015 🔴 |

### Wilcoxon (per-pair score paired test)

| model | n | W | p | median Δ | mean Δ |
|---|---|---|---|---|---|
| gpt4 | 141 | 3813 | 0.0133 ⚠️ p<0.05 | +5.0 | +1.89 |
| claude_sonnet_4_6 | 102 | 921 | 0.0000 ⚠️ p<0.05 | +7.0 | +6.25 |
| gemma4_31b_or | 141 | 4984 | 0.9636 | +0.0 | -0.37 |