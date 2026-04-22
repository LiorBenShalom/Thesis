# Baselines — Paper-ready Report

_Compares three layers for each (domain, metric):_
_  1. **Random null** — permutation mean across 1000 shuffles per cell (77 cells avg)._
_  2. **Embedding baselines** — cosine similarity of verdict-fact embeddings (OpenAI 3-large, mE5-large-instruct, BGE-M3)._
_  3. **LLM reps** — scores averaged across 11 LLM models (from the main experiment)._

_**Bottom line:** random and embedding are baselines; LLM reps (especially Manual) should clearly dominate._


## DRUGS

| category                      |   F1-Oracle (b0 strict) |   F1-Oracle (b1 lenient) |   F1-CV (b0 strict) |   F1-CV (b1 lenient) |   AP-PR (b0 strict) |   AP-PR (b1 lenient) |   QWK-Oracle |   QWK-CV (10-fold) |
|:------------------------------|------------------------:|-------------------------:|--------------------:|---------------------:|--------------------:|---------------------:|-------------:|-------------------:|
| Random (null mean)            |                   0.533 |                    0.674 |               0.505 |                0.659 |               0.377 |                0.522 |        0.098 |             -0.012 |
| Embedding: BGE-M3             |                   0.817 |                    0.796 |               0.75  |                0.757 |               0.825 |                0.847 |        0.692 |              0.623 |
| Embedding: OpenAI 3-large     |                   0.743 |                    0.725 |               0.725 |                0.667 |               0.705 |                0.788 |        0.598 |              0.581 |
| Embedding: mE5-large-instruct |                   0.816 |                    0.796 |               0.779 |                0.75  |               0.788 |                0.837 |        0.712 |              0.662 |
| LLM-rep: Manual               |                   0.912 |                    0.849 |               0.894 |                0.835 |               0.956 |                0.917 |        0.859 |              0.843 |
| LLM-rep: Hybrid-Manual        |                   0.88  |                    0.822 |               0.857 |                0.788 |               0.93  |                0.895 |        0.804 |              0.75  |
| LLM-rep: Hybrid-Full          |                   0.892 |                    0.821 |               0.874 |                0.79  |               0.92  |                0.88  |        0.808 |              0.767 |
| LLM-rep: GPT-Schema           |                   0.868 |                    0.835 |               0.842 |                0.818 |               0.936 |                0.9   |        0.823 |              0.783 |
| LLM-rep: Raw-Facts            |                   0.874 |                    0.815 |               0.841 |                0.79  |               0.903 |                0.869 |        0.795 |              0.761 |
| LLM-rep: GPT-Free             |                   0.878 |                    0.81  |               0.855 |                0.774 |               0.904 |                0.871 |        0.787 |              0.744 |
| LLM-rep: GPT-Law              |                   0.867 |                    0.798 |               0.853 |                0.764 |               0.89  |                0.861 |        0.773 |              0.741 |

## WEAPON

| category                      |   F1-Oracle (b0 strict) |   F1-Oracle (b1 lenient) |   F1-CV (b0 strict) |   F1-CV (b1 lenient) |   AP-PR (b0 strict) |   AP-PR (b1 lenient) |   QWK-Oracle |   QWK-CV (10-fold) |
|:------------------------------|------------------------:|-------------------------:|--------------------:|---------------------:|--------------------:|---------------------:|-------------:|-------------------:|
| Random (null mean)            |                   0.504 |                    0.637 |               0.482 |                0.625 |               0.349 |                0.478 |        0.083 |             -0.007 |
| Embedding: BGE-M3             |                   0.591 |                    0.717 |               0.496 |                0.694 |               0.606 |                0.703 |        0.455 |              0.345 |
| Embedding: OpenAI 3-large     |                   0.532 |                    0.64  |               0.477 |                0.584 |               0.495 |                0.6   |        0.293 |              0.157 |
| Embedding: mE5-large-instruct |                   0.603 |                    0.686 |               0.565 |                0.648 |               0.542 |                0.686 |        0.403 |              0.243 |
| LLM-rep: Manual               |                   0.787 |                    0.846 |               0.766 |                0.828 |               0.827 |                0.898 |        0.763 |              0.743 |
| LLM-rep: Hybrid-Manual        |                   0.763 |                    0.828 |               0.729 |                0.805 |               0.803 |                0.876 |        0.734 |              0.704 |
| LLM-rep: Hybrid-Full          |                   0.763 |                    0.835 |               0.742 |                0.821 |               0.783 |                0.885 |        0.74  |              0.707 |
| LLM-rep: GPT-Schema           |                   0.759 |                    0.82  |               0.737 |                0.801 |               0.807 |                0.879 |        0.726 |              0.688 |
| LLM-rep: Raw-Facts            |                   0.744 |                    0.805 |               0.721 |                0.785 |               0.741 |                0.846 |        0.69  |              0.661 |
| LLM-rep: GPT-Free             |                   0.736 |                    0.806 |               0.702 |                0.785 |               0.733 |                0.864 |        0.695 |              0.67  |
| LLM-rep: GPT-Law              |                   0.736 |                    0.813 |               0.712 |                0.793 |               0.733 |                0.847 |        0.692 |              0.66  |

## Significance vs. Random Null — cells where observed > 97.5% null CI-hi

|                             |   F1-Oracle (b0 strict) |   F1-Oracle (b1 lenient) |   F1-CV (b0 strict) |   F1-CV (b1 lenient) |   AP-PR (b0 strict) |   AP-PR (b1 lenient) |   QWK-Oracle |   QWK-CV (10-fold) |
|:----------------------------|------------------------:|-------------------------:|--------------------:|---------------------:|--------------------:|---------------------:|-------------:|-------------------:|
| ('drugs', 'GPT-Free')       |                       1 |                        1 |                   1 |                    1 |                   1 |                    1 |            1 |                  1 |
| ('drugs', 'GPT-Law')        |                       1 |                        1 |                   1 |                    1 |                   1 |                    1 |            1 |                  1 |
| ('drugs', 'GPT-Schema')     |                       1 |                        1 |                   1 |                    1 |                   1 |                    1 |            1 |                  1 |
| ('drugs', 'Hybrid-Full')    |                       1 |                        1 |                   1 |                    1 |                   1 |                    1 |            1 |                  1 |
| ('drugs', 'Hybrid-Manual')  |                       1 |                        1 |                   1 |                    1 |                   1 |                    1 |            1 |                  1 |
| ('drugs', 'Manual')         |                       1 |                        1 |                   1 |                    1 |                   1 |                    1 |            1 |                  1 |
| ('drugs', 'Raw-Facts')      |                       1 |                        1 |                   1 |                    1 |                   1 |                    1 |            1 |                  1 |
| ('weapon', 'GPT-Free')      |                       1 |                        1 |                   1 |                    1 |                   1 |                    1 |            1 |                  1 |
| ('weapon', 'GPT-Law')       |                       1 |                        1 |                   1 |                    1 |                   1 |                    1 |            1 |                  1 |
| ('weapon', 'GPT-Schema')    |                       1 |                        1 |                   1 |                    1 |                   1 |                    1 |            1 |                  1 |
| ('weapon', 'Hybrid-Full')   |                       1 |                        1 |                   1 |                    1 |                   1 |                    1 |            1 |                  1 |
| ('weapon', 'Hybrid-Manual') |                       1 |                        1 |                   1 |                    1 |                   1 |                    1 |            1 |                  1 |
| ('weapon', 'Manual')        |                       1 |                        1 |                   1 |                    1 |                   1 |                    1 |            1 |                  1 |
| ('weapon', 'Raw-Facts')     |                       1 |                        1 |                   1 |                    1 |                   1 |                    1 |            1 |                  1 |

![Baselines headline plot](headline_plot.png)
