# Embedding Baseline Report

_Cosine similarity of full verdict facts (`indicment_facts_1/2`), rescaled to [0, 100] to match LLM score scale._

_Models: `text-embedding-3-large`, `intfloat/multilingual-e5-large-instruct`, `BAAI/bge-m3`._


## DRUGS

| metric       |   BGE-M3 |   OpenAI 3-large |   mE5-large-instruct |
|:-------------|---------:|-----------------:|---------------------:|
| F1_Oracle_b0 |    0.817 |            0.743 |                0.816 |
| F1_Oracle_b1 |    0.796 |            0.725 |                0.796 |
| F1_CV_b0     |    0.75  |            0.725 |                0.779 |
| F1_CV_b1     |    0.757 |            0.667 |                0.75  |
| AP_b0        |    0.825 |            0.705 |                0.788 |
| AP_b1        |    0.847 |            0.788 |                0.837 |
| QWK_Oracle   |    0.692 |            0.598 |                0.712 |
| QWK_CV       |    0.623 |            0.581 |                0.662 |

## WEAPON

| metric       |   BGE-M3 |   OpenAI 3-large |   mE5-large-instruct |
|:-------------|---------:|-----------------:|---------------------:|
| F1_Oracle_b0 |    0.591 |            0.532 |                0.603 |
| F1_Oracle_b1 |    0.717 |            0.64  |                0.686 |
| F1_CV_b0     |    0.496 |            0.477 |                0.565 |
| F1_CV_b1     |    0.694 |            0.584 |                0.648 |
| AP_b0        |    0.606 |            0.495 |                0.542 |
| AP_b1        |    0.703 |            0.6   |                0.686 |
| QWK_Oracle   |    0.455 |            0.293 |                0.403 |
| QWK_CV       |    0.345 |            0.157 |                0.243 |