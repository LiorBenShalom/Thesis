# Embedding Baseline Report

_Cosine similarity of full verdict facts (`indicment_facts_1/2`), rescaled to [0, 100] to match LLM score scale._

_Models: `text-embedding-3-large`, `gemini-embedding-001`, `intfloat/multilingual-e5-large-instruct`, `BAAI/bge-m3`._


## DRUGS

| metric       |   BGE-M3 |   Gemini-embedding-001 |   OpenAI 3-large |   mE5-large-instruct |
|:-------------|---------:|-----------------------:|-----------------:|---------------------:|
| F1_Oracle_b0 |    0.817 |                  0.821 |            0.743 |                0.816 |
| F1_Oracle_b1 |    0.796 |                  0.818 |            0.725 |                0.796 |
| F1_CV_b0     |    0.75  |                  0.784 |            0.725 |                0.779 |
| F1_CV_b1     |    0.757 |                  0.792 |            0.667 |                0.75  |
| AP_b0        |    0.825 |                  0.862 |            0.705 |                0.788 |
| AP_b1        |    0.847 |                  0.896 |            0.788 |                0.837 |
| QWK_Oracle   |    0.692 |                  0.762 |            0.598 |                0.712 |
| QWK_CV       |    0.623 |                  0.667 |            0.581 |                0.662 |

## WEAPON

| metric       |   BGE-M3 |   Gemini-embedding-001 |   OpenAI 3-large |   mE5-large-instruct |
|:-------------|---------:|-----------------------:|-----------------:|---------------------:|
| F1_Oracle_b0 |    0.591 |                  0.638 |            0.532 |                0.603 |
| F1_Oracle_b1 |    0.717 |                  0.724 |            0.64  |                0.686 |
| F1_CV_b0     |    0.496 |                  0.593 |            0.477 |                0.565 |
| F1_CV_b1     |    0.694 |                  0.707 |            0.584 |                0.648 |
| AP_b0        |    0.606 |                  0.576 |            0.495 |                0.542 |
| AP_b1        |    0.703 |                  0.689 |            0.6   |                0.686 |
| QWK_Oracle   |    0.455 |                  0.486 |            0.293 |                0.403 |
| QWK_CV       |    0.345 |                  0.415 |            0.157 |                0.243 |