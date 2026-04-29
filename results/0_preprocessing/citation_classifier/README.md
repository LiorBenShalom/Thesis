# Citation Relevance Classifier — sentencing-policy precedents

A BERT classifier trained to identify the **specific paragraphs in a verdict
that cite "sentencing-policy precedents"** — i.e., precedents the judge cites
when determining the appropriate sentence range, as opposed to other types of
citations (procedural, evidentiary, definitional, etc.).

This is a **pre-processing step** for the citation-based similarity / sentencing-
range pipeline: we only want to use relevant citations as similarity signals,
not boilerplate procedural cites.

## Performance (held-out test, see `best_threshold.txt`)

| Metric | Value |
|---|---:|
| PR-AUC | 0.913 |
| F1 | 0.903 |
| Precision | 0.862 |
| Recall | 0.949 |
| False Positive Rate | 0.076 |
| Optimal threshold | 0.65 |

Recall-prioritized: we'd rather over-tag (later filtered downstream) than miss
sentencing-policy citations.

## Training data

`training_data.csv` — 581 paragraphs from 53 verdicts, manually labeled.

| Label | Count | Meaning |
|---|---:|---|
| 1 | 325 | Paragraph cites a sentencing-policy precedent |
| 0 | 248 | Other type of citation (or no citation at all) |

Schema: `document, paragraph_number, text, label`

## Model

- Architecture: `BertForSequenceClassification` (12-layer, 768-hidden, ~110M params)
- Base model: a Hebrew-pretrained BERT (see `tokenizer_config.json` for
  tokenizer family)
- Output: 2-class softmax → probability of being a sentencing-policy citation
- Apply with threshold 0.65 (calibrated on validation set)

## Files in this directory

| File | Purpose |
|---|---|
| `train_classifier.py` | Training script (full reproduction) |
| `training_data.csv` | Labeled training set (581 rows) |
| `best_threshold.txt` | Threshold + held-out metrics |
| `config.json` | Model config (BERT architecture) |
| `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.txt` | Tokenizer artifacts (needed to load the model) |
| `model.safetensors` | **NOT IN GIT** — 437MB, exceeds GitHub size limit. Reproduce with `python train_classifier.py`. Local copy at `new_try/innovation_submission/models/citation_classifier/`. |

## How it's used downstream

`new_try/innovation_submission/scripts/2_extract_citations.py` loads this model
and tags every paragraph in every verdict. Paragraphs with predicted_label=1
contribute their citations to the sentencing-similarity pipeline; the rest are
discarded.

## Reproducing

```bash
cd path/to/this/dir
pip install transformers datasets torch scikit-learn
python train_classifier.py \
    --train_csv training_data.csv \
    --output_dir ./output
```

Will re-train the BERT classifier and reproduce the threshold tuning.
