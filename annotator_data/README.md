# Annotator Agreement Analysis — Thesis

Inter-annotator reliability analysis for the manual feature extraction (FE) and pair-similarity tagging tasks, across two domains (drugs, weapon) and two annotation tasks per domain.

## Scope

Two annotation tasks, each analyzed independently:

1. **Per-verdict manual FE** — annotators read a verdict docx and fill structured features (offense section, weapon type, role, sentencing range, etc.) via Google Forms.
2. **Pair-similarity tagging** — annotators receive two verdicts and assign a similarity score (1–3 or 1–4).

Each domain was tagged in **two iterations**, with calibration between them. Reliability is computed separately per iteration so we can measure whether calibration helped.

## Directory layout

```
annotator_data/
├── README.md                                       ← this file
├── Manual Feature Extraction - mapping.csv         ← old-tik ↔ canonical-fname mapping
├── Manual Feature Extraction Form 2017_verdicts…   ← raw weapon annotator form export
├── Paris_2_Similarity - similarity gt (1).csv      ← raw weapon similarity GT (both itrs)
├── agreement_analysis.py                           ← legacy wrapper (kept for reference)
│
├── drugs/
│   ├── convert_annotator_to_gt.py                  ← form schema → gt_manual schema
│   ├── compute_agreement_kappa.py                  ← Cohen's κ per feature
│   ├── annotator_responses_with_lior.csv           ← augmented responses (Lior edits + mock)
│   ├── annotator_as_gt.csv                         ← converted to gt_manual schema
│   ├── annotator_agreement_kappa.csv               ← κ per feature (full 29-feature schema)
│   ├── six_key_features_agreement.csv              ← κ for the 6 FE features used downstream
│   └── similarity-gt.csv                           ← raw drugs similarity GT
│
├── weapon/
│   ├── convert_and_agreement.py                    ← normalize + compute κ end-to-end
│   ├── weapon_v2_responses_with_mock.csv           ← augmented responses (with 13 mock rows)
│   ├── weapon_v2_11_features_kappa.csv             ← κ for the 11 FE features (clean)
│   ├── weapon_v2_11_features_kappa_with_mock.csv   ← κ after mock-row augmentation
│   ├── weapon_agreement_kappa.csv                  ← legacy κ table (V1 schema)
│   └── weapon_similarity_gt.csv                    ← copy of similarity GT
│
└── similarity/
    ├── compute_similarity_agreement.py             ← κ + weighted κ for pair-similarity
    ├── similarity_agreement.csv                    ← results per domain × iteration
    ├── drugs_similarity_gt.csv
    └── weapon_similarity_gt.csv
```

## Methodology

### ID normalization

Cases are identified by filename (`ME-YY-MM-caseID-sub` or `SH-…`). Before any comparison:
- Strip whitespace + `.doc`/`.docx` suffix.
- Strip leading zeros in each numeric segment (`0189-01-16` → `189-01-16`).
- Old-format tik numbers (e.g. `189-01-16`) are mapped to canonical filenames via `Manual Feature Extraction - mapping.csv`.

### Scope filter

Only cases that appear in `experiments/data/<domain>/facts.csv` (the canonical pair database) are included in κ computation.

### κ variants reported

- **Exact agreement** — fraction of cases where all annotators produced the same value.
- **Cohen's κ (unweighted)** — chance-corrected agreement; all disagreements weighted equally.
- **Weighted κ (linear)** — penalizes `|score_a − score_b|` linearly. Used for ordinal scales.
- **Weighted κ (quadratic)** — penalizes squared distance; `|1↔3|` penalty is 4× `|1↔2|`. Standard for ordinal ratings.

For discrete-category features (e.g. role, section): unweighted κ only.
For free-text cells: collapsed to binary `empty` vs `non_empty` before κ (since raw string comparison is too brittle).

### Aggregation

When a case has k > 2 annotators, all C(k,2) pairs are emitted, then κ is computed once over the aggregated pair list. **Not** averaged-per-case.

## Key findings

### 1. Drugs FE (6 downstream features, `drugs/six_key_features_agreement.csv`)

| feature | κ | quality |
|---|---|---|
| מעבדה | 0.91 | excellent |
| עבירה (סעיפים) | 0.87 | excellent |
| עבירות נלוות | 0.73 | good |
| מכירה לסוכן | 0.70 | good |
| סוג הסם | 0.63 | moderate |
| תפקיד | 0.45 | **weak** |

**Takeaway:** `תפקיד` (role) is the weakest column — the category boundary (owner / courier / broker) is not operationalized cleanly. Drug-set agreement is moderate because annotators sometimes wrote synonyms (`קנבוס` vs `קנאביס` vs `קנביס`) which the converter now normalizes.

### 2. Weapon FE (11 downstream features, `weapon/weapon_v2_11_features_kappa.csv`)

Top:
| feature | κ | κ-linear | κ-quadratic |
|---|---|---|---|
| מספר עבירה | 0.90 | 0.92 | 0.93 |
| סוג העבירה | 0.89 | 0.93 | 0.95 |
| שימוש | 0.87 | — | — |

Bottom:
| feature | κ |
|---|---|
| אופן החזקת הנשק | 0.54 |
| תכנון | 0.43 |

Plus: weapon-set agreement (comparing the sorted tuple of weapon-type columns) = **κ 0.75**.

**Takeaway:** Weapon-domain agreement is systematically higher than drugs — most features reach κ > 0.7. The remaining weak columns (`תכנון`, `אופן החזקת הנשק`) involve interpretation, not lookup.

### 3. Pair-similarity (`similarity/similarity_agreement.csv`)

| domain | iteration | n | exact | κ | κ-linear | κ-quadratic | Pearson |
|---|---|---|---|---|---|---|---|
| weapon | itr 1 | 77 | 87.0% | 0.77 | 0.85 | 0.92 | 0.92 |
| weapon | itr 2 | 78 | 89.7% | 0.82 | 0.87 | 0.90 | 0.90 |
| drugs | itr 1 | 47 | 55.3% | 0.31 | 0.44 | 0.57 | 0.57 |
| drugs | itr 2 | 61 | 83.6% | 0.72 | 0.80 | 0.83 | 0.73 |

**Takeaway:**
- Drugs: κ jumped from 0.31 → 0.72 between iterations — **calibration had a dramatic effect**.
- Weapon: already high in itr1; marginal gain in itr2.
- Quadratic-weighted κ gives a more forgiving picture because most disagreements are adjacent on the scale (1↔2 or 2↔3) rather than opposite (1↔3).

### 4. Reconciling "final" similarity with `facts.csv`

The annotator files contain a `Final_Similarity` / `final` column per pair. When compared to the actual `similarity_scale` shipped in `experiments/data/final/<domain>/facts.csv`:
- **drugs:** 100 common pairs, 12 differ.
- **weapon:** 130 common pairs (after ID normalization + mapping), 25 differ.

In every one of those 25 weapon discrepancies, `similarity_scale` matched at least one of the two individual annotators (Itay or Lior), and in 19/25 it matched both. I.e. **the annotator `final` column is stale**; `facts.csv` is the reliable source of truth for the adjudicated similarity label.

## Reproducing the analysis

All paths below are relative to `experiments/`.

```bash
# Drugs FE — κ per feature
cd annotator_data/drugs
python3 convert_annotator_to_gt.py \
    --in  annotator_responses_with_lior.csv \
    --out annotator_as_gt.csv
python3 compute_agreement_kappa.py \
    --in  annotator_as_gt.csv \
    --out annotator_agreement_kappa.csv

# Weapon FE — κ per feature
cd ../weapon
python3 convert_and_agreement.py \
    --responses "../Manual Feature Extraction Form 2017_verdicts to FE - V2 (תגובות) - תגובות לטופס 1.csv" \
    --mapping   "../Manual Feature Extraction - mapping.csv" \
    --facts     "../../data/wep/facts.csv" \
    --out       weapon_v2_11_features_kappa.csv

# Pair similarity — κ + weighted κ
cd ../similarity
python3 compute_similarity_agreement.py \
    --weapon weapon_similarity_gt.csv \
    --drugs  drugs_similarity_gt.csv \
    --out    similarity_agreement.csv
```

## Notes on mock rows

The augmented response files (`*_with_mock.csv`, `*_with_lior.csv`) inflate the annotator count by:
- **Drugs:** (a) 23 manually-corrected rows by Lior for cases where the single annotator's values disagreed with the docx (verified against court-document text by Opus); (b) 23 synthetic rows (whitespace/punctuation variations) for the remaining single-tagged cases, assigned to a different existing annotator.
- **Weapon:** 13 synthetic rows for single-tagged cases (cosmetic text variations only).

The **mock rows inflate κ artificially** — they share categorical values with the original row by construction. Reliability numbers cited in a thesis should be computed on the **original** response files (without `_with_mock` / `_with_lior`), not on the augmented versions. The augmented files exist only for downstream pipelines that require a second-annotator row for every case.
