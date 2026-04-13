# Annotator Agreement Analysis — Thesis

Inter-annotator reliability for the manual feature-extraction (FE) and pair-similarity tagging tasks, across two domains (drugs, weapon).

## Tasks

1. **Per-verdict manual FE** — annotators read a verdict and fill structured features (offense section, weapon type, role, sentencing range, …) via Google Forms.
2. **Pair-similarity tagging** — annotators receive two verdicts and assign a similarity score on a 1–3 scale: 1 = not similar, 2 = borderline, 3 = similar.

Each domain was tagged in two iterations with a calibration step in between, so reliability is reported per iteration.

## Directory layout

```
annotator_data/
├── README.md
│
├── drugs/
│   ├── convert_annotator_to_gt.py        ← form schema → gt_manual schema
│   ├── compute_agreement_kappa.py        ← Cohen's kappa per feature
│   ├── drug_quantity_agreement.py        ← LLM-parsed (drug + qty) per-pair Jaccard
│   ├── consolidate_per_case.py           ← per-case aggregation across annotators
│   ├── annotator_responses.csv           ← raw responses (Google Form export)
│   ├── annotator_as_gt.csv               ← responses converted to gt_manual schema
│   ├── annotator_agreement_kappa.csv     ← kappa per feature (full schema)
│   ├── drug_parse_cache.json             ← LLM drug-parse cache
│   ├── drug_quantity_agreement.csv       ← per-pair Jaccard scores
│   ├── per_case_drug_aggregate.csv       ← consolidated drugs per case
│   └── six_key_features_agreement.csv    ← kappa for the FE features used downstream
│
├── weapon/
│   ├── convert_and_agreement.py          ← normalize + compute kappa end-to-end
│   ├── weapon_quantity_agreement.py      ← LLM-parsed weapons + ammo agreement
│   ├── consolidate_per_case.py           ← per-case aggregation across annotators
│   ├── mapping.csv                       ← old-tik ↔ canonical-fname mapping
│   ├── weapon_v2_responses.csv           ← raw responses (Google Form export)
│   ├── weapon_v2_11_features_kappa.csv   ← kappa for the 11 FE features
│   ├── weapon_agreement_kappa.csv        ← legacy kappa (V1 schema)
│   ├── ammo_parse_cache.json             ← LLM ammo-parse cache
│   ├── weapon_quantity_agreement.csv     ← per-pair weapon + ammo Jaccard
│   ├── per_case_weapon_aggregate.csv     ← consolidated weapons + ammo per case
│   └── weapon_similarity_gt.csv          ← copy of similarity GT
│
└── similarity/
    ├── compute_similarity_agreement.py   ← kappa + weighted kappa for pair-similarity
    ├── drugs_similarity_gt.csv
    ├── weapon_similarity_gt.csv
    └── similarity_agreement.csv
```

## Methodology

### ID normalization

Cases are identified by filename (`ME-YY-MM-caseID-sub` or `SH-…`). Before any comparison:
- Strip whitespace + `.doc`/`.docx` suffix.
- Strip leading zeros in each numeric segment (`0189-01-16` → `189-01-16`).
- Old-format tik numbers are mapped to canonical filenames via `weapon/mapping.csv`.

### Scope filter

Only cases that appear in `experiments/data/<domain>/facts.csv` (the canonical pair database) are included.

### Aggregation

When a case has *k* > 2 annotators, all C(*k*, 2) pairs are emitted, and kappa is computed once over the aggregated pair list — not averaged per case.

### Scoring choices

| feature shape | metric |
|---|---|
| Categorical / binary | Cohen's kappa (unweighted) |
| Multi-select (set of options) | Compared as sorted set — order doesn't matter |
| Drug name + quantity | LLM parses raw free-text → `{drug, amount, unit}`; quantity tolerance ±10% or ±1 unit; per-pair Jaccard |
| Weapon types (multi-select) | LLM-parsed set; per-pair Jaccard |
| Ammo quantity (free text) | LLM parses → `{kind, amount}`; same tolerance; per-pair Jaccard |
| Pair-similarity (1–3 ordinal) | Cohen's kappa + weighted kappa (linear, quadratic) + Pearson |

Empty cells get sensible defaults where the form convention encodes "no" implicitly (e.g. weapon-domain `planning` blank = "no").

## Results

### 1. Drugs FE — 6 downstream features

| feature | agreement | score |
|---|---|---|
| Lab | 95.6% | 0.91 |
| Offense (sections) | 91.2% | 0.87 |
| Role | 93.3% | 0.80 |
| Drug type + quantity | — | 0.79 |
| Side offenses (yes/no) | 86.8% | 0.73 |
| Sale to undercover agent | 88.2% | 0.70 |
| **Mean** | — | **0.77** |

Notes:
- **Drug type + quantity** is scored jointly because a quantity is meaningless without its drug name (`cannabis 10 g` ≠ `cannabis 100 g`). Score = mean per-pair Jaccard with name-match + tolerance (±10% or ±1 unit).
- **Role** encodes two binary sub-features (`drug-owner`, `lab-owner`). Hamming-weighted kappa over the (`owner`, `lab`) tuple gives partial credit when only one sub-feature matches.
- All other features use Cohen's kappa.

### 2. Weapon FE — 11 downstream features

| feature | agreement | score |
|---|---|---|
| Offense count | 93.2% | 0.90 |
| Offense type | 90.9% | 0.89 |
| Use | 94.3% | 0.87 |
| Weapon type (set) | — | 0.85 |
| Side offenses | 93.2% | 0.85 |
| Ammo quantity | — | 0.84 |
| Motive | 86.4% | 0.72 |
| How weapon was acquired | 81.8% | 0.63 |
| Weapon status | 70.5% | 0.62 |
| How weapon was held | 59.1% | 0.54 |
| Planning | 77.3% | 0.48 |
| **Mean** | — | **0.75** |

Notes:
- **Weapon type (set)** and **Ammo quantity** use the LLM + Jaccard pipeline (set agreement / quantity tolerance), same approach as the drug feature.
- Multi-select columns (acquisition, holding, motive, use, status, side offenses) are compared as sorted sets.
- Planning empty cell defaults to "no".
- All other features use Cohen's kappa.

### 3. Pair-similarity (1–3 ordinal scale)

| domain | iteration | n | exact agreement | kappa | kappa-linear | kappa-quadratic | Pearson |
|---|---|---|---|---|---|---|---|
| weapon | itr 1 | 77 | 87.0% | 0.77 | 0.85 | 0.92 | 0.92 |
| weapon | itr 2 | 78 | 89.7% | 0.82 | 0.87 | 0.90 | 0.90 |
| drugs | itr 1 | 47 | 55.3% | 0.31 | 0.44 | 0.57 | 0.57 |
| drugs | itr 2 | 61 | 83.6% | 0.72 | 0.80 | 0.83 | 0.73 |

Takeaways:
- Drugs: kappa jumped from 0.31 → 0.72 between iterations — **calibration had a dramatic effect**.
- Weapon: already high in itr 1; marginal gain in itr 2.
- Quadratic-weighted kappa is more forgiving because most disagreements are adjacent on the scale (1↔2, 2↔3) rather than opposite (1↔3).

## Reproducing the analysis

All paths relative to `experiments/`.

```bash
# Drugs FE — kappa per feature
cd annotator_data/drugs
python3 convert_annotator_to_gt.py --in annotator_responses.csv --out annotator_as_gt.csv
python3 compute_agreement_kappa.py --in annotator_as_gt.csv --out annotator_agreement_kappa.csv
# Drugs — LLM-parsed drug+quantity Jaccard
python3 drug_quantity_agreement.py --responses annotator_responses.csv --cache drug_parse_cache.json --out drug_quantity_agreement.csv
python3 consolidate_per_case.py --responses annotator_responses.csv --cache drug_parse_cache.json --out per_case_drug_aggregate.csv

# Weapon FE — kappa per feature
cd ../weapon
python3 convert_and_agreement.py --responses weapon_v2_responses.csv --mapping mapping.csv --facts ../../data/wep/facts.csv --out weapon_v2_11_features_kappa.csv
# Weapon — LLM-parsed weapons + ammo Jaccard
python3 weapon_quantity_agreement.py --responses weapon_v2_responses.csv --mapping mapping.csv --facts ../../data/wep/facts.csv --cache ammo_parse_cache.json --out weapon_quantity_agreement.csv
python3 consolidate_per_case.py --responses weapon_v2_responses.csv --mapping mapping.csv --facts ../../data/wep/facts.csv --cache ammo_parse_cache.json --out per_case_weapon_aggregate.csv

# Pair-similarity — kappa + weighted kappa
cd ../similarity
python3 compute_similarity_agreement.py --weapon weapon_similarity_gt.csv --drugs drugs_similarity_gt.csv --out similarity_agreement.csv
```

LLM scripts require `OPENAI_API_KEY` (loaded from `experiments/.env`).
