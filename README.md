# Thesis Experiments

Comparing 7 feature representations × 11 LLMs × 2 domains (drugs/weapon) × 2 binary similarity tasks.

## Pipeline

1. **`src/pipeline_docx/`** — Raw verdict DOCX → CSV + indictment facts extraction
2. **`src/extraction/`** — Build the 7 feature representations:
   - `manual_fe` (manual GT)
   - `fe_gpt_schema_v2` (GPT-extracted using manual schema)
   - `gpt_free` / `gpt_law` (GPT-extracted free/law features)
   - `hybrid_manual` / `hybrid_full_gpt`
   - `facts` (raw indictment facts)
3. **`src/scoring/`** — `v6_score_multimodel_experiment.py` runs all (rep × model) cells
4. **`src/analysis/`** — Significance tests, leaderboards, reports
5. **`src/maintenance/`** — GT relabel / sync utilities
6. **`src/common/`** — `model_config.py`

## Directories

- `data/` — input CSVs (7 representations × 2 domains)
- `v6_full_matrix/` — baseline results (old prompt)
- `v6_prompt_test/` — results with revised prompt
- `drugs_extraction/`, `weapon_extraction/` — extraction outputs
- `annotator_agreement/` — inter-annotator analysis
