# Archived directories — superseded by current canonical setup

## Canonical setup (use these — at top level of `experiments/`):

- **`v6_final/`** — the N=13 model panel for the similarity task. Contains
  `drugs/results_drugs/` and `weapon/results_weapon/` with predictions for
  all 13 models. **This is what the paper uses.** (per commit `25b836d`
  "Unify N=13 panel under v6_final/").
- **`results_paper/`** — best-F1 paper results
- **`results_paper_baselines/`** — baselines (Random + embeddings)
- **`results_paper_qwk/`** — QWK ordinal evaluation (N=13)
- **`data/`** — clean experimental data (drugs/, weapon/, final/)

## What's in `old/`

These dirs were used in earlier iterations and are no longer canonical:

| Dir | Why archived |
|---|---|
| `v6_full_matrix/` | Pre-N=13. Replaced by `v6_final/` per commit 25b836d. |
| `v6_final_hybrid_fixed/` | Transitional fix dir — now folded into `v6_final/`. |
| `v6_prompt_test_weapon/` | Prompt iteration work — done. |
| `v6_prompt_test_weapon_smoke/` | Smoke test from prompt iteration. |
| `v6_smoke_new_models/` | Smoke test for adding new models — superseded. |
| `results_paper_cv/` | Cross-validated F1 results. Kept here as audit trail; primary results are in `results_paper/` and `results_paper_qwk/`. |

## Note

These were all UNTRACKED in git (so they didn't appear in the GitHub repo
view) — but they cluttered the local filesystem. Moving here for cleaner
local navigation. None of the canonical N=13 paper outputs are affected.
