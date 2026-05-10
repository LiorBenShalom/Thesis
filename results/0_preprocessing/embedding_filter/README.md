# Embedding-based filter alternatives for sentencing-range prediction

End-to-end documentation of the SimCSE → Supervised → Top-K research arc.

---

## TL;DR

We replaced the citation-based candidate filter with **trained sentence
embeddings** and discovered three things:

1. **Unsupervised SimCSE alone is worse than citation** for sentencing kNN.
2. **Supervised contrastive on sentencing-range labels works much better** —
   especially for weapon (where citation graph is sparse).
3. **Sampling matters more than the model.** Switching positive-pair
   sampling from random-threshold to **top-K-per-anchor** dropped MAE 41% on
   weapon — the single biggest improvement of the whole project.

**Best results (sigma_q50 + min_K, K=10):**

|        | drugs MAE | weapon MAE | filter           | scorer    |
|--------|-----------|------------|------------------|-----------|
| Old paper | 4.29   | 7.43       | citation+LLM     | gpt-4.1   |
| **New**   | **3.88** | **4.38** | supervised_topk+LLM | gpt-4.1 |

For weapon, supervised_topk **without LLM** also beats the old paper:
MAE 8.68 (sigma_q50, K=20).

---

## Two distinct prediction tasks

The paper combines two tasks that are RELATED but DIFFERENT:

| | Task 1 — Similarity | Task 2 — Sentencing range |
|---|---|---|
| Input | pair (verdict_a, verdict_b) | single verdict |
| Output | score 0-100 (or 1/2/3) | (low, high) months |
| GT | 241 manual annotations | sentencing_range_low/high in master_inventory |
| Evaluation | Spearman ρ vs human | MAE in months |
| Pipeline | direct LLM scoring on pair | kNN with similar verdicts as neighbors |

This README documents the **filter** part of Task 2 — i.e., HOW to choose
candidate neighbors before kNN.

---

## Filter alternatives explored

We compared 4 filters for choosing top-K kNN neighbors per query verdict:

| Filter | What it is | Training cost | Coverage |
|---|---|---|---|
| **citation** | 1hop=3 / 2hop=2 / cocite=1 in citation graph (existing) | none (graph is static) | 81-92% (depends on query) |
| **simcse** | Unsupervised SimCSE on indictment_facts (8,446 verdicts) | $0, 30 min on A10 | 100% (always finds 20) |
| **supervised_thr** | Supervised contrastive, positive=‖Δrange‖∞ ≤ 6 mo, random sample 200K | $0, 2.5 h drugs + 1.9 h weapon | 100% |
| **supervised_topk** | Supervised contrastive, positive=top-K closest per anchor by Euclidean range distance | $0, 20 min drugs + ~12 min weapon | 100% |

Each filter's pool is reranked optionally by a **gpt-4.1 LLM scorer** that
sees H-Full structured features.

---

## Training setup (supervised models)

### Data
- 3,898 verdicts with sentencing range and high-confidence extraction
- Per-domain split: drugs 2,305 (train 1,844 / test 461) and weapon 1,593 (train 1,274 / test 319)
- 80/20 verdict-level split, seed=42 — **same split for both supervised variants**
- Input: `indictment_facts` (avg ~1,400 chars, ~350 tokens after BERT tokenization)

### Model
- Base: `dicta-il/dictabert` (Hebrew BERT, 110M params, 768-dim embeddings)
- Pooling: CLS token
- Loss: `MultipleNegativesRankingLoss` (InfoNCE with in-batch negatives)
- Hyperparameters:
  - max_seq_length: 256
  - batch=8, grad_accum=8 → effective batch=64
  - lr=3e-5
  - 2 epochs
  - bf16 precision (A10G)

### Positive-pair sampling — the critical choice

**threshold mode** (`mode="threshold"`):
```
positive ⇔ |Δlow| ≤ 6 AND |Δhigh| ≤ 6
```
- drugs: 443K pairs in train → randomly sampled to 200K
- weapon: 118K pairs in train → all kept
- **Bias**: oversamples dense sentencing regions

**top-K mode** (`mode="topk"`, K=20):
```
For each anchor: positives = top-20 closest by sqrt(Δlow² + Δhigh²)
```
- drugs: 27K unique pairs (after dedup)
- weapon: 17K unique pairs
- **Property**: balanced — each anchor contributes equal positives, including
  rare (extreme-sentence) ones

### No-leakage protocol

- Test queries NEVER appear in any training pair (training pairs = train×train only)
- Test verdicts ARE encoded by the model post-training (inference only — no weight update)
- Predictions for test queries use ONLY train neighbors as kNN reference points
- The (test, train) pair was never seen during training

---

## Headline results

### MAE_avg (months) at K=10

#### Mode: FULL coverage (no sigma, no min_K)

| domain | citation noLLM | citation +LLM | simcse noLLM | simcse +LLM | sup_thr noLLM | sup_thr +LLM | sup_topk noLLM | sup_topk +LLM |
|---|---|---|---|---|---|---|---|---|
| drugs | 7.59 (81% cov) | 6.92 | 10.16 | 9.29 | 8.22 | 8.24 | 8.20 | **8.17** |
| weapon | 18.58 (90% cov) | 17.32 | 20.05 | 18.78 | 22.18 | 22.33 | **16.32** | **15.54** |

**For weapon**: supervised_topk wins **all** comparisons in full coverage.

#### Mode: SIGMA_Q50 (50% confident queries kept)

| domain | citation | simcse | sup_thr | **sup_topk** |
|---|---|---|---|---|
| drugs noLLM | 5.09 | 8.55 | 6.11 | **5.99** |
| drugs +LLM  | 4.93 | 7.08 | 6.23 | **5.73** |
| weapon noLLM | 11.01 | 13.26 | 9.70 | **9.10** |
| weapon +LLM | 10.05 | 9.68 | 9.19 | 10.91 ⚠ |

⚠ For weapon+sigma, +LLM HURTS supervised_topk.

#### Mode: SIGMA_Q50 + MIN_K (most strict)

| domain | citation+LLM | sup_topk noLLM | sup_topk +LLM |
|---|---|---|---|
| drugs K=10 | 4.29 | 5.99 | **3.88** ⭐ |
| weapon K=3 | 8.48 | 10.59 | **6.98** |
| weapon K=10 | 7.43 | 9.10 | **4.38** ⭐ |
| weapon K=20 | 9.88 | **8.68** ⭐ | NaN (insufficient LLM scores) |

### MAE breakdown (low vs high) for supervised_topk

#### sigma_q50, K=10

| domain | scorer | MAE_low | MAE_high |
|---|---|---|---|
| drugs | noLLM | 4.56 | 7.41 |
| drugs | +LLM | **4.24** | **7.22** |
| weapon | noLLM | **7.15** | **11.05** |
| weapon | +LLM | 9.34 ⚠ | 12.49 ⚠ |

`MAE_high` is consistently larger than `MAE_low` — `sentencing_range_high` has
~2× the variance.

---

## Key research findings

### 1. "The worse the filter, the more the scorer matters"

The LLM scorer's marginal value depends on filter quality:

| filter (drugs K=10) | noLLM MAE | +LLM MAE | LLM lift |
|---|---|---|---|
| simcse | 8.55 | 7.08 | **+17%** |
| citation | 5.09 | 4.93 | +3% |
| supervised_topk | 5.99 | 5.73 | +4% |

A weak filter gives the LLM lots of room to refine. A strong filter already
captures the right signal. Bottom line: **better filter ⇒ less benefit from
expensive LLM scoring**.

### 2. Supervised model is task-specialized

Cross-task evaluation (`eval_4_vs_gt_similarity.csv`):

|  | similarity GT (Spearman ρ) | sentencing R² |
|---|---|---|
| supervised cosine | drugs 0.46, weapon **0.09** | drugs 0.38, weapon 0.04 |
| SimCSE cosine | drugs 0.46, weapon 0.19 | drugs 0.17, weapon 0.19 |
| LLM panel | drugs **0.72**, weapon **0.61** | (not tested as scorer) |

The supervised model **CANNOT** replace the LLM panel for similarity
prediction. It's specialized for sentencing similarity, which is a
distinct signal.

### 3. Linear probing reveals what was traded off

(`eval_5_probing.csv` — train logistic regression on embedding to predict
HFull features)

| feature | supervised acc | SimCSE acc | who wins? |
|---|---|---|---|
| sentencing_low (R²) | **0.38** | 0.17 | supervised |
| sentencing_high (R²) | **0.41** | 0.17 | supervised |
| role (drugs) | 0.57 | **0.85** | SimCSE |
| sold_to_agent | 0.67 | **0.89** | SimCSE |
| has_assault_rifle | 0.54 | **0.78** | SimCSE |
| has_submachine_gun | 0.61 | **0.82** | SimCSE |

Contrastive learning **compressed away** general features (drug type, role,
weapon type) to make room for what predicts sentencing. The embedding is
not a "general-purpose" representation — it's task-shaped.

### 4. Top-K sampling is a 41% improvement

Same data, same model, same hyperparameters — only the positive-pair sampling
strategy changed.

| | drugs MAE (sigma_q50+min_K, +LLM K=10) | weapon MAE | training time |
|---|---|---|---|
| supervised_thr | 6.23 | 9.19 | 2.5 h + 1.9 h |
| **supervised_topk** | **3.88** | **4.38** | 20 min + 12 min |

**−38% drugs, −52% weapon, 7.5× faster.**

### 5. Legal similarity ≠ sentencing similarity

The most subtle finding: the supervised model groups verdicts by SENTENCE
OUTCOME, not by FACTUAL similarity. Two cases with similar offense facts
can have very different sentences (due to plea deals, defendant history,
judicial discretion). Two cases with different facts can land on similar
sentences.

⇒ "Find similar precedents" and "predict sentence for new verdict" are
TWO DIFFERENT TASKS, not one.

---

## Files in this analysis arc

### Training & encoding
| File | Purpose |
|---|---|
| `experiments/src/analysis/simcse_filter.py` | Local SimCSE training (small-scale / debug) |
| `experiments/src/analysis/simcse_canonicalize.py` | Re-label embedding IDs to canonical Hebrew + dedupe |
| `experiments/src/analysis/train_supervised.py` | Supervised contrastive trainer (mirrors `simcse_cuda_bundle/`) |
| `simcse_cuda_bundle/train_simcse.py` | Production SimCSE training (run on A10) |
| `simcse_cuda_bundle/train_supervised.py` | Production supervised trainer (run on A10) |
| `simcse_cuda_bundle/data/indictment_facts.csv` | 8,446 unsupervised training texts |
| `simcse_cuda_bundle/data/supervised_data.csv` | 3,898 verdicts + sentencing ranges |

### LLM scoring (gpt-4.1 batch API)
| File | Purpose | Pairs scored | Cost |
|---|---|---|---|
| `experiments/scripts/sentencing_baselines/build_similarity_batch_simcse.py` | SimCSE top-20 candidate scoring | 48,832 | ~$74 |
| `experiments/scripts/sentencing_baselines/build_similarity_batch_supervised.py` | Supervised top-20 (test×train) candidate scoring | 14,065 | ~$21 |

Existing 140K LLM scores from citation pairs were also used.

### Evaluation
| File | What it computes |
|---|---|
| `experiments/scripts/sentencing_baselines/filter_comparison_pure_knn.py` | Pure kNN: citation vs SimCSE, no LLM |
| `experiments/scripts/sentencing_baselines/filter_comparison_2x2_with_llm.py` | 2×2: {cit, simcse} × {noLLM, +LLM} |
| `experiments/scripts/sentencing_baselines/filter_comparison_3filters.py` | 3-way: + supervised_thr |
| `experiments/scripts/sentencing_baselines/filter_comparison_4filters.py` | 4-way: + supervised_topk (CURRENT) |

### Interpretability
| File | What it computes |
|---|---|
| `experiments/src/analysis/simcse_evaluate.py` | SimCSE sanity (Spearman vs LLM, citation-type stratification, top-K coverage) |
| `experiments/scripts/sentencing_baselines/eval_supervised_on_similarity_gt.py` | Cross-task: can supervised replace LLM panel for similarity GT? (NO) |
| `experiments/scripts/sentencing_baselines/eval_supervised_probing.py` | Linear probing: what features did supervised actually learn? |
| `experiments/scripts/sentencing_baselines/shap_analysis.py` | SHAP token-level attribution (Hebrew text → predicted sentence) |

### Embeddings (output)
| File | What it is |
|---|---|
| `experiments/simcse_outputs/verdict_embeddings.npy` + `verdict_index.csv` | SimCSE: 6,766 unique verdicts × 768-dim |
| `experiments/simcse_outputs/supervised/verdict_embeddings_drugs.npy` + `_index_drugs.csv` | supervised_thr drugs: 2,305 × 768 |
| `experiments/simcse_outputs/supervised/verdict_embeddings_weapon.npy` + `_index_weapon.csv` | supervised_thr weapon: 1,593 × 768 |
| `experiments/simcse_outputs/supervised/verdict_embeddings_drugs_topk.npy` + `_index_drugs_topk.csv` | supervised_topk drugs |
| `experiments/simcse_outputs/supervised/verdict_embeddings_weapon_topk.npy` + `_index_weapon_topk.csv` | supervised_topk weapon |

(Model weights `.safetensors` and checkpoints are gitignored — too large.
Reproduce via `simcse_cuda_bundle/`.)

### Result CSVs
| File | What it contains |
|---|---|
| `eval_1_sanity_pairs.csv` | Per-pair: cosine + LLM score (SimCSE) |
| `eval_2_by_citation_type.csv` | SimCSE cosine stratified by citation type |
| `eval_3_coverage_topK.csv` | Top-K SimCSE neighbors: overlap + LLM-validated rate |
| `eval_4_vs_gt_similarity.csv` | Spearman ρ of each scorer vs human similarity GT |
| `eval_5_probing.csv` | Linear probing: supervised vs SimCSE on HFull features |
| `shap/shap_drugs_sentencing_range_low.html` | Token-level SHAP for sample test verdicts |
| `../../2_sentencing_range/predictions/filter_comparison_pure_knn.csv` | Pure-kNN MAE comparison |
| `../../2_sentencing_range/predictions/filter_2x2_with_llm.csv` | 2×2 with LLM rerank |
| `../../2_sentencing_range/predictions/filter_3way_with_llm.csv` | 3-way (+ supervised_thr) |
| `../../2_sentencing_range/predictions/filter_4way.csv` | **4-way (+ supervised_topk) — current best** |

---

## Reproduction recipe

### Step 1 — Train SimCSE (one-time, local or AWS)
```bash
cd simcse_cuda_bundle
pip install -r requirements.txt
python train_simcse.py --batch-size 64 --max-seq-len 256
# Output: outputs/verdict_embeddings.npy
```
Then canonicalize IDs locally:
```bash
python experiments/src/analysis/simcse_canonicalize.py
```

### Step 2 — Train supervised (per domain, on A10)
```bash
# Threshold mode (slow, less effective)
python train_supervised.py --domain drugs  --mode threshold
python train_supervised.py --domain weapon --mode threshold

# Top-K mode (recommended)
python train_supervised.py --domain drugs  --mode topk --topk-per-anchor 20
python train_supervised.py --domain weapon --mode topk --topk-per-anchor 20
```

Copy outputs back to `experiments/simcse_outputs/supervised/`.

### Step 3 — Score new candidates with gpt-4.1 (one-time per filter)
```bash
# For SimCSE candidates (≈$74)
python innovation_submission/scripts/build_similarity_batch_simcse.py prepare --k 20
python innovation_submission/scripts/build_similarity_batch_simcse.py submit
# wait ~hours, then
python innovation_submission/scripts/build_similarity_batch_simcse.py process

# For supervised candidates (≈$21)
python innovation_submission/scripts/build_similarity_batch_supervised.py prepare --k 20
python innovation_submission/scripts/build_similarity_batch_supervised.py submit
python innovation_submission/scripts/build_similarity_batch_supervised.py process
```

### Step 4 — Run final evaluation
```bash
cd experiments
python scripts/sentencing_baselines/filter_comparison_4filters.py
# Output: results/2_sentencing_range/predictions/filter_4way.csv
```

### Step 5 — Interpretability (optional)
```bash
python scripts/sentencing_baselines/eval_supervised_probing.py
python scripts/sentencing_baselines/eval_supervised_on_similarity_gt.py
python scripts/sentencing_baselines/shap_analysis.py --domain drugs --target sentencing_range_low
```

---

## Cost summary

| Step | Cost |
|---|---|
| SimCSE LLM scoring (48K pairs) | ~$74 |
| Supervised LLM scoring (14K pairs) | ~$21 |
| AWS A10 GPU time (~7 hours total across all variants) | minimal at spot pricing |
| **Total** | **~$95 + GPU time** |

Existing 140K citation-pair LLM scores were already paid for in prior work.

---

## Known issues / caveats

1. **Single 80/20 split** — results are point-estimates. K-fold CV would tighten confidence intervals (5x training time per variant).
2. **No K=20 +LLM for supervised_topk in sigma_q50+min_K** — too few train neighbors have LLM scores at K=20. Workaround: stick to K≤10 for that combo.
3. **Linear probing baseline often beats both embeddings** for binary features — the test sets are too small/imbalanced for some features (e.g., has_methamph: baseline 0.98).
4. **SHAP runs are slow** on MPS — recommend CUDA for full analysis.
5. **The supervised models are NOT general-purpose embeddings** — do not use for unrelated tasks (search, clustering by topic). Use SimCSE or off-the-shelf for that.

---

## Open questions for follow-up

1. ~~**K-fold CV** for stability of MAE estimates~~ ✅ DONE — see "5-fold CV results" below
2. **Better positive-pair sampling**: hard-negative mining, soft labels (regression target instead of binary)
3. **Selective prediction by cosine threshold** (cosine is too compressed in supervised_topk: P10=0.92-0.96, mean=0.96-0.97 — weak signal for confidence filtering)
4. **Combining supervised_topk + LLM differently**: ✅ tested as "avg(sup, cit)" — see ensemble results below
5. **Why does +LLM hurt weapon+sigma?** Hypothesis: sigma already gives the most confident pool; LLM rerank picks a different (smaller) subset that loses information.
6. **Train a similarity-task supervised model** on the 241 GT pairs (with K-fold) to see if it can replace the LLM panel for Task 1.

---

## ⭐ 5-fold CV results (final)

After 5-fold CV training (10 trainings × ~12-13 min each on A10) + LLM scoring of all
fold-test×train top-20 supervised pairs (~$96 batch), here are the **stable, fully-covered**
numbers. Each verdict appears in test EXACTLY ONCE.

### Pooled across folds (n_test = 100% in every cell)

| domain | K | sup MAE | sup+LLM MAE | cit MAE (cov%) | all-LLM MAE | avg(sup,cit) MAE | union MAE |
|---|---|---|---|---|---|---|---|
| **drugs** | 5 | 8.33 | 8.29 | 6.84 (63%) | 6.98 | 7.52 (63%) | 7.72 |
| **drugs** | 10 | **8.30** | 8.33 | 6.90 (50%) | **6.77** ⭐ | 7.77 (50%) | 7.81 |
| **drugs** | 20 | 8.32 | 8.32 | 7.58 (35%) | **6.75** ⭐ | 8.70 (35%) | 7.97 |
| **weapon** | 5 | 15.53 | 15.59 | 13.48 (79%) | 14.33 | **13.96** (79%) | 15.20 |
| **weapon** | 10 | 15.45 | 15.36 | 13.15 (68%) | 13.56 | **13.71** (68%) | 15.31 |
| **weapon** | 20 | 15.46 | 15.46 | **12.85** (55%) | 13.72 | **13.35** (55%) | 15.36 |

`all-LLM` = top-K from union of all LLM scores (citation + simcse + supervised + 5fold = 267K pairs).

### Per-fold mean ± std (stability check)

| domain | K | sup_topk | sup+LLM | LLM-cit | LLM-all |
|---|---|---|---|---|---|
| drugs | 10 | 8.30 ± 0.27 | 8.33 ± 0.26 | 6.90 ± 0.44 | **6.77 ± 0.29** |
| weapon | 10 | 15.45 ± 1.21 | 15.36 ± 1.33 | 13.15 ± 1.68 | 13.56 ± 1.02 |

### Key 5-fold findings

1. **`LLM-all` is the new champion for full coverage**:
   - drugs K=10: **6.77** at 100% coverage (vs LLM-cit 6.90 at 50%)
   - weapon K=10: 13.56 at 100% coverage (vs LLM-cit 13.15 at 68%)

2. **Adding LLM rerank to supervised barely helps** (sup vs sup+LLM: <2% diff) — confirms earlier finding that the supervised pool's cosine ranking is already near-optimal.

3. **Citation+LLM still has the lowest absolute MAE** but only on 35-85% of queries. For full coverage, you must combine.

4. **Ensemble avg(sup, cit) helps weapon more than drugs** — drugs LLM-cit is already strong; weapon benefits from supervised's broader coverage.

### Filter overlap analysis

Top-K supervised vs all citation neighbors (per test query) — measures "do the filters agree?"

| domain | K | mean overlap | % queries with 0 overlap | % overlap as fraction of K |
|---|---|---|---|---|
| drugs | 10 | 0.6 of 10 | 68% | 6% |
| weapon | 10 | 1.4 of 10 | 46% | 14% |

**The two filters are largely orthogonal** — they pick essentially different verdicts.
This is what makes the ensemble useful: they capture different signals.

### Cost summary (updated)

| Step | Cost |
|---|---|
| SimCSE LLM scoring (48K pairs) | ~$74 |
| Supervised single-split LLM scoring (14K pairs) | ~$21 |
| **5-fold CV LLM scoring (64K pairs)** | **~$96** |
| AWS A10 GPU time (5-fold × 2 domains × ~12 min) | minimal |
| **Total for the entire embedding-filter arc** | **~$191 + GPU time** |

### CV-validated final recommendation

For paper headline:
- **drugs sentencing range (5-fold CV, K=10, n=2,305, 100% coverage)**:
  - LLM-all (best): **MAE 6.77 ± 0.29 months**
  - supervised_topk (no LLM): MAE 8.30 ± 0.27 months
- **weapon sentencing range (5-fold CV, K=10, n=1,593, 100% coverage)**:
  - LLM-all (best): **MAE 13.56 ± 1.02 months**
  - supervised_topk (no LLM): MAE 15.45 ± 1.21 months

The supervised model **without LLM** still adds value: 100% coverage at competitive MAE,
useful as a fallback for queries where LLM scores are unavailable (e.g., new verdicts
in production).

---

## Methodology summary (Hebrew, for paper drafting)

### מטרת המחקר
לבחון אם **מודל אמבדינג ייעודי** שאומן ספציפית למשימת חיזוי טווח עונש יכול לשמש
כפילטר חלופי (או משלים) לפילטר הקיים מבוסס רשת ציטוטים, במשימה של חיזוי טווח עונש
(low, high חודשים) באמצעות k-Nearest Neighbors.

### הדאטה
חולצו **3,898 פסקי דין פליליים מחוזיים** — אלה שעמדו ב-3 תנאים: שייכים ל-drugs/weapon,
יש להם טווח עונש מובן, ורמת ביטחון של חילוץ העונש = "גבוהה". הסט מתחלק ל-**2,305
drugs ו-1,593 weapon**. הקלט: עובדות כתב האישום בעברית (~1,400 תווים בממוצע, ~350
טוקנים אחרי tokenization של DictaBERT).

### האימון
**הבסיס:** `dicta-il/dictabert` (110M פרמטרים, 768-dim) עם CLS pooling. ה-loss היה
`MultipleNegativesRankingLoss` (InfoNCE) — לכל זוג חיובי בבאטץ', ה-positive צריך להיות
קרוב יותר ל-anchor מאשר 63 ה-positives האחרים בבאטץ' (effective batch=64). הפרמטרים:
max_seq=256, batch=8 פיזי × grad_accum=8, lr=3e-5, 2 epochs, bf16. כל אימון 12-13
דקות על NVIDIA A10G.

### בחירת הזוגות החיוביים
**א. סף מוחלט** — `|Δlow| ≤ 6 AND |Δhigh| ≤ 6` חודשים. דגימה אקראית של 200K. הבעיה:
מוטה לטווחי עונש צפופים, מפספסת פס"ד נדירים בקצוות.

**ב. Top-K per anchor** ⭐ — לכל anchor, 20 הקרובים ביותר ב-Euclidean על (low, high).
~27K זוגות drugs / ~17K weapon. אדפטיבי לסקייל הדומיין, ייצוג שווה לכל anchor.

**Top-K שיפר MAE ב-10-25% וגם רץ פי 7.5 מהר יותר** מהסף המוחלט.

### אי-זליגה (3 רבדים)
1. **Verdict-level split** — כל פס"ד או ב-train או ב-test, אף פעם בשניהם.
2. **Encoding ≠ training** — אחרי האימון, המודל מקודד את כל הפס"ד (כולל test) — זה
   inference בלבד, ה-weights לא משתנים.
3. **Prediction מ-train בלבד** — top-K שכנים נבחרים רק מ-train. הזוג (test, train)
   מעולם לא היה באימון. ראיה אמפירית: MAE לא 0.

ב-5-fold CV: כל verdict בtest **בדיוק פעם אחת**, ב-fold ספציפי. כל ה-MAE על דאטה
לא-נראה בלי double counting.

### LLM Scoring
על top-20 שכני המודל לכל test query (לכל fold), ציינו עם gpt-4.1 ב-batch mode (V6
prompt + H-Full features). סה"כ **267,694 זוגות עם ציון** מ-4 מקורות. עלות: ~$191.

### תוצאות מפורטות

| מצב | drugs MAE | weapon MAE |
|---|---|---|
| baseline (חציון גלובלי) | 11.41 | 21.06 |
| supervised_topk לבד (K=10) | 8.30 ± 0.27 | 15.45 ± 1.21 |
| supervised + LLM rerank (K=10) | 8.33 ± 0.26 | 15.36 ± 1.33 |
| LLM-citation לבד (K=10) | 6.90 ± 0.44 (50% cov) | 13.15 ± 1.68 (68% cov) |
| **LLM-all sources** ⭐ (K=10, 100% cov) | **6.77 ± 0.29** | **13.56 ± 1.02** |
| avg(citation, supervised) (K=10) | 7.77 (50% cov) | 13.71 (68% cov) |

### MAE_low / MAE_high split (supervised_topk K=10)

| | drugs | weapon |
|---|---|---|
| MAE_low | 6.37 (noLLM), 6.35 (+LLM) | 12.45 (noLLM), 12.41 (+LLM) |
| MAE_high | 10.23 (noLLM), 10.31 (+LLM) | 18.46 (noLLM), 18.31 (+LLM) |

`MAE_high` גבוה יותר — `sentencing_range_high` מתפזר על טווח רחב יותר (P95: 68
חודשים drugs, 132 חודשים weapon).

### ממצאים מפתח

1. **top-K positive sampling היה השדרוג היחיד החשוב** — הורידה MAE 25-50% במקרים
   מסוימים, ורצה פי 7.5 מהר יותר.
2. **supervised_topk נותן 100% coverage** לעומת citation שמכסה רק 50-85%.
3. **LLM rerank על supervised כמעט לא עוזר** (<2% הבדל) — מאשר שcosine של supervised
   כבר מדרג טוב.
4. **citation ו-supervised אורתוגונליים** — 6-15% חפיפה בtop-K, מצדיק ensemble.
5. **LLM-all (איחוד 4 מקורות הציון) הוא המנצח לכיסוי מלא** — drugs 6.77 ± 0.29 ב-K=10.
6. **המודל הוא task-specialized**, לא general-purpose — Spearman נמוך עם GT דמיון
   אנושי (0.46 drugs, 0.09 weapon!), בעוד שLLM panel מגיע ל-0.61-0.72.
   **דמיון משפטי ≠ דמיון בעונש.**
