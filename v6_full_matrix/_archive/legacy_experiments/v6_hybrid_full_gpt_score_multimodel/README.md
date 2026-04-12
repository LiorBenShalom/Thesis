# v6 score multimodel — `hybrid_full_gpt`

ניסוי: ציון רציף 0–100 (ממדים + `SIMILARITY_SCORE`) עם אותו פרומפט v6 כמו ב־`structured_llm_comparison_experiment`, על כל ה־backends.

## מבנה

```
v6_hybrid_full_gpt_score_multimodel/
  drugs/results_drugs/     # *_v6score_*_preds.csv, *_stats.json
  weapon/results_weapon/
  analysis/                # נוצר אוטומטית אחרי finalize
    leaderboard_v6.csv
    pairwise_mcnemar_drugs.csv
    pairwise_mcnemar_weapon.csv
    shuffled_baseline_v6.csv
  v6_multimodel_summary.json   # אחרי סיום הרצה מלאה
```

## הרצה (מהתיקייה `new_try/code`)

```bash
python -u v6_score_multimodel_experiment.py \
  --domain both \
  --reps hybrid_full_gpt \
  --models gpt4 gpt5mini qwen3_235b dicta mistral nemotron3_nano llama3_70b gpt52 qwen_hf gemini_25_pro \
  --sleep 0.2 \
  --output-root ../experiments/v6_hybrid_full_gpt_score_multimodel
```

שורות תקינות ב־`*_preds.csv` נשמרות בקאש; אפשר להריץ שוב בלי `--fresh` כדי להשלים כשלונות בלבד.

## סטטיסטיקות (כמו תיקיית 9 המודלים)

אחרי עדכון תוצאות:

```bash
python v6_experiment_report.py ../experiments/v6_hybrid_full_gpt_score_multimodel
```

או:

```bash
bash finalize_v6_experiment.sh
```
