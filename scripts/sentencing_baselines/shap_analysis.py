#!/usr/bin/env python3
"""
SHAP analysis — which Hebrew tokens does the supervised model use to
predict sentencing range?

Pipeline:
  text → DictaBERT (supervised) → 768-dim → Ridge regressor → predicted months

SHAP runs token-level masking on the input text and measures how each token
shifts the predicted months. Output: per-token attributions, visualized as
HTML with red/blue token coloring.

For each domain, picks 5 representative test verdicts:
  - 2 with high actual sentencing range
  - 2 with low
  - 1 mid
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import shap
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try")
EXP  = ROOT / "experiments"

DATA_CSV = ROOT / "simcse_cuda_bundle/data/supervised_data.csv"
MODEL_DIR = {
    "drugs":  EXP / "simcse_outputs/supervised/model_drugs",
    "weapon": EXP / "simcse_outputs/supervised/model_weapon",
}
EMB_PATH = {
    "drugs":  EXP / "simcse_outputs/supervised/verdict_embeddings_drugs.npy",
    "weapon": EXP / "simcse_outputs/supervised/verdict_embeddings_weapon.npy",
}
IDX_PATH = {
    "drugs":  EXP / "simcse_outputs/supervised/verdict_index_drugs.csv",
    "weapon": EXP / "simcse_outputs/supervised/verdict_index_weapon.csv",
}
OUT = EXP / "results/0_preprocessing/embedding_filter/shap"
OUT.mkdir(parents=True, exist_ok=True)


def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def load_domain(domain):
    """Load model + embeddings + index + ranges for one domain."""
    df = pd.read_csv(DATA_CSV)
    df = df[df.domain == domain].dropna(subset=["indictment_facts"]).reset_index(drop=True)
    idx = pd.read_csv(IDX_PATH[domain])
    df = df.merge(idx[["verdict","split"]], on="verdict")
    print(f"  loaded {domain}: {len(df)} verdicts ({df.split.value_counts().to_dict()})")
    emb = np.load(EMB_PATH[domain])
    return df, emb


def train_regressor(df, emb, target):
    """Train Ridge on training set: emb → target."""
    train_mask = (df.split == "train").values
    X_train = emb[train_mask]
    y_train = df[target].values[train_mask]
    reg = Ridge(alpha=1.0)
    reg.fit(X_train, y_train)
    test_mask = (df.split == "test").values
    score = reg.score(emb[test_mask], df[target].values[test_mask])
    return reg, score


def pick_examples(df, target, n_high=2, n_low=2, n_mid=1):
    """Pick representative test verdicts: top/bottom by target + 1 medium."""
    test = df[df.split == "test"].copy()
    test = test.sort_values(target)
    low  = test.head(n_low).to_dict("records")
    high = test.tail(n_high).to_dict("records")
    mid  = test.iloc[len(test)//2 : len(test)//2 + n_mid].to_dict("records")
    return low + mid + high


def make_predictor(model, regressor, device):
    """Returns a function: list[str] → np.array of predicted months."""
    def predict(texts):
        embs = model.encode(texts, batch_size=8, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
        return regressor.predict(embs)
    return predict


def run_shap(predictor, examples, target_name, max_evals=200):
    """Run SHAP on each example using text masker + word-level."""
    masker = shap.maskers.Text(r"\s+")   # split on whitespace = word level
    explainer = shap.Explainer(predictor, masker)
    results = []
    for i, ex in enumerate(examples):
        text = ex["indictment_facts"]
        # Truncate to ~500 words to keep SHAP manageable
        words = text.split()
        if len(words) > 400:
            text = " ".join(words[:400])
        print(f"  [{i+1}/{len(examples)}] {ex['verdict']}: actual {target_name}={ex[target_name]:.0f} mo, text len={len(words)} words")
        sv = explainer([text], max_evals=max_evals, silent=True)
        results.append({
            "ex": ex,
            "shap_values": sv,
            "text": text,
        })
    return results


def save_html(results, target_name, predictions, out_html):
    """Save HTML visualization of all examples (one HTML per file, multi-example)."""
    html_parts = ["<html><head><meta charset='utf-8'><style>"
                  "body { font-family: Arial, sans-serif; direction: rtl; padding: 20px; max-width: 1200px; margin: auto; }"
                  ".example { margin: 30px 0; padding: 20px; border: 1px solid #aaa; border-radius: 8px; background: #fafafa; }"
                  ".header { font-weight: bold; margin-bottom: 12px; color: #222; font-size: 16px; }"
                  ".subheader { color: #666; margin-bottom: 15px; }"
                  "</style></head><body>"]
    html_parts.append(f"<h1>SHAP: {target_name}</h1>")
    for i, (r, pred) in enumerate(zip(results, predictions)):
        ex = r["ex"]; sv = r["shap_values"]
        html_parts.append("<div class='example'>")
        html_parts.append(f"<div class='header'>דוגמה {i+1}: {ex['verdict']}</div>")
        html_parts.append(f"<div class='subheader'>"
                          f"בפועל: {ex[target_name]:.0f} חודשים  |  "
                          f"חזוי: {pred:.1f} חודשים</div>")
        # shap.plots.text returns HTML string in display=False mode
        try:
            txt_html = shap.plots.text(sv, display=False)
            if txt_html: html_parts.append(txt_html)
        except Exception as e:
            html_parts.append(f"<div style='color:red'>SHAP plot error: {e}</div>")
        html_parts.append("</div>")
    html_parts.append("</body></html>")
    out_html.write_text("".join(html_parts), encoding="utf-8")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="sentencing_range_low",
                   choices=["sentencing_range_low","sentencing_range_high","both"])
    p.add_argument("--domain", default="both", choices=["drugs","weapon","both"])
    p.add_argument("--max-evals", type=int, default=200)
    p.add_argument("--n-examples", type=int, default=5)
    args = p.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    domains = ["drugs","weapon"] if args.domain == "both" else [args.domain]
    targets = (["sentencing_range_low","sentencing_range_high"]
               if args.target == "both" else [args.target])

    for domain in domains:
        print(f"\n=== {domain.upper()} ===")
        print(f"Loading model from {MODEL_DIR[domain]}...")
        model = SentenceTransformer(str(MODEL_DIR[domain]), device=device)
        df, emb = load_domain(domain)

        for target in targets:
            print(f"\n--- target: {target} ---")
            reg, r2 = train_regressor(df, emb, target)
            print(f"  Ridge R² on test: {r2:.3f}")
            examples = pick_examples(df, target,
                                     n_high=args.n_examples//2 + 1,
                                     n_low=args.n_examples//2 + 1, n_mid=1)[:args.n_examples]
            predictor = make_predictor(model, reg, device)

            preds = predictor([ex["indictment_facts"] for ex in examples])
            print(f"  Predictions vs actual:")
            for ex, pred in zip(examples, preds):
                print(f"    {ex['verdict']}: actual={ex[target]:.0f}, pred={pred:.1f}")

            print(f"\n  Running SHAP ({len(examples)} examples × {args.max_evals} evals)...")
            results = run_shap(predictor, examples, target, max_evals=args.max_evals)
            out_html = OUT / f"shap_{domain}_{target}.html"
            save_html(results, target, preds, out_html)
            print(f"  💾 → {out_html}")


if __name__ == "__main__":
    main()
