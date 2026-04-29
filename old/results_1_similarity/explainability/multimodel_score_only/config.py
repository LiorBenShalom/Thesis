"""
Config for the multi-model multi-rep score-only experiment.

11 models × 7 representations × 241 GT pairs = 18,557 calls total.

Provider grouping (for batch dispatch):
  - openai_batch:    gpt4, gpt5mini, gpt52, gpt51_thinking
  - anthropic_batch: claude_sonnet_4_6
  - google_batch:    gemini_25_pro, gemini_3_flash
  - sync (OpenRouter / HF): gemma3_27b, gemma4_31b_or, llama3_70b, qwen3_235b
"""
from pathlib import Path

ROOT = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!")
NEW_TRY = ROOT / "new_try"
OUT_DIR = NEW_TRY / "experiments/explainability_annotation/multimodel_score_only"
RESULTS_DIR = OUT_DIR / "results"
BATCH_DIR = OUT_DIR / "batch"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR.mkdir(parents=True, exist_ok=True)

# (rep_id, csv_filename, kind: "features" | "facts")
REPS = [
    ("manual_fe",         "similarity_database_fe.csv",                       "features"),
    ("hybrid_manual_gpt", "similarity_database_hybrid.csv",                   "features"),
    ("hybrid_full_gpt",   "similarity_database_hybrid_full_gpt.csv",          "features"),
    ("fe_gpt_schema",     "similarity_database_fe_gpt_schema.csv",            "features"),
    ("facts",             "similarity_database_with_indicment_facts.csv",     "facts"),
    ("gpt_free",          "similarity_database_with_gpt_features.csv",        "features"),
    ("gpt_law",           "similarity_database_with_gpt_law_features.csv",    "features"),
]

# (model_id, provider, model_api_id)
MODELS = [
    ("gpt4",              "openai",     "gpt-4.1"),
    ("gpt5mini",          "openai",     "gpt-5-mini-2025-08-07"),
    ("gpt52",             "openai",     "gpt-5.2"),
    ("gpt51_thinking",    "openai",     "gpt-5.1"),
    ("claude_sonnet_4_6", "anthropic",  "claude-sonnet-4-5"),
    ("gemini_25_pro",     "google",     "gemini-2.5-pro"),
    ("gemini_3_flash",    "google",     "gemini-2.5-flash"),
    ("gemma3_27b",        "openrouter", "google/gemma-3-27b-it"),
    ("gemma4_31b_or",     "openrouter", "google/gemma-4-31b-it"),
    ("llama3_70b",        "openrouter", "meta-llama/llama-3.1-70b-instruct"),
    ("qwen3_235b",        "openrouter", "qwen/qwen3-235b-a22b"),
]

# Which providers support batch API
BATCH_PROVIDERS = {"openai", "anthropic", "google"}

DOMAINS = ["drugs", "weapon"]


def domain_dir(domain: str) -> Path:
    return NEW_TRY / domain
