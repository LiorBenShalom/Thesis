"""What changes if we replace llama3_70b + gemma3_27b with qwen3_235b_or + mistral_large_or?

Compares 4 ordinal metrics × 2 domains × 6 alternatives = 48 Manual-vs-others tests
between (a) original N=11 and (b) swapped N=11.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon, spearmanr

EXP = Path("/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/experiments")

REP_PREFIX = {
    "Manual": "similarity_database_fe",
    "GPT-Schema": "similarity_database_fe_gpt_schema_v2",
    "GPT-Free": "similarity_database_with_gpt_features",
    "GPT-Law": "similarity_database_with_gpt_law_features",
    "Raw-Facts": "similarity_database_with_indicment_facts",
    "Hybrid-Manual": "similarity_database_hybrid",
    "Hybrid-Full": "similarity_database_hybrid_full_gpt",
}

ORIGINAL_MODELS = [
    "gpt4", "gpt5mini", "gpt52", "gpt51_thinking", "claude_sonnet_4_6",
    "gemini_25_pro", "gemini_3_flash", "gemma3_27b", "gemma4_31b_or",
    "llama3_70b", "qwen3_vl_235b_or",
]
# Drop the 2 weak ones, add 2 candidates
SWAPPED_MODELS = [m for m in ORIGINAL_MODELS if m not in ("llama3_70b", "gemma3_27b")]
SWAPPED_MODELS += ["qwen3_235b_or", "mistral_large_or"]

PILOT_BASE = EXP / "v6_pilot_5models"
PROD_BASE = EXP


def get_preds_path(domain: str, model: str, rep_prefix: str) -> Path:
    """For new candidates → pilot folder; for existing → v6_final."""
    if model in {"qwen3_235b_or", "mistral_large_or", "deepseek_r1_or"}:
        return PILOT_BASE / domain / f"results_{domain}" / f"{rep_prefix}_v6score_{model}_binary_0_preds.csv"
    return PROD_BASE / "v6_final" / domain / f"results_{domain}" / f"{rep_prefix}_v6score_{model}_binary_0_preds.csv"


def qwk(y_true, y_pred):
    n_r = 3
    O = np.zeros((n_r, n_r))
    for t, p in zip(y_true, y_pred):
        O[int(t) - 1, int(p) - 1] += 1
    N = len(y_true)
    ht = np.bincount(np.asarray(y_true, int) - 1, minlength=n_r)
    hp = np.bincount(np.asarray(y_pred, int) - 1, minlength=n_r)
    E = np.outer(ht, hp).astype(float) / N
    W = np.array([[((i - j) ** 2) / ((n_r - 1) ** 2) for j in range(n_r)] for i in range(n_r)])
    denom = np.sum(W * E)
    return 1.0 - (np.sum(W * O) / denom) if denom > 0 else 0.0


def best_thresholds_qwk(scores, gt):
    uniq = np.unique(scores)
    if len(uniq) < 3:
        return 0.0, 50.0, 50.0
    cands = (uniq[1:] + uniq[:-1]) / 2.0
    best_q, best_t = -1.0, (cands[0], cands[0])
    for t1 in cands:
        for t2 in cands:
            if t2 <= t1:
                continue
            pred = np.where(scores < t1, 1, np.where(scores < t2, 2, 3))
            q = qwk(gt, pred)
            if q > best_q:
                best_q, best_t = q, (t1, t2)
    return best_q, *best_t


def qwk_oracle_and_cv(scores, gt, k=10, seed=42):
    """Oracle = best thresholds on full data; CV = thresholds learned on train folds."""
    oracle, t1_o, t2_o = best_thresholds_qwk(scores, gt)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    pooled_pred = np.zeros_like(gt)
    for tr, te in skf.split(scores, gt):
        _, t1, t2 = best_thresholds_qwk(scores[tr], gt[tr])
        pooled_pred[te] = np.where(scores[te] < t1, 1, np.where(scores[te] < t2, 2, 3))
    return oracle, qwk(gt, pooled_pred)


def c_index(scores, gt):
    """P(score_i > score_j | gt_i > gt_j). Concordance for ordinal labels."""
    n_conc = n_pairs = 0
    for i in range(len(gt)):
        for j in range(len(gt)):
            if gt[i] > gt[j]:
                n_pairs += 1
                if scores[i] > scores[j]:
                    n_conc += 1
                elif scores[i] == scores[j]:
                    n_conc += 0.5
    return n_conc / n_pairs if n_pairs > 0 else 0.5


def evaluate_one(domain, rep_name, rep_prefix, model):
    fp = get_preds_path(domain, model, rep_prefix)
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df = df.dropna(subset=["score", "similarity_scale"])
    if len(df) < 50:
        return None
    scores = df["score"].astype(float).values
    gt = df["similarity_scale"].astype(int).values
    if len(np.unique(gt)) < 3:
        return None
    oracle, cv = qwk_oracle_and_cv(scores, gt)
    cidx = c_index(scores, gt)
    sp = spearmanr(scores, gt).statistic
    return {"QWK_Oracle": oracle, "QWK_CV": cv, "C_index": cidx, "Spearman": sp}


def build_results_table(model_set, label):
    """Compute all 4 metrics for model_set × 7 reps × 2 domains."""
    rows = []
    for dom in ["drugs", "weapon"]:
        for rep_name, rep_prefix in REP_PREFIX.items():
            for m in model_set:
                res = evaluate_one(dom, rep_name, rep_prefix, m)
                if res is None:
                    continue
                rows.append({"setup": label, "domain": dom, "rep": rep_name, "model": m, **res})
    return pd.DataFrame(rows)


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    r = p[o]; adj = r * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(adj, 0, 1); return out


def manual_vs_others(df, metric):
    """For each (domain), Wilcoxon Manual > each other rep, FDR over 6."""
    sigs = {}
    for dom in ["drugs", "weapon"]:
        sub = df[df.domain == dom]
        m_vec = sub[sub.rep == "Manual"].sort_values("model")[metric].values
        models = sub[sub.rep == "Manual"].sort_values("model")["model"].values
        ps, deltas = [], []
        others = [r for r in REP_PREFIX if r != "Manual"]
        for o in others:
            o_sub = sub[sub.rep == o].sort_values("model")
            o_vec = o_sub[metric].values
            o_models = o_sub["model"].values
            common = sorted(set(models) & set(o_models))
            mv = sub[(sub.rep == "Manual") & (sub.model.isin(common))].sort_values("model")[metric].values
            ov = sub[(sub.rep == o) & (sub.model.isin(common))].sort_values("model")[metric].values
            try:
                _, p = wilcoxon(mv, ov, alternative="greater", zero_method="wilcox")
            except ValueError:
                p = 1.0
            ps.append(p); deltas.append(mv.mean() - ov.mean())
        adj = bh_fdr(ps)
        sigs[dom] = list(zip(others, deltas, ps, adj))
    return sigs


def tier2_vs_tier3(df, metric):
    """Wilcoxon Tier-2 > Tier-3 (3×3 = 9 pairs per domain)."""
    T2 = ["GPT-Schema", "Hybrid-Manual", "Hybrid-Full"]
    T3 = ["Raw-Facts", "GPT-Free", "GPT-Law"]
    res = {}
    for dom in ["drugs", "weapon"]:
        sub = df[df.domain == dom]
        ps = []
        pairs = []
        for a in T2:
            for b in T3:
                a_sub = sub[sub.rep == a].sort_values("model")
                b_sub = sub[sub.rep == b].sort_values("model")
                common = sorted(set(a_sub.model) & set(b_sub.model))
                av = sub[(sub.rep == a) & (sub.model.isin(common))].sort_values("model")[metric].values
                bv = sub[(sub.rep == b) & (sub.model.isin(common))].sort_values("model")[metric].values
                try:
                    _, p = wilcoxon(av, bv, alternative="greater", zero_method="wilcox")
                except ValueError:
                    p = 1.0
                ps.append(p)
                pairs.append((a, b))
        adj = bh_fdr(ps)
        res[dom] = list(zip(pairs, ps, adj))
    return res


# ========== RUN ==========
print("Building results tables (this takes ~60s for QWK CV) ...")
df_orig = build_results_table(ORIGINAL_MODELS, "ORIGINAL N=11")
df_swap = build_results_table(SWAPPED_MODELS, "SWAPPED N=11")
print(f"Original: {df_orig.shape}, Swapped: {df_swap.shape}")

print()
print("=" * 92)
print("HEADLINE — Manual vs each alternative (4 metrics × 2 domains × 6 alts = 48 tests)")
print("=" * 92)

for metric in ["QWK_Oracle", "QWK_CV", "C_index", "Spearman"]:
    sig_orig = manual_vs_others(df_orig, metric)
    sig_swap = manual_vs_others(df_swap, metric)
    n_o_sig = sum(1 for d in sig_orig.values() for r in d if r[3] < 0.05)
    n_s_sig = sum(1 for d in sig_swap.values() for r in d if r[3] < 0.05)
    print(f"\n{metric}")
    print(f"  ORIGINAL N=11:  {n_o_sig}/12 significant (FDR)")
    print(f"  SWAPPED  N=11:  {n_s_sig}/12 significant (FDR)")

# Tier-2 vs Tier-3
print()
print("=" * 92)
print("TIER-2 vs TIER-3 — significance count (9 pairs × 2 domains = 18 per metric)")
print("=" * 92)
for metric in ["QWK_Oracle", "QWK_CV", "C_index", "Spearman"]:
    o = tier2_vs_tier3(df_orig, metric)
    s = tier2_vs_tier3(df_swap, metric)
    n_o = sum(1 for d in o.values() for r in d if r[2] < 0.05)
    n_s = sum(1 for d in s.values() for r in d if r[2] < 0.05)
    print(f"  {metric:12s}  ORIGINAL: {n_o:>2d}/18   SWAPPED: {n_s:>2d}/18")

# Mean per rep comparison
print()
print("=" * 92)
print("MEAN per rep — comparison")
print("=" * 92)
for metric in ["QWK_Oracle", "QWK_CV", "C_index", "Spearman"]:
    print(f"\n--- {metric} ---")
    for dom in ["drugs", "weapon"]:
        print(f"  {dom}:")
        for rep in REP_PREFIX:
            o = df_orig[(df_orig.domain == dom) & (df_orig.rep == rep)][metric].mean()
            s = df_swap[(df_swap.domain == dom) & (df_swap.rep == rep)][metric].mean()
            diff = s - o
            mark = "🟢" if diff > 0.005 else ("🔴" if diff < -0.005 else "⚪")
            print(f"    {rep:14s}  ORIG={o:.3f}  SWAP={s:.3f}  Δ={diff:+.4f} {mark}")

# Save tables
df_orig.to_csv("/tmp/swap_results_orig.csv", index=False)
df_swap.to_csv("/tmp/swap_results_swap.csv", index=False)
print()
print("Saved: /tmp/swap_results_orig.csv, /tmp/swap_results_swap.csv")
