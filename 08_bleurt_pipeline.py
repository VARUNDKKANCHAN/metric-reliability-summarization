# BLEURT: Stability (CNN/DailyMail) + Consistency (SummEval)
# Run: python 08_bleurt_pipeline.py
# Install: pip install git+https://github.com/google-research/bleurt.git
#          pip install datasets transformers pandas scipy tqdm nltk
# Checkpoint: download bleurt-large-512 from:
#   https://github.com/google-research/bleurt/blob/master/checkpoints.md
# Set env var: export BLEURT_CKPT=/path/to/bleurt-large-512

import os, json, random, time
from contextlib import nullcontext
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from scipy import stats
from tqdm.auto import tqdm
import nltk
nltk.download("punkt", quiet=True)

# ---------- BLEURT IMPORT ----------
try:
    from bleurt import score as bleurt_score_fn
    BLEURT_AVAILABLE = True
except ImportError:
    print("ERROR: bleurt not installed.")
    print("Install: pip install git+https://github.com/google-research/bleurt.git")
    BLEURT_AVAILABLE = False

BLEURT_CKPT = os.environ.get("BLEURT_CKPT", "bleurt-large-512")

# ---------- CONFIG ----------
FAST_MODE  = False
SMOKE_TEST = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Detected device:", device)

CNN_NAME, CNN_VER, CNN_SPLIT = "cnn_dailymail", "3.0.0", "test[:500]"
SUMMEVAL_NAME = "mteb/summeval"
GEN_MODEL     = "t5-small"
GEN_K         = 3
SEEDS         = list(range(100, 110))
BATCH_SIZE    = 16
MAX_INPUT, MAX_NEW = 512, 128
TEMP, TOP_K, TOP_P = 0.8, 50, 0.95
BOOTSTRAP_ITERS, CI_ALPHA = 1000, 0.05
OUT_DIR = "outputs/bleurt_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

if SMOKE_TEST:
    CNN_SPLIT, SEEDS, GEN_K, BATCH_SIZE, BOOTSTRAP_ITERS = "test[:10]", [SEEDS[0]], 1, 1, 50

print(f"Config -> model:{GEN_MODEL} | GEN_K:{GEN_K} | seeds:{SEEDS} | bleurt_ckpt:{BLEURT_CKPT}")

# ---------- HELPERS ----------
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device == "cuda": torch.cuda.manual_seed_all(seed)

def bootstrap_ci(values, n_iter=BOOTSTRAP_ITERS, alpha=CI_ALPHA):
    arr = np.array(values); n = len(arr)
    if n == 0: return float("nan"), float("nan"), float("nan")
    means = [arr[np.random.randint(0, n, n)].mean() for _ in range(n_iter)]
    return float(np.mean(means)), float(np.percentile(means, 100*(alpha/2))), float(np.percentile(means, 100*(1-alpha/2)))

if not BLEURT_AVAILABLE:
    raise SystemExit("Cannot run: bleurt not installed. See instructions above.")

print("Loading BLEURT scorer (checkpoint:", BLEURT_CKPT, ")...")
bleurt_scorer = bleurt_score_fn.BleurtScorer(BLEURT_CKPT)
print("BLEURT ready.")

# ---------- LOAD DATA & GEN MODEL ----------
print("\nLoading CNN/DailyMail...")
ds       = load_dataset(CNN_NAME, CNN_VER, split=CNN_SPLIT)
examples = [{"id": i, "document": ex["article"], "reference": ex["highlights"]} for i, ex in enumerate(ds)]
n_examples = len(examples)
print(f"Loaded {n_examples} examples.")

tok       = AutoTokenizer.from_pretrained(GEN_MODEL)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL).to(device)
gen_model.eval()
if device == "cuda": gen_model.half()

pretokenized = []
for ex in tqdm(examples, desc="pretokenize"):
    t = tok("summarize: "+ex["document"], truncation=True, padding="max_length", max_length=MAX_INPUT, return_tensors="pt")
    pretokenized.append({"input_ids": t["input_ids"].squeeze(0), "attention_mask": t["attention_mask"].squeeze(0)})

# ---------- STABILITY ----------
print("\n=== STABILITY: BLEURT (CNN/DailyMail) ===")
stability_records, corpus_stats = [], []
autocast_ctx = torch.amp.autocast(device_type="cuda") if device=="cuda" else nullcontext()

for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")
    set_seed(seed); per_example_hyps = []
    with torch.no_grad():
        for i in tqdm(range(0, n_examples, BATCH_SIZE), desc=f"gen(seed={seed})"):
            idxs = list(range(i, min(i+BATCH_SIZE, n_examples)))
            ids  = torch.stack([pretokenized[j]["input_ids"] for j in idxs]).to(device)
            attn = torch.stack([pretokenized[j]["attention_mask"] for j in idxs]).to(device)
            with autocast_ctx:
                out = gen_model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=MAX_NEW,
                                         do_sample=True, temperature=TEMP, top_k=TOP_K, top_p=TOP_P,
                                         num_return_sequences=GEN_K)
            dec = tok.batch_decode(out, skip_special_tokens=True)
            per_example_hyps.extend([dec[k:k+GEN_K] for k in range(0, len(dec), GEN_K)])

    assert len(per_example_hyps) == n_examples
    hyps_flat = [h for g in per_example_hyps for h in g]
    refs_flat = [examples[i]["reference"] for i, g in enumerate(per_example_hyps) for _ in g]

    print(f"Scoring {len(hyps_flat)} pairs with BLEURT (seed={seed})...")
    bleurt_scores = bleurt_scorer.score(references=refs_flat, candidates=hyps_flat)

    idx = 0; first_scores = []
    for ex_idx, gen_list in enumerate(per_example_hyps):
        ref = examples[ex_idx]["reference"]
        for sid, hyp in enumerate(gen_list):
            sc = float(bleurt_scores[idx])
            stability_records.append({"seed": seed, "example_id": ex_idx, "sample_id": sid,
                                      "hypothesis": hyp, "reference": ref, "bleurt": sc})
            if sid == 0: first_scores.append(sc)
            idx += 1

    mean_direct = float(np.mean(first_scores))
    _, bl, bh   = bootstrap_ci(first_scores)
    print(f"Seed {seed}: BLEURT={mean_direct:.4f} | CI [{bl:.4f},{bh:.4f}]")
    corpus_stats.append({"seed": seed, "bleurt_mean": mean_direct, "bleurt_boot_low": bl, "bleurt_boot_high": bh})

df_stab = pd.DataFrame(stability_records)
df_stab.to_csv(f"{OUT_DIR}/bleurt_stability_sample_level.csv", index=False)
pd.DataFrame(corpus_stats).to_csv(f"{OUT_DIR}/bleurt_corpus_per_seed.csv", index=False)

g     = df_stab.groupby(["seed","example_id"])["bleurt"].mean().reset_index()
pivot = g.pivot(index="example_id", columns="seed", values="bleurt")
pe    = pd.DataFrame({"example_id": pivot.index, "mean_across_seeds": pivot.mean(axis=1),
                      "std_across_seeds": pivot.std(axis=1, ddof=1)})
pe["cv_across_seeds"] = pe["std_across_seeds"] / pe["mean_across_seeds"].replace(0, 1e-8)
pe.to_csv(f"{OUT_DIR}/bleurt_per_example_across_seeds.csv", index=False)

try:
    lev_stat, lev_p = stats.levene(*[pivot[s].dropna().values for s in pivot.columns])
    print(f"Levene: stat={lev_stat:.4f}, p={lev_p:.4f}")
except: lev_stat, lev_p = None, None

summary = {"timestamp": time.asctime(), "device": device, "gen_model": GEN_MODEL,
           "bleurt_checkpoint": BLEURT_CKPT, "n_examples": n_examples, "gen_k": GEN_K, "seeds": SEEDS,
           "avg_std_across_examples": float(pe["std_across_seeds"].mean()),
           "avg_cv_across_examples":  float(pe["cv_across_seeds"].mean()),
           "levene_stat": float(lev_stat) if lev_stat is not None else None,
           "levene_p":    float(lev_p)    if lev_p    is not None else None}
with open(f"{OUT_DIR}/bleurt_stability_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== STABILITY COMPLETE ===")

# ---------- CONSISTENCY: SummEval ----------
print("\n=== CONSISTENCY: BLEURT vs Human (SummEval) ===")
se = load_dataset(SUMMEVAL_NAME, split="test")
records = []
for i, ex in enumerate(se):
    if not ex["human_summaries"]: continue
    ref = ex["human_summaries"][0]
    for j, cand in enumerate(ex["machine_summaries"]):
        records.append({"id": f"{i}_{j}", "candidate": cand, "reference": ref,
                        "coherence": ex["coherence"][j], "consistency": ex["consistency"][j],
                        "fluency": ex["fluency"][j], "relevance": ex["relevance"][j]})
se_df = pd.DataFrame(records)
print(f"Parsed {len(se_df)} SummEval rows. Computing BLEURT...")
se_scores = bleurt_scorer.score(references=se_df["reference"].tolist(), candidates=se_df["candidate"].tolist())
se_df["bleurt"] = [float(s) for s in se_scores]
se_df["composite_human"] = se_df[["coherence","consistency","fluency","relevance"]].mean(axis=1)
corr = {}
for h in ["coherence","consistency","fluency","relevance","composite_human"]:
    sub = se_df[[h,"bleurt"]].dropna()
    corr[h] = {"pearson": float(sub.corr("pearson").iloc[0,1]),
               "spearman": float(sub.corr("spearman").iloc[0,1]), "n": len(sub)}
se_df.to_csv(f"{OUT_DIR}/summ_eval_with_bleurt_large.csv", index=False)
with open(f"{OUT_DIR}/bleurt_vs_human_correlations.json", "w") as f:
    json.dump(corr, f, indent=2)
print("Correlations:", corr)
print("\n=== ALL BLEURT EXPERIMENTS COMPLETE ===")
print("Outputs saved to:", OUT_DIR)
