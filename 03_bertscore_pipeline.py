# BERTScore: Stability (CNN/DailyMail) + Consistency (SummEval)
# Run: python 03_bertscore_pipeline.py
# Install: pip install transformers==4.38.2 bert-score==0.3.13 datasets pandas scipy tqdm nltk
# NOTE: Python 3.10 required for bert-score==0.3.13

import os, json, random, time
from contextlib import nullcontext
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from bert_score import score as bertscore_score
from scipy import stats
from tqdm.auto import tqdm
import nltk
nltk.download('punkt', quiet=True)

# ---------- CONFIG ----------
FAST_MODE  = False
SMOKE_TEST = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Detected device:", device)

CNN_NAME, CNN_VER, CNN_SPLIT = "cnn_dailymail", "3.0.0", "test[:500]"
SUMMEVAL_NAME = "mteb/summeval"

if device == "cuda":
    GEN_MODEL = "t5-small"
    BS_MODEL  = "roberta-large"
    GEN_K, SEEDS, BATCH_SIZE = (1 if FAST_MODE else 3), ([100,101,102] if FAST_MODE else list(range(100,110))), (12 if FAST_MODE else 16)
    BOOTSTRAP_ITERS = 200 if FAST_MODE else 1000
    MAX_INPUT = 512
else:
    GEN_MODEL = "sshleifer/distilbart-cnn-12-6"
    BS_MODEL  = "distilroberta-base"
    GEN_K, SEEDS, BATCH_SIZE = 1, ([100,101] if FAST_MODE else [100,101,102]), 2
    BOOTSTRAP_ITERS = 100 if FAST_MODE else 200
    MAX_INPUT = 256

if SMOKE_TEST:
    CNN_SPLIT, SEEDS, GEN_K, BATCH_SIZE, BOOTSTRAP_ITERS = "test[:10]", [SEEDS[0]], 1, 1, 50

TEMP, TOP_K, TOP_P, MAX_NEW, CI_ALPHA = 0.8, 50, 0.95, 128, 0.05
OUT_DIR = "outputs/bertscore_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Config -> GEN_MODEL:{GEN_MODEL} | BS_MODEL:{BS_MODEL} | GEN_K:{GEN_K} | seeds:{SEEDS} | batch:{BATCH_SIZE}")

# ---------- HELPERS ----------
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device == "cuda": torch.cuda.manual_seed_all(seed)

def bertscore_corpus_ci(f1_list, n_iter=BOOTSTRAP_ITERS, alpha=CI_ALPHA):
    arr = np.array(f1_list); n = len(arr); samples = []
    for _ in range(n_iter):
        samples.append(arr[np.random.randint(0, n, size=n)].mean())
    return float(np.mean(samples)), float(np.percentile(samples, 100*(alpha/2))), float(np.percentile(samples, 100*(1-alpha/2))), samples

# ---------- STABILITY: CNN/DailyMail ----------
print("\n=== STABILITY: CNN/DailyMail (BERTScore F1) ===")
ds_cnn   = load_dataset(CNN_NAME, CNN_VER, split=CNN_SPLIT)
examples = [{"id": i, "document": ex["article"], "reference": ex["highlights"]} for i, ex in enumerate(ds_cnn)]
n_examples = len(examples)
print(f"Loaded {n_examples} examples.")

gen_tok   = AutoTokenizer.from_pretrained(GEN_MODEL)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL).to(device)
gen_model.eval()
if device == "cuda":
    try: gen_model.half(); print("Using fp16.")
    except: pass

pretokenized = []
for ex in tqdm(examples, desc="pretokenize"):
    t = gen_tok("summarize: "+ex["document"], truncation=True, padding="max_length", max_length=MAX_INPUT, return_tensors="pt")
    pretokenized.append({"input_ids": t["input_ids"].squeeze(0), "attention_mask": t["attention_mask"].squeeze(0)})

stability_records, corpus_stats = [], []
autocast_ctx = torch.cuda.amp.autocast if device == "cuda" else nullcontext

for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")
    set_seed(seed); per_example_hyps = []
    with torch.no_grad():
        for i in tqdm(range(0, n_examples, BATCH_SIZE), desc=f"gen(seed={seed})"):
            idxs = list(range(i, min(i+BATCH_SIZE, n_examples)))
            ids  = torch.stack([pretokenized[j]["input_ids"] for j in idxs]).to(device)
            attn = torch.stack([pretokenized[j]["attention_mask"] for j in idxs]).to(device)
            with autocast_ctx():
                out = gen_model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=MAX_NEW,
                                         do_sample=True, temperature=TEMP, top_k=TOP_K, top_p=TOP_P,
                                         num_return_sequences=GEN_K, use_cache=True)
            dec = gen_tok.batch_decode(out, skip_special_tokens=True)
            per_example_hyps.extend([dec[j:j+GEN_K] for j in range(0, len(dec), GEN_K)])

    assert len(per_example_hyps) == n_examples
    hyps_flat = [h for g in per_example_hyps for h in g]
    refs_flat = [examples[i]["reference"] for i, g in enumerate(per_example_hyps) for _ in g]

    print(f"Computing BERTScore on {len(hyps_flat)} pairs (seed={seed})...")
    P, R, F1 = bertscore_score(hyps_flat, refs_flat, model_type=BS_MODEL, verbose=False,
                               device=device, idf=False, rescale_with_baseline=False)
    f1_list = [float(x) for x in F1]

    idx = 0
    for ex_idx, gen_list in enumerate(per_example_hyps):
        ref = examples[ex_idx]["reference"]
        for sid, hyp in enumerate(gen_list):
            stability_records.append({"seed": seed, "example_id": ex_idx, "sample_id": sid,
                                      "hypothesis": hyp, "reference": ref, "bertscore_f1": f1_list[idx]})
            idx += 1

    first_hyps = [g[0] for g in per_example_hyps]
    refs_all   = [ex["reference"] for ex in examples]
    _, _, F1f  = bertscore_score(first_hyps, refs_all, model_type=BS_MODEL, verbose=False,
                                 device=device, idf=False, rescale_with_baseline=False)
    f1f = [float(x) for x in F1f]
    corp_mean = float(np.mean(f1f))
    bm, bl, bh, _ = bertscore_corpus_ci(f1f)
    print(f"Seed {seed}: BERTScore-F1={corp_mean:.4f} | CI [{bl:.4f},{bh:.4f}]")
    corpus_stats.append({"seed": seed, "corpus_bertscore_f1_firsthyp": corp_mean,
                         "corpus_bertscore_boot_mean": bm, "corpus_bertscore_boot_low": bl, "corpus_bertscore_boot_high": bh})

df_stab = pd.DataFrame(stability_records)
df_stab.to_csv(f"{OUT_DIR}/bertscore_stability_sample_level.csv", index=False)
pd.DataFrame(corpus_stats).to_csv(f"{OUT_DIR}/bertscore_corpus_per_seed.csv", index=False)

g     = df_stab.groupby(["seed","example_id"])["bertscore_f1"].mean().reset_index()
pivot = g.pivot(index="example_id", columns="seed", values="bertscore_f1")
pe    = pd.DataFrame({"example_id": pivot.index, "mean_across_seeds": pivot.mean(axis=1),
                      "std_across_seeds": pivot.std(axis=1, ddof=1)})
pe["cv_across_seeds"] = pe["std_across_seeds"] / pe["mean_across_seeds"].replace(0, 1e-8)
pe.to_csv(f"{OUT_DIR}/bertscore_per_example_across_seeds.csv", index=False)

try:
    lev_stat, lev_p = stats.levene(*[pivot[s].dropna().values for s in pivot.columns])
    print(f"Levene: stat={lev_stat:.4f}, p={lev_p:.4f}")
except: lev_stat, lev_p = None, None

summary = {"timestamp": time.asctime(), "device": device, "gen_model": GEN_MODEL,
           "bertscore_model": BS_MODEL, "n_examples": n_examples, "gen_k": GEN_K, "seeds": SEEDS,
           "avg_std_across_examples": float(pe["std_across_seeds"].mean()),
           "avg_cv_across_examples":  float(pe["cv_across_seeds"].mean()),
           "levene_stat": float(lev_stat) if lev_stat is not None else None,
           "levene_p":    float(lev_p)    if lev_p    is not None else None}
with open(f"{OUT_DIR}/bertscore_stability_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== STABILITY COMPLETE ===")

# ---------- CONSISTENCY: SummEval ----------
print("\n=== CONSISTENCY: BERTScore vs Human (SummEval) ===")
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
print(f"Parsed {len(se_df)} SummEval rows. Computing BERTScore...")
P_s, R_s, F1_s = bertscore_score(se_df["candidate"].tolist(), se_df["reference"].tolist(),
                                  model_type=BS_MODEL, verbose=False, device=device,
                                  idf=False, rescale_with_baseline=False)
se_df["bertscore_f1"] = [float(x) for x in F1_s]
corr = {}
for h in ["coherence","consistency","fluency","relevance"]:
    sub = se_df[[h,"bertscore_f1"]].dropna()
    corr[h] = {"pearson": float(sub.corr("pearson").iloc[0,1]),
               "spearman": float(sub.corr("spearman").iloc[0,1]), "n": len(sub)}
se_df.to_csv(f"{OUT_DIR}/summ_eval_with_bertscore.csv", index=False)
with open(f"{OUT_DIR}/bertscore_vs_human_correlations.json", "w") as f:
    json.dump(corr, f, indent=2)
print("Correlations:", corr)
print("\n=== CONSISTENCY COMPLETE ===")
print("All outputs saved to:", OUT_DIR)
