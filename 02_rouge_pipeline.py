# ROUGE-1/2/L Stability (CNN/DailyMail) + Consistency (SummEval) + Analysis Plots
# Run: python 02_rouge_pipeline.py
# Install: pip install datasets transformers rouge-score pandas scipy tqdm matplotlib nltk

import os, json, random, time
from contextlib import nullcontext
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
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
    MODEL_NAME = "t5-small"
    GEN_K, SEEDS, BATCH_SIZE, BOOTSTRAP_ITERS = (1 if FAST_MODE else 3), ([100,101,102] if FAST_MODE else list(range(100,110))), (12 if FAST_MODE else 16), (200 if FAST_MODE else 1000)
    MAX_INPUT = 512
else:
    MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
    GEN_K, SEEDS, BATCH_SIZE, BOOTSTRAP_ITERS = 1, ([100,101] if FAST_MODE else [100,101,102]), 2, (100 if FAST_MODE else 200)
    MAX_INPUT = 256

if SMOKE_TEST:
    CNN_SPLIT, SEEDS, GEN_K, BATCH_SIZE, BOOTSTRAP_ITERS = "test[:10]", [SEEDS[0]], 1, 1, 50

TEMP, TOP_K, TOP_P, MAX_NEW, CI_ALPHA = 0.8, 50, 0.95, 128, 0.05
OUT_DIR = "outputs/rouge_final_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Config -> model:{MODEL_NAME} | GEN_K:{GEN_K} | seeds:{SEEDS} | batch:{BATCH_SIZE} | max_input:{MAX_INPUT}")

# ---------- HELPERS ----------
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device == "cuda": torch.cuda.manual_seed_all(seed)

rouge_scorer_inst = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)

def compute_rouge_f1(ref, hyp):
    sc = rouge_scorer_inst.score(ref, hyp)
    return {"rouge1": float(sc["rouge1"].fmeasure), "rouge2": float(sc["rouge2"].fmeasure),
            "rougeL": float(sc["rougeL"].fmeasure)}

def corpus_rougeL_ci(hyps, refs, n_iter=BOOTSTRAP_ITERS, alpha=CI_ALPHA):
    n = len(hyps); vals = []
    for _ in range(n_iter):
        idxs = np.random.randint(0, n, size=n)
        vals.append(np.mean([rouge_scorer_inst.score(refs[i], hyps[i])["rougeL"].fmeasure for i in idxs]))
    return float(np.mean(vals)), float(np.percentile(vals, 100*(alpha/2))), float(np.percentile(vals, 100*(1-alpha/2))), vals

# ---------- STABILITY: CNN/DailyMail ----------
print("\n=== STABILITY: CNN/DailyMail (ROUGE-1/2/L) ===")
ds_cnn   = load_dataset(CNN_NAME, CNN_VER, split=CNN_SPLIT)
examples = [{"id": i, "document": ex["article"], "reference": ex["highlights"]} for i, ex in enumerate(ds_cnn)]
n_examples = len(examples)
print(f"Loaded {n_examples} examples.")

tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model.eval()
if device == "cuda":
    try: model.half(); print("Using fp16.")
    except: pass

pretokenized = []
for ex in tqdm(examples, desc="pretokenize"):
    t = tok("summarize: "+ex["document"], truncation=True, padding='max_length', max_length=MAX_INPUT, return_tensors="pt")
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
                out = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=MAX_NEW,
                                     do_sample=True, temperature=TEMP, top_k=TOP_K, top_p=TOP_P,
                                     num_return_sequences=GEN_K, use_cache=True)
            dec = tok.batch_decode(out, skip_special_tokens=True)
            per_example_hyps.extend([dec[j:j+GEN_K] for j in range(0, len(dec), GEN_K)])

    assert len(per_example_hyps) == n_examples
    first_hyps = []
    for ex_idx, gen_list in enumerate(per_example_hyps):
        ref = examples[ex_idx]["reference"]
        for sid, hyp in enumerate(gen_list):
            r = compute_rouge_f1(ref, hyp)
            stability_records.append({"seed": seed, "example_id": ex_idx, "sample_id": sid,
                                      "hypothesis": hyp, "reference": ref,
                                      "rouge1": r["rouge1"], "rouge2": r["rouge2"], "rougeL": r["rougeL"]})
        first_hyps.append(gen_list[0])

    refs = [ex["reference"] for ex in examples]
    mean_rl, lo, hi, _ = corpus_rougeL_ci(first_hyps, refs)
    mean_direct = float(np.mean([rouge_scorer_inst.score(refs[i], first_hyps[i])["rougeL"].fmeasure for i in range(n_examples)]))
    print(f"Seed {seed}: ROUGE-L={mean_direct:.4f} | CI [{lo:.4f},{hi:.4f}]")
    corpus_stats.append({"seed": seed, "rougeL_mean": mean_direct,
                         "rougeL_boot_mean": mean_rl, "rougeL_boot_low": lo, "rougeL_boot_high": hi})

df_stab = pd.DataFrame(stability_records)
df_stab.to_csv(f"{OUT_DIR}/rouge_stability_sample_level.csv", index=False)
pd.DataFrame(corpus_stats).to_csv(f"{OUT_DIR}/rouge_corpus_per_seed.csv", index=False)

g  = df_stab.groupby(["seed","example_id"]).agg({"rouge1":"mean","rouge2":"mean","rougeL":"mean"}).reset_index()
pR1 = g.pivot(index="example_id", columns="seed", values="rouge1")
pR2 = g.pivot(index="example_id", columns="seed", values="rouge2")
pRL = g.pivot(index="example_id", columns="seed", values="rougeL")
pe  = pd.DataFrame({"example_id": pRL.index,
                    "mean_r1_across_seeds": pR1.mean(axis=1), "std_r1_across_seeds": pR1.std(axis=1, ddof=1),
                    "mean_r2_across_seeds": pR2.mean(axis=1), "std_r2_across_seeds": pR2.std(axis=1, ddof=1),
                    "mean_rl_across_seeds": pRL.mean(axis=1), "std_rl_across_seeds": pRL.std(axis=1, ddof=1)})
pe["cv_rl_across_seeds"] = pe["std_rl_across_seeds"] / pe["mean_rl_across_seeds"].replace(0, 1e-8)
pe.to_csv(f"{OUT_DIR}/rouge_per_example_across_seeds.csv", index=False)

try:
    lev_stat, lev_p = stats.levene(*[pRL[s].dropna().values for s in pRL.columns])
    print(f"Levene: stat={lev_stat:.4f}, p={lev_p:.4f}")
except Exception as e:
    lev_stat, lev_p = None, None

summary = {"timestamp": time.asctime(), "device": device, "dataset": f"{CNN_NAME} {CNN_VER}",
           "n_examples": n_examples, "model": MODEL_NAME, "gen_k": GEN_K, "seeds": SEEDS,
           "bootstrap_iters": BOOTSTRAP_ITERS,
           "avg_std_rl_across_examples": float(pe["std_rl_across_seeds"].mean()),
           "avg_cv_rl_across_examples":  float(pe["cv_rl_across_seeds"].mean()),
           "levene_stat": float(lev_stat) if lev_stat is not None else None,
           "levene_p":    float(lev_p)    if lev_p    is not None else None}
with open(f"{OUT_DIR}/rouge_stability_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== STABILITY COMPLETE ===")

# ---------- CONSISTENCY: SummEval ----------
print("\n=== CONSISTENCY: ROUGE vs Human (SummEval) ===")
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
print(f"Parsed {len(se_df)} SummEval rows.")
rr = se_df.apply(lambda r: compute_rouge_f1(r["reference"], r["candidate"]), axis=1)
se_df["rouge1"] = rr.apply(lambda x: x["rouge1"])
se_df["rouge2"] = rr.apply(lambda x: x["rouge2"])
se_df["rougeL"] = rr.apply(lambda x: x["rougeL"])
corr = {}
for h in ["coherence","consistency","fluency","relevance"]:
    corr[h] = {}
    for m in ["rouge1","rouge2","rougeL"]:
        sub = se_df[[h,m]].dropna()
        corr[h][m] = {"pearson": float(sub.corr("pearson").iloc[0,1]),
                      "spearman": float(sub.corr("spearman").iloc[0,1]), "n": len(sub)}
se_df.to_csv(f"{OUT_DIR}/summ_eval_with_rouge.csv", index=False)
with open(f"{OUT_DIR}/rouge_vs_human_correlations.json", "w") as f:
    json.dump(corr, f, indent=2)
print("Correlations saved.")

# ---------- QUICK PLOTS ----------
print("\n=== PLOTS ===")
plt.figure(figsize=(6,4))
pe["cv_rl_across_seeds"].dropna().hist(bins=40)
plt.title("Per-example ROUGE-L CV across seeds"); plt.xlabel("CV"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plot_cv_histogram.png", dpi=200); plt.close()

plt.figure(figsize=(6,4))
pd.DataFrame(corpus_stats)["rougeL_mean"].plot.box()
plt.title("Corpus ROUGE-L per seed"); plt.ylabel("ROUGE-L (F1)")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plot_corpus_boxplot.png", dpi=200); plt.close()

agg = df_stab.groupby(["seed","example_id"]).agg({"rouge1":"mean","rouge2":"mean","rougeL":"mean"}).reset_index()
mm  = agg.groupby("example_id").mean().reset_index(drop=True)[["rouge1","rouge2","rougeL"]].dropna()
corr_sp = mm.corr(method="spearman")
print("Spearman correlation (ROUGE1/2/L):\n", corr_sp.round(4))
plt.figure(figsize=(5,4))
im = plt.imshow(corr_sp.values, vmin=-1, vmax=1)
plt.xticks(range(3), corr_sp.columns); plt.yticks(range(3), corr_sp.index)
plt.colorbar(im)
for i in range(3):
    for j in range(3):
        plt.text(j, i, f"{corr_sp.values[i,j]:.2f}", ha="center", va="center")
plt.title("Spearman correlation (ROUGE metrics)")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plot_rouge_spearman_heatmap.png", dpi=200); plt.close()

print("All outputs saved to:", OUT_DIR)
