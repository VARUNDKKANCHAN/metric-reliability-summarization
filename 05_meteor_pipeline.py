# METEOR: Stability (CNN/DailyMail) + Consistency (SummEval)
# Run: python 05_meteor_pipeline.py
# Install: pip install datasets transformers sacrebleu pandas scipy tqdm matplotlib nltk

import os, json, random, time, re
from contextlib import nullcontext
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.meteor_score import meteor_score
from scipy import stats
from tqdm.auto import tqdm
import nltk
nltk.download('punkt',  quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

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
    GEN_K, SEEDS, BATCH_SIZE, BOOTSTRAP_ITERS = 1, ([100,101] if FAST_MODE else [100,101,102]), 2, 200
    MAX_INPUT = 256

if SMOKE_TEST:
    CNN_SPLIT, SEEDS, GEN_K, BATCH_SIZE, BOOTSTRAP_ITERS = "test[:10]", [SEEDS[0]], 1, 1, 50

TEMP, TOP_K, TOP_P, MAX_NEW, CI_ALPHA = 0.8, 50, 0.95, 128, 0.05
OUT_DIR = "outputs/meteor_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Config -> model:{MODEL_NAME} | GEN_K:{GEN_K} | seeds:{SEEDS} | batch:{BATCH_SIZE}")

# ---------- HELPERS ----------
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device == "cuda": torch.cuda.manual_seed_all(seed)

def safe_tokenize(text):
    text = "" if text is None else str(text)
    try:
        from nltk.tokenize import word_tokenize
        toks = word_tokenize(text.lower())
        if toks: return toks
    except Exception:
        pass
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower())

def sentence_meteor_tokenized(ref, hyp):
    try:
        return float(meteor_score([safe_tokenize(ref)], safe_tokenize(hyp)))
    except Exception as e:
        if not getattr(sentence_meteor_tokenized, "_warned", False):
            print("Warning: meteor_score failed:", e)
            sentence_meteor_tokenized._warned = True
        return 0.0

def bootstrap_ci_from_values(values, n_iter=BOOTSTRAP_ITERS, alpha=CI_ALPHA):
    arr = np.array(values, dtype=float)
    if arr.size == 0: return 0.0, 0.0, 0.0, []
    n = arr.shape[0]; samples = []
    for _ in range(n_iter):
        samples.append(arr[np.random.randint(0, n, size=n)].mean())
    return float(np.mean(samples)), float(np.percentile(samples, 100*(alpha/2))), float(np.percentile(samples, 100*(1-alpha/2))), samples

# ---------- STABILITY ----------
print("\n=== STABILITY: CNN/DailyMail (METEOR) ===")
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
    per_example_meteor_firsthyp = []
    for ex_idx, gen_list in enumerate(per_example_hyps):
        ref = examples[ex_idx]["reference"]
        for sid, hyp in enumerate(gen_list):
            m = sentence_meteor_tokenized(ref, hyp)
            stability_records.append({"seed": seed, "example_id": ex_idx, "sample_id": sid,
                                      "hypothesis": hyp, "reference": ref, "meteor": float(m)})
        per_example_meteor_firsthyp.append(sentence_meteor_tokenized(ref, gen_list[0]))

    direct = float(np.mean(per_example_meteor_firsthyp)) if per_example_meteor_firsthyp else 0.0
    bm, bl, bh, _ = bootstrap_ci_from_values(per_example_meteor_firsthyp)
    print(f"Seed {seed}: METEOR={direct:.4f} | CI [{bl:.4f},{bh:.4f}]")
    corpus_stats.append({"seed": seed, "meteor_mean": direct, "meteor_boot_mean": bm,
                         "meteor_boot_low": bl, "meteor_boot_high": bh})

df_stab = pd.DataFrame(stability_records)
df_stab.to_csv(f"{OUT_DIR}/meteor_stability_sample_level.csv", index=False)
pd.DataFrame(corpus_stats).to_csv(f"{OUT_DIR}/meteor_corpus_per_seed.csv", index=False)

g     = df_stab.groupby(["seed","example_id"])["meteor"].mean().reset_index()
pivot = g.pivot(index="example_id", columns="seed", values="meteor")
pe    = pd.DataFrame({"example_id": pivot.index, "mean_across_seeds": pivot.mean(axis=1),
                      "std_across_seeds": pivot.std(axis=1, ddof=1)})
pe["cv_across_seeds"] = pe["std_across_seeds"] / pe["mean_across_seeds"].replace(0, 1e-8)
pe.to_csv(f"{OUT_DIR}/meteor_per_example_across_seeds.csv", index=False)

try:
    lev_stat, lev_p = stats.levene(*[pivot[s].dropna().values for s in pivot.columns])
    print(f"Levene: stat={lev_stat:.4f}, p={lev_p:.4f}")
except: lev_stat, lev_p = None, None

summary = {"timestamp": time.asctime(), "device": device, "model": MODEL_NAME, "n_examples": n_examples,
           "gen_k": GEN_K, "seeds": SEEDS, "bootstrap_iters": BOOTSTRAP_ITERS,
           "avg_std_across_examples": float(pe["std_across_seeds"].mean()),
           "avg_cv_across_examples":  float(pe["cv_across_seeds"].mean()),
           "levene_stat": float(lev_stat) if lev_stat is not None else None,
           "levene_p":    float(lev_p)    if lev_p    is not None else None}
with open(f"{OUT_DIR}/meteor_stability_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== STABILITY COMPLETE ===")

# ---------- CONSISTENCY: SummEval ----------
print("\n=== CONSISTENCY: METEOR vs Human (SummEval) ===")
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
se_df["meteor"] = se_df.apply(lambda r: sentence_meteor_tokenized(r["reference"], r["candidate"]), axis=1)
corr = {}
for h in ["coherence","consistency","fluency","relevance"]:
    sub = se_df[[h,"meteor"]].dropna()
    corr[h] = {"pearson": float(sub.corr("pearson").iloc[0,1]),
               "spearman": float(sub.corr("spearman").iloc[0,1]), "n": len(sub)}
se_df.to_csv(f"{OUT_DIR}/summ_eval_with_meteor.csv", index=False)
with open(f"{OUT_DIR}/meteor_vs_human_correlations.json", "w") as f:
    json.dump(corr, f, indent=2)
print("Correlations:", corr)

# ---------- QUICK PLOTS ----------
plt.figure(figsize=(6,4))
pe["cv_across_seeds"].dropna().hist(bins=40)
plt.title("Per-example METEOR CV across seeds"); plt.xlabel("CV"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plot_meteor_cv_histogram.png", dpi=200); plt.close()
plt.figure(figsize=(6,4))
pd.DataFrame(corpus_stats)["meteor_mean"].plot.box()
plt.title("Corpus METEOR per seed"); plt.ylabel("METEOR")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plot_meteor_corpus_boxplot.png", dpi=200); plt.close()
print("\n=== COMPLETE. Outputs saved to:", OUT_DIR)
