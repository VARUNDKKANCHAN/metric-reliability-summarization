# CHRF++: Stability (CNN/DailyMail) + Consistency (SummEval)
# Run: python 04_chrf_pipeline.py
# Install: pip install datasets transformers sacrebleu pandas scipy tqdm nltk

import os, json, random, time
from contextlib import nullcontext
import numpy as np, pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu
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
CHAR_ORDER, WORD_ORDER, BETA = 6, 2, 2.0
OUT_DIR = "outputs/chrfpp_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Config -> model:{MODEL_NAME} | GEN_K:{GEN_K} | seeds:{SEEDS} | char_order:{CHAR_ORDER} | word_order:{WORD_ORDER}")

# ---------- HELPERS ----------
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device == "cuda": torch.cuda.manual_seed_all(seed)

def sentence_chrf(ref, hyp, word_order=WORD_ORDER, char_order=CHAR_ORDER, beta=BETA):
    try:
        return float(sacrebleu.corpus_chrf([hyp], [[ref]], char_order=char_order, word_order=word_order, beta=beta).score)
    except Exception:
        try:
            from sacrebleu.metrics import CHRF
            return float(CHRF(word_order=word_order, char_order=char_order, beta=beta).corpus_score([hyp], [[ref]]).score)
        except Exception:
            return 0.0

def corpus_chrf_ci(hyps, refs, n_iter=BOOTSTRAP_ITERS, alpha=CI_ALPHA):
    n = len(hyps); scores = []
    for _ in range(n_iter):
        idxs = np.random.randint(0, n, size=n)
        scores.append(float(sacrebleu.corpus_chrf([hyps[i] for i in idxs], [[refs[i] for i in idxs]],
                                                  char_order=CHAR_ORDER, word_order=WORD_ORDER, beta=BETA).score))
    return float(np.mean(scores)), float(np.percentile(scores, 100*(alpha/2))), float(np.percentile(scores, 100*(1-alpha/2))), scores

# ---------- STABILITY ----------
print("\n=== STABILITY: CNN/DailyMail (CHRF++) ===")
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
            ch = sentence_chrf(ref, hyp)
            stability_records.append({"seed": seed, "example_id": ex_idx, "sample_id": sid,
                                      "hypothesis": hyp, "reference": ref, "chrfpp": float(ch)})
        first_hyps.append(gen_list[0])

    refs = [ex["reference"] for ex in examples]
    bm, bl, bh, _ = corpus_chrf_ci(first_hyps, refs)
    direct = float(np.mean([sentence_chrf(refs[i], first_hyps[i]) for i in range(n_examples)]))
    print(f"Seed {seed}: CHRF++={direct:.4f} | CI [{bl:.4f},{bh:.4f}]")
    corpus_stats.append({"seed": seed, "chrfpp_mean": direct, "chrfpp_boot_mean": bm,
                         "chrfpp_boot_low": bl, "chrfpp_boot_high": bh})

df_stab = pd.DataFrame(stability_records)
df_stab.to_csv(f"{OUT_DIR}/chrfpp_stability_sample_level.csv", index=False)
pd.DataFrame(corpus_stats).to_csv(f"{OUT_DIR}/chrfpp_corpus_per_seed.csv", index=False)

g     = df_stab.groupby(["seed","example_id"])["chrfpp"].mean().reset_index()
pivot = g.pivot(index="example_id", columns="seed", values="chrfpp")
pe    = pd.DataFrame({"example_id": pivot.index, "mean_across_seeds": pivot.mean(axis=1),
                      "std_across_seeds": pivot.std(axis=1, ddof=1)})
pe["cv_across_seeds"] = pe["std_across_seeds"] / pe["mean_across_seeds"].replace(0, 1e-8)
pe.to_csv(f"{OUT_DIR}/chrfpp_per_example_across_seeds.csv", index=False)

try:
    lev_stat, lev_p = stats.levene(*[pivot[s].dropna().values for s in pivot.columns])
    print(f"Levene: stat={lev_stat:.4f}, p={lev_p:.4f}")
except: lev_stat, lev_p = None, None

summary = {"timestamp": time.asctime(), "device": device, "model": MODEL_NAME, "n_examples": n_examples,
           "gen_k": GEN_K, "seeds": SEEDS, "char_order": CHAR_ORDER, "word_order": WORD_ORDER, "beta": BETA,
           "avg_std_across_examples": float(pe["std_across_seeds"].mean()),
           "avg_cv_across_examples":  float(pe["cv_across_seeds"].mean()),
           "levene_stat": float(lev_stat) if lev_stat is not None else None,
           "levene_p":    float(lev_p)    if lev_p    is not None else None}
with open(f"{OUT_DIR}/chrfpp_stability_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== STABILITY COMPLETE ===")

# ---------- CONSISTENCY: SummEval ----------
print("\n=== CONSISTENCY: CHRF++ vs Human (SummEval) ===")
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
se_df["chrfpp"] = se_df.apply(lambda r: sentence_chrf(r["reference"], r["candidate"]), axis=1)
corr = {}
for h in ["coherence","consistency","fluency","relevance"]:
    sub = se_df[[h,"chrfpp"]].dropna()
    corr[h] = {"pearson": float(sub.corr("pearson").iloc[0,1]),
               "spearman": float(sub.corr("spearman").iloc[0,1]), "n": len(sub)}
se_df.to_csv(f"{OUT_DIR}/summ_eval_with_chrfpp.csv", index=False)
with open(f"{OUT_DIR}/chrfpp_vs_human_correlations.json", "w") as f:
    json.dump(corr, f, indent=2)
print("Correlations:", corr)
print("\n=== COMPLETE. Outputs saved to:", OUT_DIR)
