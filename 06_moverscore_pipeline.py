# Approximate MoverScore (Hungarian proxy): Stability + Consistency
# Run: python 06_moverscore_pipeline.py
# Install: pip install sentence-transformers datasets transformers sacrebleu pandas scipy tqdm nltk

import os, json, random, re, time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy import stats as scipy_stats
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt',  quiet=True)
nltk.download('wordnet', quiet=True)

# ---------- CONFIG ----------
OUT_DIR = Path("outputs/moverscore_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CNN_NAME, CNN_VER, CNN_SPLIT = "cnn_dailymail", "3.0.0", "test[:500]"
SUMMEVAL_NAME = "mteb/summeval"
MODEL_NAME    = "t5-small"
GEN_K         = 3
SEEDS         = list(range(100, 110))
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE    = 16 if DEVICE == "cuda" else 2
MAX_INPUT     = 512 if DEVICE == "cuda" else 256
MAX_NEW       = 128
BOOTSTRAP_ITERS = 500 if DEVICE == "cuda" else 200
CI_ALPHA      = 0.05
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L12-v1"

print("Device:", DEVICE)
print(f"Config -> model:{MODEL_NAME} | GEN_K:{GEN_K} | seeds:{SEEDS} | batch:{BATCH_SIZE}")

# ---------- HELPERS ----------
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if DEVICE == "cuda": torch.cuda.manual_seed_all(s)

def safe_tokenize(text):
    text = "" if text is None else str(text)
    try:
        toks = word_tokenize(text.lower())
        if toks: return toks
    except Exception:
        pass
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower())

from sentence_transformers import SentenceTransformer
print("Loading embedder:", EMBED_MODEL)
embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
print("Embedder loaded.")

def approx_moverscore_hungarian(ref, hyp, max_tokens=64):
    r_tokens = safe_tokenize(ref)[:max_tokens]
    h_tokens = safe_tokenize(hyp)[:max_tokens]
    if not r_tokens or not h_tokens: return 0.0
    tokens = r_tokens + h_tokens
    embs   = embedder.encode(tokens, convert_to_tensor=False, show_progress_bar=False)
    re_, he_ = np.array(embs[:len(r_tokens)]), np.array(embs[len(r_tokens):])
    dist   = cdist(re_, he_, metric='cosine')
    n_ref, n_hyp = dist.shape; n = max(n_ref, n_hyp)
    if n_ref != n_hyp:
        big = float(dist.max()) + 1.0
        sq  = np.full((n, n), big)
        sq[:n_ref, :n_hyp] = dist
    else:
        sq = dist
    row_ind, col_ind = linear_sum_assignment(sq)
    matches = [(r, c) for r, c in zip(row_ind, col_ind) if r < n_ref and c < n_hyp]
    if not matches: return 0.0
    mean_cost = sum(sq[r, c] for r, c in matches) / len(matches)
    return float(max(0.0, min(1.0, 1.0 - (mean_cost / 2.0))))

# ---------- LOAD DATA & MODEL ----------
print("\nLoading CNN/DailyMail:", CNN_SPLIT)
ds       = load_dataset(CNN_NAME, CNN_VER, split=CNN_SPLIT)
examples = [{"id": i, "document": ex["article"], "reference": ex["highlights"]} for i, ex in enumerate(ds)]
n_examples = len(examples)
print(f"Loaded {n_examples} examples.")

print("Loading generation model:", MODEL_NAME)
tok       = AutoTokenizer.from_pretrained(MODEL_NAME)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
gen_model.eval()
if DEVICE == "cuda":
    try: gen_model.half()
    except: pass

pretoken = []
for ex in tqdm(examples, desc="pretokenize"):
    t = tok("summarize: "+ex["document"], truncation=True, padding="max_length", max_length=MAX_INPUT, return_tensors="pt")
    pretoken.append({"input_ids": t["input_ids"].squeeze(0), "attention_mask": t["attention_mask"].squeeze(0)})

# ---------- STABILITY ----------
stability_records, corpus_stats = [], []
start_all = time.time()

for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")
    set_seed(seed); per_example_hyps = []
    with torch.no_grad():
        for i in tqdm(range(0, n_examples, BATCH_SIZE), desc=f"gen(seed={seed})"):
            idxs = list(range(i, min(i+BATCH_SIZE, n_examples)))
            ids  = torch.stack([pretoken[j]["input_ids"] for j in idxs]).to(DEVICE)
            attn = torch.stack([pretoken[j]["attention_mask"] for j in idxs]).to(DEVICE)
            out  = gen_model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=MAX_NEW,
                                      do_sample=True, temperature=0.8, top_k=50, top_p=0.95,
                                      num_return_sequences=GEN_K)
            dec = tok.batch_decode(out, skip_special_tokens=True)
            per_example_hyps.extend([dec[j:j+GEN_K] for j in range(0, len(dec), GEN_K)])

    assert len(per_example_hyps) == n_examples
    first_scores = []
    for ex_idx, gen_list in enumerate(tqdm(per_example_hyps, desc="score_samples")):
        ref = examples[ex_idx]["reference"]
        scores = [approx_moverscore_hungarian(ref, h) for h in gen_list]
        for sid, sc in enumerate(scores):
            stability_records.append({"seed": seed, "example_id": ex_idx, "sample_id": sid,
                                      "reference": ref, "hypothesis": gen_list[sid], "moverscore": float(sc)})
        first_scores.append(float(scores[0]))

    arr = np.array(first_scores)
    direct_mean = float(arr.mean()) if arr.size > 0 else 0.0
    boot_means  = [arr[np.random.randint(0, len(arr), len(arr))].mean() for _ in range(BOOTSTRAP_ITERS)]
    bl = float(np.percentile(boot_means, 100*(CI_ALPHA/2)))
    bh = float(np.percentile(boot_means, 100*(1-CI_ALPHA/2)))
    corpus_stats.append({"seed": seed, "moverscore_mean": direct_mean,
                         "moverscore_boot_low": bl, "moverscore_boot_high": bh})
    print(f"Seed {seed}: mean={direct_mean:.4f} | CI [{bl:.4f},{bh:.4f}]")

stability_df = pd.DataFrame(stability_records)
stability_df.to_csv(OUT_DIR/"moverscore_stability_samples.csv", index=False)
pd.DataFrame(corpus_stats).to_csv(OUT_DIR/"moverscore_stability_summary.csv", index=False)

g     = stability_df.groupby(["seed","example_id"])["moverscore"].mean().reset_index()
pivot = g.pivot(index="example_id", columns="seed", values="moverscore")
pe    = pd.DataFrame({"example_id": pivot.index, "mean_across_seeds": pivot.mean(axis=1),
                      "std_across_seeds": pivot.std(axis=1, ddof=1)})
pe["cv_across_seeds"] = pe["std_across_seeds"] / pe["mean_across_seeds"].replace(0, 1e-8)
pe.to_csv(OUT_DIR/"moverscore_per_example_across_seeds.csv", index=False)

try:
    lev_stat, lev_p = scipy_stats.levene(*[pivot[c].dropna().values for c in pivot.columns])
    print(f"Levene: stat={float(lev_stat):.4f}, p={float(lev_p):.4f}")
except Exception as e:
    print("Levene skipped:", e)

print("\n=== STABILITY COMPLETE ===")

# ---------- CONSISTENCY: SummEval ----------
print("\n=== CONSISTENCY: MoverScore vs Human (SummEval) ===")
se = load_dataset(SUMMEVAL_NAME, split="test")
se_records = []
for doc_id, ex in enumerate(se):
    if not ex["human_summaries"]: continue
    ref = ex["human_summaries"][0]
    for sid, cand in enumerate(ex["machine_summaries"]):
        se_records.append({"doc_id": doc_id, "system_id": sid, "reference": ref, "candidate": cand,
                           "coherence": ex["coherence"][sid], "consistency": ex["consistency"][sid],
                           "fluency": ex["fluency"][sid], "relevance": ex["relevance"][sid]})
se_df = pd.DataFrame(se_records)
print("Parsed SummEval rows:", len(se_df))
se_df["moverscore"] = se_df.apply(lambda r: approx_moverscore_hungarian(r["reference"], r["candidate"]), axis=1)
se_df.to_csv(OUT_DIR/"summ_eval_moverscore.csv", index=False)

corr = {}
for h in ["coherence","consistency","fluency","relevance"]:
    sub = se_df[[h,"moverscore"]].dropna()
    corr[h] = {"pearson": float(sub.corr("pearson").iloc[0,1]),
               "spearman": float(sub.corr("spearman").iloc[0,1]), "n": len(sub)}
with open(OUT_DIR/"moverscore_vs_human_correlations.json", "w") as f:
    json.dump(corr, f, indent=2)
print("Correlations:", corr)
print("\nAll done. Elapsed:", round(time.time()-start_all, 1), "sec")
print("Outputs saved to:", OUT_DIR)
