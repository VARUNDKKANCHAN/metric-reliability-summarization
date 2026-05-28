# COMET: Stability (CNN/DailyMail) + Consistency (SummEval)
# Run: python 07_comet_pipeline.py
# Install: pip install unbabel-comet datasets transformers pandas scipy tqdm nltk

import os, json, random, time
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from scipy import stats
from contextlib import nullcontext
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from comet import download_model, load_from_checkpoint
import nltk
nltk.download("punkt", quiet=True)

# ---------- CONFIG ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Detected device:", device)

CNN_NAME, CNN_VER, CNN_SPLIT = "cnn_dailymail", "3.0.0", "test[:500]"
SUMMEVAL_NAME  = "mteb/summeval"
GEN_MODEL      = "t5-small"
GEN_K          = 3
SEEDS          = list(range(100, 110))
BATCH_SIZE     = 16
MAX_INPUT, MAX_NEW = 512, 128
TEMP, TOP_K, TOP_P = 0.8, 50, 0.95
BOOTSTRAP_ITERS, CI_ALPHA = 1000, 0.05
COMET_MODEL_NAME = "wmt20-comet-da"
OUT_DIR = "outputs/comet_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Config -> model:{GEN_MODEL} | seeds:{SEEDS} | GEN_K:{GEN_K}")

# ---------- HELPERS ----------
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device == "cuda": torch.cuda.manual_seed_all(seed)

def bootstrap_ci(values):
    arr = np.array(values); n = len(arr)
    if n == 0: return float("nan"), float("nan"), float("nan")
    means = [arr[np.random.randint(0, n, n)].mean() for _ in range(BOOTSTRAP_ITERS)]
    return float(np.mean(means)), float(np.percentile(means, 100*(CI_ALPHA/2))), float(np.percentile(means, 100*(1-CI_ALPHA/2)))

def comet_predict_safe(model, data, batch_size=8, use_gpu=(device=="cuda")):
    attempts = [{"batch_size": batch_size, "gpus": 1}, {"batch_size": batch_size, "devices": 1},
                {"batch_size": batch_size, "accelerator": "gpu", "devices": 1}, {"batch_size": batch_size}] if use_gpu else [{"batch_size": batch_size}]
    last_exc = None
    for kw in attempts:
        try: return model.predict(data, **kw)
        except (TypeError, Exception) as e: last_exc = e; continue
    try: return model.predict(data)
    except Exception as e: raise last_exc if last_exc else e

def extract_seg_scores(pred):
    if isinstance(pred, dict):
        raw = pred.get("scores") or pred.get("predictions") or list(pred.values())[0]
    elif isinstance(pred, (list, tuple)):
        raw = pred[0] if pred else []
    else:
        raw = pred
    if isinstance(raw, torch.Tensor):
        raw_scores = [raw.detach().cpu().item()] if raw.dim()==0 else raw.detach().cpu().numpy().tolist()
    elif isinstance(raw, np.ndarray):
        raw_scores = raw.tolist()
    elif isinstance(raw, (list, tuple)):
        if raw and isinstance(raw[0], dict):
            key = next((k for k in ("score","scores","comet") if k in raw[0]), None)
            if key: raw_scores = [r[key] for r in raw]
            else:
                raw_scores = []
                for r in raw:
                    num = next((v for v in r.values() if isinstance(v, (int,float,np.number,torch.Tensor))), None)
                    if num is None: raise ValueError("No numeric score in prediction dict")
                    raw_scores.append(num.detach().cpu().item() if isinstance(num, torch.Tensor) else num)
        else:
            raw_scores = list(raw)
    else:
        raw_scores = [raw]
    scores = []
    for s in raw_scores:
        if isinstance(s, torch.Tensor): s = s.detach().cpu().item()
        elif isinstance(s, np.generic): s = float(s)
        if isinstance(s, str):
            try: s = float(s.strip())
            except: raise ValueError(f"Non-numeric score: {s!r}")
        scores.append(float(s))
    return scores

# ---------- LOAD DATA & MODELS ----------
print("\nLoading CNN/DailyMail...")
ds       = load_dataset(CNN_NAME, CNN_VER, split=CNN_SPLIT)
examples = [{"id": i, "document": ex["article"], "reference": ex["highlights"]} for i, ex in enumerate(ds)]
print(f"Loaded {len(examples)} examples.")

print("Loading generation model:", GEN_MODEL)
tok       = AutoTokenizer.from_pretrained(GEN_MODEL)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL).to(device)
gen_model.eval()
if device == "cuda": gen_model.half()

pretokenized = []
for ex in tqdm(examples, desc="pretokenize"):
    t = tok("summarize: "+ex["document"], truncation=True, padding="max_length", max_length=MAX_INPUT, return_tensors="pt")
    pretokenized.append({"input_ids": t["input_ids"].squeeze(0), "attention_mask": t["attention_mask"].squeeze(0)})

print("\nLoading COMET model:", COMET_MODEL_NAME)
ckpt        = download_model(COMET_MODEL_NAME)
comet_model = load_from_checkpoint(ckpt)
print("COMET ready.")

# ---------- STABILITY ----------
print("\n=== STABILITY: COMET (CNN/DailyMail) ===")
stability_records, corpus_stats = [], []
autocast_ctx = torch.amp.autocast(device_type="cuda") if device=="cuda" else nullcontext()

for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")
    set_seed(seed); per_example_hyps = []
    with torch.no_grad():
        for i in tqdm(range(0, len(examples), BATCH_SIZE), desc=f"gen(seed={seed})"):
            idxs = list(range(i, min(i+BATCH_SIZE, len(examples))))
            ids  = torch.stack([pretokenized[j]["input_ids"] for j in idxs]).to(device)
            attn = torch.stack([pretokenized[j]["attention_mask"] for j in idxs]).to(device)
            with autocast_ctx:
                out = gen_model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=MAX_NEW,
                                         do_sample=True, temperature=TEMP, top_k=TOP_K, top_p=TOP_P,
                                         num_return_sequences=GEN_K)
            dec = tok.batch_decode(out, skip_special_tokens=True)
            per_example_hyps.extend([dec[k:k+GEN_K] for k in range(0, len(dec), GEN_K)])

    assert len(per_example_hyps) == len(examples)
    comet_data = [{"src": examples[i]["document"], "mt": h, "ref": examples[i]["reference"]}
                  for i, hyps in enumerate(per_example_hyps) for h in hyps]
    print(f"Scoring {len(comet_data)} hypotheses with COMET...")
    raw_pred   = comet_predict_safe(comet_model, comet_data, batch_size=8, use_gpu=(device=="cuda"))
    seg_scores = extract_seg_scores(raw_pred)
    if len(seg_scores) != len(comet_data):
        raise ValueError(f"Score mismatch: {len(seg_scores)} vs {len(comet_data)}")

    ptr = 0; first_scores = []
    for ex_idx, hyps in enumerate(per_example_hyps):
        for sid, h in enumerate(hyps):
            sc = seg_scores[ptr]
            stability_records.append({"seed": seed, "example_id": ex_idx, "sample_id": sid,
                                      "hypothesis": h, "reference": examples[ex_idx]["reference"], "comet": sc})
            if sid == 0: first_scores.append(sc)
            ptr += 1

    mean_direct = float(np.mean(first_scores))
    _, bl, bh   = bootstrap_ci(first_scores)
    print(f"Seed {seed}: COMET={mean_direct:.4f} | CI [{bl:.4f},{bh:.4f}]")
    corpus_stats.append({"seed": seed, "comet_mean": mean_direct, "comet_boot_low": bl, "comet_boot_high": bh})

pd.DataFrame(stability_records).to_csv(f"{OUT_DIR}/comet_stability_sample_level.csv", index=False)
pd.DataFrame(corpus_stats).to_csv(f"{OUT_DIR}/comet_corpus_per_seed.csv", index=False)

df_stab = pd.DataFrame(stability_records)
g       = df_stab.groupby(["seed","example_id"])["comet"].mean().reset_index()
pivot   = g.pivot(index="example_id", columns="seed", values="comet")
pe      = pd.DataFrame({"example_id": pivot.index, "mean_across_seeds": pivot.mean(axis=1),
                        "std_across_seeds": pivot.std(axis=1, ddof=1)})
pe["cv_across_seeds"] = pe["std_across_seeds"] / pe["mean_across_seeds"].replace(0, 1e-8)
pe.to_csv(f"{OUT_DIR}/comet_per_example_across_seeds.csv", index=False)

try:
    lev_stat, lev_p = stats.levene(*[pivot[s].dropna().values for s in pivot.columns])
except: lev_stat, lev_p = None, None

summary = {"timestamp": time.asctime(), "model": GEN_MODEL, "dataset": f"{CNN_NAME} {CNN_VER}",
           "split": CNN_SPLIT, "n_examples": len(pe), "seeds": SEEDS, "gen_k": GEN_K,
           "avg_std": float(pe["std_across_seeds"].mean()), "avg_cv": float(pe["cv_across_seeds"].mean()),
           "levene_stat": float(lev_stat) if lev_stat is not None else None,
           "levene_p":    float(lev_p)    if lev_p    is not None else None}
with open(f"{OUT_DIR}/comet_stability_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"Per-example stats | mean CV={pe['cv_across_seeds'].mean():.4f} | mean σ={pe['std_across_seeds'].mean():.4f}")
print("\n=== STABILITY COMPLETE ===")

# ---------- CONSISTENCY: SummEval ----------
print("\n=== CONSISTENCY: COMET vs Human (SummEval) ===")
se      = load_dataset(SUMMEVAL_NAME, split="test")
records = []
for ex in se:
    ref = ex["human_summaries"][0]
    for j, cand in enumerate(ex["machine_summaries"]):
        records.append({"source": ex["text"], "reference": ref, "candidate": cand,
                        "coherence": ex["coherence"][j], "consistency": ex["consistency"][j],
                        "fluency": ex["fluency"][j], "relevance": ex["relevance"][j]})
se_df = pd.DataFrame(records)
print(f"Parsed {len(se_df)} SummEval rows.")

raw_pred  = comet_predict_safe(comet_model,
                               [{"src": r.source, "mt": r.candidate, "ref": r.reference} for r in se_df.itertuples()],
                               batch_size=8, use_gpu=(device=="cuda"))
se_df["comet"] = extract_seg_scores(raw_pred)
se_df["composite_human"] = se_df[["coherence","consistency","fluency","relevance"]].mean(axis=1)

corr = {}
for h in ["coherence","consistency","fluency","relevance","composite_human"]:
    sub = se_df[[h,"comet"]].dropna()
    corr[h] = {"pearson": float(sub.corr("pearson").iloc[0,1]),
               "spearman": float(sub.corr("spearman").iloc[0,1]), "n": len(sub)}

se_df.to_csv(f"{OUT_DIR}/summ_eval_with_comet.csv", index=False)
with open(f"{OUT_DIR}/comet_vs_human_correlations.json", "w") as f:
    json.dump(corr, f, indent=2)

print("\nCOMET vs Human Correlations:")
for h, v in corr.items():
    print(f"  {h:<22} Pearson={v['pearson']:.4f}  Spearman={v['spearman']:.4f}")
print(f"\nComposite Pearson = {corr['composite_human']['pearson']:.4f}")
print("\n=== ALL COMET EXPERIMENTS COMPLETE ===")
print("Outputs saved to:", OUT_DIR)
