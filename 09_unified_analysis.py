# UNIFIED METRIC EVALUATION PIPELINE
# Stability + Consistency aggregation, figures, tables, Excel output
# Run AFTER all 8 metric pipelines (01-08)
# Run: python 09_unified_analysis.py
# Install: pip install pandas numpy matplotlib scipy openpyxl

import os, json, ast, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from scipy import stats as scipy_stats
from scipy.stats import levene as levene_test

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────
ROOT = "outputs"
OUT  = os.path.join(ROOT, "final_results")
os.makedirs(OUT, exist_ok=True)

# Auto-detect MoverScore folder
_mover_candidates = ["moverscore_outputs", "moverscore_results", "moverscore_real"]
MOVER_DIR = next((os.path.join(ROOT, d) for d in _mover_candidates
                  if os.path.isdir(os.path.join(ROOT, d))),
                 os.path.join(ROOT, "moverscore_outputs"))
print(f"MoverScore folder: {MOVER_DIR}")

# ── Metric file configuration ──────────────────────────────
metrics = {
    "BLEU": {
        "per_example":  f"{ROOT}/bleu_final_outputs/bleu_per_example_across_seeds.csv",
        "corpus":       f"{ROOT}/bleu_final_outputs/bleu_corpus_per_seed.csv",
        "sample_level": f"{ROOT}/bleu_final_outputs/bleu_stability_sample_level.csv",
        "sample_col":   "bleu", "cv_col": "cv_across_seeds", "std_col": "std_across_seeds",
        "summ_eval_csv": f"{ROOT}/bleu_final_outputs/summ_eval_with_bleu.csv", "metric_col": "bleu",
    },
    "ROUGE-L": {
        "per_example":  f"{ROOT}/rouge_final_outputs/rouge_per_example_across_seeds.csv",
        "corpus":       f"{ROOT}/rouge_final_outputs/rouge_corpus_per_seed.csv",
        "sample_level": f"{ROOT}/rouge_final_outputs/rouge_stability_sample_level.csv",
        "sample_col":   "rougeL", "cv_col": "cv_rl_across_seeds", "std_col": "std_rl_across_seeds",
        "summ_eval_csv": f"{ROOT}/rouge_final_outputs/summ_eval_with_rouge.csv", "metric_col": "rougeL",
    },
    "METEOR": {
        "per_example":  f"{ROOT}/meteor_outputs/meteor_per_example_across_seeds.csv",
        "corpus":       f"{ROOT}/meteor_outputs/meteor_corpus_per_seed.csv",
        "sample_level": f"{ROOT}/meteor_outputs/meteor_stability_sample_level.csv",
        "sample_col":   "meteor", "cv_col": "cv_across_seeds", "std_col": "std_across_seeds",
        "summ_eval_csv": f"{ROOT}/meteor_outputs/summ_eval_with_meteor.csv", "metric_col": "meteor",
    },
    "CHRF++": {
        "per_example":  f"{ROOT}/chrfpp_outputs/chrfpp_per_example_across_seeds.csv",
        "corpus":       f"{ROOT}/chrfpp_outputs/chrfpp_corpus_per_seed.csv",
        "sample_level": f"{ROOT}/chrfpp_outputs/chrfpp_stability_sample_level.csv",
        "sample_col":   "chrfpp", "cv_col": "cv_across_seeds", "std_col": "std_across_seeds",
        "summ_eval_csv": f"{ROOT}/chrfpp_outputs/summ_eval_with_chrfpp.csv", "metric_col": "chrfpp",
    },
    "BERTScore": {
        "per_example":  f"{ROOT}/bertscore_outputs/bertscore_per_example_across_seeds.csv",
        "corpus":       f"{ROOT}/bertscore_outputs/bertscore_corpus_per_seed.csv",
        "sample_level": f"{ROOT}/bertscore_outputs/bertscore_stability_sample_level.csv",
        "sample_col":   "bertscore_f1", "cv_col": "cv_across_seeds", "std_col": "std_across_seeds",
        "summ_eval_csv": f"{ROOT}/bertscore_outputs/summ_eval_with_bertscore.csv", "metric_col": "bertscore_f1",
    },
    "MoverScore": {
        "per_example":       f"{MOVER_DIR}/moverscore_per_example_across_seeds.csv",
        "corpus":            f"{MOVER_DIR}/moverscore_stability_summary.csv",
        "sample_level":      f"{MOVER_DIR}/moverscore_stability_samples.csv",
        "stability_summary": f"{MOVER_DIR}/moverscore_stability_summary.json",
        "sample_col":        "moverscore", "cv_col": "cv_across_seeds", "std_col": "std_across_seeds",
        "summ_eval_csv":     f"{MOVER_DIR}/summ_eval_moverscore.csv", "metric_col": "moverscore",
    },
    "COMET": {
        "per_example":  f"{ROOT}/comet_outputs/comet_per_example_across_seeds.csv",
        "corpus":       f"{ROOT}/comet_outputs/comet_corpus_per_seed.csv",
        "sample_level": f"{ROOT}/comet_outputs/comet_stability_sample_level.csv",
        "sample_col":   "comet", "cv_col": "cv_across_seeds", "std_col": "std_across_seeds",
        "summ_eval_csv": f"{ROOT}/comet_outputs/summ_eval_with_comet.csv", "metric_col": "comet",
    },
    "BLEURT": {
        "per_example":  f"{ROOT}/bleurt_outputs/bleurt_per_example_across_seeds.csv",
        "corpus":       f"{ROOT}/bleurt_outputs/bleurt_corpus_per_seed.csv",
        "sample_level": f"{ROOT}/bleurt_outputs/bleurt_stability_sample_level.csv",
        "sample_col":   "bleurt", "cv_col": "cv_across_seeds", "std_col": "std_across_seeds",
        "summ_eval_csv": f"{ROOT}/bleurt_outputs/summ_eval_with_bleurt_large.csv", "metric_col": "bleurt",
    },
}

HUMAN_COLS = ["coherence", "consistency", "fluency", "relevance"]
N_BOOT = 1000
ALPHA  = 0.05
RNG    = np.random.default_rng(42)


# ── HELPER FUNCTIONS ───────────────────────────────────────

def safe_float(val):
    if isinstance(val, (int, float, np.number)): return float(val)
    if isinstance(val, list):
        try: return float(np.mean([float(x) for x in val]))
        except: return np.nan
    if isinstance(val, str):
        t = val.strip()
        if t.startswith("["):
            try: return float(np.mean([float(x) for x in ast.literal_eval(t)]))
            except: return np.nan
        try: return float(t)
        except: return np.nan
    return np.nan


def bootstrap_ci_correlation(x, y, n_iter=N_BOOT, alpha=ALPHA):
    xy = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(xy) < 10: return np.nan, np.nan, np.nan
    x_arr, y_arr = xy["x"].values, xy["y"].values
    r_obs = float(np.corrcoef(x_arr, y_arr)[0, 1])
    boot_rs = []
    n = len(x_arr)
    for _ in range(n_iter):
        idx = RNG.integers(0, n, n)
        boot_rs.append(np.corrcoef(x_arr[idx], y_arr[idx])[0, 1])
    return r_obs, float(np.percentile(boot_rs, 100*alpha/2)), float(np.percentile(boot_rs, 100*(1-alpha/2)))


def compute_levene_from_sample(sample_path, sample_col):
    if not sample_path or not os.path.exists(sample_path): return None, None
    try:
        df    = pd.read_csv(sample_path)
        agg   = df.groupby(["seed","example_id"])[sample_col].mean().reset_index()
        pivot = agg.pivot(index="example_id", columns="seed", values=sample_col)
        groups = [pivot[s].dropna().values for s in pivot.columns]
        stat, p = levene_test(*groups)
        return float(stat), float(p)
    except Exception as e:
        print(f"    [WARN] Levene failed: {e}"); return None, None


def compute_mad_from_sample(sample_path, sample_col):
    if not sample_path or not os.path.exists(sample_path): return None
    try:
        df    = pd.read_csv(sample_path)
        agg   = df.groupby(["seed","example_id"])[sample_col].mean().reset_index()
        pivot = agg.pivot(index="example_id", columns="seed", values=sample_col)
        mad_per_doc = pivot.apply(lambda r: np.median(np.abs(r - np.median(r))), axis=1)
        return float(mad_per_doc.mean())
    except Exception as e:
        print(f"    [WARN] MAD failed: {e}"); return None


def compute_stability(metric_name, cfg):
    path    = cfg["per_example"]
    cv_col  = cfg["cv_col"]
    std_col = cfg["std_col"]
    if not os.path.exists(path):
        print(f"  [WARN] Missing per-example CSV for {metric_name}"); return None
    df = pd.read_csv(path)
    if cv_col not in df.columns or std_col not in df.columns:
        print(f"  [WARN] Missing columns for {metric_name}"); return None

    raw_mean_cv = float(df[cv_col].mean())
    mean_std    = float(df[std_col].mean())

    if raw_mean_cv < 0:
        sample_path = cfg.get("sample_level")
        sample_col  = cfg.get("sample_col")
        if sample_path and os.path.exists(sample_path):
            raw_df    = pd.read_csv(sample_path)
            agg       = raw_df.groupby(["seed","example_id"])[sample_col].mean().reset_index()
            pivot     = agg.pivot(index="example_id", columns="seed", values=sample_col)
            doc_means = pivot.mean(axis=1)
            doc_stds  = pivot.std(axis=1, ddof=1)
            cv_abs    = (doc_stds / doc_means.abs()).values
            mean_cv   = float(np.nanmean(cv_abs))
            median_cv = float(np.nanmedian(cv_abs))
        else:
            mean_cv = abs(raw_mean_cv); median_cv = abs(float(df[cv_col].median()))
        used_abs = True
    else:
        mean_cv = raw_mean_cv; median_cv = float(df[cv_col].median()); used_abs = False

    lev_stat, lev_p = compute_levene_from_sample(cfg.get("sample_level"), cfg.get("sample_col"))
    if lev_p is None and cfg.get("stability_summary") and os.path.exists(cfg["stability_summary"]):
        try:
            js = json.load(open(cfg["stability_summary"]))
            lev_stat = js.get("levene_stat"); lev_p = js.get("levene_p")
        except: pass

    mad = compute_mad_from_sample(cfg.get("sample_level"), cfg.get("sample_col"))
    if mad is None: mad = 0.6745 * mean_std

    cv_range = "CV < 0.05" if mean_cv < 0.05 else ("0.05 ≤ CV < 0.15" if mean_cv < 0.15 else "CV ≥ 0.15")
    return {"mean_cv": mean_cv, "median_cv": median_cv, "mean_std": mean_std, "mean_mad": mad,
            "n": len(df), "levene_stat": lev_stat, "levene_p": lev_p, "cv_abs": used_abs, "cv_range": cv_range}


def compute_consistency(metric_name, cfg):
    csv_path   = cfg["summ_eval_csv"]
    metric_col = cfg["metric_col"]
    if not os.path.exists(csv_path):
        print(f"  [WARN] Missing SummEval CSV for {metric_name}"); return None
    df = pd.read_csv(csv_path)
    df[metric_col] = df[metric_col].apply(safe_float)
    remap = cfg.get("consistency_col_remap", {})
    if remap: df = df.rename(columns=remap)
    available = [h for h in HUMAN_COLS if h in df.columns]
    if not available:
        print(f"  [WARN] No human score columns for {metric_name}"); return None
    df["composite"] = df[available].map(safe_float).mean(axis=1)
    sub = df[["composite", metric_col]].dropna()
    if len(sub) < 10: return None
    pearson, ci_low, ci_high = bootstrap_ci_correlation(sub["composite"], sub[metric_col])
    spearman = float(scipy_stats.spearmanr(sub["composite"], sub[metric_col]).statistic)
    return {"pearson": pearson, "ci_low": ci_low, "ci_high": ci_high, "spearman": spearman, "n": len(sub)}


# ── STEP 1: COMPUTE ALL RESULTS ────────────────────────────
print("=" * 65)
print("COMPUTING STABILITY AND CONSISTENCY FOR ALL 8 METRICS")
print("=" * 65)

stability_rows, consistency_rows = [], []

for metric, cfg in metrics.items():
    print(f"\n── {metric} ──")
    stab = compute_stability(metric, cfg)
    if stab:
        stability_rows.append({"metric": metric, **stab})
        abs_note = " [CV=σ/|mean|]" if stab["cv_abs"] else ""
        print(f"  Stability  | mean CV={stab['mean_cv']:.4f}{abs_note} | MAD={stab['mean_mad']:.4f} | Levene p={stab['levene_p']}")
    cons = compute_consistency(metric, cfg)
    if cons:
        consistency_rows.append({"metric": metric, **cons})
        print(f"  Consistency| Pearson={cons['pearson']:.4f} [{cons['ci_low']:.4f},{cons['ci_high']:.4f}] | Spearman={cons['spearman']:.4f}")


# ── STEP 2: SUMMARY DATAFRAMES ─────────────────────────────
stability_df   = pd.DataFrame(stability_rows).sort_values("mean_cv")
consistency_df = pd.DataFrame(consistency_rows).sort_values("pearson", ascending=False)
stability_df.to_csv(os.path.join(OUT, "global_stability_summary.csv"), index=False)
consistency_df.to_csv(os.path.join(OUT, "global_consistency_summary.csv"), index=False)

print("\n" + "=" * 65)
print("TABLE II — STABILITY")
print("=" * 65)
print(f"{'Metric':<12} {'Mean CV':>9} {'Median CV':>10} {'Mean σ':>8} {'MAD':>9} {'Levene p':>9} {'CV range':<20} {'|mean|?'}")
print("-" * 90)
for _, r in stability_df.iterrows():
    lp   = f"{r['levene_p']:.3f}" if r['levene_p'] is not None else "—"
    flag = "YES‡" if r["cv_abs"] else ""
    print(f"{r['metric']:<12} {r['mean_cv']:>9.4f} {r['median_cv']:>10.4f} {r['mean_std']:>8.4f} "
          f"{r['mean_mad']:>9.4f} {lp:>9} {r['cv_range']:<20} {flag}")

print("\n" + "=" * 65)
print("TABLE I — CONSISTENCY")
print("=" * 65)
print(f"{'Metric':<12} {'Pearson r':>10} {'95% CI':>22} {'Spearman ρ':>12}")
print("-" * 60)
for _, r in consistency_df.iterrows():
    ci_str = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
    print(f"{r['metric']:<12} {r['pearson']:>10.3f} {ci_str:>22} {r['spearman']:>12.3f}")


# ── STEP 3: PAIRWISE MANN–WHITNEY U ────────────────────────
print("\n" + "=" * 65)
print("PAIRWISE MANN–WHITNEY U TESTS (28 metric pairs)")
print("=" * 65)

cv_vectors = {}
for metric, cfg in metrics.items():
    sample_path = cfg.get("sample_level"); sample_col = cfg.get("sample_col")
    row = stability_df[stability_df["metric"] == metric]
    if len(row) == 0: continue
    use_abs = bool(row["cv_abs"].values[0])
    if sample_path and os.path.exists(sample_path):
        raw_df    = pd.read_csv(sample_path)
        agg       = raw_df.groupby(["seed","example_id"])[sample_col].mean().reset_index()
        pivot     = agg.pivot(index="example_id", columns="seed", values=sample_col)
        doc_means = pivot.mean(axis=1); doc_stds = pivot.std(axis=1, ddof=1)
        cv_vec    = (doc_stds / (doc_means.abs() if use_abs else doc_means)).values
        cv_vectors[metric] = cv_vec[~np.isnan(cv_vec)]

metric_list = list(cv_vectors.keys()); pairs = []
for i in range(len(metric_list)):
    for j in range(i+1, len(metric_list)):
        m1, m2 = metric_list[i], metric_list[j]
        stat, p = scipy_stats.mannwhitneyu(cv_vectors[m1], cv_vectors[m2], alternative="two-sided")
        n_less  = int(np.sum(cv_vectors[m1] < cv_vectors[m2]))
        pairs.append({"metric_1": m1, "metric_2": m2, "W": int(stat), "p_raw": p,
                      "docs_m1_less": f"{n_less}/{len(cv_vectors[m1])}"})

pairs_df = pd.DataFrame(pairs).sort_values("p_raw").reset_index(drop=True)
m_tests  = len(pairs_df)
pairs_df["p_holm"] = [min(1.0, (m_tests - i) * p) for i, p in enumerate(pairs_df["p_raw"])]
pairs_df["significant"] = pairs_df["p_holm"] < 0.05
pairs_df.to_csv(os.path.join(OUT, "pairwise_stability_tests.csv"), index=False)
print(pairs_df[["metric_1","metric_2","W","p_holm","significant","docs_m1_less"]].to_string(index=False))


# ── STEP 4: FIGURE 1 — Consistency ─────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
c_sorted = consistency_df.sort_values("pearson", ascending=False)
names = c_sorted["metric"].tolist(); vals = c_sorted["pearson"].tolist()
ci_lo = [v - lo for v, lo in zip(vals, c_sorted["ci_low"].tolist())]
ci_hi = [hi - v for v, hi in zip(vals, c_sorted["ci_high"].tolist())]
colors = ['#2166ac' if v > 0.18 else '#92c5de' if v > 0.10 else '#d1e5f0' for v in vals]
bars = ax.bar(names, vals, color=colors, edgecolor='white', linewidth=0.8, zorder=3)
ax.errorbar(names, vals, yerr=[ci_lo, ci_hi], fmt='none', color='#333333', capsize=4, linewidth=1.2, zorder=4)
ax.axhline(0.15, color='gray', linestyle='--', linewidth=0.9, label='r = 0.15 reference')
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ci_hi) + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8.5)
ax.set_ylabel('Pearson r with Composite Human Score', fontsize=10)
ax.set_title('Figure 1. Metric–Human Consistency (SummEval, n = 1,600)\nError bars: bootstrap 95% CI (1,000 iterations)',
             fontsize=11, fontweight='bold')
ax.set_ylim(0, max(vals) + 0.10); ax.tick_params(axis='x', rotation=20)
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3, zorder=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "figure1_consistency.png"), dpi=300, bbox_inches='tight'); plt.close()
print("\nSaved: figure1_consistency.png")


# ── STEP 5: FIGURE 2 — Stability ───────────────────────────
stab_sorted = stability_df.sort_values("mean_cv")
fig, ax = plt.subplots(figsize=(11, 5))
s_names = stab_sorted["metric"].tolist(); s_vals = stab_sorted["mean_cv"].tolist(); s_abs = stab_sorted["cv_abs"].tolist()
s_colors = ['#1a9641' if v < 0.05 else '#fdae61' if v < 0.15 else '#d7191c' for v in s_vals]
hatches  = ['///' if a else '' for a in s_abs]
for i, (val, col, hatch) in enumerate(zip(s_vals, s_colors, hatches)):
    ax.bar(i, val, color=col, edgecolor='white', linewidth=0.8, hatch=hatch, zorder=3)
for i, (val, ab) in enumerate(zip(s_vals, s_abs)):
    ax.text(i, val + 0.005, f'{val:.4f}{"‡" if ab else ""}', ha='center', va='bottom', fontsize=9)
ax.set_xticks(range(len(s_names))); ax.set_xticklabels(s_names, rotation=15)
ax.set_ylabel('Mean CV (σ / |mean|)  — lower = more stable', fontsize=10)
ax.set_title('Figure 2. Metric Stability Under Stochastic Decoding (CNN/DailyMail, n=500, 10 seeds)\n'
             '‡ CV = σ/|mean| for metrics with negative mean (COMET, BLEURT)', fontsize=10, fontweight='bold')
ax.set_ylim(0, max(s_vals) + 0.06); ax.grid(axis='y', alpha=0.3, zorder=0)
legend_elements = [Patch(facecolor='#1a9641', label='Stable (CV < 0.05)'),
                   Patch(facecolor='#fdae61', label='Moderate (0.05 ≤ CV < 0.15)'),
                   Patch(facecolor='#d7191c', label='Unstable (CV ≥ 0.15)'),
                   Patch(facecolor='white', edgecolor='gray', hatch='///', label='CV = σ/|mean|')]
ax.legend(handles=legend_elements, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "figure2_stability.png"), dpi=300, bbox_inches='tight'); plt.close()
print("Saved: figure2_stability.png")


# ── STEP 6: FIGURE 3 — Joint plot ──────────────────────────
merged = pd.merge(stability_df[["metric","mean_cv","cv_abs"]],
                  consistency_df[["metric","pearson","ci_low","ci_high"]], on="metric", how="inner")
fig, ax = plt.subplots(figsize=(9, 6.5))
point_colors = ['#d95f02' if a else '#2166ac' for a in merged["cv_abs"]]
ax.errorbar(merged["mean_cv"], merged["pearson"],
            yerr=[merged["pearson"]-merged["ci_low"], merged["ci_high"]-merged["pearson"]],
            fmt='none', color='#888888', capsize=4, linewidth=1.0, zorder=4)
ax.scatter(merged["mean_cv"], merged["pearson"], c=point_colors, s=90, zorder=5)
ax.axvspan(-0.005, 0.05, alpha=0.07, color='green', zorder=1)
ax.axhspan(0.18, merged["pearson"].max()+0.04, alpha=0.07, color='blue', zorder=1)
offsets = {'MoverScore': (-82, 8), 'COMET': (8, 8), 'METEOR': (-72, 8), 'ROUGE-L': (8, 8),
           'CHRF++': (8, -16), 'BERTScore': (-82, -16), 'BLEU': (8, 8), 'BLEURT': (8, -16)}
for _, row in merged.iterrows():
    ox, oy = offsets.get(row["metric"], (8, 8))
    ax.annotate(row["metric"] + ("‡" if row["cv_abs"] else ""),
                xy=(row["mean_cv"], row["pearson"]), xytext=(ox, oy), textcoords='offset points',
                fontsize=10, arrowprops=dict(arrowstyle='-', color='gray', lw=0.6, alpha=0.6))
legend_elements = [Patch(facecolor='#2166ac', label='CV = σ / mean'),
                   Patch(facecolor='#d95f02', label='CV = σ / |mean| (negative mean)‡')]
ax.legend(handles=legend_elements, fontsize=9)
ax.set_xlabel('Mean CV (lower = more stable)', fontsize=11)
ax.set_ylabel('Pearson r with Composite Human Score\n(error bars = bootstrap 95% CI)', fontsize=11)
ax.set_title('Figure 3. Stability–Consistency Trade-Off across All 8 Metrics\n'
             '‡ CV = σ/|mean| for COMET and BLEURT (negative mean scores)', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3, zorder=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "figure3_joint.png"), dpi=300, bbox_inches='tight'); plt.close()
print("Saved: figure3_joint.png")


# ── STEP 7: FIGURE 4 — Joint plot (6 positive-mean metrics) ──
merged6 = merged[~merged["cv_abs"]].copy()
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.errorbar(merged6["mean_cv"], merged6["pearson"],
            yerr=[merged6["pearson"]-merged6["ci_low"], merged6["ci_high"]-merged6["pearson"]],
            fmt='none', color='#888888', capsize=4, linewidth=1.0, zorder=4)
ax.scatter(merged6["mean_cv"], merged6["pearson"], c='#2166ac', s=90, zorder=5)
ax.axvspan(-0.005, 0.05, alpha=0.07, color='green', zorder=1)
ax.axhspan(0.18, merged6["pearson"].max()+0.04, alpha=0.07, color='blue', zorder=1)
for _, row in merged6.iterrows():
    ox, oy = offsets.get(row["metric"], (8, 8))
    ax.annotate(row["metric"], xy=(row["mean_cv"], row["pearson"]),
                xytext=(ox, oy), textcoords='offset points', fontsize=10,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.6, alpha=0.6))
ax.set_xlabel('Mean CV = σ / mean (lower = more stable)', fontsize=11)
ax.set_ylabel('Pearson r with Composite Human Score\n(error bars = bootstrap 95% CI)', fontsize=11)
ax.set_title('Figure 4. Stability–Consistency Trade-Off (6 positive-mean metrics)\nNote: COMET and BLEURT excluded (CV = σ/|mean|)',
             fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3, zorder=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "figure4_joint_6metrics.png"), dpi=300, bbox_inches='tight'); plt.close()
print("Saved: figure4_joint_6metrics.png")


# ── STEP 8: EXCEL WORKBOOK ─────────────────────────────────
excel_path = os.path.join(OUT, "metric_global_summary.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    stability_df.to_excel(writer,   sheet_name="Stability",     index=False)
    consistency_df.to_excel(writer, sheet_name="Consistency",   index=False)
    pairs_df.to_excel(writer,       sheet_name="Pairwise_Tests", index=False)
    for metric, cfg in metrics.items():
        cpath = cfg["corpus"]
        if os.path.exists(cpath):
            try: pd.read_csv(cpath).to_excel(writer, sheet_name=f"{metric}_corpus", index=False)
            except Exception as e: print(f"[WARN] corpus sheet {metric}: {e}")

print(f"\nSaved Excel workbook: {excel_path}")
print("\n" + "=" * 65)
print("PIPELINE COMPLETE — all 8 metrics included in all outputs")
print(f"Outputs saved to: {OUT}")
print("=" * 65)
