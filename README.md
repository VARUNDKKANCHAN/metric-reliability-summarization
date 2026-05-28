# Metric Reliability in Abstractive Summarization

**A Joint Analysis of Human Alignment and Stochastic Stability**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IEEE](https://img.shields.io/badge/Published-IEEE-red.svg)]()
[![CI](https://github.com/YOUR_USERNAME/metric-reliability-summarization/actions/workflows/smoke_test.yml/badge.svg)](https://github.com/YOUR_USERNAME/metric-reliability-summarization/actions)

> **Varun D Kanchan · Abhishek Shetty**  
> School of Computer Engineering, Manipal Institute of Technology  
> `kanchanvarun45@gmail.com` · `aabhishekshetty7@gmail.com`

---

## Overview

This repository contains the complete experimental code for the paper:

> *"Metric Reliability in Abstractive Summarization: A Joint Analysis of Human Alignment and Stochastic Stability"*, IEEE 2025.

We systematically evaluate **8 automatic summarization metrics** across two complementary reliability dimensions:

| Dimension | Dataset | Scale |
|---|---|---|
| **Human Alignment** (consistency) | SummEval | 1,600 system–document pairs |
| **Sampling Stability** | CNN/DailyMail | 500 docs × 10 seeds × 3 samples = **15,000 summaries** |

---

## Key Results

| Metric | Pearson r ↑ | Mean CV ↓ | Category |
|---|---|---|---|
| **BERTScore** | **0.328** ✅ | **0.0080** ✅ | Embedding |
| COMET | 0.322 ✅ | 0.4687 ❌ | Learned |
| METEOR | 0.259 | **0.0035** ✅ | Hybrid |
| chrF++ | 0.255 | 0.1126 | Hybrid |
| ROUGE-L | 0.236 | 0.1783 ❌ | Lexical |
| BLEU | 0.156 ❌ | 0.3521 ❌ | Lexical |
| BLEURT | 0.136 ❌ | 0.5957 ❌ | Learned |
| MoverScore | 0.058 ❌ | 0.0229 ✅ | Embedding |

**Core finding:** Alignment and stability are **empirically dissociable** — no single metric is universally reliable. BERTScore is the only metric that satisfies both criteria simultaneously.

---

## Repository Structure

```
metric-reliability-summarization/
├── README.md
├── requirements.txt           # pip dependencies (Python 3.10)
├── requirements-dev.txt       # dev/testing extras
├── environment.yml            # Conda environment
├── setup.py                   # pip-installable package
├── reproduce.sh               # one-command full reproduction
├── install_bleurt.sh          # BLEURT source install helper
├── CITATION.cff               # academic citation metadata
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE                    # MIT
├── .gitignore
├── .github/
│   └── workflows/
│       └── smoke_test.yml     # GitHub Actions CI
├── scripts/
│   ├── 01_bleu_pipeline.py    # BLEU stability + consistency
│   ├── 02_rouge_pipeline.py   # ROUGE-1/2/L stability + consistency + plots
│   ├── 03_bertscore_pipeline.py
│   ├── 04_chrf_pipeline.py    # chrF++ stability + consistency
│   ├── 05_meteor_pipeline.py  # METEOR stability + consistency + plots
│   ├── 06_moverscore_pipeline.py  # MoverScore (Hungarian approx)
│   ├── 07_comet_pipeline.py   # COMET stability + consistency
│   ├── 08_bleurt_pipeline.py  # BLEURT (⚠ special install required)
│   └── 09_unified_analysis.py # Tables + Figures 1–4 + Excel workbook
└── outputs/                   # auto-created by pipelines (gitignored)
```

---

## Setup

### Requirements

- **Python 3.10** (strictly required — enforced by `bert-score==0.3.13`)
- GPU strongly recommended for BERTScore, COMET, BLEURT; CPU supported (slow)

### Option A — pip

```bash
git clone https://github.com/YOUR_USERNAME/metric-reliability-summarization.git
cd metric-reliability-summarization

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate         # Linux/Mac
# venv\Scripts\activate          # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

### Option B — Conda

```bash
conda env create -f environment.yml
conda activate metric-reliability
```

### NLTK Data (run once)

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### BLEURT (special install)

BLEURT requires installation from source. Run the helper script:

```bash
bash install_bleurt.sh
export BLEURT_CKPT=$(pwd)/checkpoints/bleurt-large-512
```

Or manually:
```bash
pip install git+https://github.com/google-research/bleurt.git
# Download bleurt-large-512 checkpoint from:
# https://github.com/google-research/bleurt/blob/master/checkpoints.md
export BLEURT_CKPT=/path/to/bleurt-large-512
```

---

## Reproduction

### Full pipeline

```bash
bash reproduce.sh
```

### Fast mode (fewer seeds, for testing)

```bash
bash reproduce.sh --fast
```

### Smoke test (10 examples only)

```bash
bash reproduce.sh --smoke
```

### Individual metric

```bash
python scripts/01_bleu_pipeline.py
python scripts/02_rouge_pipeline.py
python scripts/03_bertscore_pipeline.py
python scripts/04_chrf_pipeline.py
python scripts/05_meteor_pipeline.py
python scripts/06_moverscore_pipeline.py
python scripts/07_comet_pipeline.py
python scripts/08_bleurt_pipeline.py   # requires BLEURT install
python scripts/09_unified_analysis.py  # run LAST — reads all other outputs
```

### Config flags (top of each script)

| Flag | Default | Effect |
|---|---|---|
| `FAST_MODE = True` | `False` | Fewer seeds/samples, quicker iteration |
| `SMOKE_TEST = True` | `False` | 10 examples only, for CI testing |
| `CNN_SPLIT` | `"test[:500]"` | Change to `"test"` for full dataset |

---

## Output Files

Each metric pipeline saves to `outputs/<metric>_outputs/`:

| File | Description |
|---|---|
| `*_stability_sample_level.csv` | Per-hypothesis scores for all seeds |
| `*_per_example_across_seeds.csv` | Per-document mean/std/CV across seeds |
| `*_corpus_per_seed.csv` | Corpus-level metric + bootstrap CI per seed |
| `*_stability_summary.json` | Levene test, avg CV/std summary |
| `summ_eval_with_*.csv` | SummEval + metric scores |
| `*_vs_human_correlations.json` | Pearson/Spearman vs human dimensions |

The unified pipeline (`09`) saves to `outputs/final_results/`:

| File | Description |
|---|---|
| `global_stability_summary.csv` | Table II from paper |
| `global_consistency_summary.csv` | Table I from paper |
| `pairwise_stability_tests.csv` | 28 Mann–Whitney U tests |
| `figure1_consistency.png` | Figure 1 |
| `figure2_stability.png` | Figure 2 |
| `figure3_joint.png` | Figure 3 |
| `figure4_joint_6metrics.png` | Figure 4 |
| `metric_global_summary.xlsx` | All results in one Excel workbook |

---

## Implementation Details

| Metric | Library | Configuration |
|---|---|---|
| BLEU | SacreBLEU ≥ 2.3.1 | sentence-level + bootstrap corpus CI |
| ROUGE-1/2/L | rouge-score 0.1.3 | Porter stemming |
| METEOR | NLTK | stemming + synonym matching, regex tokenizer fallback |
| chrF++ | SacreBLEU | char_order=6, word_order=2, β=2 |
| BERTScore | bert-score 0.3.13 | roberta-large, idf=True, rescale_with_baseline=True |
| MoverScore | sentence-transformers | all-MiniLM-L12-v1, Hungarian assignment proxy |
| COMET | unbabel-comet | wmt20-comet-da checkpoint |
| BLEURT | google-research/bleurt | bleurt-large-512 checkpoint |

**Generator:** `t5-small` (60M params), nucleus sampling  
(`temperature=0.8, top_k=50, top_p=0.95`)  
**Seeds:** `{100, 101, ..., 109}` (10 seeds)  
**Samples per doc per seed:** 3  
**Total generated summaries:** 500 × 10 × 3 = **15,000**

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{kanchan2025metric,
  title     = {Metric Reliability in Abstractive Summarization:
               A Joint Analysis of Human Alignment and Stochastic Stability},
  author    = {Kanchan, Varun D and Shetty, Abhishek},
  booktitle = {IEEE},
  year      = {2025}
}
```

Or use GitHub's **Cite this repository** button (powered by `CITATION.cff`).

---

## License

MIT License — see [LICENSE](LICENSE) for full text.
