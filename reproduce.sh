#!/usr/bin/env bash
# ============================================================
# reproduce.sh — Full experiment reproduction
# Usage: bash reproduce.sh [--fast] [--smoke]
#   --fast   : FAST_MODE (fewer seeds, quicker iteration)
#   --smoke  : SMOKE_TEST (10 examples only, for testing)
# Example: bash reproduce.sh
# Example: bash reproduce.sh --smoke
# ============================================================

set -euo pipefail

# ── Parse args ──────────────────────────────────────────────
FAST=false; SMOKE=false
for arg in "$@"; do
  case $arg in
    --fast)  FAST=true ;;
    --smoke) SMOKE=true ;;
  esac
done

echo "============================================================"
echo " Metric Reliability in Abstractive Summarization"
echo " Authors: Varun D Kanchan, Abhishek Shetty"
echo " Manipal Institute of Technology, 2025"
echo "============================================================"
echo " FAST_MODE  : $FAST"
echo " SMOKE_TEST : $SMOKE"
echo "============================================================"

# ── Check Python version ────────────────────────────────────
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$PY_VER" != "3.10" ]]; then
  echo ""
  echo "⚠ WARNING: Python $PY_VER detected."
  echo "  Python 3.10 is REQUIRED for bert-score==0.3.13 and moverscore_v2."
  echo "  Some metrics may fail on other Python versions."
  echo ""
fi

# ── Check GPU ───────────────────────────────────────────────
python3 -c "import torch; gpu=torch.cuda.is_available(); print(f'GPU available: {gpu} | Device: {torch.cuda.get_device_name(0) if gpu else \"CPU\"}')"

# ── NLTK data ───────────────────────────────────────────────
echo ""
echo "Downloading NLTK data..."
python3 -c "
import nltk
for pkg in ['punkt', 'wordnet', 'omw-1.4']:
    nltk.download(pkg, quiet=True)
print('NLTK data ready.')
"

# ── Helper: run a pipeline with optional flags ───────────────
run_pipeline() {
  SCRIPT=$1
  echo ""
  echo "──────────────────────────────────────────────────────"
  echo "Running: $SCRIPT"
  echo "──────────────────────────────────────────────────────"

  if $SMOKE; then
    FAST_MODE=False SMOKE_TEST=True python3 "scripts/$SCRIPT"
  elif $FAST; then
    FAST_MODE=True  SMOKE_TEST=False python3 "scripts/$SCRIPT"
  else
    FAST_MODE=False SMOKE_TEST=False python3 "scripts/$SCRIPT"
  fi
}

# ── Run all pipelines in order ───────────────────────────────
run_pipeline "01_bleu_pipeline.py"
run_pipeline "02_rouge_pipeline.py"
run_pipeline "03_bertscore_pipeline.py"
run_pipeline "04_chrf_pipeline.py"
run_pipeline "05_meteor_pipeline.py"
run_pipeline "06_moverscore_pipeline.py"
run_pipeline "07_comet_pipeline.py"

# BLEURT needs special install — skip gracefully if not available
echo ""
echo "──────────────────────────────────────────────────────"
echo "Running: 08_bleurt_pipeline.py"
echo "  (requires bleurt installed from source — see README)"
echo "──────────────────────────────────────────────────────"
python3 -c "import bleurt" 2>/dev/null \
  && python3 scripts/08_bleurt_pipeline.py \
  || echo "  [SKIPPED] bleurt not installed. See README for install instructions."

# ── Unified analysis (must run last) ─────────────────────────
echo ""
echo "──────────────────────────────────────────────────────"
echo "Running: 09_unified_analysis.py  (figures + tables + Excel)"
echo "──────────────────────────────────────────────────────"
python3 scripts/09_unified_analysis.py

echo ""
echo "============================================================"
echo " ALL DONE"
echo " Outputs saved in:  outputs/"
echo " Figures & tables:  outputs/final_results/"
echo "============================================================"
