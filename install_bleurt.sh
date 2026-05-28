#!/usr/bin/env bash
# ============================================================
# install_bleurt.sh  —  Install BLEURT from source
# Usage: bash install_bleurt.sh
# ============================================================
# BLEURT cannot be installed from PyPI directly.
# This script clones the repo and installs it, then
# downloads the bleurt-large-512 checkpoint.
# ============================================================

set -euo pipefail

echo "============================================================"
echo " Installing BLEURT from source"
echo "============================================================"

# Step 1: Clone and install
echo "[1/3] Cloning google-research/bleurt..."
if [ -d "bleurt" ]; then
  echo "  bleurt/ already exists — skipping clone."
else
  git clone https://github.com/google-research/bleurt.git
fi

echo "[2/3] Installing bleurt package..."
cd bleurt
pip install .
cd ..

# Step 2: Download checkpoint
CKPT_NAME="bleurt-large-512"
CKPT_URL="https://storage.googleapis.com/bleurt-oss/${CKPT_NAME}.zip"
CKPT_DIR="checkpoints"

mkdir -p "$CKPT_DIR"

if [ -d "$CKPT_DIR/$CKPT_NAME" ]; then
  echo "[3/3] Checkpoint already exists at $CKPT_DIR/$CKPT_NAME — skipping download."
else
  echo "[3/3] Downloading $CKPT_NAME checkpoint (~1.2 GB)..."
  wget -q --show-progress -O "$CKPT_DIR/${CKPT_NAME}.zip" "$CKPT_URL"
  echo "  Extracting..."
  unzip -q "$CKPT_DIR/${CKPT_NAME}.zip" -d "$CKPT_DIR/"
  rm "$CKPT_DIR/${CKPT_NAME}.zip"
  echo "  Checkpoint saved to: $CKPT_DIR/$CKPT_NAME"
fi

# Step 3: Export env variable
echo ""
echo "============================================================"
echo " BLEURT install complete!"
echo " Add this to your shell or .env file:"
echo "   export BLEURT_CKPT=$(pwd)/$CKPT_DIR/$CKPT_NAME"
echo "============================================================"
echo ""
echo "Then run:"
echo "   python scripts/08_bleurt_pipeline.py"
