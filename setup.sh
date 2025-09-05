#!/usr/bin/env bash
set -euo pipefail

# Name of the conda env
ENV_NAME="mp"

echo "[*] Creating conda environment: $ENV_NAME (python=3.11)"
conda create -y -n "$ENV_NAME" python=3.11

echo "[*] Activating environment"
if ! command -v conda >/dev/null 2>&1; then
  echo "[!] 'conda' not found on PATH. Please install Miniforge/Conda and retry."
  exit 1
fi
# Initialize conda for this shell if needed
eval "$(conda shell.bash hook 2>/dev/null || conda shell.zsh hook 2>/dev/null || true)"
conda activate "$ENV_NAME"

echo "[*] Installing base dependencies with conda"
conda install -y -c conda-forge opencv numpy

echo "[*] Installing pip dependencies"
python -m pip install --upgrade pip
python -m pip install mediapipe==0.10.14 mlx-whisper

echo "[*] Installing brew dependencies (ffmpeg, wget)"
brew install ffmpeg wget

# Choose a FastVLM repo location (portable)
FASTVLM_REPO_DIR="${FASTVLM_REPO:-"$PWD/ml-fastvlm"}"
echo "[*] Cloning FastVLM repo (if not already present) into: $FASTVLM_REPO_DIR"
if [ ! -d "$FASTVLM_REPO_DIR" ]; then
  git clone https://github.com/apple/ml-fastvlm.git "$FASTVLM_REPO_DIR"
fi

echo "[*] Installing FastVLM in editable mode"
python -m pip install -e "$FASTVLM_REPO_DIR"

echo "[*] Downloading FastVLM models"
cd "$FASTVLM_REPO_DIR"
bash get_models.sh || true

echo "[*] Setup complete!"
echo ""
echo "To start using the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Then set env vars and run, e.g.:"
echo "  export FASTVLM_REPO=\"$FASTVLM_REPO_DIR\""
echo "  export FASTVLM_MODEL=\"$FASTVLM_REPO_DIR/checkpoints/llava-fastvithd_0.5b_stage3\""
echo "  python iphonerecording.py --cam-index 1 --face-mesh --show-fps"
