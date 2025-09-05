#!/usr/bin/env bash
set -euo pipefail

# Resolve script dir for portability; don't depend on CWD
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$SCRIPT_DIR/iphonerecording.py"

# Prefer env vars for FastVLM; fall back to autodetect only if repo is set
REPO="${FASTVLM_REPO:-}"
MODEL="${FASTVLM_MODEL:-}"

# If REPO is provided but MODEL is not, try to auto-pick a stage3 model
if [[ -z "${MODEL}" && -n "${REPO}" ]]; then
  MODEL="$(ls -d "$REPO"/checkpoints/*_stage3 2>/dev/null | head -n1 || true)"
fi

if [[ -z "${MODEL}" ]]; then
  echo "[run] FastVLM model not set. Captions will be disabled."
  echo "      Set env var FASTVLM_MODEL or FASTVLM_REPO to enable."
else
  echo "[run] Using model: ${MODEL}"
fi

# Build args conditionally to avoid injecting empty paths
args=(
  --cam-index 1
  --width 1920 --height 1080 --fps 30
  --face-mesh
  --show-fps
  --auto-transcribe
)

if [[ -n "${REPO}" ]]; then
  args+=( --fastvlm-repo "${REPO}" )
fi
if [[ -n "${MODEL}" ]]; then
  args+=( --fastvlm-model "${MODEL}" )
fi

conda run -n mp python "$PY" "${args[@]}"
