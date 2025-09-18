#!/usr/bin/env bash
set -euo pipefail

# Resolve script dir for portability; don't depend on CWD
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$SCRIPT_DIR/iphonerecording.py"

# Load .env if present for orchestrator/api configuration
ENV_FILE="$SCRIPT_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  echo "[run] Loaded environment from $ENV_FILE"
fi

# Prefer env vars for FastVLM; fall back to autodetect only if repo is set
REPO="${FASTVLM_REPO:-}"
MODEL="${FASTVLM_MODEL:-}"

# If env var missing, try well-known default clone
if [[ -z "${REPO}" ]]; then
  if [[ -d "$HOME/ml-fastvlm" ]]; then
    REPO="$HOME/ml-fastvlm"
    echo "[run] FASTVLM_REPO unset; defaulting to ${REPO}"
  fi
fi

# If REPO is provided but MODEL is not, try to auto-pick a stage3 model
if [[ -z "${MODEL}" && -n "${REPO}" ]]; then
  MODEL="$(ls -d "$REPO"/checkpoints/*_stage3 2>/dev/null | head -n1 || true)"
fi

if [[ -n "${REPO}" ]]; then
  echo "[run] FASTVLM_REPO=${REPO}"
  if [[ ! -f "${REPO}/predict.py" ]]; then
    echo "[run]   warn: predict.py missing at ${REPO}/predict.py"
  fi
else
  echo "[run] FASTVLM_REPO not set"
fi

if [[ -n "${MODEL}" ]]; then
  echo "[run] FASTVLM_MODEL=${MODEL}"
  if [[ ! -d "${MODEL}" ]]; then
    echo "[run]   warn: ${MODEL} is not a directory"
  else
    echo "[run] Using model: ${MODEL}"
  fi
else
  echo "[run] FastVLM model not set. Captions will be disabled."
  echo "      Set env var FASTVLM_MODEL or FASTVLM_REPO to enable."
fi

ORCH_LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$ORCH_LOG_DIR"

declare -a WORKER_PIDS=()

cleanup_workers() {
  for pid in "${WORKER_PIDS[@]}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
}

trap cleanup_workers EXIT

launch_worker() {
  local script_path="$1"
  local label="$2"
  local cmd=("${RUNNER[@]}" "$script_path")

  if [[ ! -x "$script_path" ]]; then
    echo "[run] $label script missing or not executable at $script_path"
    return
  fi

  if command -v pgrep >/dev/null 2>&1 && pgrep -f "$script_path" >/dev/null 2>&1; then
    echo "[run] $label already running; skipping auto-launch."
    return
  fi

  local log_file="$ORCH_LOG_DIR/${label}.out"
  touch "$log_file"
  "${cmd[@]}" >>"$log_file" 2>&1 &
  local pid=$!
  echo "[run] $label running in background (pid ${pid}). Logs: $log_file"
  WORKER_PIDS+=($pid)

  if [[ "$label" == "orchestrator" ]] && command -v osascript >/dev/null 2>&1; then
    local log_escaped
    printf -v log_escaped '%q' "$log_file"
    if osascript <<OSA
tell application "Terminal"
  activate
  do script "tail -f ${log_escaped}"
end tell
OSA
    then
      echo "[run] Opened Terminal window to follow orchestrator logs."
    else
      echo "[run] osascript tail launch failed; monitor logs manually." >&2
    fi
  fi
}

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

# Choose Python runner: conda env 'mp' -> local venv -> system python3
RUNNER=("python")

# 1) Try conda run if conda and env exists
if command -v conda >/dev/null 2>&1; then
  if conda env list 2>/dev/null | grep -qE "^[^#]*\bmp\b"; then
    RUNNER=("conda" "run" "-n" "mp" "python")
    echo "[run] Using conda env: mp"
  fi
fi

# 2) If no conda env selected, prefer project venv
if [[ "${RUNNER[*]}" == "python" ]]; then
  if [[ -x "$SCRIPT_DIR/venv/bin/python" ]]; then
    RUNNER=("$SCRIPT_DIR/venv/bin/python")
    echo "[run] Using local venv: $SCRIPT_DIR/venv"
  elif command -v python3 >/dev/null 2>&1; then
    RUNNER=("python3")
    echo "[run] Using system python3"
  else
    echo "[run] Using default 'python' on PATH"
  fi
fi

launch_worker "$SCRIPT_DIR/workers/orchestrator.py" "orchestrator"
launch_worker "$SCRIPT_DIR/workers/topic_aggregator.py" "topic_aggregator"
launch_worker "$SCRIPT_DIR/workers/voice_agent.py" "voice_agent"

echo "[run] Launching: ${RUNNER[*]} $PY ${args[*]}"
"${RUNNER[@]}" "$PY" "${args[@]}"
