#!/usr/bin/env sh
set -eu

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python}"
else
  echo "Python 3.11+ is required. Please install Python first." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ is required.")
PY

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
LOG_DIR="$SCRIPT_DIR/.fusion-memory-logs"
LOG_FILE="$LOG_DIR/install.log"
mkdir -p "$LOG_DIR"
: > "$LOG_FILE"

run_step() {
  step_name="$1"
  shift
  echo "$step_name..."
  {
    echo
    echo "=== $step_name ==="
    printf '%s ' "$@"
    echo
    "$@"
  } >>"$LOG_FILE" 2>&1 || {
    echo "Fusion Memory installation needs attention." >&2
    echo "Step: $step_name" >&2
    echo "Log: $LOG_FILE" >&2
    exit 1
  }
}

run_step "installer bootstrap" "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
run_step "fusion memory base" "$PYTHON_BIN" -m pip install -e "$SCRIPT_DIR"
run_step "modelscope downloader" "$PYTHON_BIN" -m pip install --only-binary=:all: "modelscope-hub>=0.1.6"
run_step "local qwen models" "$PYTHON_BIN" -m fusion_memory.windows_installer --download-models-only --script-dir "$SCRIPT_DIR" --log-dir "$LOG_DIR"
run_step "local qwen runtime" "$PYTHON_BIN" -m pip install --only-binary=:all: \
  "psycopg2-binary>=2.9" \
  "torch>=2.5" \
  "transformers>=4.51" \
  "sentence-transformers>=3.4" \
  safetensors \
  tokenizers \
  hf-xet \
  click \
  typer
if [ "${FUSION_MEMORY_USE_WIZARD:-}" = "1" ]; then
  run_step "wizard" "$PYTHON_BIN" -m fusion_memory.cli init --wizard
elif [ "${FUSION_MEMORY_SKIP_WIZARD:-}" = "1" ]; then
  run_step "install readiness" "$PYTHON_BIN" -m fusion_memory.cli install-check --force
else
  run_step "install readiness" "$PYTHON_BIN" -m fusion_memory.cli install-check --force
fi
run_step "doctor" "$PYTHON_BIN" -m fusion_memory.cli doctor

echo
echo "Fusion Memory is installed."
echo "Model paths: $SCRIPT_DIR/models/Qwen3-Embedding-0.6B and $SCRIPT_DIR/models/Qwen3-Reranker-0.6B"
echo "Log: $LOG_FILE"
echo "Start it with: fusion-memory start"
echo "Check it with: fusion-memory status"
