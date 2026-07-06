#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
LOG_DIR="$SCRIPT_DIR/.fusion-memory-logs"
LOG_FILE="$LOG_DIR/install.log"
UV_BIN="${FUSION_MEMORY_UV_BIN:-uv}"
FUSION_MEMORY_PACKAGE="${FUSION_MEMORY_PACKAGE:-$SCRIPT_DIR}"

mkdir -p "$LOG_DIR"
: > "$LOG_FILE"

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  echo "Fusion Memory installation needs uv." >&2
  echo "Set FUSION_MEMORY_UV_BIN or install uv, then rerun install.sh." >&2
  exit 1
fi

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

run_step "fusion memory tool" "$UV_BIN" tool install --force \
  --python "3.12" \
  --managed-python \
  --no-progress \
  --with "modelscope-hub>=0.1.6" \
  --with "psycopg2-binary>=2.9" \
  --with "torch>=2.5" \
  --with "transformers>=4.51" \
  --with "sentence-transformers>=3.4" \
  --with "safetensors" \
  --with "tokenizers" \
  --with "hf-xet" \
  --with "click" \
  --with "typer" \
  --no-build-package "psycopg2-binary" \
  --no-build-package "torch" \
  --no-build-package "transformers" \
  --no-build-package "sentence-transformers" \
  --no-build-package "safetensors" \
  --no-build-package "tokenizers" \
  --no-build-package "hf-xet" \
  "$FUSION_MEMORY_PACKAGE"

TOOL_BIN_DIR=$("$UV_BIN" tool dir --bin)
FUSION_MEMORY_CMD="$TOOL_BIN_DIR/fusion-memory"

run_step "local qwen models" "$FUSION_MEMORY_CMD" download-models --json
if [ "${FUSION_MEMORY_USE_WIZARD:-}" = "1" ]; then
  run_step "wizard" "$FUSION_MEMORY_CMD" init --wizard
else
  run_step "install readiness" "$FUSION_MEMORY_CMD" install-check --force
fi
run_step "doctor" "$FUSION_MEMORY_CMD" doctor

echo
echo "Fusion Memory is installed."
echo "Model paths: Fusion Memory home models directory, or FUSION_MEMORY_HOME/models when FUSION_MEMORY_HOME is set."
echo "Log: $LOG_FILE"
echo "Start it with: fusion-memory start"
echo "Check it with: fusion-memory status"
