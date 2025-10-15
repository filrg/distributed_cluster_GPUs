#!/usr/bin/env bash
# ============================================================
# 📊 Plot-only runner for Ubuntu (bash)
# - Quét tất cả thư mục con trong LOG_FOLDER và truyền vào:
#     plot_sim_result.py --run <name>=<path> ...
# - Không chạy mô phỏng; chỉ vẽ hình.
#
# Quick start:
#   chmod +x run_plots.sh
#   ./run_plots.sh
#
# Tuỳ chọn CLI (đơn giản):
#   --python <auto|python|python3|/path/to/python>
#   --log-folder <path>
#   --outdir <path>
#   --bin <int>
#
# Ví dụ:
#   ./run_plots.sh --python auto --log-folder test_run --outdir test_run/figs_all --bin 20
# ============================================================

set -euo pipefail

# ----------------------------
# 🔧 Default config
# ----------------------------
PY="auto"                   # auto | python | python3 | /path/to/python
LOG_FOLDER="test_run"
OUTDIR="${LOG_FOLDER}/figs_all"
BIN="20"

# ----------------------------
# 🧭 Parse simple CLI flags
# ----------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PY="${2:-}"; shift 2 ;;
    --log-folder)
      LOG_FOLDER="${2:-}"; shift 2 ;;
    --outdir)
      OUTDIR="${2:-}"; shift 2 ;;
    --bin)
      BIN="${2:-}"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--python auto|python|python3|/path/python] [--log-folder PATH] [--outdir PATH] [--bin N]"
      exit 0 ;;
    *)
      echo "[WARN] Unknown arg: $1 (ignored)"; shift ;;
  esac
done

# ----------------------------
# 🐍 Resolve Python interpreter
# ----------------------------
resolve_python() {
  local choice="$1"
  local cmd=""
  if [[ "$choice" == "auto" ]]; then
    if command -v python3 >/dev/null 2>&1; then
      cmd="python3"
    elif command -v python >/dev/null 2>&1; then
      cmd="python"
    else
      echo "[ERROR] python/python3 not found in PATH." >&2
      exit 1
    fi
  elif [[ "$choice" == "python" || "$choice" == "python3" ]]; then
    if command -v "$choice" >/dev/null 2>&1; then
      cmd="$choice"
    else
      echo "[ERROR] Interpreter not found: $choice" >&2
      exit 1
    fi
  else
    # Explicit path
    if [[ -x "$choice" ]]; then
      cmd="$choice"
    else
      echo "[ERROR] PY points to non-executable: $choice" >&2
      exit 1
    fi
  fi
  echo "$cmd"
}

PY_CMD="$(resolve_python "$PY")"
echo "Using Python interpreter: $PY_CMD"
echo "LOG_FOLDER: $LOG_FOLDER"
echo "OUTDIR    : $OUTDIR"
echo "BIN       : $BIN"

# ----------------------------
# 🗂️ Collect runs
# ----------------------------
if [[ ! -d "$LOG_FOLDER" ]]; then
  echo "[ERROR] LOG_FOLDER does not exist: $LOG_FOLDER" >&2
  exit 1
fi

mapfile -t RUN_DIRS < <(find "$LOG_FOLDER" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ "${#RUN_DIRS[@]}" -eq 0 ]]; then
  echo "[ERROR] No run directories found under: $LOG_FOLDER" >&2
  echo "Make sure your logs are like: ${LOG_FOLDER}/baseline, ${LOG_FOLDER}/bandit, ..." >&2
  exit 1
fi

RUN_ARGS=()
for d in "${RUN_DIRS[@]}"; do
  name="$(basename "$d")"
  RUN_ARGS+=(--run "${name}=${d}")
done

# ----------------------------
# 🖼️ Plot
# ----------------------------
mkdir -p "$OUTDIR"
echo "[PLOT] $PY_CMD plot_sim_result.py ${RUN_ARGS[*]} --outdir \"$OUTDIR\" --bin $BIN"
"$PY_CMD" plot_sim_result.py \
  "${RUN_ARGS[@]}" \
  --outdir "$OUTDIR" \
  --bin "$BIN"
