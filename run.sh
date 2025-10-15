#!/usr/bin/env bash
# ============================================
# 🔧 GLOBAL CONFIG
# ============================================

# --- Python interpreter ---
#   auto   : ưu tiên python3, không có thì dùng python
#   python : bắt buộc dùng python
#   python3: bắt buộc dùng python3
#   hoặc để đường dẫn tuyệt đối tới python
PY="auto"

# --- Chọn 1 thuật toán cố định ---
#   baseline | bandit | joint_nf | rl_energy | rl_energy_adv | rl_energy_upgr
ALGO="baseline"

# --- Nếu dùng rl_energy_adv: weighted | constrained ---
RL_MODE="weighted"

# --- Tham số mô phỏng chung ---
DURATION="604800"
INF_MODE="off"
TRN_RATE="0.02"
LOG_INTERVAL="20"
LOG_FOLDER="test_run"
ELASTIC="False"
SKIP_SIM="1"   # 0=chạy mô phỏng, 1=bỏ qua mô phỏng

# --- Cài đặt vẽ hình ---
PLOT_ENABLE="0"  # 1=bật vẽ, 0=tắt
BIN="20"

# --- Upgraded (Deep RL) / Energy constraint ---
SLA_P99_MS="500.0"
ENERGY_BUDGET="0"  # 0=không ràng buộc; >0 thì truyền --energy_budget_j

# ============================================
# 🤖 RL PARAMS
# ============================================

# --- Common RL params ---
RL_ALPHA="0.1"
RL_GAMMA="0.1"
RL_EPS="0.05"
RL_EPS_DECAY="0.999"
RL_EPS_MIN="0.01"
RL_N_CAND="8"

# --- Advanced RL params ---
RL_TAU="0.1"
RL_CLIP_GRAD="5.0"
RL_BASELINE_BETA="0.01"

# --- Upgraded (Deep RL) specific params ---
UPGR_BUFFER="200000"
UPGR_BATCH="256"
UPGR_WARMUP="1000"
UPGR_DEVICE="cpu"

# ============================================
# 🐍 Resolve Python interpreter
# ============================================
resolve_python() {
  local choice="$1"
  local cmd=""
  if [[ "$choice" == "auto" ]]; then
    if command -v python3 >/dev/null 2>&1; then
      cmd="python3"
    elif command -v python >/dev/null 2>&1; then
      cmd="python"
    else
      echo "[ERROR] Không tìm thấy python/python3 trong PATH."
      exit 1
    fi
  elif [[ "$choice" == "python" || "$choice" == "python3" ]]; then
    if command -v "$choice" >/dev/null 2>&1; then
      cmd="$choice"
    else
      echo "[ERROR] Không tìm thấy interpreter: $choice"
      exit 1
    fi
  else
    # Giả định là đường dẫn tuyệt đối/tương đối tới python
    if [[ -x "$choice" ]]; then
      cmd="$choice"
    else
      echo "[ERROR] PY chỉ tới file không thực thi: $choice"
      exit 1
    fi
  fi
  echo "$cmd"
}

PY_CMD="$(resolve_python "$PY")"
echo "Using Python interpreter: $PY_CMD"

# ============================================
# 🏷️  Derive run name & dirs
# ============================================
RUN_NAME="$ALGO"
if [[ "$ALGO" == "rl_energy_adv" ]]; then
  RUN_NAME="${ALGO}_${RL_MODE}"
fi
RUN_DIR="${LOG_FOLDER}/${RUN_NAME}"
OUTDIR="${RUN_DIR}/figs"

# ============================================
# 🧰 Build algo-specific args
# ============================================
ALGO_ARGS=(--algo "$ALGO")

case "$ALGO" in
  baseline)
    # no extra args
    ;;
  bandit)
    # no extra args
    ;;
  joint_nf)
    # no extra args
    ;;
  rl_energy)
    ALGO_ARGS+=(
      --rl-alpha "$RL_ALPHA"
      --rl-gamma "0.0"
      --rl-eps "0.2"
      --rl-eps-decay "0.995"
      --rl-eps-min "0.02"
      --rl-n-cand "$RL_N_CAND"
      --elastic-scaling "$ELASTIC"
    )
    ;;
  rl_energy_adv)
    ALGO_ARGS+=(
      --rl-mode "$RL_MODE"
      --rl-alpha "$RL_ALPHA"
      --rl-gamma "$RL_GAMMA"
      --rl-eps "$RL_EPS"
      --rl-eps-decay "$RL_EPS_DECAY"
      --rl-eps-min "$RL_EPS_MIN"
      --rl-tau "$RL_TAU"
      --rl-clip-grad "$RL_CLIP_GRAD"
      --rl-baseline-beta "$RL_BASELINE_BETA"
      --rl-n-cand "$RL_N_CAND"
      --elastic-scaling "$ELASTIC"
    )
    ;;
  rl_energy_upgr)
    ALGO_ARGS+=(
      --upgr-buffer "$UPGR_BUFFER"
      --upgr-batch "$UPGR_BATCH"
      --upgr-warmup "$UPGR_WARMUP"
      --upgr-device "$UPGR_DEVICE"
      --sla_p99_ms "$SLA_P99_MS"
      --elastic-scaling "$ELASTIC"
    )
    if [[ "$ENERGY_BUDGET" != "0" ]]; then
      ALGO_ARGS+=(--energy_budget_j "$ENERGY_BUDGET")
    fi
    ;;
  *)
    echo "[ERROR] ALGO \"$ALGO\" không hỗ trợ."
    exit 1
    ;;
esac

# ============================================
# 🚀 RUN (single algorithm)
# ============================================
if [[ "$SKIP_SIM" == "0" ]]; then
  echo "==============================="
  echo "Running simulation: $RUN_NAME"
  echo "Log path         : \"$RUN_DIR\""
  echo "==============================="

  # Tạo thư mục log trước để script plot biết nơi đọc
  mkdir -p "$RUN_DIR"

  if ! "$PY_CMD" run_sim_paper.py \
      "${ALGO_ARGS[@]}" \
      --inf-mode "$INF_MODE" \
      --trn-rate "$TRN_RATE" \
      --duration "$DURATION" \
      --log-interval "$LOG_INTERVAL" \
      --log-path "$RUN_DIR"
  then
    echo "[ERROR] Simulation failed."
    # tiếp tục xuống bước plot (nếu bật) như bản .bat gốc
  fi
else
  echo "==============================="
  echo "Skipping simulations (SKIP_SIM=1)"
  echo "Expected run dir: \"$RUN_DIR\""
  echo "==============================="
fi

# ============================================
# 📊 PLOT (toggle by PLOT_ENABLE)
# ============================================
if [[ "$PLOT_ENABLE" == "1" ]]; then
  echo
  echo "[PLOT] Generating plots for \"$RUN_NAME\" ..."
  if [[ -d "$RUN_DIR" ]]; then
    mkdir -p "$OUTDIR"
    echo "[PLOT] $PY_CMD plot_sim_result.py --run ${RUN_NAME}=${RUN_DIR} --outdir \"$OUTDIR\" --bin $BIN"
    "$PY_CMD" plot_sim_result.py \
      --run "${RUN_NAME}=${RUN_DIR}" \
      --outdir "$OUTDIR" \
      --bin "$BIN"
  else
    echo "[WARN] Run directory not found: \"$RUN_DIR\". Skip plotting."
  fi
else
  echo "[PLOT] Disabled (PLOT_ENABLE=0)"
fi

# ============================================
# 💾 SAVE SCRIPT COPY
# ============================================
mkdir -p "$LOG_FOLDER"
cp "$0" "${LOG_FOLDER}/run_single_copy.txt" 2>/dev/null || true
echo
echo "Script copy saved to \"${LOG_FOLDER}/run_single_copy.txt\""
