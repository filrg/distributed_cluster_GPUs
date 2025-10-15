@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ============================================
rem 🔧 GLOBAL CONFIG
rem ============================================

rem --- Python interpreter ---
rem   auto  : ưu tiên python3, không có thì dùng python
rem   python: bắt buộc dùng python
rem   python3 hoặc đường dẫn tuyệt đối tới python.exe cũng được
set "PY=auto"

rem --- Chọn 1 thuật toán cố định ---
rem   baseline | bandit | joint_nf | rl_energy | rl_energy_adv | rl_energy_upgr
set "ALGO=baseline"

rem --- Nếu dùng rl_energy_adv: weighted | constrained ---
set "RL_MODE=weighted"

rem --- Tham số mô phỏng chung ---
set "DURATION=604800"
set "INF_MODE=off"
set "TRN_RATE=0.02"
set "LOG_INTERVAL=20"
set "LOG_FOLDER=test_run"
set "ELASTIC=False"
set "SKIP_SIM=1"   rem 0=chạy mô phỏng, 1=bỏ qua mô phỏng

rem --- Cài đặt vẽ hình ---
set "PLOT_ENABLE=0"            rem 1=bật vẽ, 0=tắt
set "BIN=20"

rem --- Upgraded (Deep RL) / Energy constraint ---
set "SLA_P99_MS=500.0"
set "ENERGY_BUDGET=0"          rem 0=không ràng buộc; >0 thì truyền --energy_budget_j

rem ============================================
rem 🤖 RL PARAMS
rem ============================================

rem --- Common RL params ---
set "RL_ALPHA=0.1"
set "RL_GAMMA=0.1"
set "RL_EPS=0.05"
set "RL_EPS_DECAY=0.999"
set "RL_EPS_MIN=0.01"
set "RL_N_CAND=8"

rem --- Advanced RL params ---
set "RL_TAU=0.1"
set "RL_CLIP_GRAD=5.0"
set "RL_BASELINE_BETA=0.01"

rem --- Upgraded (Deep RL) specific params ---
set "UPGR_BUFFER=200000"
set "UPGR_BATCH=256"
set "UPGR_WARMUP=1000"
set "UPGR_DEVICE=cpu"

rem ============================================
rem 🐍 Resolve Python interpreter
rem ============================================
if /I "%PY%"=="auto" (
  where python3 >nul 2>nul
  if not errorlevel 1 ( set "PY=python3" ) else ( set "PY=python" )
)

echo Using Python interpreter: "%PY%"

rem ============================================
rem 🏷️  Derive run name & dirs
rem ============================================
set "RUN_NAME=%ALGO%"
if /I "%ALGO%"=="rl_energy_adv" set "RUN_NAME=%ALGO%_%RL_MODE%"
set "RUN_DIR=%LOG_FOLDER%\%RUN_NAME%"
set "OUTDIR=%RUN_DIR%\figs"

rem ============================================
rem 🧰 Build algo-specific args
rem ============================================
set "ALGO_ARGS=--algo %ALGO%"

if /I "%ALGO%"=="baseline" (
  rem no extra args
) else if /I "%ALGO%"=="bandit" (
  rem no extra args
) else if /I "%ALGO%"=="joint_nf" (
  rem no extra args
) else if /I "%ALGO%"=="rl_energy" (
  set "ALGO_ARGS=!ALGO_ARGS! --rl-alpha %RL_ALPHA% --rl-gamma 0.0 --rl-eps 0.2 --rl-eps-decay 0.995 --rl-eps-min 0.02 --rl-n-cand %RL_N_CAND% --elastic-scaling %ELASTIC%"
) else if /I "%ALGO%"=="rl_energy_adv" (
  set "ALGO_ARGS=!ALGO_ARGS! --rl-mode %RL_MODE% --rl-alpha %RL_ALPHA% --rl-gamma %RL_GAMMA% --rl-eps %RL_EPS% --rl-eps-decay %RL_EPS_DECAY% --rl-eps-min %RL_EPS_MIN% --rl-tau %RL_TAU% --rl-clip-grad %RL_CLIP_GRAD% --rl-baseline-beta %RL_BASELINE_BETA% --rl-n-cand %RL_N_CAND% --elastic-scaling %ELASTIC%"
) else if /I "%ALGO%"=="rl_energy_upgr" (
  set "ALGO_ARGS=!ALGO_ARGS! --upgr-buffer %UPGR_BUFFER% --upgr-batch %UPGR_BATCH% --upgr-warmup %UPGR_WARMUP% --upgr-device %UPGR_DEVICE% --sla_p99_ms %SLA_P99_MS% --elastic-scaling %ELASTIC%"
  if not "%ENERGY_BUDGET%"=="0" set "ALGO_ARGS=!ALGO_ARGS! --energy_budget_j %ENERGY_BUDGET%"
) else (
  echo [ERROR] ALGO "%ALGO%" khong ho tro.
  goto :END
)

rem ============================================
rem 🚀 RUN (single algorithm)
rem ============================================
if "%SKIP_SIM%"=="0" (
  echo ===============================
  echo Running simulation: %RUN_NAME%
  echo Log path         : "%RUN_DIR%"
  echo ===============================

  "%PY%" run_sim_paper.py ^
    %ALGO_ARGS% ^
    --inf-mode %INF_MODE% ^
    --trn-rate %TRN_RATE% ^
    --duration %DURATION% ^
    --log-interval %LOG_INTERVAL% ^
    --log-path "%RUN_DIR%"

  if errorlevel 1 (
    echo [ERROR] Simulation failed.
    goto :AFTER_RUN
  )
) else (
  echo ===============================
  echo Skipping simulations (SKIP_SIM=1)
  echo Expected run dir: "%RUN_DIR%"
  echo ===============================
)

:AFTER_RUN

rem ============================================
rem 📊 PLOT (toggle by PLOT_ENABLE)
rem ============================================
if "%PLOT_ENABLE%"=="1" (
  echo.
  echo [PLOT] Generating plots for "%RUN_NAME%" ...
  if exist "%RUN_DIR%" (
    set "RUN_ARGS=--run %RUN_NAME%=%RUN_DIR%"
    echo [PLOT] "%PY%" plot_sim_result.py !RUN_ARGS! --outdir "%OUTDIR%" --bin %BIN%
    "%PY%" plot_sim_result.py !RUN_ARGS! --outdir "%OUTDIR%" --bin %BIN%
  ) else (
    echo [WARN] Run directory not found: "%RUN_DIR%". Skip plotting.
  )
) else (
  echo [PLOT] Disabled (PLOT_ENABLE=0)
)

rem ============================================
rem 💾 SAVE SCRIPT COPY
rem ============================================
if not exist "%LOG_FOLDER%" mkdir "%LOG_FOLDER%" >nul 2>nul
copy "%~f0" "%LOG_FOLDER%\run_single_copy.txt" >nul
echo.
echo Script copy saved to "%LOG_FOLDER%\run_single_copy.txt"

:END
echo.
pause
