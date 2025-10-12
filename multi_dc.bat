@echo off
setlocal enabledelayedexpansion

rem ============================================
rem 🔧 GLOBAL CONFIGURATION SECTION
rem ============================================

rem === Simulation parameters ===
set "DURATION=86400"
set "INF_MODE=off"
set "TRN_RATE=0.02"
set "LOG_INTERVAL=20"
set "LOG_FOLDER=test_run"
set "ELASTIC=False"
set "SKIP_SIM=1"
rem 0 = run simulations, 1 = skip simulations

rem === Plot parameters ===
set "BIN=20"
set "OUTDIR=%LOG_FOLDER%\figs_all"

rem ============================================
rem 🤖 RL CONFIGURATION SECTION
rem ============================================

rem --- Basic & Advanced RL common params ---
rem Basic & Advanced RL are now different at this
set "RL_ALPHA=0.1"
set "RL_GAMMA=0.1"
set "RL_EPS=0.05"
set "RL_EPS_DECAY=0.999"
set "RL_EPS_MIN=0.01"

rem -- Advanced RL params --
set "RL_TAU=0.1"
set "RL_CLIP_GRAD=5.0"
set "RL_BASELINE_BETA=0.01"
set "RL_N_CAND=8"

rem --- Upgraded (Deep RL) specific params ---
set "UPGR_BUFFER=200_000"
set "UPGR_BATCH=256"
set "UPGR_WARMUP=1_000"
set "UPGR_DEVICE=cpu"
set "SLA_P99_MS=500.0"
set "ENERGY_BUDGET=0"

rem ============================================
rem 🚀 RUN SIMULATIONS
rem ============================================

if "%SKIP_SIM%"=="0" (
    echo ===============================
    echo Running all RL simulations...
    echo ===============================

    rem --- 1️⃣ Baseline ---
    echo.
    echo [RUN] python run_sim_paper.py --algo baseline --trn-rate %TRN_RATE% --inf-mode %INF_MODE% --duration %DURATION% --log-interval %LOG_INTERVAL% --log-path "%LOG_FOLDER%"
    python run_sim_paper.py ^
        --algo baseline ^
        --inf-mode %INF_MODE% ^
        --trn-rate %TRN_RATE% ^
        --duration %DURATION% ^
        --log-interval %LOG_INTERVAL% ^
        --log-path "%LOG_FOLDER%"

    rem --- 2️⃣ Bandit ---
    echo.
    echo [RUN] python run_sim_paper.py --algo bandit --trn-rate %TRN_RATE% --inf-mode %INF_MODE% --duration %DURATION% --log-interval %LOG_INTERVAL% --log-path "%LOG_FOLDER%"
    python run_sim_paper.py ^
        --algo bandit ^
        --inf-mode %INF_MODE% ^
        --trn-rate %TRN_RATE% ^
        --duration %DURATION% ^
        --log-interval %LOG_INTERVAL% ^
        --log-path "%LOG_FOLDER%"

    rem --- 3️⃣ Joint NF ---
    echo.
    echo [RUN] python run_sim_paper.py --algo joint_nf --trn-rate %TRN_RATE% --inf-mode %INF_MODE% --duration %DURATION% --log-interval %LOG_INTERVAL% --log-path "%LOG_FOLDER%"
    python run_sim_paper.py ^
        --algo joint_nf ^
        --inf-mode %INF_MODE% ^
        --trn-rate %TRN_RATE% ^
        --duration %DURATION% ^
        --log-interval %LOG_INTERVAL% ^
        --log-path "%LOG_FOLDER%"

    rem --- 4️⃣ RL Energy (Basic Q-learning) ---
    if 0==1 (
    echo.
    echo [RUN] python run_sim_paper.py --algo rl_energy --rl-alpha %RL_ALPHA% --rl-gamma 0.0 --rl-eps 0.2 --rl-eps-decay 0.995 --rl-eps-min 0.02 --rl-n-cand %RL_N_CAND% --elastic-scaling %ELASTIC% --trn-rate %TRN_RATE% --inf-mode %INF_MODE% --duration %DURATION% --log-interval %LOG_INTERVAL% --log-path "%LOG_FOLDER%"
    python run_sim_paper.py ^
        --algo rl_energy ^
        --rl-alpha %RL_ALPHA% ^
        --rl-gamma 0.0 ^
        --rl-eps 0.2 ^
        --rl-eps-decay 0.995 ^
        --rl-eps-min 0.02 ^
        --rl-n-cand %RL_N_CAND% ^
        --elastic-scaling %ELASTIC% ^
        --trn-rate %TRN_RATE% ^
        --inf-mode %INF_MODE% ^
        --duration %DURATION% ^
        --log-interval %LOG_INTERVAL% ^
        --log-path "%LOG_FOLDER%"
    )
    rem --- 5️⃣ RL Energy Advanced (Weighted) ---
    echo.
    echo [RUN] python run_sim_paper.py --algo rl_energy_adv --rl-mode weighted --rl-alpha %RL_ALPHA% --rl-gamma %RL_GAMMA% --rl-eps %RL_EPS% --rl-eps-decay %RL_EPS_DECAY% --rl-eps-min %RL_EPS_MIN% --rl-tau %RL_TAU% --rl-clip-grad %RL_CLIP_GRAD% --rl-baseline-beta %RL_BASELINE_BETA% --rl-n-cand %RL_N_CAND% --elastic-scaling %ELASTIC% --trn-rate %TRN_RATE% --inf-mode %INF_MODE% --duration %DURATION% --log-interval %LOG_INTERVAL% --log-path "%LOG_FOLDER%\rl_adv_w"
    python run_sim_paper.py ^
        --algo rl_energy_adv ^
        --rl-mode weighted ^
        --rl-alpha %RL_ALPHA% ^
        --rl-gamma %RL_GAMMA% ^
        --rl-eps %RL_EPS% ^
        --rl-eps-decay %RL_EPS_DECAY% ^
        --rl-eps-min %RL_EPS_MIN% ^
        --rl-tau %RL_TAU% ^
        --rl-clip-grad %RL_CLIP_GRAD% ^
        --rl-baseline-beta %RL_BASELINE_BETA% ^
        --rl-n-cand %RL_N_CAND% ^
        --elastic-scaling %ELASTIC% ^
        --trn-rate %TRN_RATE% ^
        --inf-mode %INF_MODE% ^
        --duration %DURATION% ^
        --log-interval %LOG_INTERVAL% ^
        --log-path "%LOG_FOLDER%\rl_adv_w"

    rem --- 6️⃣ RL Energy Advanced (Constrained) ---
    echo.
    echo [RUN] python run_sim_paper.py --algo rl_energy_adv --rl-mode constrained --rl-alpha %RL_ALPHA% --rl-gamma %RL_GAMMA% --rl-eps %RL_EPS% --rl-eps-decay %RL_EPS_DECAY% --rl-eps-min %RL_EPS_MIN% --rl-tau %RL_TAU% --rl-clip-grad %RL_CLIP_GRAD% --rl-baseline-beta %RL_BASELINE_BETA% --rl-n-cand %RL_N_CAND% --elastic-scaling %ELASTIC% --trn-rate %TRN_RATE% --inf-mode %INF_MODE% --duration %DURATION% --log-interval %LOG_INTERVAL% --log-path "%LOG_FOLDER%\rl_adv_c"
    python run_sim_paper.py ^
        --algo rl_energy_adv ^
        --rl-mode constrained ^
        --rl-alpha %RL_ALPHA% ^
        --rl-gamma %RL_GAMMA% ^
        --rl-eps %RL_EPS% ^
        --rl-eps-decay %RL_EPS_DECAY% ^
        --rl-eps-min %RL_EPS_MIN% ^
        --rl-tau %RL_TAU% ^
        --rl-clip-grad %RL_CLIP_GRAD% ^
        --rl-baseline-beta %RL_BASELINE_BETA% ^
        --rl-n-cand %RL_N_CAND% ^
        --elastic-scaling %ELASTIC% ^
        --trn-rate %TRN_RATE% ^
        --inf-mode %INF_MODE% ^
        --duration %DURATION% ^
        --log-interval %LOG_INTERVAL% ^
        --log-path "%LOG_FOLDER%\rl_adv_c"

    rem --- 7️⃣ RL Energy Upgraded (Deep RL) ---
    echo.
    if "%ENERGY_BUDGET%"=="0" (
    echo [RUN] python run_sim_paper.py --algo rl_energy_upgr --upgr-buffer %UPGR_BUFFER% --upgr-batch %UPGR_BATCH% --upgr-warmup %UPGR_WARMUP% --upgr-device %UPGR_DEVICE% --sla_p99_ms %SLA_P99_MS% --elastic-scaling %ELASTIC% --trn-rate %TRN_RATE% --inf-mode %INF_MODE% --duration %DURATION% --log-interval %LOG_INTERVAL% --log-path "%LOG_FOLDER%"
    python run_sim_paper.py ^
        --algo rl_energy_upgr ^
        --upgr-buffer %UPGR_BUFFER% ^
        --upgr-batch %UPGR_BATCH% ^
        --upgr-warmup %UPGR_WARMUP% ^
        --upgr-device %UPGR_DEVICE% ^
        --sla_p99_ms %SLA_P99_MS% ^
        --elastic-scaling %ELASTIC% ^
        --trn-rate %TRN_RATE% ^
        --inf-mode %INF_MODE% ^
        --duration %DURATION% ^
        --log-interval %LOG_INTERVAL% ^
        --log-path "%LOG_FOLDER%"
    ) else (
    echo [RUN] python run_sim_paper.py --algo rl_energy_upgr --upgr-buffer %UPGR_BUFFER% --upgr-batch %UPGR_BATCH% --upgr-warmup %UPGR_WARMUP% --upgr-device %UPGR_DEVICE% --sla_p99_ms %SLA_P99_MS% --energy_budget_j %ENERGY_BUDGET% --elastic-scaling %ELASTIC% --trn-rate %TRN_RATE% --inf-mode %INF_MODE% --duration %DURATION% --log-interval %LOG_INTERVAL% --log-path "%LOG_FOLDER%"
    python run_sim_paper.py ^
        --algo rl_energy_upgr ^
        --upgr-buffer %UPGR_BUFFER% ^
        --upgr-batch %UPGR_BATCH% ^
        --upgr-warmup %UPGR_WARMUP% ^
        --upgr-device %UPGR_DEVICE% ^
        --sla_p99_ms %SLA_P99_MS% ^
        --energy_budget_j %ENERGY_BUDGET% ^
        --elastic-scaling %ELASTIC% ^
        --trn-rate %TRN_RATE% ^
        --inf-mode %INF_MODE% ^
        --duration %DURATION% ^
        --log-interval %LOG_INTERVAL% ^
        --log-path "%LOG_FOLDER%"
    )
    echo.
    echo All simulations completed!
) else (
    echo ===============================
    echo Skipping simulations SKIP_SIM=1
    echo ===============================
)

rem ============================================
rem 📊 GENERATE PLOTS
rem ============================================
echo.
echo Generating plots from all runs inside "%LOG_FOLDER%" ...

set "RUN_ARGS="
for /d %%d in ("%LOG_FOLDER%\*") do (
    set "name=%%~nxd"
    set "RUN_ARGS=!RUN_ARGS! --run !name!=%%d"
)

echo [PLOT] python plot_sim_result.py !RUN_ARGS! --outdir "%OUTDIR%" --bin %BIN%
python plot_sim_result.py !RUN_ARGS! --outdir "%OUTDIR%" --bin %BIN%

rem ============================================
rem 💾 SAVE SCRIPT COPY
rem ============================================
copy "%~f0" "%LOG_FOLDER%\run_all_copy.txt" >nul
echo.
echo Script copy saved to "%LOG_FOLDER%\run_all_copy.txt"
pause