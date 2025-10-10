@echo off
setlocal enabledelayedexpansion

rem ==== CONFIG SECTION ==========
set "ALGO=debug"
set "NUM_FIXED_GPUS=1 2 3 4 5 6 7 8
set "INF_MODE=off"
set "TRN_MODE=poisson"
set "DURATION=86400"
set "LOG_INTERVAL=10"
set "TRN_RATE=0.02"
set "LOG_FOLDER=energy_freq"
set "BIN=5"
set "OUTDIR=%LOG_FOLDER%\figs_all"
set "SKIP_SIM=1"


rem ==== RUN SIMULATION ====
if "%SKIP_SIM%"=="0" (
    echo ===============================
    echo Running simulations...
    echo ===============================
    for %%r in (%NUM_FIXED_GPUS%) do (
        set "RUN_PATH=%LOG_FOLDER%\%ALGO%_%%r"
        mkdir "!RUN_PATH!" 2>nul

        echo python run_sim_paper.py --algo %ALGO% --num_fixed_gpus %%r --inf-mode %INF_MODE% --trn-mode %TRN_MODE% --trn-rate %TRN_RATE% --duration %DURATION% --log-interval %LOG_INTERVAL% --log-path "!RUN_PATH!"
        echo.

        python run_sim_paper.py ^
            --algo %ALGO% ^
            --num_fixed_gpus %%r ^
            --inf-mode %INF_MODE% ^
            --trn-mode %TRN_MODE% ^
            --trn-rate %TRN_RATE% ^
            --duration %DURATION% ^
            --log-interval %LOG_INTERVAL% ^
            --log-path "!RUN_PATH!"
    )
) else (
    echo ===============================
    echo Skipping simulations SKIP_SIM=1
    echo ===============================
)


rem ==== GENERATE PLOTS ====
echo.
echo Generating plots from all folders in "%LOG_FOLDER%" ...

set "RUN_ARGS="

for /d %%d in ("%LOG_FOLDER%\*") do (
    set "name=%%~nxd"
    set "RUN_ARGS=!RUN_ARGS! --run !name!=%%d"
)

python plot_sim_result.py !RUN_ARGS! --outdir "%OUTDIR%" --bin %BIN%


rem ==== SAVE SCRIPT COPY ====
echo.
echo Saving script copy to "%LOG_FOLDER%\run_all_copy.txt" ...
copy "%~f0" "%LOG_FOLDER%\run_all_copy.txt" >nul
echo Done.

pause