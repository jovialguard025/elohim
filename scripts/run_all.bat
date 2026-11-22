@echo off
echo ========================================================
echo  Running Unit Tests
echo ========================================================
run_tests.exe
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Unit tests failed!
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================================
echo  Executing Full Experiment Sequence (Experiments 1 - 8)
echo ========================================================
msar_reproduce.exe
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Experiment execution failed!
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================================
echo  Generating Visualization Graphs
echo ========================================================
python scripts\plot_results.py
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Plot generation encountered an issue. Ensure matplotlib is installed.
)

echo.
echo [DONE] Reproduction suite finished successfully.
