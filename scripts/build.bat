@echo off
echo ========================================================
echo  Building Confidence-Gated MSAR C++ Research Framework
echo ========================================================

g++ -std=c++14 -O2 -Iinclude src/main.cpp -o msar_reproduce.exe
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to compile msar_reproduce.exe
    exit /b %ERRORLEVEL%
)
echo [SUCCESS] Compiled msar_reproduce.exe

g++ -std=c++14 -O2 -Iinclude tests/test_pipeline.cpp -o run_tests.exe
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to compile run_tests.exe
    exit /b %ERRORLEVEL%
)
echo [SUCCESS] Compiled run_tests.exe

echo All binaries built successfully!
