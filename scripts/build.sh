#!/bin/bash
set -e

echo "========================================================"
echo " Building Confidence-Gated MSAR C++ Research Framework"
echo "========================================================"

g++ -std=c++14 -O2 -Iinclude src/main.cpp -o msar_reproduce
echo "[SUCCESS] Compiled msar_reproduce"

g++ -std=c++14 -O2 -Iinclude tests/test_pipeline.cpp -o run_tests
echo "[SUCCESS] Compiled run_tests"

echo "All binaries built successfully!"
