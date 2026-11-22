#!/bin/bash
set -e

echo "========================================================"
echo " Running Unit Tests"
echo "========================================================"
./run_tests

echo ""
echo "========================================================"
echo " Executing Full Experiment Sequence (Experiments 1 - 8)"
echo "========================================================"
./msar_reproduce

echo ""
echo "========================================================"
echo " Generating Visualization Graphs"
echo "========================================================"
python3 scripts/plot_results.py || python scripts/plot_results.py

echo ""
echo "[DONE] Reproduction suite finished successfully."
