# Scripts (`scripts/`)

This directory contains standalone, reproducible CLI scripts for data preparation, model training, and evaluation.

## Available Scripts

1. **`prepare_data.py`**:
   - Downloads, validates, and partitions raw datasets into approved train/val/test splits.
   - Usage:
     ```bash
     python scripts/prepare_data.py --input data/raw/sample.json --output data/processed/
     ```

2. **`train.py`**:
   - Trains confidence gating heads or fine-tunes base models with logged random seeds and hyperparameters.
   - Usage:
     ```bash
     python scripts/train.py --config configs/train_config.yaml --seed 42
     ```

3. **`evaluate.py`**:
   - Evaluates the MAR pipeline against test benchmarks, tracks context-budget consumption, and outputs sanitized metrics to `results/`.
   - Usage:
     ```bash
     python scripts/evaluate.py --data data/processed/test.json --output results/metrics/summary.json
     ```
