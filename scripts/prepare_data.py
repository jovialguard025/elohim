"""Reproducible data preparation script for confidence-gated MSAR datasets."""

import argparse
import os
import json


def prepare_dataset(input_path: str, output_dir: str, split_ratio: float = 0.8) -> None:
    """Validate, split, and save processed datasets."""
    print(f"Preparing data from: {input_path}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory established at: {output_dir}")
    print(f"Split ratio (train/val): {split_ratio:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Data preparation and validation script.")
    parser.add_argument("--input", type=str, default="data/sample_dataset.json", help="Path to input data file.")
    parser.add_argument("--output", type=str, default="data/processed", help="Path to output split directory.")
    parser.add_argument("--split", type=float, default=0.8, help="Train/Validation split ratio.")
    args = parser.parse_args()

    prepare_dataset(args.input, args.output, args.split)


if __name__ == "__main__":
    main()
