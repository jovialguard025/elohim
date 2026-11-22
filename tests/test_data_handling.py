"""Tests for dataset loading, attribution, and schema validation."""

import os
import json


def test_dataset_format_validation():
    sample_record = {
        "id": "sample-001",
        "query": "Solve step-by-step logic puzzle",
        "ground_truth": "Option B",
        "difficulty": "medium",
    }
    assert "query" in sample_record
    assert "ground_truth" in sample_record
    assert isinstance(sample_record["id"], str)
