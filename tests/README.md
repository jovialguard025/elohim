# Tests (`tests/`)

This directory contains test cases for:
- **Feature Extraction** (`test_features.py`): Validates entropy computation and token signal extraction.
- **Data Handling** (`test_data_handling.py`): Tests dataset loading, splitting, and schema validation.
- **Predictions** (`test_predictions.py`): Tests base model predictions and output formatting.
- **Confidence Gating** (`test_confidence_gating.py`): Tests threshold gating, branch decisions, and fallback conditions.
- **Context-Budget Compliance** (`test_context_budget.py`): Tests token ceiling tracking and budget boundary enforcement.

## Running Tests
Execute the entire test suite with:
```bash
pytest tests/ -v
```
