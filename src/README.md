# Source Code (`src/`)

This directory contains the reviewed, complete source code for the **Confidence-Gated MSAR** system.

## Module Overview

- **`features/`**: Feature extraction pipelines, context tokenizers, and uncertainty signal extractors.
- **`gating/`**: Confidence gating mechanisms, calibrated thresholds, and adaptive branch selectors.
- **`mar/`**: Multi-Agent Routing (MAR) system orchestration and state management.
- **`models/`**: Predictive model definitions, neural architectures, and training wrappers.
- **`evaluation/`**: Evaluation pipelines, accuracy/latency metrics, and context-budget trackers.

## Code Standards
- All code should include type hints (`typing`) and informative docstrings.
- Ensure new methods are tested under `tests/`.
- Maintain modular decoupling between feature extraction, gating logic, and external LLM/agent providers.
