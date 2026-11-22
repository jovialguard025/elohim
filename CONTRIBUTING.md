# Contributing Guidelines

Welcome to the **Confidence-Gated MSAR** research project! To maintain high code quality, reproducibility, and structured collaboration across team members, please adhere to these guidelines.

---

## 1. Branching Strategy

- **`main`**: Protected branch containing stable, reviewed, and tested code. Direct commits to `main` should be avoided for experimental code.
- **Branch Naming Conventions**:
  - `feature/<short-description>`: New functional components (e.g., `feature/entropy-gating`).
  - `experiment/<exp-id-or-topic>`: Experimental pipelines, hypothesis testing, and exploratory models.
  - `fix/<issue-description>`: Bug fixes or corrections in data/code pipelines.
  - `docs/<topic>`: Documentation updates, methodology notes, or reports.

---

## 2. Pull Request (PR) Workflow

1. **Create Branch**: Always branch off the latest `main`.
2. **Commit Early & Clearly**: Use descriptive commit messages following the Conventional Commits style:
   - `feat: add entropy-based confidence threshold calculation`
   - `fix: resolve zero-division edge case in precision calculation`
   - `docs: update experimental protocol for Phase 2 benchmarks`
3. **PR Description**: Include:
   - Objective of the PR.
   - Summary of changes.
   - Verification evidence (e.g., test outputs, metrics).
4. **Code Review**: At least one review/approval from a fellow collaborator ([jovialguard025](https://github.com/jovialguard025), [mayukh437](https://github.com/mayukh437), [KOUSTAV-04](https://github.com/KOUSTAV-04), [KousambiGhosh](https://github.com/KousambiGhosh)) is recommended before merging.

---

## 3. Testing Requirements

Before opening a PR or merging code:
- **Run Unit Tests**: All tests in `tests/` must pass:
  ```bash
  pytest tests/
  ```
- **Add Tests for New Features**: Any new feature in `src/` (feature extractors, gating thresholds, model heads) must have corresponding test coverage in `tests/`.
- **Budget Compliance Verification**: Changes impacting multi-agent interaction must pass context-budget limit checks in `tests/test_context_budget.py`.

---

## 4. Experiment-Reporting Rules

To guarantee reproducibility and clean collaboration:
- **Never commit raw or unchecked datasets**: Place dataset scripts and descriptions in `data/` instead of dumping large raw dumps.
- **Log Hyperparameters & Random Seeds**: Every training and evaluation run must log seeds, commit hash, and configuration.
- **Sanitized Results in `results/`**: Only commit aggregated metric summaries, confusion matrices, and experiment logs. Avoid committing high-volume tensor logs or redundant checkpoints.
- **Model Checkpoints in `models/`**: Only save final, reproducible model weights and configurations. Intermediate variants must be pruned or stored off-repo.
