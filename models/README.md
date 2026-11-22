# Models Directory (`models/`)

> **Policy: Upload Selectively**  
> **Only the selected, reproducible model and its required normalization parameters, configuration, and model description.**  
> **Do not upload every experimental variant.**

---

## Guidelines for Storing Model Assets

1. **Configuration & Parameters**:
   - Store model architecture definitions and calibrated normalization/scaling parameters under `models/configs/`.
   - Include clear hyperparameter definitions (`learning_rate`, `entropy_threshold`, `budget_ceiling`).
2. **Weights & Checkpoints**:
   - Do NOT commit every intermediate epoch or trial checkpoint.
   - For reproducible benchmark models, commit the final configuration and metadata. Heavy binary weights (`*.pt`, `*.bin`) should be hosted via release assets, Git LFS, or Hugging Face Hub with download links documented in this directory.

---

## Model Inventory
| Model Identifier | Architecture / Base | Calibrated Threshold | Description |
| :--- | :--- | :--- | :--- |
| `msar-gate-v1` | Lightweight MLP / Logistic Gate | `threshold = 1.2` | Initial baseline confidence gating head |
