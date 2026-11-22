# Final Test Evaluation Report

## Configuration
- **Model**: MLP (80 -> 64 -> 3) with ReLU + Softmax
- **Features**: Clean V3 (80-dim: 40 per candidate x 2)
- **Training Data**: GAP Development (2000) + GAP Validation (454) = 2454 samples
- **Augmentation**: A<->B Swap (2x training data) + TTA at inference
- **Optimizer**: Adam (LR=0.0005, L2=0.0001)
- **Class Weighting**: Inverse-frequency^0.5
- **Calibration Multipliers**: Scale A=0.85, B=0.9, Neither=0.9
- **Best Epoch**: 10

## Test Set Performance (N = 2,000)

| Metric | Raw Inference | Calibrated Inference |
|---|:---:|:---:|
| **Accuracy** | **69.75%** | **69.75%** |
| **Macro-F1** | 0.6540 | 0.6554 |
| **Weighted-F1** | 0.6983 | 0.6982 |
| **Weighted Loss** | 0.7209 | 0.7217 |

### Per-Class Breakdown (Raw Inference)
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| A (0) | 0.8022 | 0.6405 | 0.7123 | 918 |
| B (1) | 0.6756 | 0.7965 | 0.7311 | 855 |
| Neither (2) | 0.4865 | 0.5551 | 0.5185 | 227 |

### Confusion Matrix (Raw Inference)
```
                 Predicted
                 A        B        Neither
Actual A              588      257       73
Actual B              114      681       60
Actual Neither         31       70      126
```
