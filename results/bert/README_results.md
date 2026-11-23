# Google GAP Pronoun Resolution — Benchmark Results

## Comparative Model Performance

The table below summarizes performance across the hand-crafted C++ pipelines and the transformer pipeline.
Per the project guidelines, all reported metrics derive strictly from executed scripts and saved JSON/log checkpoints.

| Model / Pipeline | Validation / OOF Accuracy | Held-Out Test Accuracy | Test Macro-F1 | Notes & Evidence |
|---|:---:|:---:|:---:|---|
| **C++ V2 Baseline (Exp 4 & 8)** | **63.22%** | Held Out | — | 36 distance/count features + 32-dim MLP. Checkpoints: `results/experiments/exp4_result.json`, `exp8_result.json`. |
| **C++ V3 (Exp 9 - Loose Mentions & Syntactic)** | **66.52%** | **68.85%** | **0.6391** | 56 features incorporating `findAllMentionsLoose`, URL title matching, syntactic heuristics. Checkpoint: `results/experiments/exp9_test_result.json`. |
| **DeBERTa-v3-large (Single Fold)** | *Pending GPU execution* | *Pending GPU execution* | *Pending GPU execution* | Expected ~89–91%. Hardware rule: Local host lacks CUDA GPU (`CUDA available: False`). Prepared in `notebooks/train_gap_kaggle_colab.ipynb`. |
| **DeBERTa-v3-large (5-Fold Ensemble + TTA)** | *Pending GPU execution* | *Pending GPU execution* | *Pending GPU execution* | Expected $\ge 91\%$. Symmetrical A $\leftrightarrow$ B swap augmentation and test-time augmentation. |
| **Hybrid (DeBERTa + V3 Handcrafted Features)** | *Pending GPU execution* | *Pending GPU execution* | *Pending GPU execution* | Feature dump ready in `data/processed/` with `--extra_features_*` flags. |

---

## Detailed C++ Evaluation Logs

### C++ V3 (Exp 9) Held-Out Test Set (N=2,000)
- **Accuracy**: 68.85% (1,377 / 2,000 correct predictions)
- **Macro-F1**: 0.6391
- **Weighted F1**: 0.6863
- **Weighted Loss**: 0.7507

#### Per-Class Breakdown:
| Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| **Candidate A (0)** | 0.7776 | 0.6285 | 0.6952 | 918 |
| **Candidate B (1)** | 0.6619 | 0.8082 | 0.7278 | 855 |
| **Neither (2)** | 0.5093 | 0.4802 | 0.4943 | 227 |

#### Confusion Matrix:
```
                 Predicted
                 A        B        Neither
Actual A         577      280      61
Actual B         120      691      44
Actual Neither   45       73       109
```

---

## Transformer Reproduction on GPU

To train the 5-fold `microsoft/deberta-v3-large` model and evaluate the held-out test set:

1. Open `notebooks/train_gap_kaggle_colab.ipynb` on Google Colab (select GPU runtime: T4 or V100/A100) or Kaggle (P100 / T4x2).
2. Run cells 1 through 7 end-to-end.
3. The script will save:
   - `models/bert_fold0.pt` ... `models/bert_fold4.pt`
   - `results/bert/train_log.csv`
   - `results/bert/oof_probs.npy`
   - `results/bert/test_probs.npy`
   - `results/bert/test_result.json`
