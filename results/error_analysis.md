# Stage 0 Diagnostic Error Analysis (5-Fold Stratified Cross-Validation)

**Evaluation Dataset**: Google GAP Development + Validation Combined ($N = 2,454$)
**Model Evaluated**: Baseline V2 36-Feature MLP (36 -> 32 -> 3, Adam $lr=0.001$, $\alpha=0.50$)

## 1. 5-Fold Cross-Validation Performance Summary

| Fold | Accuracy | Macro-F1 | Weighted-F1 | Weighted Loss |
|:---:|:---:|:---:|:---:|:---:|
| Fold 1 | 61.59% | 0.5564 | 0.6163 | 0.9258 |
| Fold 2 | 65.99% | 0.5968 | 0.6556 | 0.8991 |
| Fold 3 | 66.19% | 0.6051 | 0.6614 | 0.9057 |
| Fold 4 | 62.86% | 0.5630 | 0.6302 | 0.9730 |
| Fold 5 | 63.67% | 0.5643 | 0.6348 | 0.8911 |
| **Mean ± Std** | **64.06 ± 1.79%** | **0.5771 ± 0.0198** | **0.6396 ± 0.0166** | **0.9189 ± 0.0294** |

### Overall Out-Of-Fold (OOF) Metrics
- **OOF Accuracy**: 64.06%
- **OOF Macro-F1**: 0.5769
- **OOF Weighted-F1**: 0.6396

### Overall OOF Confusion Matrix
```
                 Predicted
                 A        B        Neither
Actual A              649      325       87
Actual B              238      819       73
Actual Neither         81       78      104
```

### Overall OOF Per-Class Breakdown
| Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| A (0) | 0.6705 | 0.6117 | 0.6397 | 1061 |
| B (1) | 0.6702 | 0.7248 | 0.6964 | 1130 |
| Neither (2) | 0.3939 | 0.3954 | 0.3947 | 263 |

---

## 2. Errors by Pronoun Type

| Pronoun | Total | Correct | Errors | Accuracy | % of Neither Correct |
|---|:---:|:---:|:---:|:---:|:---:|
| `he` | 466 | 307 | 159 | 65.88% | 38.10% (16/42) |
| `her` | 712 | 449 | 263 | 63.06% | 40.00% (26/65) |
| `him` | 124 | 64 | 60 | 51.61% | 23.53% (4/17) |
| `his` | 637 | 412 | 225 | 64.68% | 42.17% (35/83) |
| `she` | 515 | 340 | 175 | 66.02% | 41.07% (23/56) |

---

## 3. Errors by Sentence Locality (Context Distance)

| Context Configuration | Total | Correct | Accuracy | Neither Recall |
|---|:---:|:---:|:---:|:---:|
| At least one candidate in Same Sentence | 1663 | 1050 | 63.14% | 35.10% |
| Candidate in Immediately Preceding Sentence | 755 | 496 | 65.70% | 45.28% |
| Distant Sentence (> 1 sentence gap) | 36 | 26 | 72.22% | 50.00% |

---

## 4. Errors by Character Distance (Min Offset to Pronoun)

| Distance Range | Total | Correct | Accuracy |
|---|:---:|:---:|:---:|
| < 50 characters | 1612 | 1047 | 64.95% |
| 50 - 100 characters | 584 | 371 | 63.53% |
| 100 - 200 characters | 225 | 136 | 60.44% |
| 200+ characters | 33 | 18 | 54.55% |

---

## 5. Diagnostic Findings & Stage Roadmap

### Finding 1: The 'Neither' Bottleneck
- The minority class (Neither, $N = 263$) has by far the weakest F1 score (0.3947).
- 81 actual Neither samples were falsely classified as A, and 78 were falsely classified as B.
- Conversely, 87 A samples and 73 B samples were misclassified as Neither.
- **Implication for Stage 2 & 4**: We need lexical gender agreement features (e.g. matching name gender to pronoun gender) and OOF threshold/multiplier calibration to separate true coreferents from non-matching candidates.

### Finding 2: Candidate A vs. Candidate B Confusion
- Mutual confusion between A and B accounts for substantial error volume:
  - Actual A predicted as B: 325 samples.
  - Actual B predicted as A: 238 samples.
- **Implication for Stage 1**: Because A and B occupy distinct feature slots (0..17 vs 18..35), the MLP lacks structural symmetry. A shared-weight pair-scoring architecture (or A<->B swap augmentation) will enforce symmetry and improve discrimination between A and B.

### Finding 3: Gender and Grammatical Role Ambiguity
- Possessive pronouns (`his`, `her`) and ambiguous `her` (accusative vs possessive) exhibit distinct error patterns compared to nominative pronouns (`he`, `she`).
- **Implication for Stage 2**: Adding grammatical role features (nominative vs accusative vs possessive) and verb-parallelism heuristics will directly target these errors.
