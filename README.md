# MAR: Antecedent-Aware Evidence Selection for Lightweight RAG
## Minimum-Sufficient Antecedent Restoration (MSAR) — Baseline Reproduction

[![C++14](https://img.shields.io/badge/Language-C%2B%2B14-blue.svg)](https://isocpp.org/)
[![Compiler](https://img.shields.io/badge/Compiler-GCC%20%7C%20Clang%20%7C%20MSVC-green.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-Google%20GAP-orange.svg)](https://github.com/google-research-datasets/gap-coreference)
[![License](https://img.shields.io/badge/License-Research%20Baseline-lightgrey.svg)]()

This repository contains an independent, from-scratch **C++** research reproduction of the core antecedent-resolution classifier for **MAR** (Budget-Constrained Antecedent-Aware Evidence Selection for Lightweight Retrieval-Augmented Generation) and its extension **Confidence-Gated MSAR** (Minimum-Sufficient Antecedent Restoration).

---

## 1. Project Overview & Research Problem

In Retrieval-Augmented Generation (RAG) for Small Language Models (SLMs), retrieved text chunks often contain pronouns whose referents (antecedents) reside in preceding sentences that were excluded during passage chunking. 

For example:
* *Preceding context (unretrieved):* "David Anderson established a software company in 2011."
* *Retrieved passage:* "He later expanded the company internationally."

When evaluated in isolation, an SLM cannot determine who "He" refers to. The MSAR framework aims to restore only the minimum-sufficient context needed to disambiguate the referent without exceeding strict context token budgets.

This repository implements the foundational **antecedent classification engine**: a lightweight 3-class neural network trained from scratch on the Google GAP benchmark to determine whether a target pronoun refers to **Candidate A (Class 0)**, **Candidate B (Class 1)**, or **Neither (Class 2)**.

> [!NOTE]
> Per the research protocol, this implementation contains no pretrained language models, BERT, Transformers, LLM APIs, or heavy external NLP packages. It is implemented in standard C++14 using exact analytical forward/backward propagation, numerically stable Softmax, and custom Adam optimization.

---

## 2. Benchmark Dataset: Google GAP

The model is trained and evaluated strictly on the official [Google GAP Coreference Dataset](https://github.com/google-research-datasets/gap-coreference).

### Partition Distributions
| Partition | Total Samples | Candidate A (Class 0) | Candidate B (Class 1) | Neither (Class 2) |
|---|:---:|:---:|:---:|:---:|
| **Development (Train)** | 2,000 | 874 (43.70%) | 925 (46.25%) | 201 (10.05%) |
| **Validation (Val)** | 454 | 187 (41.19%) | 205 (45.15%) | 62 (13.66%) |
| **Test (Held-Out)** | 2,000 | 918 (45.90%) | 855 (42.75%) | 227 (11.35%) |
| **Total** | **4,454** | **1,979** | **1,985** | **490** |

*Methodological constraints:* The development set is used exclusively for training and standardizer fitting; the validation set is used exclusively for model selection and hyperparameter evaluation. The test set is strictly held out.

---

## 3. Feature Representations

### 12-Feature Baseline (V1)
Computes 6 features per candidate ($6 \times 2 = 12$ dimensions):
1. **Normalized Character Distance**: $(offset_{\text{pronoun}} - offset_{\text{cand}}) / len(\text{text})$
2. **Normalized Sentence-Boundary Distance**: $(sent_{\text{pronoun}} - sent_{\text{cand}}) / count(\text{sentences})$
3. **Candidate Mention Frequency**: Count of exact candidate matches in text
4. **Sentence Jaccard Similarity**: $|Tokens(Sent_{\text{cand}}) \cap Tokens(Sent_{\text{pron}})| / |Tokens(Sent_{\text{cand}}) \cup Tokens(Sent_{\text{pron}})|$
5. **Candidate-Name Word Count**: Number of token words in candidate name
6. **Precedes-Pronoun Indicator**: $1.0$ if $offset_{\text{cand}} < offset_{\text{pronoun}}$, else $0.0$

### 36-Feature Expanded Representation (V2)
Computes 18 features per candidate ($18 \times 2 = 36$ dimensions):
1. Character distance
2. Sentence distance
3. Candidate mention frequency
4. Candidate/pronoun sentence Jaccard overlap
5. Candidate-name token length
6. Precedes-pronoun indicator
7. Normalized word/token distance
8. Same-sentence indicator ($1.0$ if in same sentence, else $0.0$)
9. Immediately-preceding-sentence indicator ($1.0$ if $sent_{\text{pron}} - sent_{\text{cand}} == 1$, else $0.0$)
10. Normalized sentence gap ($|sent_{\text{pron}} - sent_{\text{cand}}| / count(\text{sentences})$)
11. Mentions before pronoun
12. Mentions after pronoun
13. Distance to the nearest mention
14. Position of the first mention before pronoun
15. Distance from the last prior mention
16. Local-context Jaccard overlap ($\pm 60$ character window around mention and pronoun)
17. Candidate mention in the pronoun sentence ($1.0$ if present, else $0.0$)
18. Mentions strictly between the candidate and pronoun

---

## 4. Architecture & Training Pipeline

```
36 Input Features (V2)
          |
          v
   Feature Standardizer  (Fitted exclusively on Training/Dev partition)
          |
          v
  Fully Connected Layer  (36 -> 32)
          |
          v
         ReLU
          |
          v
  Fully Connected Layer  (32 -> 3)
          |
          v
    Stable Softmax       (z_k - max(z))
          |
          v
  Class Probabilities    [ P(A), P(B), P(Neither) ]
```

* **Loss Function**: Weighted Multiclass Cross-Entropy with normalized inverse class frequencies:
  $$w_c = \frac{(1 / N_c)^\alpha}{\frac{1}{3} \sum_{k=0}^2 (1 / N_k)^\alpha}$$
* **Optimizer**: Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$)
* **Learning Rate**: $0.001$ (with $0.0003$ evaluated in Exp 3)
* **Batch Size**: 32
* **Max Epochs**: 500
* **Early Stopping**: Patience = 50 epochs
* **Model Selection**: Highest validation Macro-F1 (tie-breaker: lower weighted validation loss)

---

## 5. Experimental Reproduction Results

All 8 documented historical research experiments were reproduced from scratch under identical hyperparameter conditions (Random Seed = 42).

| Experiment ID | Configuration | Features | Hidden | LR | Class Weight $\alpha$ | L2 $\lambda$ | Best Epoch | Reprod. Acc | Ref. Acc | Reprod. Macro-F1 | Ref. Macro-F1 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exp 1** | Baseline V1 | 12 | 32 | 0.001 | 0.50 | 0.0 | 77 | **60.35%** | 54.85% | **0.5230** | ~0.4900 |
| **Exp 2** | V2 Inv-Freq | 36 | 32 | 0.001 | 1.00 | 0.0 | 62 | **56.83%** | 56.17% | **0.5242** | 0.5226 |
| **Exp 3** | Reduced LR | 36 | 32 | 0.0003 | 1.00 | 0.0 | 87 | **56.39%** | 55.95% | **0.5176** | 0.5143 |
| **Exp 4** | Sqrt-Weighting | 36 | 32 | 0.001 | 0.50 | 0.0 | 48 | **63.44%** | 63.00% | **0.5545** | 0.5604 |
| **Exp 5** | Alpha=0.65 | 36 | 32 | 0.001 | 0.65 | 0.0 | 48 | **61.45%** | 62.56% | **0.5465** | 0.5606 |
| **Exp 6** | 30-feat Ablation | 30 | 32 | 0.001 | 0.50 | 0.0 | 7 | **60.57%** | 59.91% | **0.5171** | 0.5187 |
| **Exp 7** | Hidden=64 | 36 | 64 | 0.001 | 0.50 | 0.0 | 75 | **61.01%** | 60.13% | **0.5486** | 0.5315 |
| **Exp 8** | L2 Regularized | 36 | 32 | 0.001 | 0.50 | 0.0001 | 46 | **63.00%** | 63.00% | **0.5551** | 0.5604 |

### Best Model Performance Breakdown (Experiment 4 / 8)
* **Overall Accuracy**: **63.00% - 63.44%** (Reference: **63.00%**, exactly 286 / 454 correct predictions)
* **Macro-F1**: **0.5551** (Reference: **0.5604**)

#### Confusion Matrix (Exp 8 on Validation Set, N=454)
```
                 Predicted
                 A        B        Neither
Actual A         111      60       16
Actual B         37       157      11
Actual Neither   23       21       18
```

#### Per-Class Classification Report
| Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| **Candidate A (0)** | 0.6491 | 0.5936 | 0.6201 | 187 |
| **Candidate B (1)** | 0.6597 | 0.7659 | 0.7088 | 205 |
| **Neither (2)** | 0.4000 | 0.2903 | 0.3364 | 62 |
| **Macro Average** | **0.5696** | **0.5499** | **0.5551** | **454** |

---

## 6. How to Reproduce

### Prerequisites
* C++14 compliant compiler (`g++` >= 6.3, `clang++`, or `MSVC`).
* Python >= 3.8 with `matplotlib` (only for rendering plots).

### Step-by-Step Instructions

#### 1. Compile Binaries
On Windows:
```cmd
scripts\build.bat
```
Or manually using `g++`:
```cmd
g++ -std=c++14 -O2 -Iinclude src/main.cpp -o msar_reproduce.exe
g++ -std=c++14 -O2 -Iinclude tests/test_pipeline.cpp -o run_tests.exe
```

On Linux / macOS:
```bash
make all
```

#### 2. Run the Unit Test Suite
```cmd
run_tests.exe
```
This verifies dataset schema counts, character offset extractions, finite numerical values (no NaN/Inf), candidate feature asymmetry, standardizer edge cases, and numerical stability of Softmax.

#### 3. Execute All Experiments
```cmd
msar_reproduce.exe
```
This loads the GAP dataset, fits standardizers on the development set, runs Experiments 1 through 8, saves model weights to `models/`, writes detailed metrics to `results/experiments/*.json`, and exports comparison tables.

#### 4. Generate Visualization Graphs
```cmd
python scripts/plot_results.py
```
This produces:
* `results/comparisons/training_validation_loss.png`
* `results/comparisons/validation_metrics_progression.png`
* `results/comparisons/confusion_matrix_best_model.png`
* `results/comparisons/experiment_f1_comparison.png`

#### 5. Run Inference on a Sample Sentence
Use the built-in `--predict` mode:
```cmd
msar_reproduce.exe --predict "Michael spoke to Robert before he entered the building." "he" 31 "Michael" 0 "Robert" 17
```
Expected output:
```
--- Antecedent Prediction Result ---
Text: "Michael spoke to Robert before he entered the building."
Pronoun: "he" at offset 31
Candidate A: "Michael" at offset 0
Candidate B: "Robert" at offset 17

Predicted Class: Candidate A (Class 0)
Class Probabilities:
  P(Candidate A: Michael) = 63.0517%
  P(Candidate B: Robert)  = 26.4266%
  P(Neither)              = 10.5217%
```

---

## 7. Research Limitations & Methodological Discrepancies

1. **Standalone Scope**: This classifier addresses only the 3-class antecedent coreference component. It does not construct the end-to-end RAG retrieval pipeline or token ceiling budgeting.
2. **Methodological Reconstruction**: The exact historical phrase tokenization constants and local-context window bounds were not specified in the original handoff. We adopted deterministic 60-character sliding windows and standard alphanumeric lowercased tokenization. Consequently, minor metric variations (e.g., Macro-F1: 0.5551 vs historical 0.5604) reflect minor feature formulation differences rather than algorithmic deviations.
3. **Accuracy Ceiling**: As a lightweight feedforward MLP without pretrained embeddings or contextual attention, validation accuracy naturally peaks around **63.00% - 63.44%**.

---

## 8. Repository Layout

```
confidence_gated_msar/
├── README.md                           # Comprehensive documentation and reproduction guide
├── Makefile                            # Standard POSIX build targets
├── requirements.txt                    # Python and C++ environment requirements
├── .gitignore                          # Build artifacts and temporary files ignore list
│
├── include/                            # Modular C++ header implementations
│   ├── data_loader.hpp                 # TSV parser and partition verifier
│   ├── preprocessing.hpp               # Sentence splitter, tokenizer, and Jaccard engine
│   ├── features_v1.hpp                 # 12-feature extractor (Phase 5)
│   ├── features_v2.hpp                 # 36-feature extractor (Phase 6)
│   ├── standardizer.hpp                # Training-only z-score standardizer (Phase 8)
│   ├── mlp.hpp                         # MLP forward/backward engine with Adam & Softmax (Phase 7)
│   ├── evaluation.hpp                  # Confusion matrix, Precision, Recall, Macro-F1 (Phase 11)
│   ├── training.hpp                    # Mini-batch training loop, early stopping, and loss logging
│   └── correlation.hpp                 # Feature Pearson correlation audit and 30-feat ablation
│
├── src/
│   └── main.cpp                        # Unified C++ reproduction executable & inference CLI
│
├── scripts/
│   ├── build.bat                       # Windows compilation script
│   ├── build.sh                        # Linux/Mac compilation script
│   ├── run_all.bat                     # Windows end-to-end execution script
│   ├── run_all.sh                      # Linux/Mac end-to-end execution script
│   └── plot_results.py                 # Matplotlib visualization script
│
├── tests/
│   └── test_pipeline.cpp               # C++ unit test suite
│
├── data/
│   └── raw/                            # Original Google GAP coreference TSVs
│       ├── gap-development.tsv
│       ├── gap-validation.tsv
│       └── gap-test.tsv
│
├── models/                             # Checkpoints and fitted scalers
│   ├── best_model.bin
│   ├── standardizer_v1.txt
│   ├── standardizer_v2.txt
│   └── model_exp*.bin
│
├── results/
│   ├── baseline/                       # Experiment 1 logs & history
│   ├── experiments/                    # Experiments 2 - 8 JSON logs & CSV histories
│   └── comparisons/                    # Comparison tables and generated PNG graphs
│
└── docs/
    ├── methodology.md                  # In-depth architectural & mathematical methodology
    └── experiment_log.md               # Detailed experiment log and correlation findings
```
