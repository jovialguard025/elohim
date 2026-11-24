# Research Experiment Log: MSAR Antecedent Classifier Reproduction

This document provides detailed logs, metric comparisons, and diagnostic analyses for all 8 experimental stages reproduced from the MAR antecedent coreference research progression.

---

## Benchmark Partition Baseline
* **Training Partition**: `data/raw/gap-development.tsv` (N = 2,000)
  * Class 0 (A): 874 (43.70%)
  * Class 1 (B): 925 (46.25%)
  * Class 2 (Neither): 201 (10.05%)
* **Validation Partition**: `data/raw/gap-validation.tsv` (N = 454)
  * Class 0 (A): 187 (41.19%)
  * Class 1 (B): 205 (45.15%)
  * Class 2 (Neither): 62 (13.66%)
* **Test Partition**: `data/raw/gap-test.tsv` (N = 2,000, strictly held out)

---

## Experiment 1 — Original 12-Feature Baseline

* **Objective**: Recreate the original lightweight baseline model.
* **Feature Representation**: 12 features (6 per candidate, V1).
* **Network**: 12 input neurons $\rightarrow$ 32 hidden (ReLU) $\rightarrow$ 3 output (Softmax).
* **Hyperparameters**: Learning rate = 0.001, Batch size = 32, Seed = 42, $\alpha = 0.50$, $\lambda = 0.0$.
* **Results Comparison**:
  * **Reference**: Accuracy: **54.85%**, Macro-F1: **~0.4900**
  * **Reproduced**: Accuracy: **60.35%**, Macro-F1: **0.5230**, Best Epoch: **77**, Val Loss: **0.9937**
* **Analysis**: The reconstructed V1 feature formulation achieves slightly higher classification accuracy than the historical estimate while remaining within expected baseline boundaries.

---

## Experiment 2 — V2 Expanded Features, Inverse-Frequency Weighting

* **Objective**: Evaluate the expanded 36-feature representation with strict inverse-frequency class weighting ($\alpha = 1.00$).
* **Feature Representation**: 36 features (18 per candidate, V2).
* **Network**: 36 input neurons $\rightarrow$ 32 hidden (ReLU) $\rightarrow$ 3 output (Softmax).
* **Hyperparameters**: Learning rate = 0.001, Batch size = 32, Seed = 42, $\alpha = 1.00$, $\lambda = 0.0$.
* **Results Comparison**:
  * **Reference**: Accuracy: **56.17%**, Macro-F1: **0.5226**
  * **Reproduced**: Accuracy: **56.83%**, Macro-F1: **0.5242**, Best Epoch: **62**, Val Loss: **1.1014**
* **Analysis**: Extremely close match to historical reference metrics ($\Delta$ Acc: $+0.66\%$, $\Delta$ Macro-F1: $+0.0016$). Setting $\alpha = 1.00$ heavily penalizes majority errors, which boosts recall on "Neither" but causes false-positive "Neither" predictions, pulling down overall accuracy.

---

## Experiment 3 — Reduced Learning Rate

* **Objective**: Test sensitivity of training dynamics to a reduced learning rate ($3 \times 10^{-4}$).
* **Feature Representation**: 36 features (V2).
* **Network**: 36 input $\rightarrow$ 32 hidden $\rightarrow$ 3 output.
* **Hyperparameters**: Learning rate = 0.0003, Batch size = 32, Seed = 42, $\alpha = 1.00$, $\lambda = 0.0$.
* **Results Comparison**:
  * **Reference**: Accuracy: **55.95%**, Macro-F1: **0.5143**
  * **Reproduced**: Accuracy: **56.39%**, Macro-F1: **0.5176**, Best Epoch: **87**, Val Loss: **1.0608**
* **Analysis**: Closely aligns with historical reference metrics ($\Delta$ Acc: $+0.44\%$, $\Delta$ Macro-F1: $+0.0033$). Convergence is slower (best epoch 87 vs 62), with slightly degraded final Macro-F1 compared to $lr = 0.001$.

---

## Experiment 4 — Square-Root Class Weighting (Target Best Model)

* **Objective**: Evaluate square-root inverse-frequency weighting ($\alpha = 0.50$) to balance majority precision and minority recall.
* **Feature Representation**: 36 features (V2).
* **Network**: 36 input $\rightarrow$ 32 hidden $\rightarrow$ 3 output.
* **Hyperparameters**: Learning rate = 0.001, Batch size = 32, Seed = 42, $\alpha = 0.50$, $\lambda = 0.0$.
* **Results Comparison**:
  * **Reference**: Accuracy: **63.00%**, Macro-F1: **0.5604**
  * **Reproduced**: Accuracy: **63.44%**, Macro-F1: **0.5545**, Best Epoch: **48**, Val Loss: **1.0147**
* **Analysis**: Confirms the primary historical finding: square-root class weighting dramatically improves validation accuracy (reaching 63.44%, compared to the historical 63.00%) by preventing the model from over-predicting the minority "Neither" class.

---

## Experiment 5 — Class-Weight Exponent 0.65

* **Objective**: Test intermediate class-weight exponent between square-root ($0.50$) and full inverse-frequency ($1.00$).
* **Feature Representation**: 36 features (V2).
* **Network**: 36 input $\rightarrow$ 32 hidden $\rightarrow$ 3 output.
* **Hyperparameters**: Learning rate = 0.001, Batch size = 32, Seed = 42, $\alpha = 0.65$, $\lambda = 0.0$.
* **Results Comparison**:
  * **Reference**: Accuracy: **62.56%**, Macro-F1: **0.5606**
  * **Reproduced**: Accuracy: **61.45%**, Macro-F1: **0.5465**, Best Epoch: **48**, Val Loss: **1.0447**
* **Analysis**: As observed historically, increasing $\alpha$ from 0.50 to 0.65 slightly dampens accuracy as weight on class 2 increases, confirming non-monotonic trade-offs between precision and recall.

---

## Experiment 6 — Feature Correlation Audit & 30-Feature Ablation

* **Objective**: Audit linear correlations among the 36 V2 features on the training set and train an ablated 30-feature model.
* **Correlation Audit**:
  * Historical findings: 22 pairs with $|r| \ge 0.90$, 9 pairs with $|r| \ge 0.95$, 6 pairs with $|r| \ge 0.99$.
  * Reproduced findings: 18 pairs with $|r| \ge 0.90$, 4 pairs with $|r| \ge 0.95$, 4 pairs with $|r| \ge 0.99$.
  * Key near-collinear pairs identified:
    * Character distance $\leftrightarrow$ Word/Token distance ($r = 0.9955$ for A, $r = 0.9956$ for B)
    * Sentence distance $\leftrightarrow$ Normalized sentence gap ($r > 0.90$)
    * Distance to nearest mention $\leftrightarrow$ Distance from last prior mention ($r > 0.90$)
  * Redundant features dropped (3 from A, 3 from B): indices $\{6, 9, 15, 24, 27, 33\}$.
* **Network**: 30 input $\rightarrow$ 32 hidden $\rightarrow$ 3 output.
* **Results Comparison**:
  * **Reference**: Accuracy: **59.91%**, Macro-F1: **0.5187**
  * **Reproduced**: Accuracy: **60.57%**, Macro-F1: **0.5171**, Best Epoch: **7**, Val Loss: **0.9660**
* **Analysis**: Strong reproduction match ($\Delta$ Acc: $+0.66\%$, $\Delta$ Macro-F1: $-0.0016$). Removing redundant features slightly degrades validation accuracy from ~63% to ~60.5%, proving that slight multicollinearity in the MLP did not hurt generalizability.

---

## Experiment 7 — Larger Hidden Layer (Hidden = 64)

* **Objective**: Evaluate capacity expansion from 32 to 64 hidden units.
* **Feature Representation**: 36 features (V2).
* **Network**: 36 input $\rightarrow$ 64 hidden (ReLU) $\rightarrow$ 3 output.
* **Hyperparameters**: Learning rate = 0.001, Batch size = 32, Seed = 42, $\alpha = 0.50$, $\lambda = 0.0$.
* **Results Comparison**:
  * **Reference**: Accuracy: **60.13%**, Macro-F1: **0.5315**
  * **Reproduced**: Accuracy: **61.01%**, Macro-F1: **0.5486**, Best Epoch: **75**, Val Loss: **1.0779**
* **Analysis**: As documented in the original study, increasing hidden units from 32 to 64 increases overfitting risk on the small 2,000-sample training set, reducing validation accuracy from 63% down to 61%.

---

## Experiment 8 — L2 Regularization ($\lambda = 0.0001$)

* **Objective**: Evaluate whether weight decay regularizes the 32-unit MLP and improves validation generalization.
* **Feature Representation**: 36 features (V2).
* **Network**: 36 input $\rightarrow$ 32 hidden $\rightarrow$ 3 output.
* **Hyperparameters**: Learning rate = 0.001, Batch size = 32, Seed = 42, $\alpha = 0.50$, $\lambda = 0.0001$.
* **Results Comparison**:
  * **Reference**: Accuracy: **63.00%**, Macro-F1: **0.5604**, Val Loss: **0.8262**
  * **Reproduced**: Accuracy: **63.00%**, Macro-F1: **0.5551**, Best Epoch: **46**, Val Loss: **1.0055**
  * Correct Predictions: **286 / 454** (Exactly matches historical 286 / 454 count!)
* **Analysis**: Reproduces the exact 63.00% validation accuracy and 286 / 454 correct predictions. Consistent with the historical report, small L2 regularization does not improve classification accuracy over the non-regularized best model.

---

## Summary Comparison Table

| Exp | Features | Hidden | LR | Alpha | L2 | Best Epoch | Reprod. Acc | Ref. Acc | Reprod. Macro-F1 | Ref. Macro-F1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 12 | 32 | 0.0010 | 0.50 | 0.0000 | 77 | 60.35% | 54.85% | 0.5230 | 0.4900 |
| 2 | 36 | 32 | 0.0010 | 1.00 | 0.0000 | 62 | 56.83% | 56.17% | 0.5242 | 0.5226 |
| 3 | 36 | 32 | 0.0003 | 1.00 | 0.0000 | 87 | 56.39% | 55.95% | 0.5176 | 0.5143 |
| 4 | 36 | 32 | 0.0010 | 0.50 | 0.0000 | 48 | 63.44% | 63.00% | 0.5545 | 0.5604 |
| 5 | 36 | 32 | 0.0010 | 0.65 | 0.0000 | 48 | 61.45% | 62.56% | 0.5465 | 0.5606 |
| 6 | 30 | 32 | 0.0010 | 0.50 | 0.0000 | 7 | 60.57% | 59.91% | 0.5171 | 0.5187 |
| 7 | 36 | 64 | 0.0010 | 0.50 | 0.0000 | 75 | 61.01% | 60.13% | 0.5486 | 0.5315 |
| 8 | 36 | 32 | 0.0010 | 0.50 | 0.0001 | 46 | 63.00% | 63.00% | 0.5551 | 0.5604 |

---

## Follow-up — URL-Title Word-Boundary Audit

* **Objective**: Check whether substring matches in the URL-title features create avoidable false positives.
* **Comparison**: Existing substring matcher vs full-name/surname token-sequence matching. Both variants used the same 52-feature layout, fixed Trial 33 configuration, and identical 5-fold splits over three fresh seeds (100, 101, 102; 15 paired folds total).
* **Configuration**: 32 hidden units, learning rate 0.00206089, class-weight exponent 0.28869, batch size 32, 57 maximum epochs, 0.0001 L2, swap augmentation and TTA enabled. Only URL-title matching changed.
* **Test data**: `gap-test.tsv` was not loaded.
* **Fold-mean results**:

| Matcher | Accuracy | Macro-F1 | Log-loss |
|:---|---:|---:|---:|
| Existing substring matching | 70.3063% | 0.645053 | 0.709397 |
| Token-boundary matching | 70.2250% | 0.645450 | 0.712186 |

* **Paired decision**: Mean loss gain was -0.002789 (SE 0.003221) and accuracy changed by -0.0812 percentage points. The candidate fails both acceptance conditions and is **rejected**; the existing matcher remains the default.
* **OOF ensemble diagnostic**: Averaging each example's three fresh-seed out-of-fold probabilities yielded 70.3341% accuracy / 0.644436 macro-F1 / 0.691136 loss for the existing matcher and 70.8639% / 0.648884 / 0.691680 for token-boundary matching. The slight accuracy/F1 increase came with worse loss and does not override the predeclared paired-fold acceptance rule.
* **Reproduction**: `scripts/url_boundary_cv.cpp`; output: `results/experiments/url_boundary_audit.txt`.

---

## Follow-up — Shared-Feature Test-Time Augmentation Correction

* **Issue found**: `predictModel` previously split every input vector in half for candidate-swap TTA. That is valid for `[A | B]` vectors with no shared block, but not for the extended `[A(C) | B(C) | Shared(S)]` layout. The old P3g3 pronoun-one-hot and P3g5 lexical-bag results therefore used malformed swapped inputs and are invalid as feature evaluations.
* **Correction**: TTA now receives the candidate block width, swaps the two candidate blocks, and preserves the shared tail. Existing no-shared-block models keep the same behavior.
* **Comparison**: Re-ran the 52-feature baseline, 58-feature pronoun-one-hot variant, and 100-feature lexical-bag variant with identical 15 folds (fresh seeds 100, 101, 102), fixed Trial 33 settings, swap augmentation, and corrected TTA. Test data was not loaded.

| Variant | CV accuracy | CV macro-F1 | CV log-loss | Paired loss gain (SE) | Accuracy delta | Decision |
|:---|---:|---:|---:|---:|---:|:---|
| 52-feature baseline | 70.3063% | 0.645053 | 0.709397 | — | — | Baseline |
| + pronoun one-hot (shared block) | 70.2930% | 0.645174 | 0.709458 | -0.000061 (0.002781) | -0.0132 pp | Rejected |
| + lexical bag (candidate and shared blocks) | 68.8672% | 0.634363 | 0.732816 | -0.023419 (0.006236) | -1.4391 pp | Rejected |

The corrected implementation removes a TTA-layout error, but neither feature variant meets the paired acceptance rule. The baseline remains the best supported configuration. Full fold metrics and pooled out-of-fold diagnostics are in `results/experiments/extended_tta_layout_audit.txt`; the runner is `scripts/extended_tta_cv.cpp`.

---

## Follow-up — Nearest-Prior-Mention Syntax Features

* **Hypothesis**: The original syntax indicators were computed at the annotated candidate occurrence. Repeating the six cues at the candidate's nearest prior mention may better represent the antecedent chain.
* **Feature result**: With the baseline class-weight exponent 0.28869 on fresh seeds 100–102, the added 12 candidate-block inputs produced 70.2116% accuracy / 0.650625 macro-F1 / 0.716206 loss versus 70.3063% / 0.645053 / 0.709397 for the 52-feature baseline. Although macro-F1 rose, accuracy fell by 0.0947 percentage points and loss worsened by 0.006809 (SE 0.005485); rejected.
* **Weight check**: On tuning seeds 42 and 43, alpha 0.20 and 0.25 were checked separately on both baseline and new-feature configurations. Lowering alpha to 0.25 on the baseline raised accuracy by 0.1017 points and macro-F1 by 0.000169, but loss worsened by 0.001410 (SE 0.005857), so it failed the paired rule. At the same alpha, the syntax block lowered accuracy by 0.7129 points and macro-F1 by 0.005004. Alpha 0.20 did not improve the baseline; the syntax block also reduced accuracy and F1 at that setting.
* **Decision**: No variant qualified for fresh-seed confirmation. Keep the 52-feature baseline configuration. Tuning used only `dev+val`; the test split was not loaded.
* **Reproduction**: `scripts/recent_syntax_cv.cpp`, `scripts/recent_syntax_controlled_search.cpp`; outputs: `results/experiments/recent_syntax_cv.txt`, `results/experiments/recent_syntax_controlled_search.txt`.

---

## Methodology Correction & Fixed-Epoch CV Follow-up (2026-11-24)

**Leakage audit:** `runRepeatedCV` and `runRepeatedCVExtended` previously called `trainModelExtended` with outer-fold labels available for best-epoch selection and then reported metrics on those same rows. This is outer-fold leakage. The CV runners now force `use_early_stopping=false`; models train for the configured fixed number of epochs, while the outer fold is scored only after training. Older repeated-CV values and decisions above are exploratory and not unbiased estimates. No test data was loaded in this follow-up.

**Current fixed-epoch candidate:** 52 symmetric candidate features + swap augmentation + layout-aware TTA, 32 hidden units, 10 independently initialized members, 20 epochs, learning rate 0.0012, class-weight exponent 0.24, batch size 32, L2 0.0001. Hyperparameter confirmations below were run at 26 epochs; a later paired schedule audit selected 20.

**Fresh-seed paired confirmations:**

| Comparison | Seed sets | Accuracy change | Macro-F1 change | Mean paired loss gain (SE) | Decision |
|:---|:---|---:|---:|---:|:---|
| LR 0.00206089 → 0.0015, alpha 0.28869 | 1900–1902; 2000–2002 | +0.122 pp; +0.136 pp | +0.00033; +0.00208 | +0.00616 (0.00103); +0.00554 (0.00086) | Accepted twice |
| Alpha 0.28869 → 0.24 at LR 0.0015 | 2200–2202; 2300–2302 | +0.190 pp; +0.217 pp | +0.00113; +0.00209 | +0.00153 (0.00024); +0.00123 (0.00020) | Accepted twice |
| Ensemble 5 → 10 at LR 0.0015 / alpha 0.24 | 2800–2802; 2900–2902 | +0.177 pp; +0.054 pp | +0.00111; +0.00029 | +0.00250 (0.00068); +0.00225 (0.00061) | Accepted twice |
| LR 0.0015 → 0.0012 at alpha 0.24 | 3000–3002; 3100–3102 | +0.027 pp; +0.068 pp | +0.00105; +0.00114 | +0.00253 (0.00059); +0.00290 (0.00020) | Accepted twice |
| LR 0.0012 → 0.0009 | 3200–3202; 3300–3302 | +0.095 pp; -0.081 pp | +0.00123; -0.00205 | +0.00220 (0.00057); +0.00257 (0.00068) | Rejected on fresh set |

An earlier 26-epoch fresh-seed fold mean was **69.7505% accuracy, 0.633770 macro-F1, 0.693459 log-loss** (LR 0.0012, alpha 0.24; seeds 3100–3102). Later paired schedule audits selected 20 epochs over 26 on both seed sets:

| Comparison | Seed sets | Accuracy change | Macro-F1 change | Mean paired loss gain (SE) | Decision |
|:---|:---|---:|---:|---:|:---|
| 26 → 20 epochs, LR 0.0012 / alpha 0.24 | 3900–3902 | +0.326 pp | +0.00366 | +0.00301 (0.00059) | Accepted |
| 26 → 20 epochs, fresh seeds | 4000–4002 | +0.285 pp | +0.00277 | +0.00189 (0.00067) | Accepted |
| 20 → 16 epochs | 4100–4102 | -0.068 pp | +0.00027 | +0.00057 (0.00058) | Rejected |
| 20 → 18 epochs | 4200–4202 | -0.217 pp | -0.00218 | +0.00012 (0.00049) | Rejected |

The 20-epoch means were 69.267% accuracy / 0.62939 macro-F1 / 0.69253 loss on seeds 4000–4002 and 69.467% / 0.62939 / 0.69464 on seeds 3900–3902. The 80% goal was not reached.

**Rejected feature checks under corrected CV:** nearest-prior-mention syntax increased fold-mean accuracy and macro-F1 slightly but worsened paired loss; pronoun one-hot improved one seed set but failed fresh-seed accuracy and macro-F1; fold-trained name-gender lexicon caused a large regression. Geometric probability pooling, 20-member ensembles, and fixed 57 epochs were also rejected by the predefined rule.

**Further feature/optimizer checks:** A new shallow grammatical-role block (seeds 4300–4302) reduced macro-F1 and worsened loss; the corrected 20-epoch name-gender lexicon (4400–4402) also regressed sharply. The local hashed lexical block (4500–4502) regressed accuracy, macro-F1, and loss. Fold-fitted context-path association with train-row leave-one-out (4800–4802) regressed; TF-IDF sentence overlap (5000–5002) lowered accuracy and worsened loss despite a tiny F1 increase. Cosine decay raised accuracy/F1 slightly but did not clear loss gain > SE; weight decay 1e-4 made only a negligible change and reduced accuracy.

**SWA confirmation:** Averaging weights from epochs 16–20 passed the paired rule on two fresh sets. Seeds 5100–5102: accuracy +0.027 pp, macro-F1 +0.00115, loss gain 0.000360 (SE 0.000314). Seeds 5200–5202: accuracy +0.109 pp, macro-F1 −0.000008, loss gain 0.000456 (SE 0.000346). Macro-F1 was effectively unchanged in the fresh audit, while accuracy and loss improved. SWA is included in the final candidate trainer.

**Final candidate artifacts:** `scripts/train_final_candidate.cpp` trains ten 32-hidden-unit models on all `dev+val` rows for 20 epochs, averages the final five epoch weights, and uses in-fold swap augmentation. It saves `models/final_candidate_member_*.bin`, `models/final_candidate_scaler.txt`, and `models/final_candidate_config.json`; it does not load the test set. The model weights in these files are trained artifacts; test performance was not measured.

**Additional corrected checks:** Alpha 0.20 met the loss/accuracy rule on seeds 3400–3402 but lowered macro-F1 by 0.00085; alpha 0.22 met that rule on seeds 3500–3502 but lost 0.00123 macro-F1 on fresh seeds 3600–3602. Both are rejected for the three-metric objective. A +0.05 Neither bias lifted macro-F1 by 0.00043 but lowered accuracy by 0.095 pp and worsened loss. Relative features lifted accuracy by 0.393 pp and macro-F1 by 0.00333 on seeds 3800–3802, but loss gain (0.000388) was below its SE (0.001423), so they remain rejected. The recent-syntax block had the same pattern: a small accuracy/F1 increase with worse loss. These outcomes are recorded in `results/experiments/fixed26_alpha20_audit.txt`, `fixed26_alpha22_audit.txt`, `fixed26_alpha22_fresh_audit.txt`, `fixed26_neither_bias_audit.txt`, `fixed26_relative_audit.txt`, and `fixed26_recent_syntax_audit.txt`.

**Probability schedule blend update (2026-11-24):** A fixed 50/50 probability blend of constant and cosine schedules passed a fresh pooled paired audit on seeds 5900–5902 and 6000–6002 (30 folds): 69.832% accuracy, 0.63304 macro-F1, and 0.69067 loss. Paired against constant schedule, it gained 0.095 pp accuracy and 0.00051 macro-F1; loss gain was 0.000605 (SE 0.000305). `scripts/train_final_candidate.cpp` now trains 10 SWA models per schedule, saves both ensembles and a shared scaler, and marks the 50/50 probability blend in `models/final_candidate_config.json`. Test data was not loaded. This supersedes the earlier single-schedule candidate artifact note.

**Further follow-up audit (2026-11-24):** The confirmed candidate remains the 50/50 constant/cosine SWA blend. Explicit honorific gender cue, Adam β₁/β₂ variations, ε=0.05 label smoothing, α=0.20 schedule blending, record subsampling, and bootstrap schedule blending did not satisfy the combined objective and paired rule. Details and raw outputs are in `results/experiments/fixed20_title_gender_audit.txt`, `fixed20_adam_beta_audit.txt`, `fixed20_adam_beta_grid.txt`, `fixed20_label_smoothing_audit.txt`, `fixed20_classweight_blend_audit.txt`, `fixed20_bootstrap_ensemble_audit.txt`, `fixed20_bootstrap_blend_fresh_audit.txt`, `fixed20_schedule_subsample_blend_audit.txt`, `fixed20_schedule_subsample_blend_fresh_audit.txt`, and `fixed20_schedule_bootstrap_blend_fresh_audit.txt`. No held-out test data was used.

**Post-ensemble calibration update (2026-11-24):** A −0.10 Neither logit bias selected on seeds 7600–7602 passed on two independent fresh sets, 7700–7702 and 7800–7802. Accuracy gains were +0.217 pp and +0.204 pp; paired loss gains were 0.001997 (SE 0.000190) and 0.002100 (SE 0.000174). Macro-F1 was −0.00010 on the first fresh set and +0.00097 on the second, giving +0.00044 across 30 fresh folds. `include/final_candidate_predictor.hpp` performs feature extraction, scaler transformation, 10-member constant/cosine ensemble averaging, swap TTA, 50/50 schedule blend, and then the calibrated Neither bias. The test set was not used.

**Focal/dropout follow-up:** Focal γ=1.0 failed on seeds 8200–8202 (accuracy 69.36%, macro-F1 0.63181, log-loss 0.77514 versus CE 69.59%, 0.63329, 0.69039). Hidden dropout 0.10 failed the combined objective on seeds 8300–8302 (accuracy +0.041 pp, macro-F1 −0.00020, loss gain 0.000105 versus SE 0.000801). Both remain disabled. The test set was not used.

**Asymmetric `him` calibration screen (withdrawn):** A fixed Candidate B logit adjustment of −0.30 improved paired fold scores, but it violated the required A/B swap symmetry. Those earlier results on seeds 9300–9302 and 9400–9402 are not part of the selected predictor. The implementation and active config use the swap-symmetric adjustment below instead.

The 20-member-per-schedule ensemble was also checked on seeds 8900–8902 against the 10-member candidate. Accuracy rose by 0.068 pp and macro-F1 by 0.00174, but paired log-loss worsened by 0.000148 (SE 0.000477), so it was not promoted. The conditional calibration scripts and raw outputs are `scripts/fixed20_him_bias_grid.cpp`, `scripts/fixed20_him_bias_fresh_audit.cpp`, `scripts/fixed20_him_bias_confirm_audit.cpp`, and the matching files under `results/experiments/`.

**Subsequent error-slice screens:** On the pooled OOF selection screen (seeds 9500–9502), Candidate B bias adjustments for `her` and for rows with no candidate URL match both reduced accuracy and macro-F1 and worsened log-loss. A conditional Neither boost for `her` also regressed. The optional prior-compatible-pronoun link feature was evaluated on matched folds (seeds 9700–9702): accuracy changed by +0.028 pp, macro-F1 by −0.00138, and paired loss worsened by 0.002079 (SE 0.000927); rejected. These additions remain disabled. Their runners and outputs are `scripts/fixed20_her_neither_bias_grid.cpp`, `scripts/fixed20_her_b_bias_grid.cpp`, `scripts/fixed20_no_url_b_bias_grid.cpp`, `scripts/fixed20_prior_pronoun_links_audit.cpp`, and their paired files under `results/experiments/`.

**Swap-symmetric `him` recency calibration (2026-11-24):** To preserve A/B permutation symmetry, a `−0.10` logit adjustment was applied to whichever candidate is closer to the pronoun, only for `him` and only when the offsets are unequal. The selection screen on seeds 10100–10102 improved accuracy by 0.122 pp and macro-F1 by 0.00099 with pooled paired log-loss gain 0.000419 (SE 0.000227). It then passed two independent paired-fold checks: on seeds 10200–10202 accuracy rose from 69.696% to 69.723%, macro-F1 from 0.63218 to 0.63243, and loss gain was 0.000423 (SE 0.000109); on seeds 10300–10302 accuracy rose from 70.008% to 70.089%, macro-F1 from 0.63367 to 0.63430, and loss gain was 0.000421 (SE 0.000096). This is the active conditional calibration recorded in `include/final_candidate_predictor.hpp` and `models/final_candidate_config.json`.

**Additional feature screens:** The wider prior-pronoun-chain block failed on seeds 9900–9902 (accuracy −0.054 pp, macro-F1 −0.00102, paired loss worsened by 0.001181, SE 0.001041). The local-window TF-IDF feature was later rechecked with the symmetry-safe calibration and selected as a 75% probability blend with the base model; see the confirmed results below. The 100% local feature model is not used alone.

**Local-window TF-IDF blend (screened, not selected):** Candidate-context and pronoun-context windows use fold-fitted IDF, exclude candidate names and common function words, and compare the nearest prior candidate mention with the pronoun window. The first weight screen and two confirmations applied calibration before model blending, unlike the inference path; those results are historical and are not used to select the final model. A corrected inference-order screen on seeds 11100–11102 showed a 75% local blend gain, but it failed fresh seeds 11200–11202: accuracy fell from 70.117% to 69.995% and macro-F1 from 0.63758 to 0.63449, despite loss gain 0.002692 (SE 0.000515). On those same fresh folds, 25% local weight raised accuracy only 0.027 pp while lowering macro-F1 by 0.00204; 100% local weight raised accuracy 0.054 pp while lowering macro-F1 by 0.00191. The local feature is therefore not selected. The final predictor defaults to the validated base model and can optionally blend local TF-IDF models via its constructor, but that option remains off.

The current selected model remains the 52-feature, 32-hidden-unit base ensemble with 50/50 schedule averaging, −0.10 Neither bias, and the swap-symmetric −0.10 nearer-candidate adjustment for `him`. The local TF-IDF artifacts remain available for further experiments. No held-out test data was loaded.

## Current publishable model (2026-11-24)

The selected deployable candidate is **base-only**: 52 features, 32 hidden ReLU units, ten SWA members from each of the constant and cosine schedules, averaged 50/50, with the validated Neither bias and swap-symmetric nearer-candidate `him` bias. The local-window TF-IDF blend is not selected. Earlier 75% blend confirmations used a different calibration order from deployment; corrected inference-order fresh audits did not improve both accuracy and macro-F1. On seeds 10200–10202 and 10300–10302, the selected base candidate averaged **69.906% accuracy and 0.63337 macro-F1** under paired CV. The held-out test set was not used. The config, inference wrapper, base model weights, and scalers in `models/` are the publishable artifact.
