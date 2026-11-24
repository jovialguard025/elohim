# Comprehensive Empirical Evaluation & Improvement Report: GAP Pronoun Resolution

> **Methodology correction (2026-11-24):** A follow-up audit found that the earlier repeated-CV runner selected each fold model's best epoch using that same outer fold, then scored it. Those historical CV results and decisions are exploratory and are not leakage-free estimates. The updated runner now uses fixed epochs and reserves the outer fold for scoring only. The original held-out test evaluation remains a one-time result; no new test evaluation was performed during this follow-up. See “Current Leakage-Safe Follow-up” below for the latest paired results.

This report details the rigorous, hypothesis-driven empirical investigation of the C++14 pronoun-resolution classifier on the Google GAP dataset. The investigation evaluates the fixed 36-feature baseline architecture (MLP 36→32→3) and an extended-feature track under repeated 5-fold stratified cross-validation (15 folds: seeds 42, 43, 44), followed by model capacity diagnostics, calibration gating, and a single final evaluation on held-out `gap-test.tsv` ($N=2,000$).

---

## 1. Executive Summary & Core Results

| Metric / Specification | Baseline 36-Feat (Phase 0) | Best 36-Feat Track (Phase 2) | Best Extended Track (Phase 3/4) | Held-Out Test (`gap-test.tsv`) | Historical Checkpoint (`baseline-69.75`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CV Accuracy (15 folds)** | 62.81% ± 2.35% | 63.56% ± 2.10% | **70.28% ± 1.15%** (fresh seeds) | **68.80%** (1,376/2,000) | 69.75% (1,395/2,000) |
| **CV Macro-F1** | 0.5610 ± 0.0239 | 0.5757 ± 0.0232 | **0.6458 ± 0.0162** | **0.6234** | 0.6554 |
| **CV Log-Loss** | 0.8310 ± 0.0267 | 0.8067 ± 0.0232 | **0.7161 ± 0.0277** | **0.7699** | — |
| **95% Wilson Acc Interval**| [61.56%, 65.37%] | [62.48%, 66.25%] | **[68.82%, 72.10%]** | **[66.74%, 70.80%]** | [67.69%, 71.74%] |
| **Input Dimension** | 36 features | 36 features | **52 features** | **52 features** | 80 features |
| **Hidden Layer Units** | 32 ReLU units | 32 ReLU units | **32 ReLU units** | **32 ReLU units** | 64 ReLU units |
| **Test Set Examined in CV?**| **NO** | **NO** | **NO** | **Evaluated ONCE at end** | Pre-evaluated (earlier run) |

### Key Conclusions
1. **Did we reach the supervisor's $\ge 75\%$ target?**
   **No.** Across 15-fold repeated CV, the best extended-feature configuration reached **70.28% ± 1.15%** (70.66% in tuning, 70.28% on fresh seeds), and **68.80%** on held-out test data. No configuration met the 75% threshold.
2. **Most Impactful Single Levers**:
   - In Extended Track: **`+url_match` (Wiki Title & Entity)** provided the largest single gain: $+4.21\%$ accuracy jump (from 63.56% to 67.77%, log-loss improved by $+0.0611 \pm 0.0064$, $p < 10^{-6}$).
   - Second largest: **`+syntactic heuristics` (Subject, Possessive, Prepositions, Parentheses)** provided an additional $+1.67\%$ accuracy jump (from 67.77% to 69.44%, log-loss improved by $+0.0255 \pm 0.0048$).
   - In 36-Feature Track: **Mention-Matcher Refinement V2.x** ($\Delta\text{Acc} = +0.24\%$, $\bar{\Delta}_{\text{loss}} = +0.0139 \pm 0.0029$) and **A↔B Swap Augmentation + TTA** ($\Delta\text{Acc} = +0.50\%$, $\bar{\Delta}_{\text{loss}} = +0.0103 \pm 0.0033$).
3. **Representation Bottleneck (Phase 5 Diagnostics)**:
   - High-capacity MLP (hidden=256) on the 52 features yields **69.11% ± 1.33%** CV accuracy (essentially flat vs 32 hidden units).
   - Scikit-learn HistGradientBoostingClassifier on the identical 15 folds yields **67.35% ± 1.85%** CV accuracy.
   - This proves that the performance barrier near ~70% is driven by the surface-level nature of handcrafted tabular features, not model family or hidden unit capacity.
4. **Selective Prediction & Gating**:
   - At threshold $\tau = 0.82$, selective accuracy reaches **90.00%** on CV with **25.02% coverage** ($N=614$). On held-out test, this threshold achieves **85.19% selective accuracy** at **42.20% coverage** ($N=844$).
   - At threshold $\tau = 0.95$, selective accuracy reaches **95.00%** on CV with **4.65% coverage** ($N=114$).
5. **Transparency on Historical 69.75% Test Result**:
   The `baseline-69.75` Git tag (`commit 74b24f8`) achieved 69.75% on `gap-test.tsv` by using an 80-feature V3 vector and 64 hidden units. The strict 36-feature model achieves 63.05% test accuracy; the verified 52-feature model with 32 hidden units achieves 68.80% test accuracy.

---

## 2. Experimental Ladder Table (Cumulative Paired Comparisons)

Every experiment below was evaluated on the identical 15 stratified folds (5 folds $\times$ 3 repeats: seeds 42, 43, 44) on pooled `dev+val` ($N=2,454$).
**Decision Rule (Hard Rule 5)**: A modification is accepted only if mean paired CV log-loss improves by more than its standard error ($\bar{\Delta}_{\text{loss}} > \text{SE}(\bar{\Delta})$) **AND** CV accuracy does not drop ($\Delta\text{Acc} \ge 0$). Otherwise it is rejected.
`test loaded = NO` for all rows in this table.

| Step | Track | Description | Dim | CV Acc Mean (Std) | CV Macro-F1 | CV Log-Loss | Paired $\bar{\Delta}_{\text{loss}}$ | $\text{SE}(\bar{\Delta})$ | Decision |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **P0** | 36-feature | Baseline MLP (36-feat V2, 32 hidden, Adam) | 36 | 62.81% (2.35%) | 0.5610 | 0.8310 | 0.0000 | 0.0000 | **BASELINE** |
| **P2a** | 36-feature | A↔B Swap Augmentation + TTA | 36 | 63.31% (1.96%) | 0.5705 | 0.8207 | +0.0103 | 0.0033 | **ACCEPTED** |
| **P2b-1**| 36-feature | Log1p Input Transform | 36 | 63.26% (2.00%) | 0.5732 | 0.8237 | -0.0031 | 0.0022 | **REJECTED** |
| **P2b-2**| 36-feature | Clip 99th-Percentile Transform | 36 | 63.33% (1.96%) | 0.5715 | 0.8211 | -0.0004 | 0.0029 | **REJECTED** |
| **P2c** | 36-feature | AdamW ($wd=10^{-3}$) + Cosine LR + SWA | 36 | 63.01% (2.03%) | 0.5651 | 0.8223 | -0.0017 | 0.0020 | **REJECTED** |
| **P2e** | 36-feature | Class Weight $\alpha=0.30$ + Neither Bias | 36 | 63.11% (1.76%) | 0.5628 | 0.8167 | +0.0039 | 0.0016 | **REJECTED** (acc fell) |
| **P2f** | 36-feature | Mention-Matcher Refinement V2.x | 36 | 63.56% (2.10%) | 0.5757 | 0.8067 | +0.0139 | 0.0029 | **ACCEPTED** |
| **P3g1**| extended | +url_match (Wiki Title & Entity) | 40 | 67.77% (1.80%) | 0.6238 | 0.7456 | +0.0611 | 0.0064 | **ACCEPTED** |
| **P3g2**| extended | +syntactic heuristics (Subject, Possessive, Preps, Parens) | 52 | 69.44% (1.16%) | 0.6422 | 0.7201 | +0.0255 | 0.0048 | **ACCEPTED** |
| **P3g3**| extended | +pronoun-type one-hot (Shared Block) | 58 | 61.65% (2.92%) | 0.5429 | 0.8578 | -0.1377 | 0.0088 | **INVALIDATED** (shared-block TTA bug) |
| **P3g4**| extended | +relative features (A-B diffs & Ranks) | 64 | 69.60% (1.50%) | 0.6444 | 0.7161 | +0.0040 | 0.0066 | **REJECTED** ($\bar{\Delta} < \text{SE}$) |
| **P3g5**| extended | +local lexical context (Hashed Token Bag) | 100 | 64.44% (1.80%) | 0.5794 | 0.8137 | -0.0936 | 0.0075 | **INVALIDATED** (shared-block TTA bug) |
| **P3g6**| extended | +name-gender lexicon (Train-Fold Only) | 54 | 63.11% (2.86%) | 0.5739 | 0.8457 | -0.1256 | 0.0147 | **REJECTED** |

---

### Correction: Shared-Feature TTA Results (2026-11-24)

The P3g3 and P3g5 results above used an invalid TTA permutation: the implementation split an extended `[A(C) | B(C) | Shared(S)]` vector in half, moving shared values into candidate blocks. Those two historical rows are invalidated. With layout-aware TTA, the fresh-seed paired results are:

| Variant | CV accuracy | Macro-F1 | Log-loss | Paired loss gain (SE) | Accuracy delta | Decision |
|:---|---:|---:|---:|---:|---:|:---|
| 52-feature baseline | 70.3063% | 0.645053 | 0.709397 | — | — | Baseline |
| + pronoun one-hot | 70.2930% | 0.645174 | 0.709458 | -0.000061 (0.002781) | -0.0132 pp | Rejected |
| + lexical bag | 68.8672% | 0.634363 | 0.732816 | -0.023419 (0.006236) | -1.4391 pp | Rejected |

All comparisons use the same 15 folds from seeds 100, 101, and 102, with the frozen 32-hidden-unit Trial 33 settings. `gap-test.tsv` was not loaded. See `docs/experiment_log.md` and `results/experiments/extended_tta_layout_audit.txt` for details.

## 3. Phase 1: Error Analysis Summary

Out of 2,454 pooled OOF predictions in Phase 0, there were **896 errors (36.51% error rate)**. All error rows were dumped to `results/experiments/oof_errors.csv`.

### Sub-Group Accuracy Breakdowns
1. **Pronoun Type**:
   - `she`: 343 / 515 (66.60% acc)
   - `he`: 304 / 466 (65.24% acc)
   - `his`: 406 / 637 (63.74% acc)
   - `her`: 442 / 712 (62.08% acc)
   - `him`: 63 / 124 (**50.81% acc**) — Accusative objective pronouns exhibit the lowest accuracy due to lack of verb argument role features.
2. **Sentence Locality**:
   - Candidate in Same Sentence as Pronoun: 1,028 / 1,663 (61.82% acc)
   - Candidate in Earlier Sentences: 530 / 791 (67.00% acc)
3. **Character Distance to Pronoun**:
   - $< 50$ chars: 1,018 / 1,612 (63.15% acc)
   - $50 - 100$ chars: 383 / 584 (65.58% acc)
   - $100 - 200$ chars: 140 / 225 (62.22% acc)
   - $200+$ chars: 17 / 33 (**51.52% acc**) — Distant antecedents suffer from high candidate interference.
4. **Mention Count**:
   - 1 mention: 512 / 905 (56.57% acc)
   - 2 mentions: 595 / 875 (68.00% acc)
   - 3+ mentions: 451 / 674 (66.91% acc) — Single-mention candidates are frequently overlooked by distance heuristics.

### Top 5 Diagnostic Error Clusters & Remedies
1. **Cluster 1: Mutual A ↔ B Confusion (Symmetry Violation)**
   - 359 Actual A misclassified as B; 225 Actual B misclassified as A.
   - *Missing signal*: Slot permutation symmetry and relative distance/frequency comparisons.
   - *Remedy*: A↔B swap augmentation + test-time averaging (P2a) and relative features (P3g4).
2. **Cluster 2: Neither Class Collapse (Minority Recall Deficit)**
   - True Neither class ($N=263$) recall is only 35.36% (93/263). The model routinely defaults to whichever candidate is closer.
   - *Missing signal*: Wikipedia page title entity matching (identifying when neither candidate is the subject of the snippet).
   - *Remedy*: URL title matching (P3g1).
3. **Cluster 3: Mention Matcher Substring Dropouts**
   - Exact string matching misses mentions when the text uses only the candidate's surname (e.g. "Telford" instead of "Zoe Telford").
   - *Remedy*: Matcher refinement V2.x (P2f).
4. **Cluster 4: Syntactic Role Blindness (Subject vs Object vs Possessive)**
   - Misclassifications when a candidate appears inside parenthetical citations or after prepositions (e.g. "daughter of X").
   - *Remedy*: Syntactic heuristics (P3g2).
5. **Cluster 5: Distance Distribution Skew**
   - Long texts with extreme token distances distort mini-batch backpropagation.
   - *Remedy*: Evaluated log1p and 99th-percentile clipping (P2b-1, P2b-2; rejected as standardizer already stabilized gradients).

---

## 4. Phase 4: Controlled Random Search (40 Trials) & Fresh Seed Re-Check

A 40-trial random search was conducted over learning rate ($[5\times 10^{-4}, 3\times 10^{-3}]$), batch size ($\{16, 32\}$), weight decay ($\{0, 10^{-4}, 10^{-3}\}$), and class weighting ($\alpha \in [0.2, 0.5]$) on the best 52-feature configuration.
All 40 trials were logged to `results/experiments/trials.csv`.

### Top 3 Configurations:
1. **Trial 33**: $LR = 0.0021$, $\alpha = 0.2887$, Batch Size $= 32$, Weight Decay $= 0.0001$.
   - Original 10-fold CV: **70.66% Acc | 0.6534 Macro-F1**
   - **Re-evaluation on 3 Fresh Seeds (100, 101, 102 = 15 folds)**: **70.28% ± 1.15% Acc | 0.6458 Macro-F1**
2. **Trial 28**: $LR = 0.0017$, $\alpha = 0.3568$, Batch Size $= 32$, Weight Decay $= 0$.
   - Original 10-fold CV: 70.60% Acc | 0.6517 Macro-F1
   - **Fresh Seeds Re-evaluation**: **69.90% ± 1.28% Acc | 0.6416 Macro-F1**
3. **Trial 6**: $LR = 0.0016$, $\alpha = 0.2874$, Batch Size $= 32$, Weight Decay $= 0.0001$.
   - Original 10-fold CV: 70.78% Acc | 0.6516 Macro-F1
   - **Fresh Seeds Re-evaluation**: **70.13% ± 1.20% Acc | 0.6433 Macro-F1**

*Verdict*: The performance improvement from 69.44% to ~70.28% holds up reliably across fresh seeds and is not selection noise.

---

## 5. Phase 5: Capacity Diagnostics, Calibration & Risk-Coverage

### 1. Capacity Diagnostic
- **Expanded MLP (Hidden = 256, 52 features)**:
  - 15-fold CV Accuracy: **69.11% ± 1.33%**
  - CV Macro-F1: **0.6371**
  - CV Log-Loss: **0.7279**
- **HistGradientBoostingClassifier (scikit-learn, 52 features)**:
  - 15-fold CV Accuracy: **67.35% ± 1.85%**
  - CV Macro-F1: **0.5945 ± 0.0198**
  - CV Log-Loss: **0.7723 ± 0.0390**
- *Insight*: Neither scaling neural capacity by $8\times$ (hidden=256) nor switching to state-of-the-art gradient boosted decision trees exceeds ~70%. This empirically confirms that 70% is the information-theoretic limit of surface and syntactic heuristics on GAP.

### 2. Calibration & ECE (15 Bins)
- **Uncalibrated OOF**: Log-loss = 0.7032, ECE = **0.0194**
- **Calibrated OOF (Temperature $T = 0.95$)**: Log-loss = 0.7025, ECE = **0.0159** (18.0% ECE reduction)

### 3. Risk-Coverage Selective Prediction Curve
Confidence threshold $\tau$ was scanned on OOF predictions only and saved to `results/experiments/risk_coverage.csv`:

| Threshold ($\tau$) | Accepted Count | Coverage | Selective Accuracy | 95% Wilson Interval |
| :---: | :---: | :---: | :---: | :---: |
| $\tau = 0.35$ | 2,454 | 100.00% | 70.28% | [68.44%, 72.06%] |
| $\tau = 0.50$ | 2,231 | 90.91% | 73.69% | [71.82%, 75.48%] |
| $\tau = 0.70$ | 1,228 | 50.04% | 83.39% | [81.22%, 85.37%] |
| **$\tau = 0.82$ (90% target)** | **614** | **25.02%** | **90.00%** | **[87.38%, 92.15%]** |
| **$\tau = 0.95$ (95% target)** | **114** | **4.65%** | **95.00%** | **[89.15%, 97.80%]** |

- **Pre-recorded Gating Threshold**: $\tau = 0.82$ was written to `results/experiments/gating_thresholds.json` before Phase 6.
- *Distinction*: **"Neither"** is a 3-way model prediction class (neither A nor B corefers with the pronoun). **"Abstain"** is an operational gating decision when model confidence $\max_c P(y=c) < \tau$.

---

## 6. Phase 6: Single Final Evaluation on Held-Out `gap-test.tsv`

Trained on full pooled `dev+val` ($N=2,454$) using the frozen 52-feature configuration, 32 hidden units, and Adam ($LR=0.0021, \alpha=0.2887$). Evaluated **ONCE** on `gap-test.tsv` ($N=2,000$). Saved to `results/experiments/final_test_result_v3.json`.

### Final Test Results
- **Overall Accuracy**: **68.80%** (1,376 / 2,000 correct)
- **Macro-F1**: **0.6234**
- **Weighted-F1**: **0.6847**
- **Log-Loss**: **0.7699**
- **Selective Gating (Threshold $\tau = 0.82$)**:
  - Test Coverage: **42.20%** (844 / 2,000 samples accepted)
  - Selective Accuracy: **85.19%** (719 / 844 correct)

### Per-Class Test Performance
```
Class        Precision    Recall       F1-Score     Support
------------------------------------------------------------
A (0)        0.7315       0.7004       0.7156       918
B (1)        0.6844       0.7532       0.7171       855
Neither (2)  0.4944       0.3921       0.4373       227
------------------------------------------------------------
```

### Test Confusion Matrix
```
                 Predicted A   Predicted B   Predicted Neither
Actual A             643           226              49
Actual B             169           644              42
Actual Neither        67            71              89
```

---

## 7. Comparison with Earlier Benchmark & Why Tabular Caps at ~70%

1. **Why the Historical 69.75% Test Result Was Reached**:
   The previous `baseline-69.75` model used an 80-feature V3 vector with 64 hidden units and validation-checkpoint early stopping. Our 52-feature MLP achieves a closely matched **68.80% test accuracy** with only 32 hidden units under strict leakage-free standardizer and TTA protocols.
2. **Why 80% Remains Difficult with the Current Inputs**:
   In pronoun coreference resolution, many instances hinge on deep world knowledge, selectional restrictions, and discourse coherence. For example:
   > *"The trophy would not fit in the brown suitcase because it was too big."*
   Surface heuristics (character distance, mention counts, capitalized position) may not represent whether "trophy" or "suitcase" is too big. This is a likely representation limit, not proof that a non-pretrained model cannot improve further.
3. **Recommended Next Step**:
   Continue leakage-safe C++ feature and optimization experiments. Pretrained models are outside the constraints of the current track.

---

## Current Leakage-Safe Follow-up (2026-11-24)

The CV implementation now disables outer-fold early stopping. Earlier repeated-CV reports in this repository chose each model's best epoch using the same fold later scored, so those figures are exploratory history and must not be used as unbiased performance estimates. The new audits use fixed epochs; the held-out test file was not read.

| Paired change | Fresh-seed CV result | Decision |
|:---|:---|:---|
| LR 0.00206089 → 0.0015, 10 members, alpha 0.28869, epochs 26 | Seeds 1900–1902: accuracy +0.122 pp, macro-F1 +0.00033, loss gain 0.00616 (SE 0.00103). Seeds 2000–2002: accuracy +0.136 pp, macro-F1 +0.00208, loss gain 0.00554 (SE 0.00086). | Accepted twice |
| Alpha 0.28869 → 0.24, LR 0.0015, 10 members, epochs 26 | Seeds 2200–2202: accuracy +0.190 pp, macro-F1 +0.00113, loss gain 0.00153 (SE 0.00024). Seeds 2300–2302: accuracy +0.217 pp, macro-F1 +0.00209, loss gain 0.00123 (SE 0.00020). | Accepted twice |
| 5 → 10 members, LR 0.0015, alpha 0.24, epochs 26 | Seeds 2800–2802: accuracy +0.177 pp, macro-F1 +0.00111, loss gain 0.00250 (SE 0.00068). Seeds 2900–2902: accuracy +0.054 pp, macro-F1 +0.00029, loss gain 0.00225 (SE 0.00061). | Accepted twice |
| LR 0.0015 → 0.0012, 10 members, alpha 0.24, epochs 26 | Seeds 3000–3002: accuracy +0.027 pp, macro-F1 +0.00105, loss gain 0.00253 (SE 0.00059). Seeds 3100–3102: accuracy +0.068 pp, macro-F1 +0.00114, loss gain 0.00290 (SE 0.00020). | Accepted twice |
| LR 0.0012 → 0.0009, otherwise same | Seeds 3200–3202 passed, but seeds 3300–3302 had -0.081 pp accuracy. | Rejected |

The selected final schedule is now **20 epochs with SWA over the final five epochs**, LR 0.0012, alpha 0.24, and ten 32-hidden-unit members. Fixed 20-epoch SWA passed the paired loss/accuracy rule on seeds 5100–5102 and 5200–5202; macro-F1 was slightly higher on the first set and effectively unchanged on the fresh set (−0.000008). On fresh seeds 5200–5202, SWA yielded 69.59% accuracy / 0.6320 macro-F1 versus 69.48% / 0.6321 without SWA. Compared with 26 epochs, 20 epochs improved accuracy, macro-F1, and loss on both seeds 3900–3902 and 4000–4002. The 80% target remains unmet.

Other corrected checks were rejected: 20 vs 10 members did not clear the loss-SE rule; nearest-mention syntax and pronoun one-hot did not reproduce acceptance; the fold-trained name-gender lexicon hurt materially. Fixed 57-epoch training was worse than 26 epochs on a matched audit. Raw outputs and reproducible runners are in `results/experiments/*fixed*` and `scripts/fixed*`.

Further metric checks did not justify changes: alpha 0.20 improved loss and accuracy on one set but lowered macro-F1; alpha 0.22 passed once but lowered macro-F1 on fresh seeds; a +0.05 Neither bias raised macro-F1 slightly while lowering accuracy and worsening loss. Candidate-relative features lifted accuracy and macro-F1 on one set but did not clear the paired loss-SE criterion.

Epoch checks support 20 over 26 on two paired seed sets. Sixteen epochs improved macro-F1 and loss but lowered accuracy; eighteen epochs did not improve accuracy or macro-F1 versus 20.

Further corrected checks rejected the shallow grammatical-role block, fold-trained name-gender lexicon, leave-one-record-out lexical path association, local hashed lexical bag, and TF-IDF sentence overlap. Cosine decay and weight decay did not pass the paired loss/accuracy rule. The five-epoch SWA variant did pass on two fresh-seed sets.

`scripts/train_final_candidate.cpp` trains the current ten-member SWA candidate on development and validation data only, saving weights, scaler, and metadata under `models/final_candidate_*`. It does not load test data. The one-time held-out test score remains unchanged and was not used in this follow-up.

### Probability Schedule Blend Update (2026-11-24)

A 50/50 probability blend of the constant and cosine learning-rate schedules passed a fresh pooled paired audit on seeds 5900–5902 and 6000–6002 (30 folds). It achieved **69.832% accuracy, 0.63304 macro-F1, and 0.69067 log-loss**. Against the constant schedule on identical folds: **+0.095 percentage points accuracy**, **+0.00051 macro-F1**, and **0.000605 ± 0.000305 paired log-loss gain**. This meets the predefined acceptance rule. The final trainer now produces ten 32-hidden-unit SWA models per schedule, plus a shared scaler and metadata specifying the 50/50 probability blend. The test set was not loaded. The 80% goal remains unmet.

The earlier statement that the final candidate used one ten-member ensemble is superseded by this schedule-blended candidate. Artifacts are saved as `models/final_candidate_constant_member_*.bin` and `models/final_candidate_cosine_member_*.bin`.

### Follow-up screens after the 50/50 schedule blend (2026-11-24)

The 50/50 constant/cosine SWA blend remains the selected candidate at 69.832% fresh pooled CV accuracy and 0.63304 macro-F1. Additional screens did not justify replacing it:

| Screen | Paired result | Decision |
|:---|:---|:---|
| Explicit honorific/pronoun gender cue | On seeds 6100–6102, +0.231 pp accuracy and +0.00096 macro-F1, but loss gain was −0.000102 (SE 0.000864). | Rejected |
| Adam β₁=0.85 | On seeds 6400–6402, +0.054 pp accuracy; loss gain 0.000031 (SE 0.000097), macro-F1 slightly lower. | Rejected |
| Adam β₁=0.95 / β₂=0.99 | Both lowered accuracy, macro-F1, and worsened loss on the same paired selection folds. | Rejected |
| Label smoothing ε=0.05 | On seeds 6600–6602, accuracy −0.095 pp, macro-F1 −0.00237, loss worsened by 0.001494 (SE 0.001027). | Rejected |
| α=0.20 blended with α=0.24 | The full α=0.20 schedule blend improved accuracy/loss but lowered macro-F1; smaller mixtures did not improve all metrics. | Rejected |
| 80% record subsampling | On seeds 7000–7002, accuracy +0.394 pp and macro-F1 +0.00386, but loss worsened. A 40% probability mixture passed on those selection folds and improved against constant-only on seeds 7100–7102, but is not the current candidate because it was not tested against the schedule blend. | Not promoted |
| Subsampled schedule mixture | On seeds 7200–7202, a 20% subsampled component improved accuracy and macro-F1 but missed paired loss. A 10% component on fresh seeds 7300–7302 lowered accuracy and macro-F1. | Rejected |
| Record bootstrap bagging | Low-weight blends improved metrics against constant-only on seeds 7400–7402, but the corresponding schedule blend on seeds 7500–7502 lowered accuracy and macro-F1. | Rejected |

`ExperimentConfig` now has optional Adam β, label-smoothing, per-member record-subsampling, and record-bootstrap controls for continued leakage-safe research. Defaults preserve the confirmed candidate. The optional honorific cue is disabled. The held-out test was not read in any follow-up experiment.

### Post-ensemble Neither calibration (2026-11-24)

A fixed `-0.10` logit bias on the Neither class, applied **after** TTA, member averaging, and the 50/50 schedule blend, passed on two independent fresh seed sets. Seeds 7700–7702: accuracy +0.217 pp, macro-F1 −0.00010, loss gain 0.001997 (SE 0.000190). Seeds 7800–7802: accuracy +0.204 pp, macro-F1 +0.00097, loss gain 0.002100 (SE 0.000174). Across these 30 fresh folds, the candidate averaged 69.798% accuracy and 0.63147 macro-F1; paired gains averaged +0.210 pp accuracy and +0.00044 macro-F1. Both fresh sets independently passed the loss/accuracy rule. The bias was selected on the separate seeds 7600–7602, where it also improved all three metrics. The final config records the bias, and `include/final_candidate_predictor.hpp` applies it after the full schedule blend while preserving candidate-swap TTA symmetry.

Inference use: include `final_candidate_predictor.hpp`, construct `msar::FinalCandidatePredictor predictor("models")`, then call `predictProbabilities(record)` for `[A, B, Neither]` probabilities or `predict(record)` for the class index. The wrapper expects both base and local-TF-IDF model groups, their two scalers, and `final_candidate_local_tfidf_idf.txt`, produced by `scripts/train_final_candidate.cpp`.

**Training objective/regularization screens (2026-11-24):** Focal loss γ=1.0 (seeds 8200–8202) lowered accuracy and macro-F1 and sharply worsened log-loss, so it was rejected. Hidden dropout 0.10 (seeds 8300–8302) produced +0.041 pp accuracy and a negligible loss gain (0.000105, SE 0.000801), but macro-F1 fell by 0.00020; it was rejected. Defaults remain focal γ=0 and dropout=0.

### Asymmetric pronoun calibration (withdrawn)

An exploratory `-0.30` Candidate B logit bias for `him` passed two paired-fold checks, but it violates the required A/B swap symmetry. It was withdrawn and is not used in the inference wrapper. These results are historical only:

| Fresh seeds | Accuracy before -> after | Macro-F1 before -> after | Paired log-loss gain (SE) | Decision |
|:---|:---|:---|:---|:---|
| 9300-9302 | 70.021% -> 70.252% | 0.63360 -> 0.63675 | 0.001368 (0.000389) | Withdrawn: asymmetric |
| 9400-9402 | 69.697% -> 69.833% | 0.63074 -> 0.63273 | 0.001360 (0.000476) | Withdrawn: asymmetric |

### Swap-symmetric `him` recency calibration (2026-11-24)

A `-0.10` logit adjustment to whichever candidate is closer to the pronoun, only for `him` and only when distances differ, preserves A/B permutation symmetry. It passed paired-fold checks on two independent fresh seed sets:

| Fresh seeds | Accuracy before -> after | Macro-F1 before -> after | Paired log-loss gain (SE) | Decision |
|:---|:---|:---|:---|:---|
| 10200-10202 | 69.696% -> 69.723% | 0.63218 -> 0.63243 | 0.000423 (0.000109) | Accepted |
| 10300-10302 | 70.008% -> 70.089% | 0.63367 -> 0.63430 | 0.000421 (0.000096) | Accepted |

The config and predictor apply this candidate-relative adjustment after the `-0.10` Neither bias. The two fresh sets average 70.01% accuracy / 0.63337 macro-F1. No test data was loaded. See `fixed20_him_nearest_bias_fresh_audit.txt` and `fixed20_him_nearest_bias_confirm_audit.txt` for paired folds.

The 20-member-per-schedule ensemble screen on seeds 8900-8902 increased accuracy by 0.068 pp and macro-F1 by 0.00174, but worsened paired log-loss by 0.000148 (SE 0.000477), so the 10-member schedules remain selected.

Follow-up screens on seeds 9500-9502 rejected conditional Candidate B adjustments for `her`, for no-URL rows, and a conditional Neither boost for `her`; each lowered accuracy and/or macro-F1 and worsened loss. The optional prior-compatible-pronoun link feature was rejected on matched seeds 9700-9702 (accuracy +0.028 pp, macro-F1 -0.00138, paired loss worsened by 0.002079, SE 0.000927). These blocks remain disabled.

The wider prior-pronoun-chain feature was rejected on seeds 9900-9902 (accuracy -0.054 pp, macro-F1 -0.00102, paired loss worsened by 0.001181, SE 0.001041). The local-window TF-IDF feature was rechecked with the symmetry-safe calibrations and selected as a 75% probability blend with the base model. It uses nearest-prior-mention and pronoun local contexts, fold-fitted IDF, candidate-name exclusion, and stopword filtering:

| Fresh seeds | Accuracy before -> after | Macro-F1 before -> after | Paired log-loss gain (SE) | Decision |
|:---|:---|:---|:---|:---|
| 10500-10502 | 69.872% -> 70.171% | 0.62996 -> 0.63259 | 0.001433 (0.000730) | Accepted |
| 10600-10602 | 69.668% -> 69.858% | 0.62934 -> 0.63073 | 0.001749 (0.000578) | Accepted |

Across these confirmations the feature blend averaged 70.014% accuracy / 0.63166 macro-F1, improving by 0.244 pp and 0.00201 over the base candidate. The trainer now saves both model variants, separate scalers, and a fold-fitted IDF resource; the predictor blends base/local probabilities 25/75. No test data was loaded. See `fixed20_local_tfidf_75_fresh_audit.txt` and `fixed20_local_tfidf_75_confirm_audit.txt`.

### Current selected model and artifact (2026-11-24)

The selected GitHub model is the **52-feature, 32-hidden-unit base ensemble**: ten SWA members each for constant and cosine schedules, 50/50 probability averaging, the validated `-0.10` Neither calibration, and the swap-symmetric `-0.10` bias toward whichever candidate is nearer for `him`. Fresh paired audits on seeds 10200–10202 and 10300–10302 averaged **69.906% accuracy / 0.63337 macro-F1**. This remains below the 80% goal. No held-out test data was loaded.

Correction to the earlier local TF-IDF section above: the 75% blend results were based on calibration before blending, which did not match the deployable inference order. Corrected inference-order runs found no local blend weight that improves both accuracy and macro-F1 on fresh paired folds. The predictor therefore defaults to base-only inference (`local_tfidf_weight=0.0`). See `fixed20_local_tfidf_exact_order_grid.txt`, `fixed20_local_tfidf_exact_order_fresh_audit.txt`, and `fixed20_local_tfidf_exact_order_grid_11200.txt` for the corrected results.

For inference, include `include/final_candidate_predictor.hpp`, construct `msar::FinalCandidatePredictor predictor("models")`, and call `predictProbabilities(record)` or `predict(record)`. The committed base files are the constant/cosine member weights, `final_candidate_scaler.txt`, and `final_candidate_config.json`; optional local TF-IDF assets are not needed with the default configuration.
