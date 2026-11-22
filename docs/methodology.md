# Research Methodology: MAR Antecedent Classifier

## 1. Research Context & Formulation

In Retrieval-Augmented Generation (RAG) for Small Language Models (SLMs), retrieved passages frequently exhibit anaphora where an antecedent is missing because of passage truncation. Minimum-Sufficient Antecedent Restoration (MSAR) determines the minimal preceding context to restore to resolve ambiguous pronouns. 

The core predictive task is formulated as a three-class coreference classification problem over candidate antecedents:
* $\mathcal{Y} \in \{0, 1, 2\}$, where:
  * $y = 0$: Pronoun refers to Candidate A.
  * $y = 1$: Pronoun refers to Candidate B.
  * $y = 2$: Pronoun refers to Neither candidate.

---

## 2. Feature Extraction Mechanics

Features are computed using exact character boundaries provided in the Google GAP benchmark. All candidate features are computed symmetrically and independently for Candidate A and Candidate B.

### 2.1 Sentence Segmentation & Tokenization
Text is partitioned into discrete sentence spans $\mathcal{S} = \{[s_i, e_i)\}_{i=0}^{S-1}$ using sentence-terminating punctuation (`.`, `?`, `!`) followed by whitespace or string termination. Given candidate character offset $offset_{\text{cand}}$ and pronoun offset $offset_{\text{pron}}$, the respective sentence indices $s_{\text{cand}}$ and $s_{\text{pron}}$ are located via binary/linear search.

Tokenization maps string inputs to lowercase alphanumeric tokens $\mathcal{T} = [t_1, t_2, \dots, t_T]$. Jaccard similarity between two token multisets $U$ and $V$ is evaluated as:
$$J(U, V) = \frac{|U \cap V|}{|U \cup V|}$$
with $J(\emptyset, \emptyset) = 0$.

### 2.2 Feature Specifications

#### V1: 12-Feature Baseline Representation ($6 \times 2$)
For each candidate $C \in \{A, B\}$:
1. Normalized character distance: $\frac{offset_{\text{pron}} - offset_C}{len(\text{text})}$
2. Normalized sentence distance: $\frac{s_{\text{pron}} - s_C}{S}$
3. Mention frequency: Count of exact phrase occurrences of name $C$ in the text.
4. Sentence Jaccard similarity: $J(Tokens(Sent_{s_C}), Tokens(Sent_{s_{\text{pron}}}))$
5. Candidate name word count: $|Tokens(Name_C)|$
6. Precedes pronoun indicator: $\mathbf{1}(offset_C < offset_{\text{pron}})$

#### V2: 36-Feature Expanded Representation ($18 \times 2$)
For each candidate $C \in \{A, B\}$:
1. Character distance: $(offset_{\text{pron}} - offset_C) / len(\text{text})$
2. Sentence distance: $s_{\text{pron}} - s_C$
3. Candidate mention frequency: $|M_C|$, where $M_C$ is the set of character start offsets of all occurrences of $Name_C$.
4. Sentence Jaccard overlap: $J(Tokens(Sent_{s_C}), Tokens(Sent_{s_{\text{pron}}}))$
5. Candidate name token length: $|Tokens(Name_C)|$
6. Precedes pronoun indicator: $\mathbf{1}(offset_C < offset_{\text{pron}})$
7. Normalized token distance: $(count\_tokens(text[0:offset_{\text{pron}}]) - count\_tokens(text[0:offset_C])) / total\_tokens$
8. Same sentence indicator: $\mathbf{1}(s_C == s_{\text{pron}})$
9. Immediately preceding sentence indicator: $\mathbf{1}(s_{\text{pron}} - s_C == 1)$
10. Normalized sentence gap: $|s_{\text{pron}} - s_C| / S$
11. Mentions before pronoun: $|\{m \in M_C \mid m < offset_{\text{pron}}\}|$
12. Mentions after pronoun: $|\{m \in M_C \mid m > offset_{\text{pron}}\}|$
13. Distance to nearest mention: $\min_{m \in M_C} |m - offset_{\text{pron}}| / len(\text{text})$
14. Position of first mention before pronoun: $\min(\{m \in M_C \mid m < offset_{\text{pron}}\}) / len(\text{text})$ (or $-1.0$ if empty)
15. Distance from last prior mention: $(offset_{\text{pron}} - \max(\{m \in M_C \mid m < offset_{\text{pron}}\})) / len(\text{text})$ (or $-1.0$ if empty)
16. Local-context Jaccard overlap: $J(Tokens(text[offset_C \pm 60]), Tokens(text[offset_{\text{pron}} \pm 60]))$
17. Mention in pronoun sentence: $\mathbf{1}(\exists m \in M_C : s_m == s_{\text{pron}})$
18. Mentions between selected occurrence and pronoun: $|\{m \in M_C \mid \min(offset_C, offset_{\text{pron}}) < m < \max(offset_C, offset_{\text{pron}})\}|$

---

## 3. Standardization & Leakage Prevention

To prevent data snooping and target leakage, standardizer statistics are derived exclusively from the development partition $\mathcal{D}_{\text{train}}$:
$$\mu_j = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} X_{i, j}$$
$$\sigma_j = \sqrt{\frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} (X_{i, j} - \mu_j)^2}$$

If $\sigma_j < 10^{-12}$, $\sigma_j \leftarrow 1.0$. Transformation of any feature vector $x$ (including validation samples) is computed as:
$$\tilde{x}_j = \frac{x_j - \mu_j}{\sigma_j}$$

Validation examples are never used to compute or update $\mu$ or $\sigma$.

---

## 4. Neural Network Architecture & Optimization

### 4.1 Forward Formulation
Given normalized input vector $x \in \mathbb{R}^D$ ($D \in \{12, 30, 36\}$):
1. Hidden representation ($H = 32$ or $64$):
   $$h = \text{ReLU}(W_1 x + b_1) = \max(0, W_1 x + b_1)$$
2. Output logits:
   $$z = W_2 h + b_2$$
3. Numerically stabilized Softmax:
   $$p_k = \frac{\exp(z_k - \max_j z_j)}{\sum_{j=0}^2 \exp(z_j - \max_j z_j)}$$

### 4.2 Class Weighting & Objective Function
To address class imbalance (Class 0: 874, Class 1: 925, Class 2: 201), class weights are parameterized by exponent $\alpha$:
$$w_c = \frac{(1 / N_c)^\alpha}{\frac{1}{3} \sum_{k=0}^2 (1 / N_k)^\alpha}$$

For mini-batch $\mathcal{B} \subset \mathcal{D}_{\text{train}}$:
$$\mathcal{L}(\mathcal{B}) = \frac{1}{\sum_{i \in \mathcal{B}} w_{y_i}} \sum_{i \in \mathcal{B}} w_{y_i} \left( -\log(p_{i, y_i}) \right) + \frac{\lambda}{2} \left( \|W_1\|_F^2 + \|W_2\|_F^2 \right)$$

### 4.3 Gradient Propagation
For sample $i$ with target $y_i$ and sample weight $\tilde{w}_i = \frac{w_{y_i}}{\sum_{j \in \mathcal{B}} w_{y_j}}$:
$$\frac{\partial \mathcal{L}}{\partial z_{i, k}} = \tilde{w}_i (p_{i, k} - \mathbf{1}(k == y_i))$$
$$\frac{\partial \mathcal{L}}{\partial W_2} = \sum_{i \in \mathcal{B}} h_i \left(\frac{\partial \mathcal{L}}{\partial z_i}\right)^T + \lambda W_2, \quad \frac{\partial \mathcal{L}}{\partial b_2} = \sum_{i \in \mathcal{B}} \frac{\partial \mathcal{L}}{\partial z_i}$$
$$\frac{\partial \mathcal{L}}{\partial h_i} = W_2 \frac{\partial \mathcal{L}}{\partial z_i}$$
$$\delta_{1, i} = \frac{\partial \mathcal{L}}{\partial h_i} \odot \mathbf{1}(h_i > 0)$$
$$\frac{\partial \mathcal{L}}{\partial W_1} = \sum_{i \in \mathcal{B}} x_i \delta_{1, i}^T + \lambda W_1, \quad \frac{\partial \mathcal{L}}{\partial b_1} = \sum_{i \in \mathcal{B}} \delta_{1, i}$$

### 4.4 Adam Optimizer
Moments are updated with $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_t = \theta_{t-1} - \frac{\eta \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

---

## 5. Model Selection & Early Stopping Protocol

* Training terminates when validation macro-F1 does not improve for 50 consecutive epochs (`patience = 50`), up to a ceiling of 500 epochs.
* Checkpoints are evaluated after every epoch. A checkpoint is accepted if:
  $$\text{Macro-F1}_{\text{new}} > \text{Macro-F1}_{\text{best}} + 10^{-6}$$
  or if $\text{Macro-F1}_{\text{new}} == \text{Macro-F1}_{\text{best}}$ and $\mathcal{L}_{\text{val, new}} < \mathcal{L}_{\text{val, best}}$.
* The best checkpoint is restored to memory and saved as `models/best_model.bin`.
