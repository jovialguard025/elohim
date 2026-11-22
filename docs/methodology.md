# Research Methodology

## 1. Problem Formulation
In multi-agent and multi-step reasoning systems, naive iterative interaction often suffers from:
- **Error Compounding**: Early errors cascade into subsequent agent turns, deteriorating final output quality.
- **Context Bloat & Cost Explosion**: Redundant exploratory steps burn token budgets without yielding higher solution fidelity.

## 2. Confidence Gating
We define a confidence gating operator $G(s_t)$ evaluated at reasoning step $t$:
$$G(s_t) = \begin{cases} 
\text{Direct Execution}, & \text{if } H(s_t) < \tau_1 \text{ and } C(s_t) \ge \tau_2 \\
\text{Fallback Routing}, & \text{if } \tau_1 \le H(s_t) < \tau_{high} \\
\text{Retrieval Augmentation}, & \text{if } H(s_t) \ge \tau_{high}
\end{cases}$$

Where:
- $H(s_t)$ is the predictive token entropy over the candidate generation.
- $C(s_t)$ is the calibrated confidence probability.
- $\tau_1, \tau_2, \tau_{high}$ are calibrated decision thresholds.

## 3. Context-Budget Constraint
Let $B$ be the maximum context-budget ceiling. The system enforces:
$$\sum_{t=1}^{T} \text{tokens}(s_t) \le B$$
If a step threatens to breach $B$, the MAR system triggers an early summarization or immediate conclusion fallback.
