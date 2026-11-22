# Literature Review

## 1. Multi-Agent Reasoning Systems
- **Iterative Refinement & Society of Mind**: Collaborative agent frameworks (e.g., AutoGen, MetaGPT, ChatDev) demonstrate enhanced problem decomposition but frequently run into conversational loops and excessive token utilization.

## 2. Uncertainty Estimation in LLMs
- **Predictive Entropy & Token Log-Probabilities**: Measuring the Shannon entropy of output token distributions provides a lightweight surrogate for model certainty without requiring expensive ensemble generation (Kadavath et al., Kuhn et al. - Semantic Entropy).

## 3. Dynamic Routing & Context Management
- **Cascaded Inference & Speculative Execution**: Dynamically switching between smaller models (e.g., SLMs) and larger frontier models when uncertainty is detected significantly reduces operational inference costs while preserving quality.
