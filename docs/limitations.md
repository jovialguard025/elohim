# Limitations and Edge Cases

## 1. Uncalibrated LLM Probabilities
- Overconfident incorrect generations: LLMs can display high certainty (low entropy) on hallucinated statements, leading the gate to bypass necessary verification steps. Temperature scaling or auxiliary gating heads may be required.

## 2. Threshold Sensitivity
- A fixed threshold $\tau$ across distinct domains (e.g., mathematics vs open-ended conversational QA) can lead to either under-gating or over-triggering fallback paths.

## 3. Latency Trade-offs
- While confidence gating reduces overall token consumption, computing step-level uncertainty and branching introduces minor orchestrator overhead compared to an unmonitored streaming pipeline.
