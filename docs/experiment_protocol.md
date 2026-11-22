# Experiment Protocol

## 1. Experimental Objectives
- Evaluate whether **Confidence-Gated MSAR** outperforms fixed static routing architectures.
- Measure total tokens consumed per correct reasoning trajectory (Token Efficiency).
- Measure accuracy across varying levels of question difficulty.

## 2. Baselines
1. **Single Agent Direct Prompting**: Zero-shot / few-shot baseline.
2. **Static Multi-Agent Routing**: Fixed chaining of specialized agents without confidence feedback.
3. **Confidence-Gated MSAR (Ours)**: Dynamic routing based on runtime entropy and confidence gating.

## 3. Metrics
- **Accuracy**: Percentage of correct solutions verified against ground truth.
- **Budget Compliance Rate**: Percentage of trajectories completing within context ceiling $B$.
- **Average Tokens per Query**: Total tokens consumed divided by total benchmark instances.
- **Precision / Recall of Gating Decisions**: Gating trigger precision against actual agent error points.
