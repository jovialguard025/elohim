# Results Directory (`results/`)

> **Policy: Upload Selectively**  
> Store **only sanitized metric tables, confusion matrices, experiment comparisons, and error-analysis summaries**. Do not commit raw runtime logs, megabytes of intermediate step dumps, or unsanitized outputs.

---

## Directory Organization

```text
results/
├── metrics/       # Sanitized metric tables (JSON/CSV) showing accuracy, F1, precision, recall, and budget stats
├── figures/       # Confusion matrix plots, ROC/PR curves, and budget vs accuracy tradeoffs
└── experiments/   # Markdown experiment writeups, hypothesis validation logs, and comparative analyses
```

---

## Reporting Guidelines
When adding experimental results:
1. Include git commit hash and configuration parameters in the summary.
2. Provide key comparison baselines (e.g. *Static Multi-Agent Routing* vs *Confidence-Gated MSAR*).
3. Document budget impact: reduction in average tokens used per successful solution.
