"""Evaluation package for metrics, budget compliance, and error analysis."""

from .budget import ContextBudgetTracker
from .metrics import compute_classification_metrics

__all__ = ["ContextBudgetTracker", "compute_classification_metrics"]
