"""Confidence Gating Module.

Evaluates predictive uncertainty and determines whether to allow direct execution,
route to a high-capacity fallback, or trigger retrieval.
"""

from typing import Dict, Any


class ConfidenceGate:
    """Confidence gating evaluator."""

    def __init__(self, entropy_threshold: float = 1.2, min_confidence: float = 0.75):
        """Initialize the confidence gate with decision thresholds.

        Args:
            entropy_threshold: Maximum allowed entropy before gating triggers fallback.
            min_confidence: Minimum calibrated probability required to pass gate.
        """
        self.entropy_threshold = entropy_threshold
        self.min_confidence = min_confidence

    def evaluate(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate extracted features against calibrated thresholds.

        Args:
            features: Dictionary containing 'entropy', 'mean_logprob', etc.

        Returns:
            Dictionary with gate decision ('pass', 'fallback', 'trigger_retrieval') and score.
        """
        entropy = features.get("entropy", 0.0)
        mean_logprob = features.get("mean_logprob", 0.0)

        # High entropy indicates high uncertainty
        is_confident = entropy < self.entropy_threshold and mean_logprob > -0.8

        if is_confident:
            decision = "pass"
        elif entropy >= self.entropy_threshold * 1.5:
            decision = "trigger_retrieval"
        else:
            decision = "fallback"

        return {
            "decision": decision,
            "passed": is_confident,
            "entropy": entropy,
            "confidence_score": max(0.0, 1.0 - (entropy / (self.entropy_threshold * 2))),
        }
