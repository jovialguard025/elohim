"""Feature extraction module for reasoning step confidence and context signals."""

from typing import Any, Dict, List
import math


class FeatureExtractor:
    """Extracts confidence, entropy, and context statistics from inference steps."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size

    def compute_token_entropy(self, logprobs: List[float]) -> float:
        """Calculate Shannon entropy from token log-probabilities.

        Args:
            logprobs: List of log-probabilities for predicted tokens.

        Returns:
            Calculated entropy value (float).
        """
        if not logprobs:
            return 0.0
        entropy = 0.0
        for lp in logprobs:
            p = math.exp(lp)
            if p > 0:
                entropy -= p * lp
        return entropy

    def extract_step_features(self, step_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract multi-dimensional signal vector from a reasoning step.

        Args:
            step_data: Dictionary containing step metrics (e.g. logprobs, token_count).

        Returns:
            Dictionary of computed features.
        """
        logprobs = step_data.get("logprobs", [])
        tokens_used = step_data.get("tokens_used", 0)
        entropy = self.compute_token_entropy(logprobs)
        mean_logprob = sum(logprobs) / len(logprobs) if logprobs else 0.0

        return {
            "entropy": entropy,
            "mean_logprob": mean_logprob,
            "tokens_used": float(tokens_used),
        }
