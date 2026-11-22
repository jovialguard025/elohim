"""Context Budget Tracker.

Monitors token consumption and step count to ensure context-budget compliance.
"""

from typing import Dict, Any


class ContextBudgetTracker:
    """Tracks token consumption against a maximum ceiling."""

    def __init__(self, max_token_budget: int = 8192, max_steps: int = 10):
        self.max_token_budget = max_token_budget
        self.max_steps = max_steps
        self.tokens_consumed = 0
        self.steps_taken = 0

    def reset(self) -> None:
        """Reset the budget counters for a new trajectory."""
        self.tokens_consumed = 0
        self.steps_taken = 0

    def consume_tokens(self, tokens: int) -> Dict[str, Any]:
        """Record token usage for a step and check compliance.

        Args:
            tokens: Number of tokens consumed in this step.

        Returns:
            Dictionary with compliance status and remaining budget.
        """
        self.tokens_consumed += tokens
        self.steps_taken += 1

        is_compliant = (
            self.tokens_consumed <= self.max_token_budget
            and self.steps_taken <= self.max_steps
        )

        return {
            "compliant": is_compliant,
            "tokens_consumed": self.tokens_consumed,
            "remaining_budget": max(0, self.max_token_budget - self.tokens_consumed),
            "steps_taken": self.steps_taken,
            "max_steps": self.max_steps,
        }
