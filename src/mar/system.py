"""MAR (Multi-Agent Routing) System Coordinator.

Coordinates agents, evaluates confidence gates, and enforces context budgets across
reasoning trajectories.
"""

from typing import Dict, Any, List
from src.features.extractor import FeatureExtractor
from src.gating.confidence_gate import ConfidenceGate
from src.evaluation.budget import ContextBudgetTracker


class MARSystem:
    """Orchestrates multi-agent reasoning with confidence gating and budget management."""

    def __init__(
        self,
        gate: ConfidenceGate = None,
        extractor: FeatureExtractor = None,
        budget_tracker: ContextBudgetTracker = None,
    ):
        self.gate = gate or ConfidenceGate()
        self.extractor = extractor or FeatureExtractor()
        self.budget_tracker = budget_tracker or ContextBudgetTracker()

    def process_step(self, step_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single reasoning step in the MAR pipeline.

        Args:
            step_input: Input dictionary containing step data, tokens used, etc.

        Returns:
            Dictionary with routing outcome, budget compliance, and action to take.
        """
        tokens_used = step_input.get("tokens_used", 0)
        budget_status = self.budget_tracker.consume_tokens(tokens_used)

        if not budget_status["compliant"]:
            return {
                "action": "terminate_budget_exceeded",
                "gate_status": None,
                "budget_status": budget_status,
            }

        features = self.extractor.extract_step_features(step_input)
        gate_result = self.gate.evaluate(features)

        return {
            "action": gate_result["decision"],
            "gate_status": gate_result,
            "budget_status": budget_status,
        }
