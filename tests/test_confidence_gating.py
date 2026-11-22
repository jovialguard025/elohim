"""Tests for confidence gating and threshold evaluation."""

from src.gating.confidence_gate import ConfidenceGate


def test_confidence_gate_pass():
    gate = ConfidenceGate(entropy_threshold=1.0)
    features = {"entropy": 0.3, "mean_logprob": -0.2}
    result = gate.evaluate(features)

    assert result["passed"] is True
    assert result["decision"] == "pass"


def test_confidence_gate_fallback():
    gate = ConfidenceGate(entropy_threshold=1.0)
    features = {"entropy": 1.2, "mean_logprob": -1.5}
    result = gate.evaluate(features)

    assert result["passed"] is False
    assert result["decision"] == "fallback"


def test_confidence_gate_trigger_retrieval():
    gate = ConfidenceGate(entropy_threshold=1.0)
    features = {"entropy": 2.0, "mean_logprob": -2.5}
    result = gate.evaluate(features)

    assert result["passed"] is False
    assert result["decision"] == "trigger_retrieval"
