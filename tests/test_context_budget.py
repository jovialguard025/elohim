"""Tests for context budget compliance and token limit enforcement."""

from src.evaluation.budget import ContextBudgetTracker


def test_budget_within_limits():
    tracker = ContextBudgetTracker(max_token_budget=1000, max_steps=5)
    status = tracker.consume_tokens(200)

    assert status["compliant"] is True
    assert status["remaining_budget"] == 800
    assert status["steps_taken"] == 1


def test_budget_exceeded():
    tracker = ContextBudgetTracker(max_token_budget=500, max_steps=3)
    tracker.consume_tokens(300)
    status = tracker.consume_tokens(250)

    assert status["compliant"] is False
    assert status["remaining_budget"] == 0
    assert status["tokens_consumed"] == 550
