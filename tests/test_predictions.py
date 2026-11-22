"""Tests for model prediction and output structure."""

from src.models.base import BaseModelWrapper


def test_base_model_prediction():
    model = BaseModelWrapper()
    response = model.predict("Test prompt for reasoning")

    assert "output" in response
    assert "tokens_used" in response
    assert "logprobs" in response
    assert response["tokens_used"] > 0
