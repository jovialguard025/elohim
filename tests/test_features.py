"""Tests for feature extraction and entropy calculation."""

from src.features.extractor import FeatureExtractor


def test_compute_token_entropy_empty():
    extractor = FeatureExtractor()
    assert extractor.compute_token_entropy([]) == 0.0


def test_extract_step_features():
    extractor = FeatureExtractor()
    step_data = {
        "logprobs": [-0.1, -0.3, -0.05],
        "tokens_used": 15,
    }
    features = extractor.extract_step_features(step_data)
    assert "entropy" in features
    assert "mean_logprob" in features
    assert features["tokens_used"] == 15.0
    assert features["entropy"] >= 0.0
