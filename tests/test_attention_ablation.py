"""Unit tests for Milestone 9 attention ablation and safety inspection."""

import pytest
import torch

from src.evaluation.attention_ablation import check_uniform_attention_intervention_safety
from src.models.detector import DeepfakeDetector
from src.models.detector import DeepfakeDetector
from src.utils.config import ModelConfig


def test_uniform_attention_safety_on_mock_fail():
    class IncompleteModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 1)

    is_safe, reason = check_uniform_attention_intervention_safety(IncompleteModel(), device="cpu")
    assert is_safe is False
    assert "missing" in reason.lower() or "not expose" in reason.lower()


def test_uniform_attention_safety_on_detector():
    # Instantiate lightweight detector
    cfg = ModelConfig()
    cfg.backbone.weights = None  # don't download weights for test
    model = DeepfakeDetector(cfg)

    is_safe, reason = check_uniform_attention_intervention_safety(model, device="cpu")
    assert is_safe is True
    assert "Safe inference-only substitution point verified" in reason
