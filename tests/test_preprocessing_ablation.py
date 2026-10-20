"""Unit tests for Milestone 9 preprocessing ablation."""

from pathlib import Path
import pytest
import torch

from src.evaluation.preprocessing_ablation import evaluate_ofat_experiment
from src.utils.config import AppConfig


class MockDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        return torch.tensor([[1.2]])


def test_ofat_experiment_contract_labeling():
    # Verify standard config
    cfg_std = AppConfig()
    cfg_std.data.normalization = "imagenet"
    res_std = evaluate_ofat_experiment(
        exp_name="margin_0.1",
        config=cfg_std,
        is_out_of_contract=False,
        videos=[],
        baseline_predictions={},
        model=MockDetector(),
        device="cpu",
    )
    assert res_std["contract_label"] == "production contract"
    assert res_std["is_out_of_contract"] is False

    # Verify out of contract labeling
    cfg_out = AppConfig()
    cfg_out.data.normalization = "none"
    res_out = evaluate_ofat_experiment(
        exp_name="normalization_none",
        config=cfg_out,
        is_out_of_contract=True,
        videos=[],
        baseline_predictions={},
        model=MockDetector(),
        device="cpu",
    )
    assert res_out["contract_label"] == "out_of_contract diagnostic"
    assert res_out["is_out_of_contract"] is True
