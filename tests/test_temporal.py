"""Unit tests for Temporal Modeling modules (LSTM / BiLSTM) in Milestone 4."""

import pytest
import torch
import torch.nn as nn

from src.models.temporal import LSTMTemporalModel, build_temporal_model
from src.utils.config import TemporalConfig


def test_temporal_factory_modes():
    """Verify temporal factory instantiates correct models with valid invariants."""
    # 1. Default LSTM
    cfg_lstm = TemporalConfig(type="lstm", bidirectional=False, hidden_dim=2048)
    model_lstm = build_temporal_model(cfg_lstm)
    assert isinstance(model_lstm, LSTMTemporalModel)
    assert not model_lstm.config.bidirectional
    assert model_lstm.config.hidden_dim == 2048
    assert model_lstm.config.output_dim == 2048
    assert model_lstm.config.dropout == 0.5
    assert model_lstm.config.pooling == "last"

    # 2. Ablation BiLSTM
    cfg_bilstm = TemporalConfig(type="bilstm", bidirectional=True, hidden_dim=1024)
    model_bilstm = build_temporal_model(cfg_bilstm)
    assert isinstance(model_bilstm, LSTMTemporalModel)
    assert model_bilstm.config.bidirectional
    assert model_bilstm.config.hidden_dim == 1024
    assert model_bilstm.config.output_dim == 2048

    # 3. None defaults to paper-aligned LSTM
    default_model = build_temporal_model(None)
    assert isinstance(default_model, LSTMTemporalModel)
    assert default_model.config.type == "lstm"
    assert default_model.config.output_dim == 2048


def test_temporal_parameter_counts_formula():
    """Verify LSTM and BiLSTM parameter counts adhere strictly to PyTorch formulas."""
    # Unidirectional LSTM (I=2048, H=2048)
    # Total = 4 * (I*H + H^2 + 2H) = 4 * (2048*2048 + 2048^2 + 2*2048) = 33,570,816
    cfg_lstm = TemporalConfig(type="lstm", bidirectional=False, hidden_dim=2048)
    model_lstm = build_temporal_model(cfg_lstm)
    expected_lstm_params = 4 * (2048 * 2048 + 2048**2 + 2 * 2048)
    actual_lstm_params = sum(p.numel() for p in model_lstm.parameters())
    assert actual_lstm_params == expected_lstm_params == 33_570_816

    # BiLSTM (I=2048, H=1024)
    # Total = 2 * 4 * (I*H + H^2 + 2H) = 8 * (2048*1024 + 1024^2 + 2*1024) = 25,182,208
    cfg_bilstm = TemporalConfig(type="bilstm", bidirectional=True, hidden_dim=1024)
    model_bilstm = build_temporal_model(cfg_bilstm)
    expected_bilstm_params = 2 * 4 * (2048 * 1024 + 1024**2 + 2 * 1024)
    actual_bilstm_params = sum(p.numel() for p in model_bilstm.parameters())
    assert actual_bilstm_params == expected_bilstm_params == 25_182_208

    # Ensure post_dropout has 0 parameters
    assert sum(p.numel() for p in model_lstm.post_dropout.parameters()) == 0


def test_temporal_input_validation():
    """Verify strict input validation on tensor dimensionality and feature shapes."""
    model = build_temporal_model()

    # Rejects 2D tensor
    with pytest.raises(ValueError, match="Expected 3D input tensor"):
        model(torch.randn(4, 2048))

    # Rejects 4D tensor
    with pytest.raises(ValueError, match="Expected 3D input tensor"):
        model(torch.randn(2, 16, 2048, 4))

    # Rejects B=0 or T=0
    with pytest.raises(ValueError, match="Batch size B and sequence length T must be > 0"):
        model(torch.randn(0, 16, 2048))

    with pytest.raises(ValueError, match="Batch size B and sequence length T must be > 0"):
        model(torch.randn(2, 0, 2048))

    # Rejects C != 2048
    with pytest.raises(ValueError, match="Expected input feature dimension C=2048"):
        model(torch.randn(2, 16, 512))


def test_temporal_pooling_correctness():
    """Verify 'last' and 'mean' pooling compute exact reductions."""
    torch.manual_seed(42)
    x = torch.randn(2, 8, 2048)

    # 1. Unidirectional LSTM last pooling
    cfg_last = TemporalConfig(type="lstm", dropout=0.0, pooling="last")
    model_last = build_temporal_model(cfg_last)
    model_last.eval()

    out_last = model_last(x)
    raw_lstm_out, _ = model_last.lstm(x)
    assert torch.allclose(out_last, raw_lstm_out[:, -1, :], atol=1e-6)

    # 2. BiLSTM last pooling: [h_T_fwd; h_0_bwd]
    cfg_bi = TemporalConfig(type="bilstm", bidirectional=True, hidden_dim=1024, dropout=0.0, pooling="last")
    model_bi = build_temporal_model(cfg_bi)
    model_bi.eval()

    out_bi = model_bi(x)
    raw_bi_out, _ = model_bi.lstm(x)
    expected_bi = torch.cat([raw_bi_out[:, -1, :1024], raw_bi_out[:, 0, 1024:]], dim=1)
    assert torch.allclose(out_bi, expected_bi, atol=1e-6)

    # 3. Mean pooling
    cfg_mean = TemporalConfig(type="lstm", dropout=0.0, pooling="mean")
    model_mean = build_temporal_model(cfg_mean)
    model_mean.eval()

    out_mean = model_mean(x)
    raw_mean_out, _ = model_mean.lstm(x)
    assert torch.allclose(out_mean, raw_mean_out.mean(dim=1), atol=1e-6)


def test_temporal_explicit_post_pooling_dropout():
    """Verify post-pooling dropout behaves stochastically in train() and deterministically in eval()."""
    cfg = TemporalConfig(type="lstm", dropout=0.5)
    model = build_temporal_model(cfg)

    # Internal LSTM dropout must be 0.0 to prevent PyTorch runtime warning
    assert model.lstm.dropout == 0.0

    x = torch.randn(2, 8, 2048)

    # In eval() mode, forward passes are strictly deterministic
    model.eval()
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
        assert torch.equal(out1, out2)

    # In train() mode, dropout induces stochastic zeroing
    model.train()
    out_train1 = model(x)
    out_train2 = model(x)
    assert not torch.equal(out_train1, out_train2)
    # Check that some values are zeroed out
    assert (out_train1 == 0.0).any()


def test_temporal_dtype_device_preservation():
    """Verify output preserves float32 / float64 dtypes and executes on CUDA when available."""
    model = build_temporal_model()
    model.eval()

    # Float32
    x32 = torch.randn(2, 4, 2048, dtype=torch.float32)
    assert model(x32).dtype == torch.float32

    # Float64
    model64 = build_temporal_model().to(torch.float64)
    model64.eval()
    x64 = torch.randn(2, 4, 2048, dtype=torch.float64)
    assert model64(x64).dtype == torch.float64

    # CUDA if available
    if torch.cuda.is_available():
        model_cuda = build_temporal_model().cuda()
        model_cuda.eval()
        x_cuda = torch.randn(2, 4, 2048, device="cuda")
        out_cuda = model_cuda(x_cuda)
        assert out_cuda.device.type == "cuda"
        assert out_cuda.shape == (2, 2048)


def test_temporal_gradient_flow():
    """Verify gradients propagate through LSTM parameters during backpropagation."""
    model = build_temporal_model()
    model.train()

    x = torch.randn(2, 6, 2048, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.abs().sum() > 0.0
    for name, p in model.lstm.named_parameters():
        assert p.grad is not None, f"Parameter {name} did not receive gradients"
        assert p.grad.abs().sum() > 0.0, f"Parameter {name} has all-zero gradients"
