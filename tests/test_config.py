"""Tests for configuration parsing, validation, and cross-validation constraints."""

from pathlib import Path
import pytest

from src.utils.config import AppConfig, load_config, save_config


def test_load_default_config():
    """Verify loading default.yaml succeeds and yields valid configuration."""
    config_path = Path("configs/default.yaml")
    assert config_path.exists(), "configs/default.yaml must exist"

    cfg = load_config(config_path)
    assert cfg.seed == 42
    assert cfg.data.sampling.num_frames == 16
    assert cfg.data.sampling.method == "random"
    assert cfg.data.sampling.short_video_policy == "reject"
    assert cfg.data.crop.image_size == 112
    assert cfg.data.crop.margin_ratio == 0.20
    assert cfg.data.crop.min_valid_frames == 12
    assert cfg.data.crop.max_fallback_ratio == 0.50
    assert cfg.data.normalization == "imagenet"


def test_config_cross_validation_min_valid_frames():
    """Verify rejection when min_valid_frames > num_frames."""
    cfg = AppConfig()
    cfg.data.sampling.num_frames = 10
    cfg.data.crop.min_valid_frames = 12

    with pytest.raises(ValueError, match="min_valid_frames .* cannot exceed sampling.num_frames"):
        cfg.validate()


def test_config_cross_validation_fallback_ratio_bounds():
    """Verify rejection when max_fallback_ratio is outside [0.0, 1.0]."""
    cfg = AppConfig()
    cfg.data.crop.max_fallback_ratio = 1.2
    with pytest.raises(ValueError, match="crop.max_fallback_ratio must be in"):
        cfg.validate()

    cfg.data.crop.max_fallback_ratio = -0.1
    with pytest.raises(ValueError, match="crop.max_fallback_ratio must be in"):
        cfg.validate()


def test_config_cross_validation_short_video_policy():
    """Verify short_video_policy strictly rejects non-'reject' options."""
    cfg = AppConfig()
    cfg.data.sampling.short_video_policy = "pad"
    with pytest.raises(ValueError, match="Must be 'reject'"):
        cfg.validate()


def test_config_cross_validation_normalization():
    """Verify normalization rejects options outside ('imagenet', 'none')."""
    cfg = AppConfig()
    cfg.data.normalization = "minmax"
    with pytest.raises(ValueError, match="Invalid data.normalization"):
        cfg.validate()


def test_config_save_and_reload(tmp_path: Path):
    """Verify saving and re-loading produces an equivalent configuration."""
    cfg = AppConfig(seed=99)
    save_path = tmp_path / "test_saved_config.yaml"
    save_config(cfg, save_path)

    reloaded = load_config(save_path)
    assert reloaded.seed == 99
    assert reloaded.to_dict() == cfg.to_dict()


def test_config_cross_validation_temporal_and_classifier():
    """Verify cross-validation between attention, temporal, and classifier dimensions."""
    cfg = AppConfig()
    cfg.validate()

    # Mismatch between attention output and temporal input
    cfg.model.temporal.input_dim = 1024
    with pytest.raises(ValueError, match="model.temporal.input_dim .* must match"):
        cfg.validate()

    # Reset
    cfg.model.temporal.input_dim = 2048

    # Mismatch between temporal output and classifier in_features
    cfg.model.classifier.in_features = 1024
    with pytest.raises(ValueError, match="model.classifier.in_features .* must match"):
        cfg.validate()


def test_config_temporal_lstm_vs_bilstm():
    """Verify TemporalConfig validation rules for LSTM and BiLSTM."""
    from src.utils.config import TemporalConfig

    # Valid LSTM
    t_lstm = TemporalConfig(type="lstm", bidirectional=False, hidden_dim=2048)
    assert t_lstm.output_dim == 2048

    # Invalid LSTM with bidirectional=True
    with pytest.raises(ValueError, match="temporal.type='lstm' requires bidirectional=False"):
        TemporalConfig(type="lstm", bidirectional=True, hidden_dim=2048)

    # Invalid LSTM with wrong hidden_dim
    with pytest.raises(ValueError, match="temporal.type='lstm' requires hidden_dim=2048"):
        TemporalConfig(type="lstm", bidirectional=False, hidden_dim=1024)

    # Valid BiLSTM
    t_bilstm = TemporalConfig(type="bilstm", bidirectional=True, hidden_dim=1024)
    assert t_bilstm.output_dim == 2048

    # Invalid BiLSTM with bidirectional=False
    with pytest.raises(ValueError, match="temporal.type='bilstm' requires bidirectional=True"):
        TemporalConfig(type="bilstm", bidirectional=False, hidden_dim=1024)

    # Invalid BiLSTM with wrong hidden_dim
    with pytest.raises(ValueError, match="temporal.type='bilstm' requires hidden_dim=1024"):
        TemporalConfig(type="bilstm", bidirectional=True, hidden_dim=2048)

