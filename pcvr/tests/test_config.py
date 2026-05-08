"""Tests for the Config dataclass — schema validation and defaults."""
import pytest
from dataclasses import asdict


def test_config_defaults_are_sane():
    from configs.baseline import Config
    cfg = Config()
    assert cfg.d_model == 64
    assert cfg.use_bf16 is True
    assert cfg.dense_optimizer == "adamw"
    assert cfg.sparse_optimizer == "adagrad"
    assert cfg.loss_type == "bce"
    assert cfg.warmup_steps >= 0
    assert 0 < cfg.min_lr_factor <= 1


def test_config_rejects_bad_optimizer():
    from configs.baseline import Config
    with pytest.raises(ValueError, match="dense_optimizer"):
        Config(dense_optimizer="lion")
    with pytest.raises(ValueError, match="sparse_optimizer"):
        Config(sparse_optimizer="adam")


def test_config_rejects_bad_loss():
    from configs.baseline import Config
    with pytest.raises(ValueError, match="loss_type"):
        Config(loss_type="hinge")


def test_config_rejects_bad_ns_tokenizer():
    from configs.baseline import Config
    with pytest.raises(ValueError):
        Config(ns_tokenizer_type="unknown")


def test_config_round_trips_to_dict():
    from configs.baseline import Config
    cfg = Config(seed=7, d_model=32)
    d = asdict(cfg)
    assert d["seed"] == 7
    assert d["d_model"] == 32
    cfg2 = Config(**d)
    assert asdict(cfg2) == d


def test_config_rejects_bad_seq_encoder_and_rank_mixer():
    from configs.baseline import Config
    with pytest.raises(ValueError, match="seq_encoder_type"):
        Config(seq_encoder_type="lstm")
    with pytest.raises(ValueError, match="rank_mixer_mode"):
        Config(rank_mixer_mode="custom")


def test_config_rejects_bad_bounds():
    from configs.baseline import Config
    with pytest.raises(ValueError, match="min_lr_factor"):
        Config(min_lr_factor=0.0)
    with pytest.raises(ValueError, match="min_lr_factor"):
        Config(min_lr_factor=1.5)
    with pytest.raises(ValueError, match="warmup_steps"):
        Config(warmup_steps=-1)
