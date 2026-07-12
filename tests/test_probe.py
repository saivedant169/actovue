"""ProbeConfig validation, reference-shape guards, and save/load round-trip."""

from __future__ import annotations

import pytest
import torch

from actovue.probe import Probe, ProbeConfig


def make_probe(hidden_size: int = 16, layer_index: int = 12) -> Probe:
    torch.manual_seed(0)
    cfg = ProbeConfig(hidden_size=hidden_size, layer_index=layer_index, base_model="test/model")
    return Probe(config=cfg, weight=torch.randn(hidden_size), bias=torch.randn(()), probe_id="test")


def test_config_defaults_and_roundtrip():
    cfg = ProbeConfig(hidden_size=8, layer_index=4, base_model="m")
    assert cfg.threshold == 0.5
    assert cfg.probe_type == "linear"
    assert ProbeConfig.from_dict(cfg.to_dict()) == cfg


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_size": 0, "layer_index": 1, "base_model": "m"},
        {"hidden_size": 8, "layer_index": -1, "base_model": "m"},
        {"hidden_size": 8, "layer_index": 1, "base_model": ""},
        {"hidden_size": 8, "layer_index": 1, "base_model": "m", "threshold": 1.5},
        {"hidden_size": 8, "layer_index": 1, "base_model": "m", "probe_type": "lora"},
    ],
)
def test_config_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        ProbeConfig(**kwargs)


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        ProbeConfig.from_dict({"hidden_size": 8, "layer_index": 1, "base_model": "m", "mystery": 1})


def test_weight_and_bias_are_normalized():
    cfg = ProbeConfig(hidden_size=4, layer_index=1, base_model="m")
    # weight given as [1, hidden_size], bias as [1]: both get flattened.
    p = Probe(config=cfg, weight=torch.randn(1, 4), bias=torch.tensor([0.3]))
    assert p.weight.shape == (4,)
    assert p.bias.shape == ()
    assert p.weight.dtype == torch.float32


def test_wrong_weight_size_raises():
    cfg = ProbeConfig(hidden_size=4, layer_index=1, base_model="m")
    with pytest.raises(ValueError):
        Probe(config=cfg, weight=torch.randn(5), bias=torch.zeros(()))


def test_reference_shape_guards():
    p = make_probe(hidden_size=16)
    with pytest.raises(ValueError):
        p.reference(torch.randn(16))  # not 2D
    with pytest.raises(ValueError):
        p.reference(torch.randn(3, 8))  # wrong hidden size


def test_reference_range_and_flag():
    p = make_probe(hidden_size=16)
    scores = p.reference(torch.randn(20, 16))
    assert scores.shape == (20,)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    flags = p.flagged(scores)
    assert flags.dtype == torch.bool
    assert (flags == (scores >= p.config.threshold)).all()


def test_save_and_load_roundtrip(tmp_path):
    p = make_probe(hidden_size=32, layer_index=30)
    p.save(tmp_path)
    loaded = Probe.from_pretrained(str(tmp_path))
    assert loaded.config == p.config
    assert torch.equal(loaded.weight, p.weight)
    assert torch.equal(loaded.bias, p.bias)
    assert loaded.probe_id == tmp_path.name
    # Loaded probe scores identically.
    h = torch.randn(5, 32)
    assert torch.allclose(loaded.reference(h), p.reference(h))
