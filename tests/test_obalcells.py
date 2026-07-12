"""Loading anchor probes in the obalcells/hallucination-probes layout.

Their format is not ours: probe_config.json plus training_config.json plus a
probe_head.bin torch state dict with weight [1, hidden] and bias [1]. The offline
test pins the parsing against a fixture in that exact shape. The network test
loads the real Qwen2.5-7B linear probe and is skipped in CI and offline.
"""

from __future__ import annotations

import json
from collections import OrderedDict

import pytest
import torch

from actovue.probe import Probe, load_probe


def _write_fixture(pdir, hidden_size, with_training=True):
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "probe_config.json").write_text(
        json.dumps(
            {"target_layer_name": "LlamaDecoderLayer", "layer_idx": 30, "hidden_size": hidden_size}
        )
    )
    if with_training:
        (pdir / "training_config.json").write_text(
            json.dumps(
                {
                    "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    "layer": 30,
                    "probe_threshold": 0.5,
                    "probe_id": pdir.name,
                }
            )
        )
    state = OrderedDict(
        weight=torch.randn(1, hidden_size, dtype=torch.bfloat16),
        bias=torch.randn(1, dtype=torch.bfloat16),
    )
    torch.save(state, pdir / "probe_head.bin")


def test_from_obalcells_by_probe_name(tmp_path):
    h = 4096
    _write_fixture(tmp_path / "llama3_1_8b_linear", h)
    p = Probe.from_obalcells(str(tmp_path), probe_name="llama3_1_8b_linear")
    assert p.config.hidden_size == h
    assert p.config.layer_index == 30
    assert p.config.base_model == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    assert p.config.threshold == 0.5
    assert p.weight.shape == (h,) and p.bias.shape == ()
    scores = p.reference(torch.randn(8, h))
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_from_obalcells_direct_dir(tmp_path):
    h = 2048
    pdir = tmp_path / "some_linear"
    _write_fixture(pdir, h)
    p = Probe.from_obalcells(str(pdir))  # dir is the probe itself, no probe_name
    assert p.config.hidden_size == h and p.probe_id == "some_linear"


def test_from_obalcells_without_training_config(tmp_path):
    h = 512
    pdir = tmp_path / "bare_linear"
    _write_fixture(pdir, h, with_training=False)
    p = Probe.from_obalcells(str(pdir))
    assert p.config.threshold == 0.5
    assert p.config.base_model.startswith("obalcells/")


def test_from_obalcells_repo_requires_probe_name():
    with pytest.raises(ValueError):
        Probe.from_obalcells("obalcells/hallucination-probes")


def test_load_probe_dispatches_on_double_colon(tmp_path):
    # "dir::name" routes to the obalcells loader; a plain path uses the actovue one.
    h = 256
    _write_fixture(tmp_path / "qwen2_5_7b_linear", h)
    p = load_probe(f"{tmp_path}::qwen2_5_7b_linear")
    assert p.config.hidden_size == h and p.probe_id == "qwen2_5_7b_linear"

    canonical = tmp_path / "ours"
    Probe(config=p.config, weight=p.weight, bias=p.bias, probe_id="ours").save(canonical)
    p2 = load_probe(str(canonical))  # no "::" -> actovue format
    assert p2.config.hidden_size == h


@pytest.mark.network
def test_from_obalcells_real_qwen():
    p = Probe.from_obalcells("obalcells/hallucination-probes", probe_name="qwen2_5_7b_linear")
    assert p.config.hidden_size > 0
    assert p.config.layer_index >= 0
    assert p.weight.shape == (p.config.hidden_size,)
    scores = p.reference(torch.randn(4, p.config.hidden_size))
    assert scores.min() >= 0.0 and scores.max() <= 1.0
