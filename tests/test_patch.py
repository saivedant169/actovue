"""Layer selection and the forward-wrap injection, on a plain nn.Module."""

from __future__ import annotations

import pytest
import torch

from actovue import ops
from actovue.patch import install_probe, target_layer_index


@pytest.mark.parametrize(
    "num_layers,expected",
    [(40, 38), (24, 22), (64, 60), (32, 30), (1, 0), (80, 76)],
)
def test_target_layer_index(num_layers, expected):
    assert target_layer_index(num_layers) == expected


def test_target_layer_index_rejects_bad_input():
    with pytest.raises(ValueError):
        target_layer_index(0)
    with pytest.raises(ValueError):
        target_layer_index(10, fraction=1.5)


class FakeLayer(torch.nn.Module):
    """Stands in for a decoder layer: identity-ish, returns a hidden state."""

    def __init__(self, hidden_size: int, as_tuple: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.as_tuple = as_tuple

    def forward(self, x):
        h = torch.tanh(x)
        return (h, None) if self.as_tuple else h


def test_install_probe_writes_scores_and_preserves_output():
    torch.manual_seed(0)
    H = 16
    layer = FakeLayer(H)
    weight, bias = torch.randn(H), torch.randn(())
    buf = ops.make_buffer(8)

    uninstall = install_probe(layer, weight, bias, buf)
    x = torch.randn(4, H)
    out = layer(x)

    # Output is unchanged by the wrap.
    assert torch.equal(out, torch.tanh(x))
    # Scores were written for the layer's hidden state.
    expected = torch.sigmoid(torch.tanh(x).float() @ weight.float() + bias.float())
    assert torch.allclose(buf[:4], expected, atol=1e-6)

    uninstall()
    assert layer.forward(x) is not None  # restored, still callable


def test_install_probe_handles_tuple_output():
    torch.manual_seed(1)
    H = 12
    layer = FakeLayer(H, as_tuple=True)
    weight, bias = torch.randn(H), torch.randn(())
    buf = ops.make_buffer(6)
    install_probe(layer, weight, bias, buf)

    x = torch.randn(3, H)
    out = layer(x)
    assert isinstance(out, tuple)
    expected = torch.sigmoid(torch.tanh(x).float() @ weight.float() + bias.float())
    assert torch.allclose(buf[:3], expected, atol=1e-6)


def test_install_probe_uses_callable_buffer():
    torch.manual_seed(2)
    H = 8
    layer = FakeLayer(H)
    weight, bias = torch.randn(H), torch.randn(())
    buffers = {"buf": ops.make_buffer(4)}
    install_probe(layer, weight, bias, lambda: buffers["buf"])

    # Swap the buffer after install; the wrap must pick up the new one.
    buffers["buf"] = ops.make_buffer(4)
    buffers["buf"].fill_(-1.0)
    layer(torch.randn(2, H))
    assert (buffers["buf"][:2] != -1.0).all()


def test_install_probe_rejects_wrong_hidden_size():
    layer = FakeLayer(16)
    weight, bias = torch.randn(8), torch.randn(())  # mismatched
    install_probe(layer, weight, bias, ops.make_buffer(4))
    with pytest.raises(ValueError):
        layer(torch.randn(2, 16))
