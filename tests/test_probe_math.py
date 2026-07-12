"""The correctness contract: the in-graph op equals the reference score.

Every other path that produces scores is only allowed to exist if it matches
Probe.reference() on the same inputs. These tests pin that on CPU, including the
two properties the whole design depends on: the op survives torch.compile without
a graph break, and it is not deleted as dead code when its result feeds nothing.
"""

from __future__ import annotations

import pytest
import torch

from actovue import ops
from actovue.probe import Probe, ProbeConfig

CASES = [(1, 8), (5, 16), (32, 64), (128, 128)]
DTYPES = [torch.float32, torch.bfloat16, torch.float16]


def build(hidden_size: int) -> Probe:
    torch.manual_seed(hidden_size)
    cfg = ProbeConfig(hidden_size=hidden_size, layer_index=1, base_model="m")
    return Probe(config=cfg, weight=torch.randn(hidden_size), bias=torch.randn(()))


@pytest.mark.parametrize("n,hidden_size", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_op_matches_reference(n, hidden_size, dtype):
    p = build(hidden_size)
    hidden = torch.randn(n, hidden_size, dtype=dtype)
    ref = p.reference(hidden)
    buf = ops.make_buffer(n + 3)
    ops.write_scores(hidden, p.weight, p.bias, buf)
    # Both cast hidden to fp32 the same way, so this is exact, not merely close.
    assert torch.allclose(buf[:n], ref, atol=1e-6)


def test_op_leaves_padding_untouched():
    p = build(16)
    buf = ops.make_buffer(10)
    buf.fill_(-1.0)  # sentinel
    ops.write_scores(torch.randn(4, 16), p.weight, p.bias, buf)
    assert (buf[4:] == -1.0).all()


def test_op_rejects_oversized_input():
    p = build(16)
    buf = ops.make_buffer(3)
    with pytest.raises(ValueError):
        ops.write_scores(torch.randn(4, 16), p.weight, p.bias, buf)


def test_op_survives_compile_and_dce():
    """fullgraph compile with the op's result unused: it must still run.

    This is the property the CUDA-graph path relies on. If register_fake caused a
    graph break, fullgraph=True would raise. If the op were dead-code eliminated
    because nothing reads the buffer downstream, the buffer would stay zero.
    """
    p = build(32)

    def region(x, weight, bias, buf):
        h = torch.tanh(x)
        ops.write_scores(h, weight, bias, buf)  # side effect only, result unused
        return h.sum()

    compiled = torch.compile(region, backend="inductor", fullgraph=True)
    x = torch.randn(6, 32)
    buf = ops.make_buffer(8)
    compiled(x, p.weight, p.bias, buf)

    expected = p.reference(torch.tanh(x))
    assert (buf[:6] != 0).any(), "op was dead-code eliminated: buffer never written"
    assert torch.allclose(buf[:6], expected, atol=1e-5)


def test_compiled_matches_eager():
    p = build(24)
    x = torch.randn(7, 24)

    eager = ops.make_buffer(7)
    ops.write_scores(x, p.weight, p.bias, eager)

    def region(h, w, b, buf):
        ops.write_scores(h, w, b, buf)
        return buf.sum()

    compiled = torch.compile(region, backend="inductor", fullgraph=True)
    comp = ops.make_buffer(7)
    compiled(x, p.weight, p.bias, comp)

    assert torch.allclose(eager, comp, atol=1e-6)
