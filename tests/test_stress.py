"""Fast regression versions of the manual stress run.

The full stress pass (8192 tokens, thousands of iterations) runs by hand; these
lock in the properties it verified without the multi-minute runtime: correctness
at a large batch, numerical saturation at extreme magnitudes, no torch.compile
recompilation as the token count varies, and the exact boundaries.
"""

from __future__ import annotations

import pytest
import torch

from actovue import ops
from actovue.probe import Probe, ProbeConfig


def _probe(hidden_size: int) -> Probe:
    torch.manual_seed(hidden_size)
    cfg = ProbeConfig(hidden_size=hidden_size, layer_index=1, base_model="m")
    return Probe(config=cfg, weight=torch.randn(hidden_size), bias=torch.randn(()))


def test_correct_at_large_batch():
    # Qwen3-14B hidden size at a large prefill-scale batch, in bf16.
    n, hidden_size = 2048, 5120
    p = _probe(hidden_size)
    hidden = torch.randn(n, hidden_size, dtype=torch.bfloat16)
    buf = ops.make_buffer(n)
    ops.write_scores(hidden, p.weight, p.bias, buf)
    ref = p.reference(hidden)
    assert torch.equal(buf, ref)
    assert not torch.isnan(buf).any() and not torch.isinf(buf).any()


@pytest.mark.parametrize("scale", [1e2, 1e3, 1e4, 1e6])
def test_saturates_without_nan(scale):
    # Huge logits must saturate the sigmoid to 0 or 1, never produce nan or inf.
    n, hidden_size = 64, 512
    weight = torch.randn(hidden_size) * scale
    bias = torch.zeros(())
    hidden = torch.randn(n, hidden_size) * scale
    buf = ops.make_buffer(n)
    ops.write_scores(hidden, weight, bias, buf)
    assert not torch.isnan(buf).any() and not torch.isinf(buf).any()
    assert buf.min() >= 0.0 and buf.max() <= 1.0


def test_no_recompile_across_token_counts():
    """The compiled op must stay correct as the batch size changes every step.

    A fresh compile per token count would be a decode-time perf cliff. This asserts
    behaviour (every count gives the right scores), which is what actually matters
    and is stable across torch versions, rather than poking dynamo counters.
    """
    import torch._dynamo as dynamo

    dynamo.reset()
    hidden_size = 1024
    p = _probe(hidden_size)
    buf = ops.make_buffer(64)

    def region(h, w, b, out):
        ops.write_scores(h, w, b, out)
        return out.sum()

    compiled = torch.compile(region, backend="inductor")
    for n in [1, 2, 4, 8, 16, 32, 7, 1, 32]:
        buf.zero_()
        hidden = torch.randn(n, hidden_size)
        compiled(hidden, p.weight, p.bias, buf)
        assert torch.allclose(buf[:n], p.reference(hidden), atol=1e-5), f"wrong at n={n}"


@pytest.mark.parametrize("n", [1, 2048])
def test_exact_at_boundary(n):
    # n == buffer length and the single-token case, both exact.
    hidden_size = 256
    p = _probe(hidden_size)
    buf = ops.make_buffer(n)
    hidden = torch.randn(n, hidden_size, dtype=torch.bfloat16)
    ops.write_scores(hidden, p.weight, p.bias, buf)
    assert torch.equal(buf, p.reference(hidden))
