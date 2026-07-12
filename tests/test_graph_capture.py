"""CUDA-graph capture parity. Runs on a CUDA device, skipped on CPU.

This is the Stage-1 correctness contract. The op is only useful if it produces
the same scores after being captured into a CUDA graph and replayed on fresh
data as it does eagerly. Capture is only safe if the op does no host sync and no
allocation and writes into a fixed-address static buffer, which is exactly how
ops.probe and make_buffer are built.

Nothing here needs vLLM. It runs on any CUDA box, including a laptop GPU, so the
capture-safety of the op can be checked long before renting a serving-class card.
The end-to-end serving overhead number lives in bench/run_overhead.py.
"""

from __future__ import annotations

import pytest
import torch

from actovue import ops
from actovue.probe import Probe, ProbeConfig

pytestmark = pytest.mark.gpu

CUDA = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
def test_op_capture_replay_matches_reference():
    device = "cuda"
    torch.manual_seed(0)
    n, hidden_size = 32, 128
    cfg = ProbeConfig(hidden_size=hidden_size, layer_index=1, base_model="m")
    probe = Probe(
        config=cfg,
        weight=torch.randn(hidden_size, device=device),
        bias=torch.randn((), device=device),
    )

    static_hidden = torch.randn(n, hidden_size, device=device, dtype=torch.bfloat16)
    buf = ops.make_buffer(n, device=device)

    # Warm up on a side stream before capture, as CUDA-graph capture requires.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            ops.write_scores(static_hidden, probe.weight, probe.bias, buf)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.write_scores(static_hidden, probe.weight, probe.bias, buf)

    # Replay on fresh data written into the same static input tensor.
    static_hidden.copy_(torch.randn(n, hidden_size, device=device, dtype=torch.bfloat16))
    buf.zero_()
    graph.replay()
    torch.cuda.synchronize()

    ref = probe.reference(static_hidden)
    max_abs_diff = (buf[:n] - ref).abs().max().item()
    assert max_abs_diff <= 1e-2, f"capture/replay diverged from reference by {max_abs_diff}"


@pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
def test_op_capture_leaves_padding_untouched():
    device = "cuda"
    n, hidden_size = 8, 64
    weight = torch.randn(hidden_size, device=device)
    bias = torch.zeros((), device=device)
    static_hidden = torch.randn(n, hidden_size, device=device, dtype=torch.bfloat16)
    buf = ops.make_buffer(n + 4, device=device)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            ops.write_scores(static_hidden, weight, bias, buf)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.write_scores(static_hidden, weight, bias, buf)

    buf.fill_(-1.0)
    graph.replay()
    torch.cuda.synchronize()
    assert (buf[n:] == -1.0).all(), "op wrote into padding positions"
