"""The in-graph probe op.

The probe is a matvec, a sigmoid, and a write into a preallocated buffer. That is
trivial to run eagerly, but the whole point of actovue is to run it during decode
inside the captured CUDA graph, where Python forward hooks never fire. So the work
is wrapped in a torch.library custom op:

  - It has a real side effect (mutating out_buf) and returns nothing, so the
    compiler cannot dead-code-eliminate it just because its output feeds no other
    tensor. A plain function would be deleted the moment nothing reads its result.
  - Its fake implementation is a no-op, so Dynamo can trace through it without a
    graph break and Inductor emits an opaque call inside the compiled region.

The kernel is backend-generic: the same code runs on CPU for tests and on CUDA in
production. Computation is fp32 so it matches Probe.reference() exactly.
"""

from __future__ import annotations

import torch


@torch.library.custom_op("actovue::probe", mutates_args=("out_buf",))
def probe(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out_buf: torch.Tensor,
) -> None:
    """Write sigmoid(hidden @ weight + bias) into out_buf[:num_tokens].

    hidden is [num_tokens, hidden_size]. weight is [hidden_size], bias is a scalar.
    out_buf is a preallocated fp32 [buffer_len] tensor with buffer_len at least
    num_tokens. Positions past num_tokens are left as they were; the host masks
    them out using the request layout after the step.
    """
    n = hidden.shape[0]
    scores = torch.sigmoid(
        hidden.to(torch.float32) @ weight.to(torch.float32) + bias.to(torch.float32)
    )
    out_buf[:n] = scores


@probe.register_fake
def _probe_fake(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out_buf: torch.Tensor,
) -> None:
    # No output tensor to fake-allocate; the effect is the mutation of out_buf,
    # which the compiler already knows about from mutates_args. Nothing to do.
    return None


def make_buffer(buffer_len: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Allocate the static score buffer the op writes into.

    In serving this is sized to max_num_batched_tokens and allocated once, then
    reused every step so no allocation happens inside the graph.
    """
    if buffer_len <= 0:
        raise ValueError(f"buffer_len must be positive, got {buffer_len}")
    return torch.zeros(buffer_len, dtype=torch.float32, device=device)


def write_scores(
    hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, out_buf: torch.Tensor
) -> None:
    """Invoke the op. Thin wrapper so callers do not touch torch.ops directly."""
    if hidden.shape[0] > out_buf.shape[0]:
        raise ValueError(
            f"buffer of length {out_buf.shape[0]} too small for {hidden.shape[0]} tokens"
        )
    torch.ops.actovue.probe(hidden, weight, bias, out_buf)
