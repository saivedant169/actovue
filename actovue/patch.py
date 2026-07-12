"""Injecting the probe op into a model.

The probe runs by wrapping one decoder layer's forward before the model is
compiled, so the custom op is captured into the compiled graph and replayed with
it. This is not a register_forward_hook: hooks do not run during CUDA-graph
replay, which is the whole reason the naive approach forces eager mode.

The wrapping logic and the layer choice live here as plain functions with no vLLM
import, so they can be exercised on CPU with an ordinary nn.Module. The part that
knows about vLLM (finding model.model.layers and wrapping before compilation)
lives in worker_ext.py.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from actovue.ops import write_scores

BufferSource = torch.Tensor | Callable[[], torch.Tensor]


def target_layer_index(num_layers: int, fraction: float = 0.95) -> int:
    """The layer the probe reads, floor(fraction * num_layers), clamped in range.

    Matches the training convention (floor(0.95 * L)). A probe config still pins
    its own absolute layer_index; this helper is for choosing where to train.
    """
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    return min(math.floor(fraction * num_layers), num_layers - 1)


def install_probe(
    layer: torch.nn.Module,
    weight: torch.Tensor,
    bias: torch.Tensor,
    buffer: BufferSource,
) -> Callable[[], None]:
    """Wrap layer.forward so it writes probe scores for its output hidden state.

    The original output is returned unchanged; the only added effect is the op
    writing into the buffer. buffer may be a tensor or a zero-argument callable
    returning the current buffer, so the buffer can be resized without rewrapping.

    Returns an uninstall callable that restores the original forward.
    """
    original = layer.forward

    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        hidden = out[0] if isinstance(out, tuple) else out
        buf = buffer() if callable(buffer) else buffer
        _write(hidden, weight, bias, buf)
        return out

    wrapped.__actovue_original__ = original  # type: ignore[attr-defined]
    layer.forward = wrapped  # type: ignore[method-assign]

    def uninstall() -> None:
        layer.forward = original  # type: ignore[method-assign]

    return uninstall


def _write(
    hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, buffer: torch.Tensor
) -> None:
    if not isinstance(hidden, torch.Tensor):
        raise TypeError(f"expected the layer to output a tensor, got {type(hidden).__name__}")
    if hidden.dim() != 2:
        raise ValueError(
            f"expected hidden state [num_tokens, hidden_size], got shape {tuple(hidden.shape)}; "
            "actovue v1 targets vLLM V1 flattened decode tensors"
        )
    if hidden.shape[1] != weight.shape[0]:
        raise ValueError(
            f"hidden size {hidden.shape[1]} does not match probe weight {weight.shape[0]}; "
            "wrong probe for this model"
        )
    write_scores(hidden, weight, bias, buffer)
