"""vLLM worker-side integration.

vLLM mixes a worker-extension class into its Worker when the server is started
with --worker-extension-cls actovue.worker_ext.ProbeWorkerExtension. Its methods
are reachable from the engine through collective_rpc, which is how scores get
pulled back after each step.

Everything vLLM-specific is imported lazily inside methods so that `import
actovue` and the CPU test suite never need vLLM present. The reusable, model-free
logic (reading the probe target from the environment, sizing the buffer, and
slicing the score buffer back into per-request lists) is written as plain
functions and is covered by the CPU tests.

The exact points at which vLLM exposes the model, the compilation boundary, and
the per-step request layout are validated directly against the installed vLLM
source on the GPU host before first use. The plan calls for that read; this module
is the shape it plugs into, not a claim that the wiring has run.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

ENV_PROBE = "ACTOVUE_PROBE"


def probe_source_from_env(env: dict | None = None) -> str | None:
    """The probe repo or path from ACTOVUE_PROBE, or None if unset."""
    value = (env or os.environ).get(ENV_PROBE, "").strip()
    return value or None


def buffer_length(max_num_batched_tokens: int) -> int:
    """Score buffer length: one slot per token that can be in a batch."""
    if max_num_batched_tokens <= 0:
        raise ValueError(f"max_num_batched_tokens must be positive, got {max_num_batched_tokens}")
    return max_num_batched_tokens


def extract_scores(buffer: Sequence[float], query_start_loc: Sequence[int]) -> list[list[float]]:
    """Slice the flat score buffer into per-request score lists.

    query_start_loc is the cumulative token boundary array vLLM uses to lay out a
    batch: request r owns positions [query_start_loc[r], query_start_loc[r + 1]).
    Everything past the last boundary is padding and is dropped.
    """
    starts = list(query_start_loc)
    if len(starts) < 2:
        return []
    if starts[0] != 0:
        raise ValueError(f"query_start_loc must start at 0, got {starts[0]}")
    out: list[list[float]] = []
    for r in range(len(starts) - 1):
        s, e = starts[r], starts[r + 1]
        if e < s:
            raise ValueError(f"query_start_loc not non-decreasing at {r}: {s} then {e}")
        if e > len(buffer):
            raise ValueError(f"query_start_loc {e} runs past buffer length {len(buffer)}")
        out.append([float(buffer[i]) for i in range(s, e)])
    return out


class ProbeWorkerExtension:
    """Mixed into the vLLM Worker via --worker-extension-cls.

    Lifecycle, driven from the engine over collective_rpc:
      actovue_load     load the probe named by ACTOVUE_PROBE onto the worker device
      actovue_install  choose the target layer, allocate the buffer, wrap forward
                       before the model is compiled and its graph captured
      actovue_drain    after execute_model, slice the buffer into per-request scores
    """

    def actovue_load(self) -> dict:
        from actovue.probe import Probe

        source = probe_source_from_env()
        if source is None:
            raise RuntimeError(f"{ENV_PROBE} is not set; nothing to serve")
        device = getattr(self, "device", "cuda")
        probe = Probe.from_pretrained(source)
        self._actovue_probe = probe
        self._actovue_weight = probe.weight.to(device)
        self._actovue_bias = probe.bias.to(device)
        return {"probe_id": probe.probe_id, "layer_index": probe.config.layer_index}

    def actovue_install(self) -> None:
        import torch

        from actovue.ops import make_buffer
        from actovue.patch import install_probe

        model = self._actovue_model()
        layers = model.model.layers
        probe = self._actovue_probe
        layer = layers[probe.config.layer_index]

        max_tokens = int(self._actovue_max_num_batched_tokens())
        device = getattr(self, "device", "cuda")
        self._actovue_buffer = make_buffer(buffer_length(max_tokens), device=device)
        self._actovue_uninstall = install_probe(
            layer,
            self._actovue_weight,
            self._actovue_bias,
            lambda: self._actovue_buffer,
        )
        # Match probe head dtype handling to the buffer device without forcing a
        # host sync; nothing is computed here, the op does the work in-graph.
        assert isinstance(self._actovue_buffer, torch.Tensor)

    def actovue_drain(self, query_start_loc: Sequence[int]) -> list[list[float]]:
        buf = self._actovue_buffer.detach().to("cpu")
        return extract_scores(buf.tolist(), list(query_start_loc))

    # The two accessors below name the vLLM attributes this class expects to find
    # on the Worker it is mixed into. They are isolated so the pod-side source read
    # only has to confirm or adjust these two lookups.
    def _actovue_model(self):
        return self.model_runner.model

    def _actovue_max_num_batched_tokens(self) -> int:
        return self.model_runner.scheduler_config.max_num_batched_tokens
