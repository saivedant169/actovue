"""actovue: per-token hallucination-probe scores from vLLM.

The public surface is small on purpose. Importing the package registers the
custom op. register() is the vLLM plugin entry point; the real model surgery
happens in the worker extension, not here, so it runs in the right process.
"""

from __future__ import annotations

from actovue import ops as ops  # noqa: F401  (import registers actovue::probe)
from actovue.api import (
    ActovueParams,
    FlaggedSpan,
    build_response,
    group_spans,
    parse_params,
)
from actovue.probe import Probe, ProbeConfig, load_probe

__version__ = "0.1.0"

__all__ = [
    "ActovueParams",
    "FlaggedSpan",
    "Probe",
    "ProbeConfig",
    "build_response",
    "group_spans",
    "load_probe",
    "parse_params",
    "register",
]


def register() -> None:
    """Called by vLLM through the vllm.general_plugins entry point at startup.

    Importing this package has already registered the actovue::probe op, which is
    all that must happen at plugin-load time. Layer wrapping, buffer allocation,
    and score draining run in the worker via
    actovue.worker_ext.ProbeWorkerExtension, which vLLM instantiates when started
    with --worker-extension-cls. Doing model surgery here would run it in the
    wrong process, before the worker holds the model.
    """
    return None
