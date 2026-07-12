"""Probe weights, config, and the reference score.

A probe is a single linear head over one hidden state: score = sigmoid(w . h + b),
one value per token. The reference() method here is the numerical oracle. Every
other path that computes scores, including the in-graph custom op, is correct only
if it matches reference() within tolerance on the same inputs.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

CONFIG_NAME = "config.json"
HEAD_NAME = "head.safetensors"


@dataclass
class ProbeConfig:
    """Everything needed to place and interpret a probe, and to reproduce it.

    layer_index is absolute, not a fraction. Probes are trained at
    floor(0.95 * num_layers), but the trained-at layer is pinned here so serving
    never has to recompute it and never drifts if a base model changes depth.
    """

    hidden_size: int
    layer_index: int
    base_model: str
    threshold: float = 0.5
    training_data_hash: str | None = None
    probe_type: str = "linear"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.layer_index < 0:
            raise ValueError(f"layer_index must be non-negative, got {self.layer_index}")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        if not self.base_model:
            raise ValueError("base_model must be set")
        if self.probe_type != "linear":
            raise ValueError(f"only linear probes are supported in v1, got {self.probe_type!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ProbeConfig:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)


@dataclass
class Probe:
    """A loaded linear probe plus its config.

    weight is [hidden_size], bias is a scalar, both fp32. The head is kept in fp32
    on purpose: it is a single matvec per token, so the cost is negligible, and
    fp32 removes head precision as a variable when comparing the graph op against
    this reference.
    """

    config: ProbeConfig
    weight: torch.Tensor
    bias: torch.Tensor
    probe_id: str = "unknown"
    _validated: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        self.weight = _as_vector(self.weight, self.config.hidden_size, "weight")
        self.bias = _as_scalar(self.bias, "bias")
        self._validated = True

    @torch.no_grad()
    def reference(self, hidden: torch.Tensor) -> torch.Tensor:
        """Reference score for each row of hidden.

        hidden is [num_tokens, hidden_size] in any dtype. Returns [num_tokens]
        fp32 scores in [0, 1]. Computation happens in fp32 regardless of input
        dtype so this stays the single source of truth.
        """
        if hidden.dim() != 2:
            raise ValueError(f"hidden must be 2D [num_tokens, hidden_size], got shape {tuple(hidden.shape)}")
        if hidden.shape[1] != self.config.hidden_size:
            raise ValueError(
                f"hidden dim {hidden.shape[1]} does not match probe hidden_size {self.config.hidden_size}"
            )
        w = self.weight.to(device=hidden.device, dtype=torch.float32)
        b = self.bias.to(device=hidden.device, dtype=torch.float32)
        logits = hidden.to(torch.float32) @ w + b
        return torch.sigmoid(logits)

    def flagged(self, scores: torch.Tensor) -> torch.Tensor:
        """Boolean mask of tokens at or above the probe threshold."""
        return scores >= self.config.threshold

    @classmethod
    def from_pretrained(cls, source: str, revision: str | None = None) -> Probe:
        """Load a probe from a local directory or a Hugging Face repo id.

        A local path is used directly. Anything else is treated as a hub repo and
        the config plus head file are downloaded on demand.
        """
        config_path, head_path, probe_id = _resolve(source, revision)
        config = ProbeConfig.from_dict(json.loads(Path(config_path).read_text()))
        state = _load_safetensors(head_path)
        if "weight" not in state:
            raise ValueError(f"{head_path} has no 'weight' tensor (keys: {sorted(state)})")
        weight = state["weight"]
        bias = state.get("bias", torch.zeros((), dtype=torch.float32))
        return cls(config=config, weight=weight, bias=bias, probe_id=probe_id)

    def save(self, directory: str | os.PathLike) -> None:
        """Write config.json and head.safetensors in the canonical layout."""
        from safetensors.torch import save_file

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        (out / CONFIG_NAME).write_text(json.dumps(self.config.to_dict(), indent=2))
        save_file(
            {"weight": self.weight.contiguous(), "bias": self.bias.contiguous()},
            str(out / HEAD_NAME),
        )


def _as_vector(t: torch.Tensor, hidden_size: int, name: str) -> torch.Tensor:
    t = t.reshape(-1).to(torch.float32)
    if t.numel() != hidden_size:
        raise ValueError(f"{name} has {t.numel()} elements, expected {hidden_size}")
    return t


def _as_scalar(t: torch.Tensor, name: str) -> torch.Tensor:
    t = t.reshape(-1).to(torch.float32)
    if t.numel() != 1:
        raise ValueError(f"{name} must be a single value, got {t.numel()} elements")
    return t.reshape(())


def _load_safetensors(path: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(path)


def _resolve(source: str, revision: str | None) -> tuple[str, str, str]:
    local = Path(source)
    if local.is_dir():
        return str(local / CONFIG_NAME), str(local / HEAD_NAME), local.name
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(source, CONFIG_NAME, revision=revision)
    head_path = hf_hub_download(source, HEAD_NAME, revision=revision)
    return config_path, head_path, source
