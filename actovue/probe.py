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

# The obalcells/hallucination-probes repo uses its own layout: one directory per
# probe (for example qwen2_5_7b_linear), with probe_config.json, training_config.json,
# and a probe_head.bin torch state dict holding weight [1, hidden] and bias [1].
OBALCELLS_CONFIG = "probe_config.json"
OBALCELLS_TRAIN = "training_config.json"
OBALCELLS_HEAD = "probe_head.bin"


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
            raise ValueError(
                f"hidden must be 2D [num_tokens, hidden_size], got shape {tuple(hidden.shape)}"
            )
        if hidden.shape[1] != self.config.hidden_size:
            raise ValueError(
                f"hidden dim {hidden.shape[1]} does not match "
                f"probe hidden_size {self.config.hidden_size}"
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

    @classmethod
    def from_obalcells(
        cls, source: str, probe_name: str | None = None, revision: str | None = None
    ) -> Probe:
        """Load an anchor probe from the obalcells/hallucination-probes layout.

        source is either that hub repo id (then probe_name selects the subdir, for
        example 'qwen2_5_7b_linear') or a local probe directory. Their head is a
        torch state dict with weight [1, hidden] and bias [1]; the layer and hidden
        size come from probe_config.json and the threshold and base model from
        training_config.json when present.
        """
        cfg_path, train_path, head_path, probe_id = _resolve_obalcells(source, probe_name, revision)
        probe_config = json.loads(Path(cfg_path).read_text())
        train_config = json.loads(Path(train_path).read_text()) if train_path else {}

        hidden_size = int(probe_config["hidden_size"])
        layer_index = int(probe_config.get("layer_idx", train_config.get("layer")))
        base_model = train_config.get("model_name") or f"obalcells/{probe_id}"
        threshold = float(train_config.get("probe_threshold", 0.5))

        # weights_only avoids executing arbitrary pickle from a downloaded file.
        state = torch.load(head_path, map_location="cpu", weights_only=True)
        if "weight" not in state:
            raise ValueError(f"{head_path} has no 'weight' tensor (keys: {sorted(state)})")
        bias = state.get("bias", torch.zeros((), dtype=torch.float32))

        config = ProbeConfig(
            hidden_size=hidden_size,
            layer_index=layer_index,
            base_model=base_model,
            threshold=threshold,
        )
        return cls(config=config, weight=state["weight"], bias=bias, probe_id=probe_id)

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


def load_probe(source: str, revision: str | None = None) -> Probe:
    """Load a probe, picking the format from the source string.

    "repo::name" (or "dir::name") loads the obalcells layout, selecting the named
    subdir. Anything else loads the canonical actovue layout. This is what
    ACTOVUE_PROBE accepts, so the Stage-0 anchor
    ("obalcells/hallucination-probes::qwen2_5_7b_linear") and a published probe
    ("actovue/qwen3-14b-halu-probe-v1") both work through one entry point.
    """
    if "::" in source:
        repo, probe_name = source.split("::", 1)
        return Probe.from_obalcells(repo, probe_name=probe_name, revision=revision)
    return Probe.from_pretrained(source, revision=revision)


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


def _resolve_obalcells(
    source: str, probe_name: str | None, revision: str | None
) -> tuple[str, str | None, str, str]:
    """Return (probe_config, training_config or None, probe_head, probe_id)."""
    local = Path(source)
    if local.is_dir():
        pdir = local / probe_name if probe_name else local
        train = pdir / OBALCELLS_TRAIN
        return (
            str(pdir / OBALCELLS_CONFIG),
            str(train) if train.exists() else None,
            str(pdir / OBALCELLS_HEAD),
            probe_name or pdir.name,
        )
    if not probe_name:
        raise ValueError(
            "probe_name is required when loading from a hub repo, e.g. 'qwen2_5_7b_linear'"
        )
    from huggingface_hub import hf_hub_download

    cfg = hf_hub_download(source, f"{probe_name}/{OBALCELLS_CONFIG}", revision=revision)
    head = hf_hub_download(source, f"{probe_name}/{OBALCELLS_HEAD}", revision=revision)
    try:
        train = hf_hub_download(source, f"{probe_name}/{OBALCELLS_TRAIN}", revision=revision)
    except Exception:
        train = None
    return cfg, train, head, probe_name
