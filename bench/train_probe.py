"""Fit a linear probe head from cached hidden states.

The full recipe is: run the target model over labeled long-form text, cache the
hidden state at the target layer for each token, then fit a single linear head
that maps that hidden state to a hallucination score. This script owns the last
step, which is the numerically load-bearing one and needs no serving stack: given
activations and per-token labels, it fits the head and writes it in the canonical
probe layout.

The activation-capture step is a separate GPU pass (transformers or vLLM) that
writes acts.pt of shape [N, hidden] and labels.pt of shape [N] into a directory.
A helper for the transformers path is included and imported lazily.

Layer convention: train at floor(0.95 * num_layers) unless told otherwise, and
pin the chosen absolute index in the probe config.

Run:
    python bench/train_probe.py --acts-dir runs/qwen3-14b --base-model Qwen/Qwen3-14B \
        --layer-index 38 --out probes/qwen3-14b-halu-probe-v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from actovue.probe import Probe, ProbeConfig


def fit_linear_head(
    acts: torch.Tensor,
    labels: torch.Tensor,
    l2: float = 1e-3,
    max_iter: int = 200,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Logistic regression head over acts, fit in fp32 with LBFGS.

    acts is [N, hidden], labels is [N] in {0, 1}. Returns (weight [hidden], bias []).
    fp32 throughout so the trained head matches how it is served and scored.
    """
    if acts.dim() != 2:
        raise ValueError(f"acts must be [N, hidden], got {tuple(acts.shape)}")
    if labels.shape != (acts.shape[0],):
        raise ValueError(f"labels must be [N]={acts.shape[0]}, got {tuple(labels.shape)}")

    x = acts.to(torch.float32)
    y = labels.to(torch.float32)
    hidden = x.shape[1]
    weight = torch.zeros(hidden, requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS(
        [weight, bias], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe"
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        logits = x @ weight + bias
        loss = loss_fn(logits, y) + l2 * weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return weight.detach(), bias.detach().reshape(())


def capture_with_transformers(
    model_name: str, layer_index: int, prompts: list[str]
) -> torch.Tensor:
    """Optional GPU capture path: last-token hidden state at layer_index per prompt.

    Kept lazy so this file imports without transformers. For long-form token-level
    labels the caller captures every generated position; this helper covers the
    simple prompt-level case used for quick checks.
    """
    import torch as _torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=_torch.bfloat16, device_map="auto"
    )
    model.eval()
    out = []
    with _torch.no_grad():
        for prompt in prompts:
            ids = tok(prompt, return_tensors="pt").to(model.device)
            hs = model(**ids, output_hidden_states=True).hidden_states[layer_index]
            out.append(hs[0, -1].to("cpu", _torch.float32))
    return _torch.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit a linear probe head from cached activations")
    ap.add_argument(
        "--acts-dir", required=True, help="dir with acts.pt [N, hidden] and labels.pt [N]"
    )
    ap.add_argument("--base-model", required=True, help="base model id the probe is for")
    ap.add_argument(
        "--layer-index", type=int, required=True, help="absolute layer the acts came from"
    )
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--out", required=True, help="output probe directory")
    args = ap.parse_args()

    acts_dir = Path(args.acts_dir)
    acts = torch.load(acts_dir / "acts.pt")
    labels = torch.load(acts_dir / "labels.pt")

    weight, bias = fit_linear_head(acts, labels, l2=args.l2)

    config = ProbeConfig(
        hidden_size=acts.shape[1],
        layer_index=args.layer_index,
        base_model=args.base_model,
        threshold=args.threshold,
    )
    probe = Probe(config=config, weight=weight, bias=bias, probe_id=Path(args.out).name)
    probe.save(args.out)

    scores = probe.reference(acts.to(torch.float32))
    flagged = probe.flagged(scores).float().mean().item()
    print(f"wrote {args.out}  (N={acts.shape[0]}, hidden={acts.shape[1]}, flagged={flagged:.1%})")


if __name__ == "__main__":
    main()
