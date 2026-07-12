"""Measure the decode-throughput cost of running the probe.

This is the Stage-1 number: with a fixed decode workload at a fixed batch size,
how much slower is generation with the probe op in the graph than without it. It
runs the same workload twice through vLLM, once with the probe worker extension
active and once without, and reports tokens per second for each plus the overhead
percentage.

The overhead is (baseline_tps - probed_tps) / baseline_tps. The target is at or
below 3 percent. Every published number comes from the same GPU class in one
sitting, because throughput is not comparable across cards or driver versions.

vLLM is imported lazily. This runs on the GPU host; the exact worker-extension
launch flags are confirmed against the installed vLLM before the first run.

Run (baseline then probed handled internally):
    python bench/run_overhead.py --model Qwen/Qwen2.5-7B-Instruct \
        --probe actovue/qwen2.5-7b-halu-probe-v1 --batch 32 --max-tokens 256
"""

from __future__ import annotations

import argparse
import os
import time


def build_prompts(batch: int) -> list[str]:
    base = "Write a detailed paragraph about the history of the following topic: "
    topics = [
        "the printing press",
        "deep-sea exploration",
        "the standardization of time zones",
        "vaccination",
        "the transistor",
        "medieval trade routes",
        "photography",
        "antibiotics",
    ]
    return [base + topics[i % len(topics)] for i in range(batch)]


def run_once(model: str, prompts: list[str], max_tokens: int, probe: str | None) -> float:
    """Generate the workload once and return decode tokens per second."""
    from vllm import LLM, SamplingParams

    if probe:
        os.environ["ACTOVUE_PROBE"] = probe
    else:
        os.environ.pop("ACTOVUE_PROBE", None)

    # With a probe set, launch with the worker extension so the op is installed.
    kwargs = {"model": model, "enforce_eager": False}
    if probe:
        kwargs["worker_extension_cls"] = "actovue.worker_ext.ProbeWorkerExtension"
    llm = LLM(**kwargs)

    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True, temperature=0.0)
    # Warm up so compilation and capture are not timed.
    llm.generate(prompts[:1], params)

    start = time.perf_counter()
    outputs = llm.generate(prompts, params)
    elapsed = time.perf_counter() - start

    generated = sum(len(o.outputs[0].token_ids) for o in outputs)
    return generated / elapsed


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure decode overhead of the probe op")
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", required=True, help="probe repo or path used for the probed run")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    prompts = build_prompts(args.batch)

    baseline_tps = run_once(args.model, prompts, args.max_tokens, probe=None)
    probed_tps = run_once(args.model, prompts, args.max_tokens, probe=args.probe)
    overhead = (baseline_tps - probed_tps) / baseline_tps

    print(f"model            {args.model}")
    print(f"batch            {args.batch}")
    print(f"max_tokens       {args.max_tokens}")
    print(f"baseline tok/s   {baseline_tps:.1f}")
    print(f"probed   tok/s   {probed_tps:.1f}")
    print(f"overhead         {overhead:.2%}  (target <= 3.00%)")


if __name__ == "__main__":
    main()
