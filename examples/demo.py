"""End-to-end demo: generate with a probe and print per-token risk scores.

Runs on a GPU host with a CUDA build of vLLM installed. It starts the engine with
the actovue worker extension, generates a short completion, pulls the per-token
scores back, and shapes them with the same api helpers the server uses, so the
output here matches what a client sees on choices[0].actovue.

    ACTOVUE_PROBE=actovue/qwen2.5-7b-halu-probe-v1 \
        python examples/demo.py --model Qwen/Qwen2.5-7B-Instruct

vLLM is imported lazily so this file imports on any machine.
"""

from __future__ import annotations

import argparse

from actovue.api import build_response


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate with a probe and print token scores")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Who won the 1998 Nobel Prize in Physics, and for what?")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, worker_extension_cls="actovue.worker_ext.ProbeWorkerExtension")
    llm.collective_rpc("actovue_load")
    llm.collective_rpc("actovue_install")

    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    output = llm.generate([args.prompt], params)[0]

    tokens = [output.outputs[0].text]  # per-token text is filled in on the pod path
    token_ids = output.outputs[0].token_ids
    scores = _pull_scores(llm, len(token_ids))

    resp = build_response(scores, None, probe_id="demo", threshold=args.threshold)
    print("text:", output.outputs[0].text)
    print("token_scores:", [round(s, 3) for s in resp["token_scores"]])
    print("flagged_spans:", resp["flagged_spans"])
    _ = tokens  # placeholder until token strings are threaded through on the pod


def _pull_scores(llm, num_tokens: int) -> list[float]:
    """Drain scores for a single-request batch. Layout confirmed on the pod."""
    query_start_loc = [0, num_tokens]
    per_request = llm.collective_rpc("actovue_drain", args=(query_start_loc,))[0]
    return per_request[0] if per_request else []


if __name__ == "__main__":
    main()
