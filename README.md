# actovue

Per-token hallucination-probe scores from vLLM, computed on decode-phase hidden
states and streamed back through the OpenAI-compatible API.

actovue runs a small linear probe on a model's mid-layer hidden state for every
generated token, turning each token into a risk score in the range 0 to 1. The
probe matmul is baked into the compiled model as a custom torch op, so it runs
inside the CUDA graph during decode instead of forcing eager mode. Scores ride
back to the caller as an extra field on the OpenAI-compatible response.

This is measurement infrastructure. The probes are useful, and they also break
out of distribution. actovue treats those failure modes as a first-class part of
the project, not a footnote: a companion benchmark reports where the scores stop
being trustworthy, alongside the cases where they hold.

## Status

Early. The repo is being built stage by stage, and each stage lands with one
number that either passes a threshold or triggers a documented fallback.

| Stage | What it proves | The number | State |
|-------|----------------|-----------|-------|
| 0 | Reference probe scores decode tokens in eager vLLM | max abs diff vs reference <= 1e-2 (bf16) | code + CPU oracle done, GPU run pending |
| 1 | The probe op survives CUDA-graph capture | decode overhead at batch 32, target <= 3% | needs a CUDA device |
| 2 | Scores stream through the OpenAI API | end-to-end throughput delta <= 5% | needs stage 1 |
| 3 | New probes for Qwen3-14B and gpt-oss-20b | held-out AUROC >= 0.80 | needs a GPU |
| 4 | Honest benchmark of where probes break | the in-distribution minus out-of-distribution gap | needs stage 3 |
| 5 | Fresh-machine reproduction | headline number reproduces within 1 percent | needs stages 1 to 4 |

No performance number is published here until it has been measured on the
hardware named next to it. Anything marked pending has not been run yet.

## Install

Core probe math and the custom op run on plain torch, so you can install and test
without a GPU or vLLM:

    pip install actovue

To serve behind vLLM (needs a CUDA build of vLLM):

    pip install "actovue[serve]"

## Quickstart

Point the plugin at a probe and start vLLM as usual. vLLM discovers actovue
through its plugin entry point and loads it before the model is compiled.

    export ACTOVUE_PROBE=actovue/qwen2.5-7b-halu-probe-v1
    vllm serve Qwen/Qwen2.5-7B-Instruct

Ask for scores per request through `extra_args`:

    curl http://localhost:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "Who won the 1998 Nobel Prize in Physics?"}],
        "extra_args": {"actovue": {"enabled": true, "threshold": 0.5}}
      }'

The response carries `choices[0].actovue` with a score per generated token and
the spans that crossed the threshold.

## How it works

vLLM captures the decode step into a CUDA graph and replays it. Python forward
hooks do not run during that replay, which is why the naive way to read hidden
states forces eager mode and pays a large throughput cost. actovue avoids that by
registering the probe as a `torch.library` custom op and wrapping the target
layer's forward before compilation, so the op is captured into the graph and
replayed with it. Each replay writes scores into a static GPU buffer, which is
drained asynchronously after the step and matched back to requests.

Design in one breath:

- One layer, one linear head per served model, chosen at `floor(0.95 * num_layers)`.
- The op computes for every position and reports selectively, so per-request
  on and off never changes the graph shape.
- fp32 head, tensor-parallel size 1 for v1. Anything else is a hard error, not a
  silent wrong answer.

## Probes

Weights live on the Hugging Face hub as `actovue/<model>-halu-probe-v1`: a
safetensors head plus a config with the layer index, hidden size, threshold, base
model, and a hash of the training data. The v1 anchor reuses the Apache-2.0
probes from the hallucination-probes work by Obeso and colleagues (Nanda group);
new probes for Qwen3-14B and gpt-oss-20b are trained here.

## License

Apache-2.0. See [LICENSE](LICENSE).
