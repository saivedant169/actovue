"""Request parsing, span grouping, and response shaping.

None of this touches the model. It turns the per-request knob out of
SamplingParams.extra_args into a small validated object, groups a list of token
scores into flagged spans, and shapes the field that rides back on the
OpenAI-compatible response. Keeping it here means it is all testable on a laptop
with no GPU and no vLLM.

Request in:  extra_args = {"actovue": {"enabled": true, "threshold": 0.5}}
Response out: choices[0].actovue = {probe_id, threshold, token_scores, flagged_spans}

flagged_spans use half-open [start_token, end_token) indices, so
tokens[start_token:end_token] is exactly the flagged run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

FIELD = "actovue"


@dataclass(frozen=True)
class ActovueParams:
    """Per-request settings after validation."""

    enabled: bool
    threshold: float


@dataclass(frozen=True)
class FlaggedSpan:
    start_token: int
    end_token: int  # exclusive
    max_score: float
    text: str

    def to_dict(self) -> dict:
        return {
            "start_token": self.start_token,
            "end_token": self.end_token,
            "max_score": self.max_score,
            "text": self.text,
        }


def parse_params(extra_args: dict | None, default_threshold: float = 0.5) -> ActovueParams | None:
    """Read the actovue block out of extra_args.

    Returns None when the caller did not ask for scores, so the caller can skip
    attaching anything. Raises on a malformed block rather than guessing, because
    a silently ignored threshold is worse than a clear error.
    """
    if not extra_args:
        return None
    block = extra_args.get(FIELD)
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError(f"extra_args[{FIELD!r}] must be an object, got {type(block).__name__}")

    unknown = set(block) - {"enabled", "threshold"}
    if unknown:
        raise ValueError(f"unknown {FIELD} keys: {sorted(unknown)}")

    if not block.get("enabled", False):
        return None

    threshold = block.get("threshold", default_threshold)
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValueError(f"{FIELD}.threshold must be a number, got {threshold!r}")
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{FIELD}.threshold must be in [0, 1], got {threshold}")

    return ActovueParams(enabled=True, threshold=threshold)


def group_spans(
    scores: Sequence[float],
    tokens: Sequence[str] | None,
    threshold: float,
) -> list[FlaggedSpan]:
    """Group contiguous tokens at or above threshold into spans.

    A span is a maximal run of positions i where scores[i] >= threshold. max_score
    is the peak inside the run; text is the joined token strings when tokens is
    given, else an empty string. This is deliberately dumb: smart claim
    segmentation is a client-side or later-version concern.
    """
    if tokens is not None and len(tokens) != len(scores):
        raise ValueError(f"tokens length {len(tokens)} does not match scores length {len(scores)}")

    spans: list[FlaggedSpan] = []
    start: int | None = None
    for i, s in enumerate(scores):
        if s >= threshold:
            if start is None:
                start = i
        elif start is not None:
            spans.append(_close_span(scores, tokens, start, i))
            start = None
    if start is not None:
        spans.append(_close_span(scores, tokens, start, len(scores)))
    return spans


def build_response(
    scores: Sequence[float],
    tokens: Sequence[str] | None,
    probe_id: str,
    threshold: float,
    round_ndigits: int | None = None,
) -> dict:
    """Shape the actovue response field.

    round_ndigits, when set, rounds the reported token scores for a smaller
    payload. Span max_score is left unrounded so a rounded display never hides a
    token that actually crossed the threshold.
    """
    reported = [round(float(s), round_ndigits) if round_ndigits is not None else float(s) for s in scores]
    spans = group_spans(scores, tokens, threshold)
    return {
        "probe_id": probe_id,
        "threshold": threshold,
        "token_scores": reported,
        "flagged_spans": [span.to_dict() for span in spans],
    }


def _close_span(
    scores: Sequence[float],
    tokens: Sequence[str] | None,
    start: int,
    end: int,
) -> FlaggedSpan:
    max_score = max(float(scores[i]) for i in range(start, end))
    text = "".join(tokens[start:end]) if tokens is not None else ""
    return FlaggedSpan(start_token=start, end_token=end, max_score=max_score, text=text)
