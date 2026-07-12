"""Request parsing, span grouping, response shaping, and buffer slicing."""

from __future__ import annotations

import pytest

from actovue.api import build_response, group_spans, parse_params
from actovue.worker_ext import extract_scores

# parse_params


def test_parse_none_and_missing_block():
    assert parse_params(None) is None
    assert parse_params({}) is None
    assert parse_params({"other": 1}) is None


def test_parse_disabled_returns_none():
    assert parse_params({"actovue": {"enabled": False}}) is None
    assert parse_params({"actovue": {}}) is None  # enabled defaults to False


def test_parse_enabled_defaults_threshold():
    p = parse_params({"actovue": {"enabled": True}}, default_threshold=0.4)
    assert p is not None and p.enabled and p.threshold == 0.4


def test_parse_enabled_with_threshold():
    p = parse_params({"actovue": {"enabled": True, "threshold": 0.7}})
    assert p.threshold == 0.7


@pytest.mark.parametrize(
    "block",
    [
        {"enabled": True, "threshold": 1.5},
        {"enabled": True, "threshold": -0.1},
        {"enabled": True, "threshold": "high"},
        {"enabled": True, "threshold": True},  # bool is not a valid number here
        {"enabled": True, "mystery": 1},
    ],
)
def test_parse_rejects_bad_block(block):
    with pytest.raises(ValueError):
        parse_params({"actovue": block})


def test_parse_rejects_non_object_block():
    with pytest.raises(ValueError):
        parse_params({"actovue": [1, 2, 3]})


# group_spans


def test_group_spans_empty_and_all_below():
    assert group_spans([], None, 0.5) == []
    assert group_spans([0.1, 0.2, 0.3], None, 0.5) == []


def test_group_spans_single_run():
    spans = group_spans([0.1, 0.9, 0.8, 0.2], ["a", "b", "c", "d"], 0.5)
    assert len(spans) == 1
    s = spans[0]
    assert (s.start_token, s.end_token) == (1, 3)
    assert s.text == "bc"
    assert s.max_score == pytest.approx(0.9)


def test_group_spans_multiple_and_trailing_run():
    scores = [0.9, 0.1, 0.6, 0.7, 0.2, 0.95]
    spans = group_spans(scores, None, 0.5)
    assert [(s.start_token, s.end_token) for s in spans] == [(0, 1), (2, 4), (5, 6)]
    assert spans[-1].max_score == pytest.approx(0.95)


def test_group_spans_threshold_is_inclusive():
    spans = group_spans([0.5, 0.5], None, 0.5)
    assert len(spans) == 1 and (spans[0].start_token, spans[0].end_token) == (0, 2)


def test_group_spans_length_mismatch_raises():
    with pytest.raises(ValueError):
        group_spans([0.1, 0.9], ["only-one"], 0.5)


# build_response


def test_build_response_shape():
    scores = [0.1, 0.8, 0.9]
    resp = build_response(scores, ["x", "y", "z"], probe_id="p1", threshold=0.5)
    assert resp["probe_id"] == "p1"
    assert resp["threshold"] == 0.5
    assert resp["token_scores"] == scores
    assert len(resp["flagged_spans"]) == 1
    span = resp["flagged_spans"][0]
    assert span["start_token"] == 1 and span["end_token"] == 3 and span["text"] == "yz"


def test_build_response_rounds_scores_only():
    resp = build_response([0.123456, 0.987654], None, probe_id="p", threshold=0.5, round_ndigits=2)
    assert resp["token_scores"] == [0.12, 0.99]
    # max_score is not rounded, so a display never hides a real crossing.
    assert resp["flagged_spans"][0]["max_score"] == pytest.approx(0.987654)


# extract_scores


def test_extract_scores_splits_by_request():
    buf = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert extract_scores(buf, [0, 2, 5]) == [[0.1, 0.2], [0.3, 0.4, 0.5]]


def test_extract_scores_drops_padding():
    buf = [0.1, 0.2, 0.9, 0.9]  # last two are padding beyond the layout
    assert extract_scores(buf, [0, 1, 2]) == [[0.1], [0.2]]


def test_extract_scores_edge_cases():
    assert extract_scores([1.0], [0]) == []  # no requests
    with pytest.raises(ValueError):
        extract_scores([0.1], [1, 2])  # must start at 0
    with pytest.raises(ValueError):
        extract_scores([0.1, 0.2], [0, 5])  # runs past buffer
