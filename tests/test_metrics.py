"""Metric correctness against hand-computable cases."""

from __future__ import annotations

import math

import pytest

from actovue.metrics import auroc, brier, ece, recall_at_fpr


def test_auroc_perfect_and_reversed():
    labels = [0, 0, 1, 1]
    assert auroc(labels, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert auroc(labels, [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)


def test_auroc_random_is_half():
    # Two positives and two negatives fully interleaved with ties handled.
    assert auroc([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_auroc_known_value():
    # pos scores {0.6, 0.4}, neg scores {0.5, 0.3}. Pairs where pos > neg:
    # 0.6>0.5, 0.6>0.3, 0.4>0.3 = 3 of 4 -> 0.75.
    assert auroc([1, 0, 1, 0], [0.6, 0.5, 0.4, 0.3]) == pytest.approx(0.75)


def test_auroc_needs_both_classes():
    with pytest.raises(ValueError):
        auroc([1, 1, 1], [0.2, 0.5, 0.9])


def test_brier():
    assert brier([1, 0], [1.0, 0.0]) == pytest.approx(0.0)
    assert brier([1, 0], [0.0, 1.0]) == pytest.approx(1.0)
    assert brier([1, 0], [0.5, 0.5]) == pytest.approx(0.25)


def test_ece_perfectly_calibrated():
    # In a bin around 0.5, half the labels positive: confidence 0.5 == accuracy 0.5.
    assert ece([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5], n_bins=10) == pytest.approx(0.0)


def test_ece_worst_case():
    # Confident and always wrong: confidence 1.0, accuracy 0.0 -> ECE 1.0.
    assert ece([0, 0], [1.0, 1.0], n_bins=10) == pytest.approx(1.0)


def test_recall_at_fpr():
    # 4 positives, 4 negatives. target_fpr 0.25 allows 1 false positive of 4.
    labels = [1, 1, 1, 0, 0, 1, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.65, 0.5, 0.35, 0.3, 0.1]
    # The top 3 positives (0.9, 0.8, 0.7) are caught with 0 FP. The 4th positive
    # (0.35) sits below two negatives (0.65, 0.5), so catching it needs fpr 0.5.
    # Best recall while fpr stays at or below 0.25 is therefore 3/4 = 0.75.
    assert recall_at_fpr(labels, scores, target_fpr=0.25) == pytest.approx(0.75)


def test_recall_at_fpr_zero_budget():
    # With a clean margin, fpr 0 still catches every positive.
    assert recall_at_fpr([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], target_fpr=0.0) == pytest.approx(1.0)


def test_metrics_reject_bad_labels():
    with pytest.raises(ValueError):
        brier([2, 0], [0.5, 0.5])
    with pytest.raises(ValueError):
        auroc([1, 0], [0.5])  # shape mismatch


def test_ece_in_unit_range():
    labels = [1, 0, 1, 1, 0, 0, 1, 0]
    scores = [0.9, 0.1, 0.7, 0.6, 0.3, 0.2, 0.8, 0.4]
    e = ece(labels, scores)
    assert 0.0 <= e <= 1.0 and not math.isnan(e)
