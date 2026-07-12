"""Scoring metrics for the honest benchmark.

The point of the benchmark is not a single AUROC. Probes look strong in
distribution and can collapse to near-random out of distribution, and a score
that is well ranked can still be badly calibrated. So the benchmark reports
ranking (AUROC), calibration (ECE, Brier), and an operating point
(Recall at a fixed false-positive rate) side by side, in distribution and out.

These are plain numpy so they run anywhere and are cheap to test. No sklearn or
scipy dependency.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _as_arrays(labels: Sequence[int], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError(f"labels and scores must match, got {y.shape} and {s.shape}")
    if y.size == 0:
        raise ValueError("need at least one example")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    return y, s


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the ROC curve, tie-aware, via the rank-sum identity.

    Equivalent to the probability that a random positive outranks a random
    negative, with ties counted as half.
    """
    y, s = _as_arrays(labels, scores)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs both a positive and a negative example")
    ranks = _average_ranks(s)
    rank_sum_pos = ranks[y == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def brier(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Mean squared error between probability and outcome. Lower is better."""
    y, s = _as_arrays(labels, scores)
    return float(np.mean((s - y) ** 2))


def ece(labels: Sequence[int], scores: Sequence[float], n_bins: int = 15) -> float:
    """Expected Calibration Error over equal-width probability bins.

    The gap between confidence and accuracy in each bin, weighted by bin count.
    """
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    y, s = _as_arrays(labels, scores)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = y.size
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        # Last bin is closed on the right so scores of exactly 1.0 are counted.
        in_bin = (s >= lo) & (s < hi) if hi < 1.0 else (s >= lo) & (s <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        confidence = float(s[in_bin].mean())
        accuracy = float(y[in_bin].mean())
        error += (count / total) * abs(confidence - accuracy)
    return error


def recall_at_fpr(labels: Sequence[int], scores: Sequence[float], target_fpr: float = 0.1) -> float:
    """Best recall (true-positive rate) reachable without exceeding target_fpr.

    This is the operating point that matters when a false flag is expensive: how
    much of the hallucinated content can be caught while keeping the false-alarm
    rate on good content at or below the budget.
    """
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError(f"target_fpr must be in [0, 1], got {target_fpr}")
    y, s = _as_arrays(labels, scores)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("recall_at_fpr needs both a positive and a negative example")

    order = np.argsort(-s, kind="stable")  # high score first
    y_sorted = y[order]
    s_sorted = s[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    tpr = tp / n_pos
    fpr = fp / n_neg

    # Only compare at threshold boundaries, so a run of tied scores is one point.
    boundary = np.ones(y.size, dtype=bool)
    boundary[:-1] = s_sorted[1:] != s_sorted[:-1]
    allowed = boundary & (fpr <= target_fpr)
    if not allowed.any():
        return 0.0  # predicting all-negative gives fpr 0, tpr 0
    return float(tpr[allowed].max())


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks starting at 1, with tied values sharing their average rank."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1)
    # Average ranks within groups of equal value.
    sorted_vals = values[order]
    i = 0
    n = values.size
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2  # average of ranks (i+1)..j
            ranks[order[i:j]] = avg
        i = j
    return ranks
