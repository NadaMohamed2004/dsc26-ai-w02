"""Threshold and ranking metrics for the Rawaj Pay mule score.

A case is flagged when ``score >= threshold``. All functions accept lists or
1-D numpy arrays and raise ``ValueError`` on bad input.
"""

from __future__ import annotations

import math

import numpy as np


def _prepare(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    """Validate inputs and return (labels as int array, scores as float array)."""
    y = np.asarray(y_true)
    s = np.asarray(scores)
    if y.ndim != 1 or s.ndim != 1:
        raise ValueError("y_true and scores must be 1-D")
    if len(y) != len(s):
        raise ValueError("y_true and scores must have the same length")
    if len(y) == 0:
        raise ValueError("inputs must not be empty")
    try:
        y = y.astype(float)
        s = s.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true and scores must be numeric") from exc
    if np.isnan(s).any():
        raise ValueError("scores must not contain NaN")
    if not np.isin(y, (0.0, 1.0)).all():  # NaN labels fail this check too
        raise ValueError("labels must be 0 or 1")
    return y.astype(int), s


def _check_threshold(threshold) -> float:
    try:
        t = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be a number") from exc
    if math.isnan(t):
        raise ValueError("threshold must not be NaN")
    return t


def confusion_at(y_true, scores, threshold: float) -> dict:
    """Return {"tp", "fp", "tn", "fn"} when flagging score >= threshold."""
    y, s = _prepare(y_true, scores)
    t = _check_threshold(threshold)
    flagged = s >= t
    pos = y == 1
    return {
        "tp": int((flagged & pos).sum()),
        "fp": int((flagged & ~pos).sum()),
        "tn": int((~flagged & ~pos).sum()),
        "fn": int((~flagged & pos).sum()),
    }


def precision_recall_at(y_true, scores, threshold: float) -> tuple[float, float]:
    """Precision and recall at a threshold (0.0 instead of dividing by zero)."""
    c = confusion_at(y_true, scores, threshold)
    flagged = c["tp"] + c["fp"]
    positives = c["tp"] + c["fn"]
    precision = c["tp"] / flagged if flagged else 0.0
    recall = c["tp"] / positives if positives else 0.0
    return float(precision), float(recall)


def precision_at_k(y_true, scores, k: int) -> float:
    """Share of true positives among the k highest scores.

    Ties keep their original order; k larger than n is capped at n.
    """
    y, s = _prepare(y_true, scores)
    if not k > 0:  # also catches NaN
        raise ValueError("k must be positive")
    k = min(int(k), len(y))
    order = np.argsort(-s, kind="stable")
    return float(y[order[:k]].mean())


def expected_cost(y_true, scores, threshold: float, *, cost_fp: float, cost_fn: float) -> float:
    """cost_fp * false positives + cost_fn * false negatives."""
    c = confusion_at(y_true, scores, threshold)
    return float(cost_fp * c["fp"] + cost_fn * c["fn"])


def _counts_per_candidate(y: np.ndarray, s: np.ndarray, candidates: np.ndarray):
    """For each candidate threshold return (tp, fp, tn, fn) arrays."""
    order = np.argsort(s, kind="stable")
    s_sorted = s[order]
    y_sorted = y[order]
    pos_cum = np.concatenate(([0], np.cumsum(y_sorted)))
    neg_cum = np.concatenate(([0], np.cumsum(1 - y_sorted)))
    below = np.searchsorted(s_sorted, candidates, side="left")  # scores < candidate
    fn = pos_cum[below]
    tn = neg_cum[below]
    tp = pos_cum[-1] - fn
    fp = neg_cum[-1] - tn
    return tp, fp, tn, fn


def best_threshold(y_true, scores, *, cost_fp: float, cost_fn: float) -> float:
    """Threshold with the lowest expected cost.

    Candidates are every distinct score plus ``math.inf`` (flag nothing).
    On a tie the highest threshold wins (fewer flags for the same cost).
    """
    y, s = _prepare(y_true, scores)
    candidates = np.unique(np.append(s, math.inf))  # ascending
    _, fp, _, fn = _counts_per_candidate(y, s, candidates)
    cost = cost_fp * fp + cost_fn * fn
    # argmin returns the first minimum, so scan from the highest threshold down
    j = int(np.argmin(cost[::-1]))
    return float(candidates[::-1][j])


def threshold_for_precision(y_true, scores, min_precision: float) -> float | None:
    """Lowest threshold (most recall) whose precision is >= min_precision.

    Precision is not monotonic in the threshold, so every distinct score is
    checked. Returns None if no threshold reaches the target.
    """
    y, s = _prepare(y_true, scores)
    candidates = np.unique(s)  # ascending: lowest threshold first
    tp, fp, _, _ = _counts_per_candidate(y, s, candidates)
    flagged = tp + fp  # >= 1 because every candidate is an actual score
    precision = tp / flagged
    ok = precision >= min_precision
    if not ok.any():
        return None
    return float(candidates[int(np.argmax(ok))])
