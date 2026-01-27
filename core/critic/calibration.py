"""Expected Calibration Error (ECE) computation and tracking.

Calibration measures how well predicted probabilities match actual
frequencies. A model that predicts 70% confidence should be correct
70% of the time. Poor calibration tightens governance via the
calibration_factor in GovernanceModulation.

Returns CalibrationMetrics dicts compatible with the TypeScript types.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import TypedDict

import torch


class CalibrationBucket(TypedDict):
    """Per-bucket calibration stats matching TS CalibrationBucket."""
    confidenceLow: float
    confidenceHigh: float
    avgConfidence: float
    avgAccuracy: float
    count: int


class CalibrationMetrics(TypedDict):
    """Calibration metrics matching TS CalibrationMetrics."""
    ece: float
    mce: float
    brierScore: float
    windowSize: int
    windowStart: str
    windowEnd: str
    buckets: list[CalibrationBucket]


def compute_ece(
    predictions: torch.Tensor,
    actuals: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    where B_b is the set of predictions in bin b, acc is accuracy,
    and conf is average confidence.

    Args:
        predictions: Predicted probabilities (N,) in [0, 1].
        actuals: Binary ground truth labels (N,) in {0, 1}.
        n_bins: Number of equal-width confidence bins.

    Returns:
        ECE value in [0, 1]. Lower is better calibrated.
    """
    predictions = predictions.detach().float()
    actuals = actuals.detach().float()
    n = predictions.numel()

    if n == 0:
        return 0.0

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=predictions.device)
    ece = 0.0

    for i in range(n_bins):
        low = bin_boundaries[i]
        high = bin_boundaries[i + 1]

        # Inclusive upper bound on last bin
        if i == n_bins - 1:
            mask = (predictions >= low) & (predictions <= high)
        else:
            mask = (predictions >= low) & (predictions < high)

        count = mask.sum().item()
        if count == 0:
            continue

        avg_conf = predictions[mask].mean().item()
        avg_acc = actuals[mask].mean().item()
        ece += (count / n) * abs(avg_acc - avg_conf)

    return ece


def compute_mce(
    predictions: torch.Tensor,
    actuals: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Compute Maximum Calibration Error (worst bucket).

    MCE = max_b |acc(B_b) - conf(B_b)|

    Args:
        predictions: Predicted probabilities (N,) in [0, 1].
        actuals: Binary ground truth labels (N,) in {0, 1}.
        n_bins: Number of equal-width confidence bins.

    Returns:
        MCE value in [0, 1]. Lower is better.
    """
    predictions = predictions.detach().float()
    actuals = actuals.detach().float()
    n = predictions.numel()

    if n == 0:
        return 0.0

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=predictions.device)
    mce = 0.0

    for i in range(n_bins):
        low = bin_boundaries[i]
        high = bin_boundaries[i + 1]

        if i == n_bins - 1:
            mask = (predictions >= low) & (predictions <= high)
        else:
            mask = (predictions >= low) & (predictions < high)

        count = mask.sum().item()
        if count == 0:
            continue

        avg_conf = predictions[mask].mean().item()
        avg_acc = actuals[mask].mean().item()
        mce = max(mce, abs(avg_acc - avg_conf))

    return mce


def compute_brier(
    predictions: torch.Tensor,
    actuals: torch.Tensor,
) -> float:
    """Compute Brier score (mean squared error of probabilities).

    Brier = (1/N) * sum (p_i - y_i)^2

    Decomposes into calibration + resolution + uncertainty.
    Lower is better.

    Args:
        predictions: Predicted probabilities (N,) in [0, 1].
        actuals: Binary ground truth labels (N,) in {0, 1}.

    Returns:
        Brier score in [0, 1].
    """
    predictions = predictions.detach().float()
    actuals = actuals.detach().float()

    if predictions.numel() == 0:
        return 0.0

    return ((predictions - actuals) ** 2).mean().item()


def _build_buckets(
    predictions: torch.Tensor,
    actuals: torch.Tensor,
    n_bins: int = 15,
) -> list[CalibrationBucket]:
    """Build per-bucket calibration statistics for reliability diagrams."""
    predictions = predictions.detach().float()
    actuals = actuals.detach().float()

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=predictions.device)
    buckets: list[CalibrationBucket] = []

    for i in range(n_bins):
        low = bin_boundaries[i].item()
        high = bin_boundaries[i + 1].item()

        if i == n_bins - 1:
            mask = (predictions >= low) & (predictions <= high)
        else:
            mask = (predictions >= low) & (predictions < high)

        count = mask.sum().item()
        avg_conf = predictions[mask].mean().item() if count > 0 else (low + high) / 2
        avg_acc = actuals[mask].mean().item() if count > 0 else 0.0

        buckets.append(CalibrationBucket(
            confidenceLow=round(low, 4),
            confidenceHigh=round(high, 4),
            avgConfidence=round(avg_conf, 4),
            avgAccuracy=round(avg_acc, 4),
            count=int(count),
        ))

    return buckets


class CalibrationTracker:
    """Rolling-window calibration tracker with per-bucket statistics.

    Maintains a sliding window of (prediction, actual) pairs and
    recomputes calibration metrics on demand. Produces CalibrationMetrics
    dicts compatible with the TypeScript types.

    Args:
        window_size: Maximum number of predictions to track.
        n_bins: Number of calibration bins.
    """

    def __init__(self, window_size: int = 1000, n_bins: int = 15) -> None:
        self.window_size = window_size
        self.n_bins = n_bins
        self._predictions: deque[float] = deque(maxlen=window_size)
        self._actuals: deque[float] = deque(maxlen=window_size)
        self._timestamps: deque[str] = deque(maxlen=window_size)

    def update(self, prediction: float, actual: float) -> None:
        """Record a new (prediction, actual) pair.

        Args:
            prediction: Predicted probability in [0, 1].
            actual: Actual outcome in {0, 1}.
        """
        self._predictions.append(prediction)
        self._actuals.append(actual)
        self._timestamps.append(
            datetime.now(timezone.utc).isoformat()
        )

    def update_batch(
        self,
        predictions: torch.Tensor,
        actuals: torch.Tensor,
    ) -> None:
        """Record a batch of (prediction, actual) pairs."""
        preds = predictions.detach().cpu().tolist()
        acts = actuals.detach().cpu().tolist()
        now = datetime.now(timezone.utc).isoformat()

        for p, a in zip(preds, acts):
            self._predictions.append(p)
            self._actuals.append(a)
            self._timestamps.append(now)

    def compute_metrics(self) -> CalibrationMetrics:
        """Compute current calibration metrics over the tracking window.

        Returns:
            CalibrationMetrics dict matching TypeScript interface.
        """
        if len(self._predictions) == 0:
            now = datetime.now(timezone.utc).isoformat()
            return CalibrationMetrics(
                ece=0.0,
                mce=0.0,
                brierScore=0.0,
                windowSize=0,
                windowStart=now,
                windowEnd=now,
                buckets=[],
            )

        preds = torch.tensor(list(self._predictions))
        acts = torch.tensor(list(self._actuals))

        return CalibrationMetrics(
            ece=round(compute_ece(preds, acts, self.n_bins), 6),
            mce=round(compute_mce(preds, acts, self.n_bins), 6),
            brierScore=round(compute_brier(preds, acts), 6),
            windowSize=len(self._predictions),
            windowStart=self._timestamps[0],
            windowEnd=self._timestamps[-1],
            buckets=_build_buckets(preds, acts, self.n_bins),
        )

    @property
    def count(self) -> int:
        """Number of tracked predictions."""
        return len(self._predictions)
