"""Adaptive Conformal Inference (ACI) for distribution-free prediction sets.

Implements Gibbs & Candes (NeurIPS 2021): provides coverage guarantees
that adapt to concept drift in non-exchangeable data.

Key update rule:
    alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)

When the model miscalibrates (concept drift), alpha increases, prediction
sets widen, and reported uncertainty increases. When calibration improves,
sets narrow. This feeds directly into ConformalPredictionSet in TypeScript.
"""

from __future__ import annotations

from collections import deque
from typing import TypedDict

import torch


class CoverageStats(TypedDict):
    """Coverage statistics for the conformal predictor."""
    coverageTarget: float
    empiricalCoverage: float
    currentAlpha: float
    windowSize: int
    avgSetSize: float


class AdaptiveConformalPredictor:
    """Adaptive Conformal Inference predictor.

    Maintains an adaptive significance level alpha that tracks concept
    drift. When predictions miss (err_t = 1), alpha increases and
    prediction sets widen. When predictions hit (err_t = 0), alpha
    decreases and sets narrow.

    The update rule ensures long-run coverage converges to the target
    even under distribution shift.

    Args:
        alpha_target: Target miscoverage rate (e.g., 0.1 for 90% coverage).
        gamma: Learning rate for alpha adaptation.
            Larger gamma = faster adaptation to drift, more volatile sets.
        window_size: Rolling window for empirical coverage tracking.
    """

    def __init__(
        self,
        alpha_target: float = 0.1,
        gamma: float = 0.01,
        window_size: int = 500,
    ) -> None:
        self.alpha_target = alpha_target
        self.gamma = gamma

        # Current adaptive alpha (starts at target)
        self.alpha: float = alpha_target

        # Rolling window for coverage tracking
        self._window_size = window_size
        self._errors: deque[int] = deque(maxlen=window_size)
        self._set_sizes: deque[int] = deque(maxlen=window_size)

    def update(self, prediction_set: list[int], actual: int) -> None:
        """Update alpha based on whether actual was in prediction set.

        Implements:
            err_t = 1 if actual not in set, else 0
            alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)

        When err_t = 1 (miss): alpha increases → sets widen next time
        When err_t = 0 (hit):  alpha decreases → sets narrow next time

        Args:
            prediction_set: Set of predicted class indices.
            actual: Actual class index (ground truth).
        """
        err = 0 if actual in prediction_set else 1
        self._errors.append(err)
        self._set_sizes.append(len(prediction_set))

        # ACI update rule
        self.alpha = self.alpha + self.gamma * (self.alpha_target - err)

        # Clip alpha to valid range
        self.alpha = max(0.001, min(0.999, self.alpha))

    def get_prediction_set(
        self,
        scores: torch.Tensor,
    ) -> list[int]:
        """Compute prediction set from softmax scores using current alpha.

        Includes classes in decreasing probability order until cumulative
        probability reaches (1 - alpha). This is the Adaptive Prediction
        Sets (APS) method.

        Args:
            scores: Softmax probability scores (n_classes,).

        Returns:
            List of class indices in the prediction set.
        """
        scores = scores.detach().cpu()
        n_classes = scores.numel()

        # Sort by descending probability
        sorted_probs, sorted_indices = scores.sort(descending=True)

        # Include classes until cumulative probability >= (1 - alpha)
        threshold = 1.0 - self.alpha
        cumsum = 0.0
        prediction_set: list[int] = []

        for i in range(n_classes):
            prediction_set.append(sorted_indices[i].item())
            cumsum += sorted_probs[i].item()
            if cumsum >= threshold:
                break

        # Always include at least one class
        if not prediction_set:
            prediction_set.append(sorted_indices[0].item())

        return prediction_set

    def get_coverage_stats(self) -> CoverageStats:
        """Get current coverage statistics.

        Returns:
            CoverageStats dict matching TS ConformalPredictionSet fields.
        """
        n = len(self._errors)
        if n == 0:
            return CoverageStats(
                coverageTarget=1.0 - self.alpha_target,
                empiricalCoverage=1.0 - self.alpha_target,
                currentAlpha=self.alpha,
                windowSize=0,
                avgSetSize=1.0,
            )

        empirical_errors = sum(self._errors) / n
        avg_set_size = sum(self._set_sizes) / n if self._set_sizes else 1.0

        return CoverageStats(
            coverageTarget=round(1.0 - self.alpha_target, 4),
            empiricalCoverage=round(1.0 - empirical_errors, 4),
            currentAlpha=round(self.alpha, 6),
            windowSize=n,
            avgSetSize=round(avg_set_size, 2),
        )
