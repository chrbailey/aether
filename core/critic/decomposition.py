"""Uncertainty decomposition into epistemic and aleatoric components.

This is the bridge to VERITY's Dempster-Shafer framework. The decomposition
determines how governance responds to uncertainty:

    - HIGH epistemic → tighten governance (more data helps → human review)
    - HIGH aleatoric → do NOT tighten (irreducible → review won't help)

Decomposition method: ensemble variance decomposition.
    Epistemic = variance of means across ensemble members (model disagreement)
    Aleatoric = mean of variances within each ensemble member (inherent noise)
    Total = epistemic + aleatoric
"""

from __future__ import annotations

from typing import TypedDict

import torch


class UncertaintyDecomposition(TypedDict):
    """Decomposed uncertainty matching TS UncertaintyDecomposition."""
    total: float
    epistemic: float
    aleatoric: float
    epistemicRatio: float
    method: str


def decompose_from_ensemble(
    predictions: list[torch.Tensor],
) -> UncertaintyDecomposition:
    """Decompose predictive uncertainty from an ensemble of predictions.

    Given M ensemble members, each producing softmax probabilities for
    N classes:

        Epistemic = Var_m[E[Y|m]]  — disagreement between models
        Aleatoric = E_m[Var[Y|m]]  — average within-model uncertainty
        Total = Epistemic + Aleatoric

    This is the law of total variance:
        Var(Y) = E[Var(Y|M)] + Var(E[Y|M])

    Args:
        predictions: List of M tensors, each of shape (n_classes,) or
            (batch, n_classes), representing softmax probabilities
            from different ensemble members.

    Returns:
        UncertaintyDecomposition dict with total, epistemic, aleatoric,
        epistemicRatio, and method fields.
    """
    if len(predictions) == 0:
        return UncertaintyDecomposition(
            total=0.0,
            epistemic=0.0,
            aleatoric=0.0,
            epistemicRatio=0.0,
            method="ensemble_variance",
        )

    # Stack predictions: (M, ..., n_classes)
    stacked = torch.stack([p.detach().float() for p in predictions])

    # Mean prediction across ensemble: (..., n_classes)
    # Note: We compute ensemble_mean for conceptual clarity in the variance formula,
    # but the actual epistemic calculation uses stacked.var() which internally computes the mean
    _ensemble_mean = stacked.mean(dim=0)  # noqa: F841

    # Epistemic: variance of the means across ensemble members
    # For each class, compute variance of predicted probability across models
    # Then average across classes
    epistemic_per_class = stacked.var(dim=0)  # (..., n_classes)
    epistemic = epistemic_per_class.mean().item()

    # Aleatoric: mean of the within-model variance
    # For categorical predictions, within-model variance = p * (1 - p)
    within_model_var = stacked * (1.0 - stacked)  # (M, ..., n_classes)
    aleatoric_per_class = within_model_var.mean(dim=0)  # (..., n_classes)
    aleatoric = aleatoric_per_class.mean().item()

    total = epistemic + aleatoric

    # Prevent division by zero
    epistemic_ratio = epistemic / total if total > 1e-10 else 0.0

    return UncertaintyDecomposition(
        total=round(total, 6),
        epistemic=round(epistemic, 6),
        aleatoric=round(aleatoric, 6),
        epistemicRatio=round(epistemic_ratio, 6),
        method="ensemble_variance",
    )


def decompose_from_mc_dropout(
    predictions: list[torch.Tensor],
) -> UncertaintyDecomposition:
    """Decompose uncertainty from Monte Carlo dropout samples.

    Identical math to ensemble decomposition, but with the predictions
    coming from multiple forward passes with dropout enabled.

    Args:
        predictions: List of T tensors from MC dropout forward passes.

    Returns:
        UncertaintyDecomposition dict.
    """
    result = decompose_from_ensemble(predictions)
    result["method"] = "mc_dropout"
    return result
