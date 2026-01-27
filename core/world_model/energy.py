"""Energy scorer for transition plausibility.

The energy function E(z_t, a_t, z_{t+1}) measures how plausible a
state transition is. Low energy = the transition matches the world
model's expectations. High energy = anomalous or implausible transition.

This is the JEPA scoring mechanism: instead of generating raw events
and comparing pixel-by-pixel, we compare predicted vs actual latent states.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EnergyScorer(nn.Module):
    """Scores transition plausibility via energy in latent space.

    E(z_t, a_t, z_{t+1}) = ||z_hat_{t+1} - z_{t+1}||^2

    where z_hat_{t+1} = TransitionModel(z_t, a_t, c_t).

    Low energy means the actual next state matches the prediction
    (plausible transition). High energy means surprise/anomaly.

    The normalized energy is scaled to [0, 1] for threshold comparison
    in the governance layer.

    Args:
        latent_dim: Dimension of latent states.
        normalize_temperature: Temperature for sigmoid normalization.
            Lower temperature = sharper normalization.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        normalize_temperature: float = 10.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.normalize_temperature = normalize_temperature

        # Learnable temperature for adaptive normalization
        self.log_temperature = nn.Parameter(
            torch.tensor(normalize_temperature).log()
        )

    def forward(
        self,
        z_predicted: torch.Tensor,
        z_actual: torch.Tensor,
    ) -> torch.Tensor:
        """Compute raw energy (squared L2 distance in latent space).

        Args:
            z_predicted: Predicted next state z_hat_{t+1} (..., latent_dim).
            z_actual: Actual next state z_{t+1} (..., latent_dim).

        Returns:
            Energy values (...,). Lower = more plausible.
        """
        diff = z_predicted - z_actual
        return (diff * diff).sum(dim=-1)

    def normalized_energy(
        self,
        z_predicted: torch.Tensor,
        z_actual: torch.Tensor,
    ) -> torch.Tensor:
        """Compute normalized energy in [0, 1] via sigmoid scaling.

        Useful for threshold-based decisions in the governance layer.
        A normalized energy near 0 = plausible, near 1 = implausible.

        Args:
            z_predicted: Predicted next state (..., latent_dim).
            z_actual: Actual next state (..., latent_dim).

        Returns:
            Normalized energy in [0, 1] (...,).
        """
        raw = self.forward(z_predicted, z_actual)
        temperature = self.log_temperature.exp()
        # Sigmoid normalization: maps R+ -> (0, 1)
        # Subtract latent_dim to center: expected squared L2 for unit-variance
        return torch.sigmoid((raw - self.latent_dim) / temperature)

    def is_plausible(
        self,
        z_predicted: torch.Tensor,
        z_actual: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Check if transitions are plausible (below energy threshold).

        Args:
            z_predicted: Predicted next state (..., latent_dim).
            z_actual: Actual next state (..., latent_dim).
            threshold: Normalized energy threshold.

        Returns:
            Boolean tensor (...,) where True = plausible.
        """
        norm_energy = self.normalized_energy(z_predicted, z_actual)
        return norm_energy < threshold
