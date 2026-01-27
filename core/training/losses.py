"""Loss functions for AETHER world model training.

Two primary losses:

1. VICReg (Bardes et al., ICLR 2022) — prevents representation collapse
   in the latent space through Variance-Invariance-Covariance regularization.

2. Energy Contrastive Loss — trains the energy scorer to distinguish
   plausible transitions (low energy) from implausible ones (high energy).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegLoss(nn.Module):
    """Variance-Invariance-Covariance regularization loss.

    Prevents latent space collapse through three complementary terms:

    1. Invariance: MSE between predicted and actual latent states.
       Ensures the transition model makes accurate predictions.

    2. Variance: Keeps the standard deviation of each latent dimension
       above a threshold. Prevents all representations from collapsing
       to a single point.

    3. Covariance: Decorrelates latent dimensions. Prevents information
       redundancy where multiple dimensions encode the same thing.

    Args:
        invariance_weight: Weight for invariance (prediction accuracy) loss.
        variance_weight: Weight for variance (anti-collapse) loss.
        covariance_weight: Weight for covariance (decorrelation) loss.
        variance_target: Target standard deviation per dimension.
    """

    def __init__(
        self,
        invariance_weight: float = 25.0,
        variance_weight: float = 25.0,
        covariance_weight: float = 1.0,
        variance_target: float = 1.0,
    ) -> None:
        super().__init__()
        self.invariance_weight = invariance_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.variance_target = variance_target

    def forward(
        self,
        z_predicted: torch.Tensor,
        z_actual: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute VICReg loss components.

        Args:
            z_predicted: Predicted latent states (batch, latent_dim).
            z_actual: Actual latent states (batch, latent_dim).

        Returns:
            Dict with 'total', 'invariance', 'variance', 'covariance' losses.
        """
        invariance = self._invariance_loss(z_predicted, z_actual)
        variance = self._variance_loss(z_predicted, z_actual)
        covariance = self._covariance_loss(z_predicted, z_actual)

        total = (
            self.invariance_weight * invariance
            + self.variance_weight * variance
            + self.covariance_weight * covariance
        )

        return {
            "total": total,
            "invariance": invariance,
            "variance": variance,
            "covariance": covariance,
        }

    def _invariance_loss(
        self,
        z_predicted: torch.Tensor,
        z_actual: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between predicted and actual latent states."""
        return F.mse_loss(z_predicted, z_actual)

    def _variance_loss(
        self,
        z_predicted: torch.Tensor,
        z_actual: torch.Tensor,
    ) -> torch.Tensor:
        """Hinge loss on per-dimension standard deviation.

        Penalizes dimensions with std below the target threshold.
        Applied to both predicted and actual to prevent collapse in either.
        """
        std_pred = z_predicted.std(dim=0)
        std_actual = z_actual.std(dim=0)

        # Hinge: max(0, target - std)
        var_loss_pred = F.relu(self.variance_target - std_pred).mean()
        var_loss_actual = F.relu(self.variance_target - std_actual).mean()

        return (var_loss_pred + var_loss_actual) / 2

    def _covariance_loss(
        self,
        z_predicted: torch.Tensor,
        z_actual: torch.Tensor,
    ) -> torch.Tensor:
        """Off-diagonal covariance penalty to decorrelate dimensions.

        Minimizes the squared off-diagonal elements of the covariance matrix.
        """
        cov_pred = self._off_diagonal_cov(z_predicted)
        cov_actual = self._off_diagonal_cov(z_actual)
        return (cov_pred + cov_actual) / 2

    @staticmethod
    def _off_diagonal_cov(z: torch.Tensor) -> torch.Tensor:
        """Compute mean squared off-diagonal covariance."""
        batch_size, dim = z.shape
        z_centered = z - z.mean(dim=0)
        # Clamp centered values to prevent overflow in matmul on MPS
        z_centered = z_centered.clamp(-10.0, 10.0)
        cov = (z_centered.T @ z_centered) / max(batch_size - 1, 1)

        # Zero out diagonal, compute mean squared off-diagonal
        off_diag = cov.pow(2)
        mask = ~torch.eye(dim, dtype=torch.bool, device=z.device)
        return off_diag[mask].mean()


class SIGRegLoss(nn.Module):
    """SIGReg regularization loss (Balestriero & LeCun, 2025).

    Unifies VICReg's separate variance + covariance terms into a single loss
    based on the log-sigmoid of eigenvalues of the representation covariance
    matrix. More numerically stable and theoretically grounded — connected to
    information maximization via entropy of a Gaussian.

    L_SIGReg = -mean(log(sigmoid(lambda_i / temperature)))

    Where lambda_i are eigenvalues of the covariance matrix.

    - When eigenvalues are small (collapse) → sigmoid ≈ 0 → -log(0) → large loss
    - When eigenvalues are large (good spread) → sigmoid ≈ 1 → -log(1) → zero loss

    Combined with MSE invariance for full JEPA-style training.

    Note: Uses CPU fallback for eigendecomposition since torch.linalg.eigvalsh
    is not yet implemented on MPS. The 128×128 covariance matrix transfer is
    negligible overhead, and autograd flows correctly across devices.

    Args:
        invariance_weight: Weight for invariance (prediction accuracy) loss.
        sigreg_weight: Weight for SIGReg (anti-collapse + decorrelation) loss.
        temperature: Scaling factor for eigenvalues before sigmoid.
    """

    def __init__(
        self,
        invariance_weight: float = 25.0,
        sigreg_weight: float = 10.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.invariance_weight = invariance_weight
        self.sigreg_weight = sigreg_weight
        self.temperature = temperature

    def forward(
        self,
        z_predicted: torch.Tensor,
        z_actual: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute SIGReg + invariance loss.

        Args:
            z_predicted: Predicted latent states (batch, latent_dim).
            z_actual: Actual latent states (batch, latent_dim).

        Returns:
            Dict with 'total', 'invariance', 'variance' (sigreg), 'covariance'
            (zero placeholder for backward compat with VICReg logging).
        """
        invariance = F.mse_loss(z_predicted, z_actual)
        sigreg = self._sigreg_loss(z_predicted, z_actual)

        total = (
            self.invariance_weight * invariance
            + self.sigreg_weight * sigreg
        )

        # Return with VICReg-compatible keys for seamless trainer integration
        return {
            "total": total,
            "invariance": invariance,
            "variance": sigreg,  # SIGReg subsumes both variance and covariance
            "covariance": torch.tensor(0.0, device=z_predicted.device),
        }

    def _sigreg_loss(
        self,
        z_predicted: torch.Tensor,
        z_actual: torch.Tensor,
    ) -> torch.Tensor:
        """Average SIGReg over both predicted and actual representations."""
        loss_pred = self._sigreg_term(z_predicted)
        loss_actual = self._sigreg_term(z_actual)
        return (loss_pred + loss_actual) / 2

    def _sigreg_term(self, z: torch.Tensor) -> torch.Tensor:
        """Compute SIGReg for one set of representations.

        -mean(log(sigmoid(eigenvalues / temperature)))
        """
        batch_size, dim = z.shape
        z_centered = z - z.mean(dim=0)
        # Clamp to prevent overflow in matmul (same as VICReg fix)
        z_centered = z_centered.clamp(-10.0, 10.0)
        cov = (z_centered.T @ z_centered) / max(batch_size - 1, 1)

        # Eigendecomposition: CPU fallback for MPS compatibility
        compute_device = cov.device
        if compute_device.type == "mps":
            cov_cpu = cov.to("cpu")
            eigenvalues = torch.linalg.eigvalsh(cov_cpu)
            eigenvalues = eigenvalues.to(compute_device)
        else:
            eigenvalues = torch.linalg.eigvalsh(cov)

        # SIGReg: -mean(log(sigmoid(lambda / temperature)))
        scaled = eigenvalues / self.temperature
        return -F.logsigmoid(scaled).mean()


class EnergyContrastiveLoss(nn.Module):
    """Contrastive loss using energy scores for transition learning.

    Trains the energy scorer to assign:
    - Low energy to positive (actual) transitions
    - High energy to negative (random/corrupted) transitions

    Uses a margin-based contrastive formulation:
        L = E_pos + max(0, margin - E_neg)

    This encourages positive energy to be low and negative energy
    to exceed the margin.

    Args:
        margin: Minimum energy gap between positive and negative pairs.
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        energy_positive: torch.Tensor,
        energy_negative: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute contrastive energy loss.

        Args:
            energy_positive: Energy of actual transitions (batch,).
            energy_negative: Energy of corrupted/random transitions (batch,).

        Returns:
            Dict with 'total', 'positive', 'negative' losses.
        """
        # Positive: actual transitions should have low energy
        loss_positive = energy_positive.mean()

        # Negative: random transitions should have energy above margin
        loss_negative = F.relu(self.margin - energy_negative).mean()

        total = loss_positive + loss_negative

        return {
            "total": total,
            "positive": loss_positive,
            "negative": loss_negative,
        }
