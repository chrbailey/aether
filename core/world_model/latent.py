"""Structured categorical latent variable for process path variants.

Business processes follow discrete path variants (standard, credit hold,
rework, etc.) rather than continuous Gaussian distributions. This module
provides a categorical latent variable with Gumbel-Softmax reparameterization
for differentiable sampling through discrete choices.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Process path variants matching TypeScript ProcessPathVariant
PATH_VARIANTS: list[str] = [
    "standard",
    "credit_hold",
    "rework",
    "expedited",
    "exception",
    "unknown",
]

NUM_VARIANTS: int = len(PATH_VARIANTS)

VARIANT_TO_INDEX: dict[str, int] = {v: i for i, v in enumerate(PATH_VARIANTS)}
INDEX_TO_VARIANT: dict[int, str] = {i: v for i, v in enumerate(PATH_VARIANTS)}


class LatentVariable(nn.Module):
    """Categorical latent variable over process path variants.

    Uses Gumbel-Softmax (Jang et al., ICLR 2017; Maddison et al., ICLR 2017)
    for differentiable sampling through the discrete categorical distribution.

    The latent variable c_t represents which process path variant a case
    is currently following. This enables multi-modal predictions: the
    transition model can produce different futures by sampling different c_t.

    Args:
        latent_dim: Dimension of the input latent state z_t.
        n_variants: Number of categorical path variants.
        temperature: Initial Gumbel-Softmax temperature (annealed during training).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        n_variants: int = NUM_VARIANTS,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_variants = n_variants
        self.temperature = temperature

        # Map latent state to path variant logits
        self.logit_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, n_variants),
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute path variant distribution from latent state.

        Args:
            z: Latent state of shape (..., latent_dim).

        Returns:
            Dict with keys:
                - logits: Raw logits (..., n_variants)
                - probs: Softmax probabilities (..., n_variants)
                - sample: Gumbel-Softmax sample (..., n_variants)
                - hard_sample: Straight-through hard sample (..., n_variants)
        """
        logits = self.logit_head(z)
        probs = F.softmax(logits, dim=-1)

        if self.training:
            sample = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
            hard_sample = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
        else:
            sample = probs
            hard_sample = F.one_hot(
                probs.argmax(dim=-1), num_classes=self.n_variants
            ).float()

        return {
            "logits": logits,
            "probs": probs,
            "sample": sample,
            "hard_sample": hard_sample,
        }

    def log_prob(self, z: torch.Tensor, variant_indices: torch.Tensor) -> torch.Tensor:
        """Compute log probability of specific path variants.

        Args:
            z: Latent state (..., latent_dim).
            variant_indices: Target variant indices (...,) as long tensor.

        Returns:
            Log probabilities (...,).
        """
        logits = self.logit_head(z)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, variant_indices.unsqueeze(-1)).squeeze(-1)

    def entropy(self, z: torch.Tensor) -> torch.Tensor:
        """Compute entropy of the path variant distribution.

        Higher entropy = more uncertain about which path the case follows.

        Args:
            z: Latent state (..., latent_dim).

        Returns:
            Entropy values (...,).
        """
        logits = self.logit_head(z)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(probs * log_probs).sum(dim=-1)

    def kl_divergence(self, z: torch.Tensor, other_logits: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence KL(q(c|z) || p(c)).

        Args:
            z: Latent state giving q(c|z).
            other_logits: Logits of the reference distribution p(c).

        Returns:
            KL divergence values (...,).
        """
        q_logits = self.logit_head(z)
        q_log_probs = F.log_softmax(q_logits, dim=-1)
        p_log_probs = F.log_softmax(other_logits, dim=-1)

        q_probs = F.softmax(q_logits, dim=-1)
        return (q_probs * (q_log_probs - p_log_probs)).sum(dim=-1)

    def anneal_temperature(self, new_temperature: float) -> None:
        """Update the Gumbel-Softmax temperature for annealing.

        Typically annealed from 1.0 toward 0.1 during training to
        sharpen the categorical distribution.
        """
        self.temperature = max(new_temperature, 0.01)

    @staticmethod
    def variant_name(index: int) -> str:
        """Get the path variant name for a given index."""
        return INDEX_TO_VARIANT.get(index, "unknown")

    @staticmethod
    def variant_index(name: str) -> int:
        """Get the index for a named path variant."""
        return VARIANT_TO_INDEX.get(name, VARIANT_TO_INDEX["unknown"])
