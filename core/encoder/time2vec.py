"""Time2Vec: Continuous-time positional encoding for irregular event sequences.

Implements the Time2Vec representation from Kazemi et al. (ICLR 2019).
Maps scalar time deltas to a learnable embedding space using both
linear and periodic (sinusoidal) components.

Reference: https://arxiv.org/abs/1907.05321
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Time2Vec(nn.Module):
    """Learnable time encoding with linear + periodic components.

    For a scalar time delta Δt, produces a d-dimensional embedding:
        t2v(Δt)[0] = ω_0 · Δt + φ_0                  (linear component)
        t2v(Δt)[i] = sin(ω_i · Δt + φ_i)  for i > 0  (periodic components)

    The linear component captures trends; periodic components capture
    cyclical patterns (hourly, daily, weekly business rhythms).

    Args:
        embed_dim: Dimensionality of the time embedding output.
            The first dimension is linear; the remaining (embed_dim - 1)
            are periodic (sinusoidal).
    """

    def __init__(self, embed_dim: int = 32) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Learnable frequency and phase for periodic components
        # omega: frequencies, phi: phase shifts
        self.omega = nn.Parameter(torch.randn(embed_dim))
        self.phi = nn.Parameter(torch.randn(embed_dim))

        # Initialize frequencies to capture business-relevant periods:
        # hours (1/24), days (1.0), weeks (7.0) scale
        nn.init.normal_(self.omega, mean=0.0, std=1.0)
        nn.init.zeros_(self.phi)

    def forward(self, time_deltas: torch.Tensor) -> torch.Tensor:
        """Encode time deltas into continuous time embeddings.

        Args:
            time_deltas: Tensor of shape (...,) or (..., 1) containing
                inter-event time deltas in hours. Scalars are treated
                as a batch of 1.

        Returns:
            Time embeddings of shape (..., embed_dim). A scalar input
            produces shape (1, embed_dim).
        """
        # Record original shape to preserve batch dimensions
        original_shape = time_deltas.shape

        # Scalars become (1,)
        if time_deltas.dim() == 0:
            time_deltas = time_deltas.unsqueeze(0)

        # Add trailing dimension for broadcasting: (...,) -> (..., 1)
        time_deltas = time_deltas.unsqueeze(-1)

        # Compute ω · Δt + φ for all dimensions
        # time_deltas: (..., 1), omega: (embed_dim,) → (..., embed_dim)
        linear_arg = time_deltas * self.omega + self.phi

        # First component is linear, rest are sinusoidal
        linear_part = linear_arg[..., :1]
        periodic_part = torch.sin(linear_arg[..., 1:])

        return torch.cat([linear_part, periodic_part], dim=-1)
