"""Multi-timescale hierarchical prediction heads.

The world model predicts at three timescales simultaneously from
the latent state z_t:

    1. Activity level  - What happens next? (fast: single event)
    2. Phase level     - Which phase transitions? (medium: group of events)
    3. Outcome level   - How does the case end? (slow: entire case)

This mirrors the HierarchicalPrediction type in TypeScript.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Default process phases (SAP O2C-inspired)
DEFAULT_PHASES: list[str] = [
    "order_creation",
    "credit_check",
    "delivery",
    "billing",
    "payment",
    "closed",
]


class ActivityHead(nn.Module):
    """Predicts next activity from latent state (classification).

    Outputs a probability distribution over all known activities.

    Args:
        latent_dim: Dimension of input latent state.
        n_activities: Number of possible activities.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        n_activities: int = 20,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_activities),
        )
        # Predict time delta to next event (in hours)
        self.time_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Ensure positive time delta
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict next activity distribution and expected time delta.

        Args:
            z: Latent state (batch, latent_dim).

        Returns:
            Dict with:
                - logits: (batch, n_activities)
                - probs: (batch, n_activities) softmax probabilities
                - delta_hours: (batch, 1) expected hours to next event
        """
        logits = self.head(z)
        probs = F.softmax(logits, dim=-1)
        delta_hours = self.time_head(z)

        return {
            "logits": logits,
            "probs": probs,
            "delta_hours": delta_hours,
        }


class PhaseHead(nn.Module):
    """Predicts current and next process phase from latent state.

    Args:
        latent_dim: Dimension of input latent state.
        n_phases: Number of process phases.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        n_phases: int = 6,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_phases = n_phases

        # Predict current phase
        self.current_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_phases),
        )
        # Predict next phase
        self.next_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_phases),
        )
        # Predict time to phase transition
        self.transition_time_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict phase-level information.

        Args:
            z: Latent state (batch, latent_dim).

        Returns:
            Dict with:
                - current_logits: (batch, n_phases)
                - current_probs: (batch, n_phases)
                - next_logits: (batch, n_phases)
                - next_probs: (batch, n_phases)
                - transition_hours: (batch, 1) expected hours to phase transition
        """
        current_logits = self.current_head(z)
        next_logits = self.next_head(z)
        transition_hours = self.transition_time_head(z)

        return {
            "current_logits": current_logits,
            "current_probs": F.softmax(current_logits, dim=-1),
            "next_logits": next_logits,
            "next_probs": F.softmax(next_logits, dim=-1),
            "transition_hours": transition_hours,
        }


class OutcomeHead(nn.Module):
    """Predicts case outcome from latent state (multi-task).

    Three simultaneous predictions:
        - On-time probability (sigmoid)
        - Rework probability (sigmoid)
        - Remaining hours (positive regression)

    Args:
        latent_dim: Dimension of input latent state.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        )
        self.ontime_head = nn.Linear(hidden_dim, 1)
        self.rework_head = nn.Linear(hidden_dim, 1)
        self.remaining_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Ensure positive remaining hours
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict case outcome.

        Args:
            z: Latent state (batch, latent_dim).

        Returns:
            Dict with:
                - ontime_prob: (batch, 1) probability of on-time completion
                - rework_prob: (batch, 1) probability of rework
                - remaining_hours: (batch, 1) expected remaining duration
        """
        shared = self.shared(z)
        return {
            "ontime_prob": torch.sigmoid(self.ontime_head(shared)),
            "rework_prob": torch.sigmoid(self.rework_head(shared)),
            "remaining_hours": self.remaining_head(shared),
        }


class HierarchicalPredictor(nn.Module):
    """Combines all three prediction timescales from a latent state.

    Mirrors TypeScript HierarchicalPrediction interface:
    activity (fast) + phase (medium) + outcome (slow).

    Args:
        latent_dim: Dimension of input latent state.
        n_activities: Number of possible activities.
        n_phases: Number of process phases.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        n_activities: int = 20,
        n_phases: int = 6,
    ) -> None:
        super().__init__()
        self.activity_head = ActivityHead(
            latent_dim=latent_dim,
            n_activities=n_activities,
        )
        self.phase_head = PhaseHead(
            latent_dim=latent_dim,
            n_phases=n_phases,
        )
        self.outcome_head = OutcomeHead(
            latent_dim=latent_dim,
        )

    def forward(self, z: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        """Produce hierarchical predictions at all three timescales.

        Args:
            z: Latent state (batch, latent_dim).

        Returns:
            Dict with keys 'activity', 'phase', 'outcome', each containing
            the respective head's output dict.
        """
        return {
            "activity": self.activity_head(z),
            "phase": self.phase_head(z),
            "outcome": self.outcome_head(z),
        }
