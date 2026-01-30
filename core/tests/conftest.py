"""Shared fixtures for AETHER Python tests.

Provides reusable test data, mock vocabularies, and synthetic batches
so that individual test files don't duplicate setup code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from core.encoder.vocabulary import ActivityVocabulary, ResourceVocabulary


# ---------------------------------------------------------------------------
# Standard activity / resource lists
# ---------------------------------------------------------------------------

ACTIVITIES: list[str] = [
    "create_purchase_requisition",
    "approve_purchase_requisition",
    "create_purchase_order",
    "create_goods_receipt",
    "execute_payment",
    "create_sales_order",
    "approve_credit",
    "ship_goods",
    "invoice",
    "close_order",
]

RESOURCES: list[str] = [
    "system",
    "user_001",
    "user_002",
    "manager",
    "auto",
]


# ---------------------------------------------------------------------------
# Event / case builders
# ---------------------------------------------------------------------------

def make_event(
    activity: str = "create_sales_order",
    resource: str = "system",
    timestamp: str = "2024-06-15T09:00:00",
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a single synthetic event dict."""
    return {
        "activity": activity,
        "resource": resource,
        "timestamp": timestamp,
        "attributes": attributes or {"amount": 100.0},
    }


def make_case(
    case_id: str = "test_case_001",
    n_events: int = 5,
    with_outcome: bool = True,
) -> dict[str, Any]:
    """Create a synthetic case in AETHER format."""
    events = []
    for i in range(n_events):
        events.append(
            make_event(
                activity=ACTIVITIES[i % len(ACTIVITIES)],
                resource=RESOURCES[i % len(RESOURCES)],
                timestamp=f"2024-06-{15 + i:02d}T{9 + i:02d}:00:00",
                attributes={"amount": 100.0 * (i + 1), "priority": i % 3},
            )
        )
    case: dict[str, Any] = {
        "caseId": case_id,
        "events": events,
    }
    if with_outcome:
        case["outcome"] = {
            "onTime": True,
            "rework": False,
            "durationHours": 96.0,
        }
    return case


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_events() -> list[dict[str, Any]]:
    """Five ordered events with valid timestamps."""
    return [
        make_event(
            activity=ACTIVITIES[i],
            resource=RESOURCES[i % len(RESOURCES)],
            timestamp=f"2024-06-{15 + i:02d}T{9 + i:02d}:00:00",
            attributes={"amount": 100.0 * (i + 1)},
        )
        for i in range(5)
    ]


@pytest.fixture
def sample_case() -> dict[str, Any]:
    """A single case with 5 events and an outcome."""
    return make_case()


@pytest.fixture
def mock_activity_vocab() -> ActivityVocabulary:
    """ActivityVocabulary pre-loaded with standard activities."""
    vocab = ActivityVocabulary(embed_dim=64)
    for act in ACTIVITIES:
        vocab.add_token(act)
    return vocab


@pytest.fixture
def mock_resource_vocab() -> ResourceVocabulary:
    """ResourceVocabulary pre-loaded with standard resources."""
    vocab = ResourceVocabulary(embed_dim=32)
    for res in RESOURCES:
        vocab.add_token(res)
    return vocab


@pytest.fixture
def synthetic_batch() -> dict[str, torch.Tensor]:
    """A ready-to-use training batch (batch_size=4, seq_len=6).

    Contains all keys expected by AetherTrainer._train_step().
    """
    batch_size, seq_len = 4, 6
    n_activities = len(ACTIVITIES) + 1  # +1 for UNK at index 0

    return {
        "activity_ids": torch.randint(1, n_activities, (batch_size, seq_len)),
        "resource_ids": torch.randint(1, len(RESOURCES) + 1, (batch_size, seq_len)),
        "attributes": torch.randn(batch_size, seq_len, 8),
        "time_deltas": torch.rand(batch_size, seq_len) * 24.0,
        "padding_mask": torch.zeros(batch_size, seq_len, dtype=torch.bool),
        "target_activities": torch.randint(1, n_activities, (batch_size, seq_len)),
        "target_ontime": torch.randint(0, 2, (batch_size,)).float(),
        "target_rework": torch.randint(0, 2, (batch_size,)).float(),
        "target_remaining": torch.rand(batch_size) * 100.0,
        "seq_lens": torch.full((batch_size,), seq_len, dtype=torch.long),
    }


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Temporary directory for test output files (checkpoints, etc.)."""
    out = tmp_path / "aether_test_output"
    out.mkdir(parents=True, exist_ok=True)
    return out
