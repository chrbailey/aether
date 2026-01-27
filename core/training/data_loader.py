"""Data loading utilities for AETHER training.

Loads event sequences from JSON files or SQLite databases,
handles variable-length sequences with padding, and provides
PyTorch DataLoader-compatible datasets.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ..encoder.vocabulary import ActivityVocabulary, ResourceVocabulary


class EventSequenceDataset(Dataset):
    """PyTorch Dataset for event sequence data.

    Loads event sequences and produces tensors suitable for the EventEncoder:
        - activity_ids: Integer activity indices
        - resource_ids: Integer resource indices
        - attributes: Numerical attribute features
        - time_deltas: Inter-event time deltas in hours
        - targets: Dict of supervision signals

    Args:
        events_path: Path to JSON file or SQLite database containing events.
        activity_vocab: Vocabulary for encoding activities.
        resource_vocab: Vocabulary for encoding resources.
        max_seq_len: Maximum sequence length (longer sequences are truncated).
        n_attribute_features: Number of numerical attribute features.
    """

    def __init__(
        self,
        events_path: Path,
        activity_vocab: ActivityVocabulary,
        resource_vocab: ResourceVocabulary,
        max_seq_len: int = 256,
        n_attribute_features: int = 8,
    ) -> None:
        self.events_path = Path(events_path)
        self.activity_vocab = activity_vocab
        self.resource_vocab = resource_vocab
        self.max_seq_len = max_seq_len
        self.n_attribute_features = n_attribute_features

        # Load cases (list of event sequences)
        self.cases: list[dict[str, Any]] = self._load_cases()

    def _load_cases(self) -> list[dict[str, Any]]:
        """Load event cases from file."""
        if self.events_path.suffix == ".json":
            return self._load_from_json()
        elif self.events_path.suffix in (".db", ".sqlite", ".sqlite3"):
            return self._load_from_sqlite()
        else:
            raise ValueError(
                f"Unsupported file format: {self.events_path.suffix}. "
                f"Expected .json or .db/.sqlite"
            )

    def _load_from_json(self) -> list[dict[str, Any]]:
        """Load cases from a JSON file.

        Expected format:
        [
            {
                "caseId": "case_001",
                "events": [
                    {"activity": "create_order", "resource": "user_01",
                     "timestamp": "2024-01-15T09:00:00Z", "attributes": {...}},
                    ...
                ],
                "outcome": {"onTime": true, "rework": false, "durationHours": 48.5}
            },
            ...
        ]
        """
        with open(self.events_path) as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "cases" in data:
            return data["cases"]
        else:
            raise ValueError(
                f"JSON must be a list of cases or dict with 'cases' key"
            )

    def _load_from_sqlite(self) -> list[dict[str, Any]]:
        """Load cases from a SQLite database.

        Expected tables:
            events(case_id, activity, resource, timestamp, attributes_json)
            outcomes(case_id, on_time, rework, duration_hours)
        """
        conn = sqlite3.connect(str(self.events_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Load all events grouped by case
        cursor.execute(
            "SELECT case_id, activity, resource, timestamp, attributes_json "
            "FROM events ORDER BY case_id, timestamp"
        )
        rows = cursor.fetchall()

        # Group events by case
        cases_dict: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            case_id = row["case_id"]
            if case_id not in cases_dict:
                cases_dict[case_id] = []

            attrs = {}
            if row["attributes_json"]:
                attrs = json.loads(row["attributes_json"])

            cases_dict[case_id].append({
                "activity": row["activity"],
                "resource": row["resource"],
                "timestamp": row["timestamp"],
                "attributes": attrs,
            })

        # Load outcomes
        outcomes: dict[str, dict[str, Any]] = {}
        try:
            cursor.execute("SELECT case_id, on_time, rework, duration_hours FROM outcomes")
            for row in cursor.fetchall():
                outcomes[row["case_id"]] = {
                    "onTime": bool(row["on_time"]),
                    "rework": bool(row["rework"]),
                    "durationHours": float(row["duration_hours"]),
                }
        except sqlite3.OperationalError:
            pass  # No outcomes table

        conn.close()

        cases = []
        for case_id, events in cases_dict.items():
            case: dict[str, Any] = {
                "caseId": case_id,
                "events": events,
            }
            if case_id in outcomes:
                case["outcome"] = outcomes[case_id]
            cases.append(case)

        return cases

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single case as tensors.

        Returns:
            Dict with keys:
                - activity_ids: (seq_len,) long tensor
                - resource_ids: (seq_len,) long tensor
                - attributes: (seq_len, n_attr) float tensor
                - time_deltas: (seq_len,) float tensor
                - target_activities: (seq_len,) long tensor (shifted by 1)
                - target_ontime: scalar float (0 or 1)
                - target_rework: scalar float (0 or 1)
                - target_remaining: scalar float (hours)
                - seq_len: actual sequence length (before truncation)
        """
        case = self.cases[idx]
        events = case["events"][:self.max_seq_len]
        seq_len = len(events)

        # Encode activities and resources
        activity_ids = torch.tensor(
            [self.activity_vocab.encode(e["activity"]) for e in events],
            dtype=torch.long,
        )
        resource_ids = torch.tensor(
            [self.resource_vocab.encode(e["resource"]) for e in events],
            dtype=torch.long,
        )

        # Extract numerical attributes (pad missing with 0)
        attributes = torch.zeros(seq_len, self.n_attribute_features)
        for i, event in enumerate(events):
            attrs = event.get("attributes", {})
            if isinstance(attrs, dict):
                for j, (_, v) in enumerate(sorted(attrs.items())):
                    if j >= self.n_attribute_features:
                        break
                    if isinstance(v, (int, float)):
                        attributes[i, j] = float(v)

        # Compute inter-event time deltas in hours
        time_deltas = self._compute_time_deltas(events)

        # Targets: next activity (shifted by 1)
        target_activities = torch.zeros(seq_len, dtype=torch.long)
        if seq_len > 1:
            target_activities[:-1] = activity_ids[1:]
            target_activities[-1] = 0  # Padding for last position

        # Outcome targets (default to False/0.0 for missing — never 0.5,
        # which is not a valid binary label and causes BCE to learn nothing)
        outcome = case.get("outcome", {})
        has_outcome = "onTime" in outcome
        target_ontime = torch.tensor(
            float(outcome.get("onTime", False)), dtype=torch.float
        )
        target_rework = torch.tensor(
            float(outcome.get("rework", False)), dtype=torch.float
        )
        target_remaining = torch.tensor(
            float(outcome.get("durationHours", 0.0)), dtype=torch.float
        )

        return {
            "activity_ids": activity_ids,
            "resource_ids": resource_ids,
            "attributes": attributes,
            "time_deltas": time_deltas,
            "target_activities": target_activities,
            "target_ontime": target_ontime,
            "target_rework": target_rework,
            "target_remaining": target_remaining,
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
        }

    def _compute_time_deltas(self, events: list[dict[str, Any]]) -> torch.Tensor:
        """Compute inter-event time deltas in hours.

        Attempts to parse ISO timestamps. Falls back to zero deltas.
        """
        from datetime import datetime

        deltas = torch.zeros(len(events))

        timestamps: list[datetime | None] = []
        for event in events:
            ts_str = event.get("timestamp", "")
            try:
                # Handle common ISO formats
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                timestamps.append(ts)
            except (ValueError, AttributeError):
                timestamps.append(None)

        for i in range(1, len(timestamps)):
            if timestamps[i] is not None and timestamps[i - 1] is not None:
                delta = timestamps[i] - timestamps[i - 1]
                deltas[i] = delta.total_seconds() / 3600.0  # Convert to hours

        return deltas


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate variable-length sequences into padded batches.

    Pads all sequences to the length of the longest sequence in the batch.
    Creates a padding mask for the attention mechanism.

    Args:
        batch: List of dicts from EventSequenceDataset.__getitem__.

    Returns:
        Batched dict with padded tensors and padding_mask.
    """
    # Sort by sequence length (descending) for efficient packing
    batch = sorted(batch, key=lambda x: x["seq_len"].item(), reverse=True)

    # Pad variable-length tensors
    activity_ids = pad_sequence(
        [b["activity_ids"] for b in batch], batch_first=True, padding_value=0
    )
    resource_ids = pad_sequence(
        [b["resource_ids"] for b in batch], batch_first=True, padding_value=0
    )
    attributes = pad_sequence(
        [b["attributes"] for b in batch], batch_first=True, padding_value=0.0
    )
    time_deltas = pad_sequence(
        [b["time_deltas"] for b in batch], batch_first=True, padding_value=0.0
    )
    target_activities = pad_sequence(
        [b["target_activities"] for b in batch], batch_first=True, padding_value=0
    )

    # Stack scalar targets
    target_ontime = torch.stack([b["target_ontime"] for b in batch])
    target_rework = torch.stack([b["target_rework"] for b in batch])
    target_remaining = torch.stack([b["target_remaining"] for b in batch])
    seq_lens = torch.stack([b["seq_len"] for b in batch])

    # Build padding mask: True where padded
    max_len = activity_ids.shape[1]
    padding_mask = torch.arange(max_len).unsqueeze(0) >= seq_lens.unsqueeze(1)

    return {
        "activity_ids": activity_ids,
        "resource_ids": resource_ids,
        "attributes": attributes,
        "time_deltas": time_deltas,
        "padding_mask": padding_mask,
        "target_activities": target_activities,
        "target_ontime": target_ontime,
        "target_rework": target_rework,
        "target_remaining": target_remaining,
        "seq_lens": seq_lens,
    }
