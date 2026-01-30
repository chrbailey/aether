"""Tests for the AETHER data loading pipeline.

Verifies:
1. EventSequenceDataset loads correctly from JSON
2. EventSequenceDataset loads correctly from SQLite
3. Time delta computation from ISO timestamps
4. Collate function pads variable-length sequences correctly
5. Tensor shapes and dtypes are correct
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import torch

from core.training.data_loader import EventSequenceDataset, collate_fn

# Import shared helpers from conftest (pytest discovers it automatically,
# but we also need direct access to the builder functions)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from conftest import make_case


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_cases_json(cases: list[dict[str, Any]], path: Path) -> None:
    """Write cases to a JSON file."""
    with open(path, "w") as f:
        json.dump(cases, f, default=str)


def _create_sqlite_db(cases: list[dict[str, Any]], path: Path) -> None:
    """Create a SQLite database from cases."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE events "
        "(case_id TEXT, activity TEXT, resource TEXT, "
        "timestamp TEXT, attributes_json TEXT)"
    )
    conn.execute(
        "CREATE TABLE outcomes "
        "(case_id TEXT, on_time INTEGER, rework INTEGER, duration_hours REAL)"
    )

    for case in cases:
        case_id = case["caseId"]
        for event in case["events"]:
            conn.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
                (
                    case_id,
                    event["activity"],
                    event["resource"],
                    event.get("timestamp", ""),
                    json.dumps(event.get("attributes", {})),
                ),
            )
        outcome = case.get("outcome", {})
        if outcome:
            conn.execute(
                "INSERT INTO outcomes VALUES (?, ?, ?, ?)",
                (
                    case_id,
                    int(outcome.get("onTime", False)),
                    int(outcome.get("rework", False)),
                    float(outcome.get("durationHours", 0.0)),
                ),
            )

    conn.commit()
    conn.close()


# ============================================================================
# TestEventSequenceDatasetJSON
# ============================================================================


class TestEventSequenceDatasetJSON:
    """Test dataset loading from JSON files."""

    @pytest.fixture
    def json_dataset(
        self, mock_activity_vocab, mock_resource_vocab, tmp_path
    ) -> EventSequenceDataset:
        """Create a dataset from a temporary JSON file."""
        cases = [make_case(case_id=f"case_{i:03d}", n_events=3 + i) for i in range(5)]
        json_path = tmp_path / "test_cases.json"
        _write_cases_json(cases, json_path)
        return EventSequenceDataset(
            events_path=json_path,
            activity_vocab=mock_activity_vocab,
            resource_vocab=mock_resource_vocab,
        )

    def test_loads_correct_number_of_cases(self, json_dataset):
        assert len(json_dataset) == 5

    def test_returns_correct_tensor_shapes(self, json_dataset):
        sample = json_dataset[0]
        seq_len = sample["seq_len"].item()

        assert sample["activity_ids"].shape == (seq_len,)
        assert sample["activity_ids"].dtype == torch.long
        assert sample["resource_ids"].shape == (seq_len,)
        assert sample["resource_ids"].dtype == torch.long
        assert sample["attributes"].shape == (seq_len, 8)
        assert sample["attributes"].dtype == torch.float
        assert sample["time_deltas"].shape == (seq_len,)
        assert sample["time_deltas"].dtype == torch.float

    def test_target_activities_shifted(self, json_dataset):
        """target_activities[i] should equal activity_ids[i+1] (shifted by 1)."""
        sample = json_dataset[0]
        seq_len = sample["seq_len"].item()
        if seq_len > 1:
            assert torch.equal(
                sample["target_activities"][:-1],
                sample["activity_ids"][1:],
            )

    def test_outcome_targets_correct_type(self, json_dataset):
        sample = json_dataset[0]
        assert sample["target_ontime"].dtype == torch.float
        assert sample["target_rework"].dtype == torch.float
        assert sample["target_remaining"].dtype == torch.float

    def test_handles_dict_format(self, mock_activity_vocab, mock_resource_vocab, tmp_path):
        """JSON can also be a dict with a 'cases' key."""
        cases = [make_case(case_id="dict_case")]
        json_path = tmp_path / "dict_format.json"
        with open(json_path, "w") as f:
            json.dump({"cases": cases}, f)
        ds = EventSequenceDataset(
            events_path=json_path,
            activity_vocab=mock_activity_vocab,
            resource_vocab=mock_resource_vocab,
        )
        assert len(ds) == 1


# ============================================================================
# TestEventSequenceDatasetSQLite
# ============================================================================


class TestEventSequenceDatasetSQLite:
    """Test dataset loading from SQLite databases."""

    @pytest.fixture
    def sqlite_dataset(
        self, mock_activity_vocab, mock_resource_vocab, tmp_path
    ) -> EventSequenceDataset:
        """Create a dataset from a temporary SQLite database."""
        cases = [make_case(case_id=f"sql_{i:03d}", n_events=4) for i in range(3)]
        db_path = tmp_path / "test_events.db"
        _create_sqlite_db(cases, db_path)
        return EventSequenceDataset(
            events_path=db_path,
            activity_vocab=mock_activity_vocab,
            resource_vocab=mock_resource_vocab,
        )

    def test_loads_cases_from_sqlite(self, sqlite_dataset):
        assert len(sqlite_dataset) == 3

    def test_groups_events_by_case_id(self, sqlite_dataset):
        sample = sqlite_dataset[0]
        assert sample["seq_len"].item() == 4

    def test_loads_outcomes(self, sqlite_dataset):
        sample = sqlite_dataset[0]
        # Outcome should be loaded from outcomes table
        assert sample["target_ontime"].dtype == torch.float

    def test_handles_missing_outcomes_table(
        self, mock_activity_vocab, mock_resource_vocab, tmp_path
    ):
        """Database without an outcomes table should still load."""
        db_path = tmp_path / "no_outcomes.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE events "
            "(case_id TEXT, activity TEXT, resource TEXT, "
            "timestamp TEXT, attributes_json TEXT)"
        )
        conn.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
            ("case_x", "create_order", "system", "2024-01-01T00:00:00", "{}"),
        )
        conn.commit()
        conn.close()

        ds = EventSequenceDataset(
            events_path=db_path,
            activity_vocab=mock_activity_vocab,
            resource_vocab=mock_resource_vocab,
        )
        assert len(ds) == 1


# ============================================================================
# TestComputeTimeDeltas
# ============================================================================


class TestComputeTimeDeltas:
    """Test ISO timestamp → inter-event hour delta computation."""

    def test_computes_hour_deltas(self, mock_activity_vocab, mock_resource_vocab, tmp_path):
        """Events 1 hour apart should produce delta of 1.0."""
        cases = [{
            "caseId": "delta_test",
            "events": [
                {"activity": "a", "resource": "r", "timestamp": "2024-01-01T09:00:00Z"},
                {"activity": "b", "resource": "r", "timestamp": "2024-01-01T10:00:00Z"},
                {"activity": "c", "resource": "r", "timestamp": "2024-01-01T12:00:00Z"},
            ],
            "outcome": {"onTime": True, "rework": False, "durationHours": 3.0},
        }]
        json_path = tmp_path / "delta_cases.json"
        _write_cases_json(cases, json_path)

        ds = EventSequenceDataset(
            events_path=json_path,
            activity_vocab=mock_activity_vocab,
            resource_vocab=mock_resource_vocab,
        )
        sample = ds[0]
        deltas = sample["time_deltas"]

        assert deltas[0].item() == pytest.approx(0.0)  # First event has no delta
        assert deltas[1].item() == pytest.approx(1.0)   # 09:00 → 10:00 = 1 hour
        assert deltas[2].item() == pytest.approx(2.0)   # 10:00 → 12:00 = 2 hours

    def test_missing_timestamps_produce_zero_deltas(
        self, mock_activity_vocab, mock_resource_vocab, tmp_path
    ):
        cases = [{
            "caseId": "no_ts",
            "events": [
                {"activity": "a", "resource": "r"},
                {"activity": "b", "resource": "r"},
            ],
            "outcome": {"onTime": True, "rework": False, "durationHours": 1.0},
        }]
        json_path = tmp_path / "no_ts.json"
        _write_cases_json(cases, json_path)

        ds = EventSequenceDataset(
            events_path=json_path,
            activity_vocab=mock_activity_vocab,
            resource_vocab=mock_resource_vocab,
        )
        sample = ds[0]
        assert torch.all(sample["time_deltas"] == 0.0)


# ============================================================================
# TestCollateFunction
# ============================================================================


class TestCollateFunction:
    """Test variable-length padding and batching."""

    def test_pads_to_longest_sequence(self, mock_activity_vocab, mock_resource_vocab, tmp_path):
        """All tensors should be padded to the longest sequence in the batch."""
        cases = [
            make_case(case_id="short", n_events=2),
            make_case(case_id="long", n_events=5),
        ]
        json_path = tmp_path / "var_len.json"
        _write_cases_json(cases, json_path)

        ds = EventSequenceDataset(
            events_path=json_path,
            activity_vocab=mock_activity_vocab,
            resource_vocab=mock_resource_vocab,
        )
        batch = collate_fn([ds[0], ds[1]])

        max_len = max(2, 5)
        assert batch["activity_ids"].shape == (2, max_len)
        assert batch["resource_ids"].shape == (2, max_len)
        assert batch["attributes"].shape == (2, max_len, 8)
        assert batch["time_deltas"].shape == (2, max_len)

    def test_padding_mask_shape(self, mock_activity_vocab, mock_resource_vocab, tmp_path):
        """Padding mask should be True where padded."""
        cases = [
            make_case(case_id="s", n_events=2),
            make_case(case_id="l", n_events=4),
        ]
        json_path = tmp_path / "mask.json"
        _write_cases_json(cases, json_path)

        ds = EventSequenceDataset(
            events_path=json_path,
            activity_vocab=mock_activity_vocab,
            resource_vocab=mock_resource_vocab,
        )
        batch = collate_fn([ds[0], ds[1]])

        assert "padding_mask" in batch
        assert batch["padding_mask"].shape == (2, 4)
        assert batch["padding_mask"].dtype == torch.bool

    def test_scalar_targets_stacked(self, mock_activity_vocab, mock_resource_vocab, tmp_path):
        """Scalar targets (ontime, rework, remaining) should be 1-D tensors."""
        cases = [make_case(case_id=f"c{i}", n_events=3) for i in range(3)]
        json_path = tmp_path / "scalars.json"
        _write_cases_json(cases, json_path)

        ds = EventSequenceDataset(
            events_path=json_path,
            activity_vocab=mock_activity_vocab,
            resource_vocab=mock_resource_vocab,
        )
        batch = collate_fn([ds[0], ds[1], ds[2]])

        assert batch["target_ontime"].shape == (3,)
        assert batch["target_rework"].shape == (3,)
        assert batch["target_remaining"].shape == (3,)

    def test_seq_lens_preserved(self, mock_activity_vocab, mock_resource_vocab, tmp_path):
        """seq_lens should report the original lengths (before padding)."""
        cases = [
            make_case(case_id="a", n_events=2),
            make_case(case_id="b", n_events=5),
        ]
        json_path = tmp_path / "seqlens.json"
        _write_cases_json(cases, json_path)

        ds = EventSequenceDataset(
            events_path=json_path,
            activity_vocab=mock_activity_vocab,
            resource_vocab=mock_resource_vocab,
        )
        batch = collate_fn([ds[0], ds[1]])

        # collate_fn sorts descending by length
        lens = batch["seq_lens"].tolist()
        assert sorted(lens, reverse=True) == lens
        assert 2 in lens
        assert 5 in lens
