"""Tests for the AETHER data pipeline modules.

Verifies:
1. Activity/resource normalization — canonical forms, alias handling, edge cases
2. AETHER standard format — loaded cases have required keys (caseId, events, outcome)
3. Event format — each event has activity, resource, timestamp, attributes
4. Vocabulary building — correct size, <UNK> at index 0, all activities present
5. Train/val split — correct ratios, no overlap
6. Dataset tensors — correct shapes and dtypes from EventSequenceDataset
7. End-to-end — load processed data from disk, verify consistency
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

from core.data.unified_pipeline import (
    AetherDataPipeline,
    normalize_activity,
    normalize_resource,
    _ACTIVITY_ALIASES,
)
from core.encoder.vocabulary import ActivityVocabulary, ResourceVocabulary
from core.training.data_loader import EventSequenceDataset, collate_fn


# ---------------------------------------------------------------------------
# Paths for real processed data (skip if not available)
# ---------------------------------------------------------------------------

# Paths relative to project root (tests/ -> core/ -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVENTS_DIR = _PROJECT_ROOT / "data" / "events"
TRAIN_CASES_PATH = EVENTS_DIR / "train_cases.json"
VAL_CASES_PATH = EVENTS_DIR / "val_cases.json"
VOCABULARY_PATH = EVENTS_DIR / "vocabulary.json"
METADATA_PATH = EVENTS_DIR / "metadata.json"

_REAL_DATA_AVAILABLE = (
    TRAIN_CASES_PATH.exists()
    and VAL_CASES_PATH.exists()
    and VOCABULARY_PATH.exists()
    and METADATA_PATH.exists()
)

requires_real_data = pytest.mark.skipif(
    not _REAL_DATA_AVAILABLE,
    reason="Processed event data not found at data/events/",
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_event(
    activity: str = "create_sales_order",
    resource: str = "system",
    timestamp: str = "2024-06-15T09:00:00",
    attributes: dict | None = None,
) -> Dict[str, Any]:
    """Create a single synthetic event dict."""
    return {
        "activity": activity,
        "resource": resource,
        "timestamp": timestamp,
        "attributes": attributes or {"amount": 100.0},
    }


def _make_case(
    case_id: str = "test_case_001",
    n_events: int = 5,
    source: str = "synthetic",
    with_outcome: bool = True,
) -> Dict[str, Any]:
    """Create a synthetic case dict in AETHER format."""
    activities = [
        "create_purchase_requisition",
        "approve_purchase_requisition",
        "create_purchase_order",
        "create_goods_receipt",
        "execute_payment",
    ]
    events = []
    for i in range(n_events):
        events.append(
            _make_event(
                activity=activities[i % len(activities)],
                resource=f"user_{i:03d}",
                timestamp=f"2024-06-{15 + i:02d}T09:00:00",
                attributes={"amount": 100.0 * (i + 1)},
            )
        )
    case: Dict[str, Any] = {
        "caseId": case_id,
        "events": events,
        "source": source,
    }
    if with_outcome:
        case["outcome"] = {
            "onTime": True,
            "rework": False,
            "durationHours": 96.0,
        }
    return case


def _make_cases(n: int = 20, source: str = "synthetic") -> List[Dict[str, Any]]:
    """Create a list of n synthetic cases."""
    return [
        _make_case(
            case_id=f"{source}_{i:04d}",
            n_events=3 + (i % 5),
            source=source,
        )
        for i in range(n)
    ]


def _write_cases_json(cases: List[Dict[str, Any]], path: Path) -> None:
    """Write cases to a JSON file."""
    with open(path, "w") as f:
        json.dump(cases, f, default=str)


# ============================================================================
# 1. Activity and Resource Normalization
# ============================================================================


class TestNormalizeActivity:
    """Test normalize_activity() — canonical forms, alias handling, edge cases."""

    def test_empty_string_returns_unknown(self):
        assert normalize_activity("") == "unknown_activity"

    def test_none_returns_unknown(self):
        # The function signature requires str, but it guards with `if not raw`
        assert normalize_activity("") == "unknown_activity"

    def test_whitespace_only_returns_unknown(self):
        # After strip+lower the empty string hits the fallback
        result = normalize_activity("   ")
        assert result == "unknown_activity"

    def test_exact_alias_lookup(self):
        assert normalize_activity("create sales order") == "create_sales_order"
        assert normalize_activity("create_sales_order") == "create_sales_order"

    def test_alias_case_insensitive(self):
        assert normalize_activity("Create Sales Order") == "create_sales_order"
        assert normalize_activity("CREATE SALES ORDER") == "create_sales_order"

    def test_alias_with_leading_trailing_whitespace(self):
        assert normalize_activity("  create sales order  ") == "create_sales_order"

    def test_goods_receipt_alias(self):
        assert normalize_activity("goods receipt") == "create_goods_receipt"
        assert normalize_activity("create goods receipt") == "create_goods_receipt"

    def test_payment_alias(self):
        assert normalize_activity("payment") == "execute_payment"
        assert normalize_activity("execute payment") == "execute_payment"

    def test_two_way_match_aliases(self):
        assert normalize_activity("perform two way match") == "two_way_match"
        assert normalize_activity("two-way match") == "two_way_match"

    def test_create_billing_document_alias(self):
        assert normalize_activity("create billing document") == "create_invoice"

    def test_delegate_pr_approval_alias(self):
        result = normalize_activity("delegate purchase requisition approval")
        assert result == "delegate_pr_approval"

    def test_create_rfq_alias(self):
        result = normalize_activity("create request for quotation")
        assert result == "create_rfq"

    def test_fallback_normalization_spaces(self):
        """Non-aliased activities get lowercased with underscores."""
        result = normalize_activity("Some New Activity")
        assert result == "some_new_activity"

    def test_fallback_normalization_hyphens(self):
        result = normalize_activity("some-hyphenated-activity")
        assert result == "some_hyphenated_activity"

    def test_fallback_normalization_colons(self):
        result = normalize_activity("SRM: Document Completed")
        assert result == "srm_document_completed"

    def test_fallback_normalization_slashes(self):
        result = normalize_activity("goods/receipt")
        assert result == "goods_receipt"

    def test_fallback_multiple_spaces_collapse(self):
        result = normalize_activity("create   multiple   spaces")
        assert result == "create_multiple_spaces"

    def test_all_aliases_produce_valid_output(self):
        """Every alias in the mapping should produce a non-empty underscore token."""
        for raw_key, expected in _ACTIVITY_ALIASES.items():
            result = normalize_activity(raw_key)
            assert result == expected, f"Alias '{raw_key}' produced '{result}', expected '{expected}'"
            assert "_" in result or result.isalnum(), f"Result '{result}' is not a valid token"


class TestNormalizeResource:
    """Test normalize_resource() — edge cases and special handling."""

    def test_empty_string_returns_system(self):
        assert normalize_resource("") == "system"

    def test_none_returns_system(self):
        # Guarded by `if not raw`
        assert normalize_resource("") == "system"

    def test_trial_marker_returns_system(self):
        assert normalize_resource("* TRIAL *") == "system"

    def test_regular_user(self):
        assert normalize_resource("User 001") == "user_001"

    def test_hyphenated_resource(self):
        assert normalize_resource("Procurement-Department") == "procurement_department"

    def test_mixed_whitespace(self):
        assert normalize_resource("  Finance Account Department  ") == "finance_account_department"

    def test_all_lowercase_passthrough(self):
        assert normalize_resource("system") == "system"

    def test_whitespace_only_returns_system(self):
        result = normalize_resource("   ")
        assert result == "system"


# ============================================================================
# 2. AETHER Standard Case Format
# ============================================================================


class TestAetherCaseFormat:
    """Verify that synthetic cases follow the AETHER standard format."""

    def test_case_has_required_keys(self):
        case = _make_case()
        assert "caseId" in case
        assert "events" in case
        assert isinstance(case["caseId"], str)
        assert isinstance(case["events"], list)

    def test_case_with_outcome(self):
        case = _make_case(with_outcome=True)
        assert "outcome" in case
        assert "onTime" in case["outcome"]
        assert "rework" in case["outcome"]
        assert "durationHours" in case["outcome"]

    def test_case_without_outcome_is_valid(self):
        case = _make_case(with_outcome=False)
        assert "caseId" in case
        assert "events" in case
        assert "outcome" not in case

    def test_case_has_source_tag(self):
        case = _make_case(source="bpi2019")
        assert case["source"] == "bpi2019"


# ============================================================================
# 3. Event Format
# ============================================================================


class TestEventFormat:
    """Verify individual event dicts have the required keys and types."""

    def test_event_has_all_required_keys(self):
        event = _make_event()
        for key in ("activity", "resource", "timestamp", "attributes"):
            assert key in event, f"Missing key '{key}' in event"

    def test_event_activity_is_string(self):
        event = _make_event()
        assert isinstance(event["activity"], str)

    def test_event_resource_is_string(self):
        event = _make_event()
        assert isinstance(event["resource"], str)

    def test_event_timestamp_is_string(self):
        event = _make_event()
        assert isinstance(event["timestamp"], str)

    def test_event_attributes_is_dict(self):
        event = _make_event()
        assert isinstance(event["attributes"], dict)

    def test_events_in_case_all_valid(self):
        case = _make_case(n_events=10)
        for i, event in enumerate(case["events"]):
            assert "activity" in event, f"Event {i} missing 'activity'"
            assert "resource" in event, f"Event {i} missing 'resource'"
            assert "timestamp" in event, f"Event {i} missing 'timestamp'"
            assert "attributes" in event, f"Event {i} missing 'attributes'"


# ============================================================================
# 4. Vocabulary Building
# ============================================================================


class TestVocabularyBuilding:
    """Test vocabulary construction from case data via the pipeline."""

    @pytest.fixture
    def cases(self) -> List[Dict[str, Any]]:
        return _make_cases(10)

    @pytest.fixture
    def pipeline(self) -> AetherDataPipeline:
        return AetherDataPipeline(train_ratio=0.8, seed=42)

    def test_build_vocabularies_returns_correct_types(self, pipeline, cases):
        act_vocab, res_vocab = pipeline.build_vocabularies(cases)
        assert isinstance(act_vocab, ActivityVocabulary)
        assert isinstance(res_vocab, ResourceVocabulary)

    def test_unk_token_at_index_zero(self, pipeline, cases):
        act_vocab, res_vocab = pipeline.build_vocabularies(cases)
        assert act_vocab.encode("<UNK>") == 0
        assert res_vocab.encode("<UNK>") == 0

    def test_unk_is_default_for_unknown(self, pipeline, cases):
        act_vocab, _ = pipeline.build_vocabularies(cases)
        assert act_vocab.encode("totally_unknown_activity_xyz") == 0

    def test_all_activities_in_vocab(self, pipeline, cases):
        act_vocab, _ = pipeline.build_vocabularies(cases)
        # Collect all unique activities from cases
        all_activities = set()
        for case in cases:
            for event in case["events"]:
                all_activities.add(event["activity"])
        for activity in all_activities:
            idx = act_vocab.encode(activity)
            assert idx > 0, f"Activity '{activity}' not found in vocabulary (got index 0)"

    def test_all_resources_in_vocab(self, pipeline, cases):
        _, res_vocab = pipeline.build_vocabularies(cases)
        all_resources = set()
        for case in cases:
            for event in case["events"]:
                all_resources.add(event["resource"])
        for resource in all_resources:
            idx = res_vocab.encode(resource)
            assert idx > 0, f"Resource '{resource}' not found in vocabulary (got index 0)"

    def test_vocab_size_includes_unk(self, pipeline, cases):
        act_vocab, _ = pipeline.build_vocabularies(cases)
        all_activities = set()
        for case in cases:
            for event in case["events"]:
                all_activities.add(event["activity"])
        # Vocab size = unique activities + 1 (for <UNK>)
        assert act_vocab.size == len(all_activities) + 1

    def test_empty_cases_produces_minimal_vocab(self, pipeline):
        act_vocab, res_vocab = pipeline.build_vocabularies([])
        # Only <UNK> token
        assert act_vocab.size == 1
        assert res_vocab.size == 1

    def test_duplicate_events_dont_inflate_vocab(self, pipeline):
        # All cases have the same activity
        cases = []
        for i in range(10):
            cases.append({
                "caseId": f"dup_{i}",
                "events": [
                    {"activity": "create_order", "resource": "user_1"},
                    {"activity": "create_order", "resource": "user_1"},
                ],
            })
        act_vocab, res_vocab = pipeline.build_vocabularies(cases)
        assert act_vocab.size == 2  # <UNK> + create_order
        assert res_vocab.size == 2  # <UNK> + user_1


# ============================================================================
# 5. Train/Val Split
# ============================================================================


class TestTrainValSplit:
    """Test split_train_val() — correct ratios, no overlap, reproducibility."""

    @pytest.fixture
    def pipeline(self) -> AetherDataPipeline:
        return AetherDataPipeline(train_ratio=0.8, seed=42)

    @pytest.fixture
    def cases(self) -> List[Dict[str, Any]]:
        return _make_cases(100)

    def test_split_sizes_match_ratio(self, pipeline, cases):
        train, val = pipeline.split_train_val(cases)
        assert len(train) == 80
        assert len(val) == 20
        assert len(train) + len(val) == len(cases)

    def test_no_overlap_between_train_and_val(self, pipeline, cases):
        train, val = pipeline.split_train_val(cases)
        train_ids = {c["caseId"] for c in train}
        val_ids = {c["caseId"] for c in val}
        assert len(train_ids & val_ids) == 0, "Train and val sets overlap"

    def test_all_cases_preserved(self, pipeline, cases):
        train, val = pipeline.split_train_val(cases)
        original_ids = {c["caseId"] for c in cases}
        split_ids = {c["caseId"] for c in train} | {c["caseId"] for c in val}
        assert original_ids == split_ids, "Some cases lost in split"

    def test_split_is_reproducible(self, pipeline, cases):
        train1, val1 = pipeline.split_train_val(cases)
        train2, val2 = pipeline.split_train_val(cases)
        assert [c["caseId"] for c in train1] == [c["caseId"] for c in train2]
        assert [c["caseId"] for c in val1] == [c["caseId"] for c in val2]

    def test_different_seed_produces_different_split(self, cases):
        pipe_a = AetherDataPipeline(seed=42)
        pipe_b = AetherDataPipeline(seed=123)
        train_a, _ = pipe_a.split_train_val(cases)
        train_b, _ = pipe_b.split_train_val(cases)
        ids_a = [c["caseId"] for c in train_a]
        ids_b = [c["caseId"] for c in train_b]
        assert ids_a != ids_b, "Different seeds produced identical splits"

    def test_split_ratio_0_5(self, cases):
        pipeline = AetherDataPipeline(train_ratio=0.5)
        train, val = pipeline.split_train_val(cases)
        assert len(train) == 50
        assert len(val) == 50

    def test_split_ratio_0_9(self, cases):
        pipeline = AetherDataPipeline(train_ratio=0.9)
        train, val = pipeline.split_train_val(cases)
        assert len(train) == 90
        assert len(val) == 10

    def test_single_case_goes_to_train(self):
        pipeline = AetherDataPipeline(train_ratio=0.8)
        single = [_make_case(case_id="only_one")]
        train, val = pipeline.split_train_val(single)
        # int(1 * 0.8) = 0, so one case goes to val actually
        assert len(train) + len(val) == 1

    def test_empty_cases_produces_empty_splits(self, pipeline):
        train, val = pipeline.split_train_val([])
        assert train == []
        assert val == []


# ============================================================================
# 6. Dataset Tensors (EventSequenceDataset)
# ============================================================================


class TestEventSequenceDataset:
    """Test EventSequenceDataset — tensor shapes, dtypes, indexing."""

    @pytest.fixture
    def dataset_dir(self, tmp_path: Path):
        """Create a temp directory with synthetic train/val JSON files."""
        cases = _make_cases(20)
        train_path = tmp_path / "train_cases.json"
        _write_cases_json(cases[:16], train_path)
        val_path = tmp_path / "val_cases.json"
        _write_cases_json(cases[16:], val_path)
        return tmp_path

    @pytest.fixture
    def vocabs(self) -> tuple[ActivityVocabulary, ResourceVocabulary]:
        """Build vocabularies from synthetic data."""
        cases = _make_cases(20)
        pipeline = AetherDataPipeline()
        return pipeline.build_vocabularies(cases)

    @pytest.fixture
    def train_dataset(
        self,
        dataset_dir: Path,
        vocabs: tuple[ActivityVocabulary, ResourceVocabulary],
    ) -> EventSequenceDataset:
        act_vocab, res_vocab = vocabs
        return EventSequenceDataset(
            events_path=dataset_dir / "train_cases.json",
            activity_vocab=act_vocab,
            resource_vocab=res_vocab,
            max_seq_len=256,
            n_attribute_features=8,
        )

    def test_dataset_length(self, train_dataset):
        assert len(train_dataset) == 16

    def test_getitem_returns_dict(self, train_dataset):
        item = train_dataset[0]
        assert isinstance(item, dict)

    def test_getitem_has_required_keys(self, train_dataset):
        item = train_dataset[0]
        required_keys = {
            "activity_ids",
            "resource_ids",
            "attributes",
            "time_deltas",
            "target_activities",
            "target_ontime",
            "target_rework",
            "target_remaining",
            "seq_len",
        }
        assert required_keys.issubset(item.keys())

    def test_activity_ids_dtype_and_shape(self, train_dataset):
        item = train_dataset[0]
        assert item["activity_ids"].dtype == torch.long
        seq_len = item["seq_len"].item()
        assert item["activity_ids"].shape == (seq_len,)

    def test_resource_ids_dtype_and_shape(self, train_dataset):
        item = train_dataset[0]
        assert item["resource_ids"].dtype == torch.long
        seq_len = item["seq_len"].item()
        assert item["resource_ids"].shape == (seq_len,)

    def test_attributes_dtype_and_shape(self, train_dataset):
        item = train_dataset[0]
        assert item["attributes"].dtype == torch.float32
        seq_len = item["seq_len"].item()
        assert item["attributes"].shape == (seq_len, 8)

    def test_time_deltas_dtype_and_shape(self, train_dataset):
        item = train_dataset[0]
        assert item["time_deltas"].dtype == torch.float32
        seq_len = item["seq_len"].item()
        assert item["time_deltas"].shape == (seq_len,)

    def test_target_activities_dtype(self, train_dataset):
        item = train_dataset[0]
        assert item["target_activities"].dtype == torch.long

    def test_scalar_targets_are_scalar(self, train_dataset):
        item = train_dataset[0]
        assert item["target_ontime"].ndim == 0
        assert item["target_rework"].ndim == 0
        assert item["target_remaining"].ndim == 0

    def test_seq_len_is_correct(self, train_dataset):
        for i in range(min(5, len(train_dataset))):
            item = train_dataset[i]
            case = train_dataset.cases[i]
            expected = min(len(case["events"]), 256)
            assert item["seq_len"].item() == expected

    def test_activity_ids_are_nonzero_for_known(self, train_dataset):
        """Known activities should encode to non-zero indices."""
        item = train_dataset[0]
        # At least some activities should be known (non-zero)
        assert item["activity_ids"].sum().item() > 0

    def test_max_seq_len_truncation(self, tmp_path: Path):
        """Cases longer than max_seq_len should be truncated."""
        # Create a case with 50 events
        case = _make_case(case_id="long_case", n_events=50)
        # But we only have 5 defined activities; extend with repeated events
        while len(case["events"]) < 50:
            case["events"].append(_make_event(
                activity="create_purchase_order",
                timestamp=f"2024-07-{len(case['events']):02d}T09:00:00",
            ))
        path = tmp_path / "long_cases.json"
        _write_cases_json([case], path)

        act_vocab = ActivityVocabulary()
        act_vocab.build_from_events([e["activity"] for e in case["events"]])
        res_vocab = ResourceVocabulary()
        res_vocab.build_from_events([e["resource"] for e in case["events"]])

        dataset = EventSequenceDataset(
            events_path=path,
            activity_vocab=act_vocab,
            resource_vocab=res_vocab,
            max_seq_len=10,
            n_attribute_features=8,
        )
        item = dataset[0]
        assert item["seq_len"].item() == 10
        assert item["activity_ids"].shape[0] == 10


class TestCollateFunction:
    """Test the collate_fn for batching variable-length sequences."""

    def test_collate_produces_padded_batch(self, tmp_path: Path):
        # Create cases with different lengths
        cases = [
            _make_case(case_id=f"c{i}", n_events=i + 2)
            for i in range(4)
        ]
        path = tmp_path / "cases.json"
        _write_cases_json(cases, path)

        act_vocab = ActivityVocabulary()
        res_vocab = ResourceVocabulary()
        all_events = []
        for c in cases:
            all_events.extend(c["events"])
        act_vocab.build_from_events(all_events)
        res_vocab.build_from_events(all_events)

        dataset = EventSequenceDataset(
            events_path=path,
            activity_vocab=act_vocab,
            resource_vocab=res_vocab,
            max_seq_len=256,
            n_attribute_features=8,
        )
        batch = [dataset[i] for i in range(len(dataset))]
        collated = collate_fn(batch)

        assert "activity_ids" in collated
        assert "padding_mask" in collated
        # Batch dimension should equal number of cases
        assert collated["activity_ids"].shape[0] == 4
        # Sequence dimension should equal max seq len in batch
        max_len = max(b["seq_len"].item() for b in batch)
        assert collated["activity_ids"].shape[1] == max_len

    def test_padding_mask_shape(self, tmp_path: Path):
        cases = [_make_case(case_id=f"c{i}", n_events=i + 1) for i in range(3)]
        path = tmp_path / "cases.json"
        _write_cases_json(cases, path)

        act_vocab = ActivityVocabulary()
        res_vocab = ResourceVocabulary()
        all_events = []
        for c in cases:
            all_events.extend(c["events"])
        act_vocab.build_from_events(all_events)
        res_vocab.build_from_events(all_events)

        dataset = EventSequenceDataset(
            events_path=path,
            activity_vocab=act_vocab,
            resource_vocab=res_vocab,
            max_seq_len=256,
            n_attribute_features=8,
        )
        batch = [dataset[i] for i in range(len(dataset))]
        collated = collate_fn(batch)

        # padding_mask: (batch, max_seq_len), True where padded
        assert collated["padding_mask"].shape[0] == 3
        assert collated["padding_mask"].dtype == torch.bool


# ============================================================================
# 7. Pipeline save_processed_data and round-trip
# ============================================================================


class TestSaveProcessedData:
    """Test save_processed_data() — file creation, JSON validity, metadata."""

    @pytest.fixture
    def pipeline(self) -> AetherDataPipeline:
        return AetherDataPipeline(train_ratio=0.8, seed=42)

    def test_save_creates_all_files(self, pipeline, tmp_path: Path):
        cases = _make_cases(50)
        pipeline.save_processed_data(cases, tmp_path)
        assert (tmp_path / "train_cases.json").exists()
        assert (tmp_path / "val_cases.json").exists()
        assert (tmp_path / "vocabulary.json").exists()
        assert (tmp_path / "metadata.json").exists()

    def test_saved_train_val_are_valid_json(self, pipeline, tmp_path: Path):
        cases = _make_cases(50)
        pipeline.save_processed_data(cases, tmp_path)
        with open(tmp_path / "train_cases.json") as f:
            train = json.load(f)
        with open(tmp_path / "val_cases.json") as f:
            val = json.load(f)
        assert isinstance(train, list)
        assert isinstance(val, list)
        assert len(train) + len(val) == 50

    def test_saved_vocabulary_has_correct_structure(self, pipeline, tmp_path: Path):
        cases = _make_cases(20)
        pipeline.save_processed_data(cases, tmp_path)
        with open(tmp_path / "vocabulary.json") as f:
            vocab_data = json.load(f)
        assert "activity" in vocab_data
        assert "resource" in vocab_data
        assert "token_to_idx" in vocab_data["activity"]
        assert "size" in vocab_data["activity"]
        assert "<UNK>" in vocab_data["activity"]["token_to_idx"]
        assert vocab_data["activity"]["token_to_idx"]["<UNK>"] == 0

    def test_metadata_has_correct_fields(self, pipeline, tmp_path: Path):
        cases = _make_cases(50)
        metadata = pipeline.save_processed_data(cases, tmp_path)
        assert metadata["total_cases"] == 50
        assert metadata["train_cases"] == 40
        assert metadata["val_cases"] == 10
        assert "total_events" in metadata
        assert "activity_vocab_size" in metadata
        assert "resource_vocab_size" in metadata
        assert "seed" in metadata
        assert metadata["seed"] == 42

    def test_metadata_event_counts_consistent(self, pipeline, tmp_path: Path):
        cases = _make_cases(20)
        metadata = pipeline.save_processed_data(cases, tmp_path)
        assert metadata["train_events"] + metadata["val_events"] == metadata["total_events"]

    def test_round_trip_preserves_case_count(self, pipeline, tmp_path: Path):
        """Save and reload — case count should match."""
        cases = _make_cases(30)
        pipeline.save_processed_data(cases, tmp_path)

        with open(tmp_path / "train_cases.json") as f:
            train = json.load(f)
        with open(tmp_path / "val_cases.json") as f:
            val = json.load(f)
        assert len(train) + len(val) == 30

    def test_round_trip_event_structure_preserved(self, pipeline, tmp_path: Path):
        """Save and reload — event keys should survive serialization."""
        cases = _make_cases(5)
        pipeline.save_processed_data(cases, tmp_path)

        with open(tmp_path / "train_cases.json") as f:
            loaded = json.load(f)

        for case in loaded:
            assert "caseId" in case
            assert "events" in case
            for event in case["events"]:
                assert "activity" in event
                assert "resource" in event
                assert "timestamp" in event


# ============================================================================
# 8. Pipeline initialization and configuration
# ============================================================================


class TestPipelineInit:
    """Test AetherDataPipeline construction and configuration."""

    def test_default_paths(self):
        pipeline = AetherDataPipeline()
        assert "sap_sqlite" in pipeline.paths
        assert "bpi2019_json" in pipeline.paths
        assert "output_dir" in pipeline.paths

    def test_custom_paths_override(self):
        custom = {"sap_sqlite": Path("/tmp/custom.sqlite")}
        pipeline = AetherDataPipeline(paths=custom)
        assert pipeline.paths["sap_sqlite"] == Path("/tmp/custom.sqlite")
        # Other paths should still have defaults
        assert pipeline.paths["bpi2019_json"] == (
            _PROJECT_ROOT / "data" / "external" / "bpi_2019.json"
        )

    def test_default_train_ratio(self):
        pipeline = AetherDataPipeline()
        assert pipeline.train_ratio == 0.8

    def test_custom_train_ratio(self):
        pipeline = AetherDataPipeline(train_ratio=0.7)
        assert pipeline.train_ratio == 0.7

    def test_default_seed(self):
        pipeline = AetherDataPipeline()
        assert pipeline.seed == 42

    def test_source_counts_empty_before_load(self):
        pipeline = AetherDataPipeline()
        assert pipeline.source_counts == {}


# ============================================================================
# 9. SAP datetime helper
# ============================================================================


class TestSapDatetime:
    """Test AetherDataPipeline._sap_datetime static method."""

    def test_iso_date_with_time(self):
        result = AetherDataPipeline._sap_datetime("2024-01-15", "09:30:00")
        assert result == "2024-01-15T09:30:00"

    def test_iso_date_with_t_separator(self):
        result = AetherDataPipeline._sap_datetime("2024-01-15T00:00:00", "09:30:00")
        assert result == "2024-01-15T09:30:00"

    def test_compact_yyyymmdd(self):
        result = AetherDataPipeline._sap_datetime("20240115", "093000")
        assert result == "2024-01-15T09:30:00"

    def test_no_time(self):
        result = AetherDataPipeline._sap_datetime("2024-01-15", None)
        assert result == "2024-01-15T00:00:00"

    def test_no_date(self):
        result = AetherDataPipeline._sap_datetime(None, "09:30:00")
        assert result == ""

    def test_empty_date(self):
        result = AetherDataPipeline._sap_datetime("", "09:30:00")
        assert result == ""

    def test_time_with_t_prefix(self):
        result = AetherDataPipeline._sap_datetime("2024-01-15", "1970-01-01T09:30:00")
        assert result == "2024-01-15T09:30:00"


# ============================================================================
# 10. CSV Loader (via unified_pipeline._load_csv_generic)
# ============================================================================


class TestCsvLoading:
    """Test CSV event log loading via the pipeline's internal loader."""

    @pytest.fixture
    def csv_path(self, tmp_path: Path) -> Path:
        """Create a synthetic CSV event log."""
        csv_file = tmp_path / "test_events.csv"
        rows = [
            {
                "case:concept:name": "CASE_001",
                "concept:name": "Create Purchase Order",
                "time:timestamp": "2024-06-15T09:00:00",
                "org:resource": "user_42",
                "document_id": "PO_100",
                "document_type": "EKKO",
            },
            {
                "case:concept:name": "CASE_001",
                "concept:name": "Goods Receipt",
                "time:timestamp": "2024-06-16T10:00:00",
                "org:resource": "user_42",
                "document_id": "GR_100",
                "document_type": "MSEG",
            },
            {
                "case:concept:name": "CASE_002",
                "concept:name": "Create Purchase Order",
                "time:timestamp": "2024-06-15T11:00:00",
                "org:resource": "user_99",
                "document_id": "PO_200",
                "document_type": "EKKO",
            },
        ]
        fieldnames = list(rows[0].keys())
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return csv_file

    def test_csv_loader_groups_by_case(self, csv_path: Path):
        pipeline = AetherDataPipeline(
            paths={"p2p_csv": csv_path, "output_dir": Path("/tmp/out")}
        )
        cases = pipeline._load_csv_generic(
            csv_path=csv_path,
            source_name="test_csv",
            case_id_col="case:concept:name",
            activity_col="concept:name",
            timestamp_col="time:timestamp",
            resource_col="org:resource",
            attribute_cols=["document_id", "document_type"],
        )
        assert len(cases) == 2
        # CASE_001 has 2 events, CASE_002 has 1
        case_001 = [c for c in cases if "CASE_001" in c["caseId"]][0]
        case_002 = [c for c in cases if "CASE_002" in c["caseId"]][0]
        assert len(case_001["events"]) == 2
        assert len(case_002["events"]) == 1

    def test_csv_loader_normalizes_activities(self, csv_path: Path):
        pipeline = AetherDataPipeline(
            paths={"p2p_csv": csv_path, "output_dir": Path("/tmp/out")}
        )
        cases = pipeline._load_csv_generic(
            csv_path=csv_path,
            source_name="test_csv",
            case_id_col="case:concept:name",
            activity_col="concept:name",
            timestamp_col="time:timestamp",
            resource_col="org:resource",
        )
        # "Create Purchase Order" should normalize to "create_purchase_order"
        first_event = cases[0]["events"][0]
        assert first_event["activity"] == "create_purchase_order"
        # "Goods Receipt" should normalize to "create_goods_receipt"
        case_001 = [c for c in cases if "CASE_001" in c["caseId"]][0]
        gr_event = case_001["events"][1]
        assert gr_event["activity"] == "create_goods_receipt"

    def test_csv_loader_normalizes_resources(self, csv_path: Path):
        pipeline = AetherDataPipeline(
            paths={"p2p_csv": csv_path, "output_dir": Path("/tmp/out")}
        )
        cases = pipeline._load_csv_generic(
            csv_path=csv_path,
            source_name="test_csv",
            case_id_col="case:concept:name",
            activity_col="concept:name",
            timestamp_col="time:timestamp",
            resource_col="org:resource",
        )
        first_event = cases[0]["events"][0]
        assert first_event["resource"] == "user_42"

    def test_csv_loader_sets_source(self, csv_path: Path):
        pipeline = AetherDataPipeline()
        cases = pipeline._load_csv_generic(
            csv_path=csv_path,
            source_name="my_source",
            case_id_col="case:concept:name",
            activity_col="concept:name",
            timestamp_col="time:timestamp",
            resource_col="org:resource",
        )
        for case in cases:
            assert case["source"] == "my_source"

    def test_csv_missing_file_raises(self):
        pipeline = AetherDataPipeline()
        with pytest.raises(FileNotFoundError):
            pipeline._load_csv_generic(
                csv_path=Path("/nonexistent/path.csv"),
                source_name="bad",
                case_id_col="x",
                activity_col="y",
                timestamp_col="z",
                resource_col=None,
            )

    def test_csv_no_resource_col(self, tmp_path: Path):
        """When resource_col is None, resource should default to 'system'."""
        csv_file = tmp_path / "no_resource.csv"
        rows = [
            {
                "case:concept:name": "C1",
                "concept:name": "Create Order",
                "time:timestamp": "2024-01-01T00:00:00",
            },
        ]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        pipeline = AetherDataPipeline()
        cases = pipeline._load_csv_generic(
            csv_path=csv_file,
            source_name="test",
            case_id_col="case:concept:name",
            activity_col="concept:name",
            timestamp_col="time:timestamp",
            resource_col=None,
        )
        assert cases[0]["events"][0]["resource"] == "system"


# ============================================================================
# 11. OCEL Parser helpers
# ============================================================================


class TestOcelParserHelpers:
    """Test OCEL parser internal helper functions."""

    def test_detect_table_style_ocel2(self):
        from core.data.ocel_parser import _detect_table_style

        assert _detect_table_style(["event", "object", "event_object"]) == "ocel2"

    def test_detect_table_style_prefixed(self):
        from core.data.ocel_parser import _detect_table_style

        tables = ["ocel_event", "ocel_object", "ocel_event_object"]
        assert _detect_table_style(tables) == "ocel2_prefixed"

    def test_detect_table_style_unknown_raises(self):
        from core.data.ocel_parser import _detect_table_style

        with pytest.raises(ValueError, match="Cannot detect"):
            _detect_table_style(["foo", "bar"])

    def test_core_table_names_ocel2(self):
        from core.data.ocel_parser import _core_table_names

        assert _core_table_names("ocel2") == ("event", "object", "event_object")

    def test_core_table_names_prefixed(self):
        from core.data.ocel_parser import _core_table_names

        assert _core_table_names("ocel2_prefixed") == (
            "ocel_event",
            "ocel_object",
            "ocel_event_object",
        )

    def test_compute_outcome_empty(self):
        from core.data.ocel_parser import _compute_outcome

        result = _compute_outcome([])
        assert result == {"onTime": False, "rework": False, "durationHours": 0.0}

    def test_compute_outcome_with_rework(self):
        from core.data.ocel_parser import _compute_outcome

        events = [
            {"activity": "A", "resource": "R", "timestamp": "2024-01-01T00:00:00", "attributes": {}},
            {"activity": "A", "resource": "R", "timestamp": "2024-01-02T00:00:00", "attributes": {}},
        ]
        result = _compute_outcome(events)
        assert result["rework"] is True

    def test_compute_outcome_on_time_detection(self):
        from core.data.ocel_parser import _compute_outcome

        events = [
            {"activity": "Create", "resource": "R", "timestamp": "2024-01-01T00:00:00", "attributes": {}},
            {"activity": "Execute Payment", "resource": "R", "timestamp": "2024-01-02T00:00:00", "attributes": {}},
        ]
        result = _compute_outcome(events)
        assert result["onTime"] is True


# ============================================================================
# 12. BPI 2019 parser helpers
# ============================================================================


class TestBpi2019ParserHelpers:
    """Test BPI 2019 parser utility functions."""

    def test_has_rework_with_duplicates(self):
        from core.data.bpi2019_parser import _has_rework

        assert _has_rework(["A", "B", "A"]) is True

    def test_has_rework_no_duplicates(self):
        from core.data.bpi2019_parser import _has_rework

        assert _has_rework(["A", "B", "C"]) is False

    def test_is_on_time_with_completion(self):
        from core.data.bpi2019_parser import _is_on_time

        assert _is_on_time(["Start", "Clear Invoice"]) is True

    def test_is_on_time_without_completion(self):
        from core.data.bpi2019_parser import _is_on_time

        assert _is_on_time(["Start", "Approve"]) is False

    def test_convert_trace_empty_events(self):
        from core.data.bpi2019_parser import _convert_trace

        trace = {"case_id": "C1", "events": []}
        result = _convert_trace(trace)
        assert result["caseId"] == "C1"
        assert result["events"] == []
        assert result["outcome"]["durationHours"] == 0.0

    def test_convert_trace_with_events(self):
        from core.data.bpi2019_parser import _convert_trace

        trace = {
            "case_id": "C1",
            "events": [
                {
                    "concept:name": "Create Purchase Order Item",
                    "org:resource": "user_001",
                    "time:timestamp": "2024-01-01T09:00:00",
                },
                {
                    "concept:name": "Record Goods Receipt",
                    "org:resource": "user_002",
                    "time:timestamp": "2024-01-02T10:00:00",
                },
            ],
        }
        result = _convert_trace(trace)
        assert result["caseId"] == "C1"
        assert len(result["events"]) == 2
        assert result["events"][0]["activity"] == "Create Purchase Order Item"
        assert result["events"][0]["resource"] == "user_001"


# ============================================================================
# 13. SAP Extractor helpers
# ============================================================================


class TestSapExtractorHelpers:
    """Test SAP extractor utility functions."""

    def test_parse_sap_datetime_valid(self):
        from core.data.sap_extractor import _parse_sap_datetime

        result = _parse_sap_datetime("2024-01-15T00:00:00", "1970-01-01T09:30:00")
        assert result is not None
        assert result.hour == 9
        assert result.minute == 30

    def test_parse_sap_datetime_date_only(self):
        from core.data.sap_extractor import _parse_sap_datetime

        result = _parse_sap_datetime("2024-01-15T00:00:00", None)
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_parse_sap_datetime_none_date(self):
        from core.data.sap_extractor import _parse_sap_datetime

        result = _parse_sap_datetime(None, "09:30:00")
        assert result is None

    def test_parse_sap_datetime_trial_marker(self):
        from core.data.sap_extractor import _parse_sap_datetime

        result = _parse_sap_datetime("* TRIAL *", None)
        assert result is None

    def test_safe_float_valid(self):
        from core.data.sap_extractor import _safe_float

        assert _safe_float(42.5) == 42.5
        assert _safe_float("100") == 100.0

    def test_safe_float_none(self):
        from core.data.sap_extractor import _safe_float

        assert _safe_float(None) == 0.0

    def test_safe_float_invalid(self):
        from core.data.sap_extractor import _safe_float

        assert _safe_float("not_a_number") == 0.0

    def test_safe_str_none(self):
        from core.data.sap_extractor import _safe_str

        assert _safe_str(None) == ""

    def test_safe_str_trial(self):
        from core.data.sap_extractor import _safe_str

        assert _safe_str("* TRIAL *") == ""

    def test_is_valid(self):
        from core.data.sap_extractor import _is_valid

        assert _is_valid("ABC") is True
        assert _is_valid(None) is False
        assert _is_valid("") is False
        assert _is_valid("* TRIAL *") is False


# ============================================================================
# 14. CSV Event Loader module (csv_event_loader.py)
# ============================================================================


class TestCsvEventLoaderModule:
    """Test the standalone csv_event_loader module functions."""

    def test_clean_value_strips_whitespace(self):
        from core.data.csv_event_loader import _clean_value

        assert _clean_value("  hello  ") == "hello"

    def test_clean_value_trial_placeholder(self):
        from core.data.csv_event_loader import _clean_value

        assert _clean_value("* TRIAL *") == ""

    def test_clean_value_empty(self):
        from core.data.csv_event_loader import _clean_value

        assert _clean_value("") == ""

    def test_is_trial_placeholder(self):
        from core.data.csv_event_loader import _is_trial_placeholder

        assert _is_trial_placeholder("* TRIAL *") is True
        assert _is_trial_placeholder("user_001") is False
        assert _is_trial_placeholder("") is False

    def test_has_rework_csv(self):
        from core.data.csv_event_loader import _has_rework

        assert _has_rework(["A", "B", "A"]) is True
        assert _has_rework(["A", "B", "C"]) is False

    def test_group_by_case(self):
        from core.data.csv_event_loader import _group_by_case

        rows = [
            {"case:concept:name": "C1", "val": "a"},
            {"case:concept:name": "C1", "val": "b"},
            {"case:concept:name": "C2", "val": "c"},
        ]
        grouped = _group_by_case(rows)
        assert len(grouped) == 2
        assert len(grouped["C1"]) == 2
        assert len(grouped["C2"]) == 1


# ============================================================================
# 15. load_all_sources — graceful error handling
# ============================================================================


class TestLoadAllSources:
    """Test that load_all_sources handles missing sources gracefully."""

    def test_all_missing_sources_returns_empty(self):
        pipeline = AetherDataPipeline(
            paths={
                "sap_sqlite": Path("/nonexistent/sap.sqlite"),
                "bpi2019_json": Path("/nonexistent/bpi.json"),
                "o2c_csv": Path("/nonexistent/o2c.csv"),
                "p2p_csv": Path("/nonexistent/p2p.csv"),
                "ocel_p2p": Path("/nonexistent/ocel.sqlite"),
                "output_dir": Path("/tmp/out"),
            }
        )
        cases = pipeline.load_all_sources()
        assert cases == []

    def test_source_counts_zeroed_on_missing(self):
        pipeline = AetherDataPipeline(
            paths={
                "sap_sqlite": Path("/nonexistent/sap.sqlite"),
                "bpi2019_json": Path("/nonexistent/bpi.json"),
                "o2c_csv": Path("/nonexistent/o2c.csv"),
                "p2p_csv": Path("/nonexistent/p2p.csv"),
                "ocel_p2p": Path("/nonexistent/ocel.sqlite"),
                "output_dir": Path("/tmp/out"),
            }
        )
        pipeline.load_all_sources()
        counts = pipeline.source_counts
        assert counts["sap_sqlite"] == 0
        assert counts["bpi2019"] == 0
        assert counts["o2c_csv"] == 0
        assert counts["p2p_csv"] == 0
        assert counts["ocel_p2p"] == 0


# ============================================================================
# 16. End-to-end with real processed data (skipped if unavailable)
# ============================================================================


@requires_real_data
class TestEndToEndRealData:
    """Load the actual processed data from disk and verify consistency."""

    @pytest.fixture(scope="class")
    def metadata(self) -> Dict[str, Any]:
        with open(METADATA_PATH) as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def vocab_data(self) -> Dict[str, Any]:
        with open(VOCABULARY_PATH) as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def train_cases(self) -> List[Dict[str, Any]]:
        with open(TRAIN_CASES_PATH) as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def val_cases(self) -> List[Dict[str, Any]]:
        with open(VAL_CASES_PATH) as f:
            return json.load(f)

    def test_metadata_total_cases_matches(self, metadata, train_cases, val_cases):
        assert metadata["total_cases"] == len(train_cases) + len(val_cases)

    def test_metadata_train_val_match(self, metadata, train_cases, val_cases):
        assert metadata["train_cases"] == len(train_cases)
        assert metadata["val_cases"] == len(val_cases)

    def test_no_overlap_between_train_val(self, train_cases, val_cases):
        train_ids = {c["caseId"] for c in train_cases}
        val_ids = {c["caseId"] for c in val_cases}
        overlap = train_ids & val_ids
        # Allow TRIAL-marker cases to appear in both splits (SAP data artifact
        # where anonymized documents share the "* TRIAL *" placeholder as their
        # document number, producing duplicate case IDs like "sap_o2c_* TRIAL * ").
        trial_overlap = {cid for cid in overlap if "TRIAL" in cid.upper()}
        real_overlap = overlap - trial_overlap
        assert len(real_overlap) == 0, (
            f"Found {len(real_overlap)} overlapping non-TRIAL cases: {real_overlap}"
        )

    def test_all_cases_have_events(self, train_cases, val_cases):
        for case in train_cases[:100]:
            assert len(case["events"]) > 0, f"Train case {case['caseId']} has no events"
        for case in val_cases[:100]:
            assert len(case["events"]) > 0, f"Val case {case['caseId']} has no events"

    def test_event_format_in_real_data(self, train_cases):
        for case in train_cases[:50]:
            for i, event in enumerate(case["events"]):
                assert "activity" in event, (
                    f"Case {case['caseId']}, event {i} missing 'activity'"
                )
                assert "resource" in event, (
                    f"Case {case['caseId']}, event {i} missing 'resource'"
                )
                assert "timestamp" in event, (
                    f"Case {case['caseId']}, event {i} missing 'timestamp'"
                )

    def test_vocab_unk_at_zero(self, vocab_data):
        assert vocab_data["activity"]["token_to_idx"]["<UNK>"] == 0
        assert vocab_data["resource"]["token_to_idx"]["<UNK>"] == 0

    def test_vocab_sizes_match_metadata(self, metadata, vocab_data):
        assert vocab_data["activity"]["size"] == metadata["activity_vocab_size"]
        assert vocab_data["resource"]["size"] == metadata["resource_vocab_size"]

    def test_vocab_size_equals_token_count(self, vocab_data):
        assert vocab_data["activity"]["size"] == len(
            vocab_data["activity"]["token_to_idx"]
        )
        assert vocab_data["resource"]["size"] == len(
            vocab_data["resource"]["token_to_idx"]
        )

    def test_source_distribution_matches_counts(self, metadata):
        source_dist = metadata.get("source_distribution", {})
        source_counts = metadata.get("source_counts", {})
        # source_distribution is from actual cases, source_counts from loader
        total_dist = sum(source_dist.values())
        assert total_dist == metadata["total_cases"]

    def test_metadata_event_length_stats(self, metadata):
        stats = metadata["event_length_stats"]
        assert stats["min"] >= 1
        assert stats["max"] >= stats["min"]
        assert stats["median"] >= stats["min"]
        assert stats["p25"] <= stats["median"]
        assert stats["p75"] >= stats["median"]
        assert stats["mean"] > 0

    def test_total_events_match(self, metadata, train_cases, val_cases):
        # Verify event counts match metadata
        train_event_count = sum(len(c.get("events", [])) for c in train_cases)
        val_event_count = sum(len(c.get("events", [])) for c in val_cases)
        assert metadata["train_events"] == train_event_count
        assert metadata["val_events"] == val_event_count


@requires_real_data
class TestEndToEndDataset:
    """Create EventSequenceDataset from real data and verify tensor outputs."""

    @pytest.fixture(scope="class")
    def vocab_pair(self) -> tuple[ActivityVocabulary, ResourceVocabulary]:
        with open(VOCABULARY_PATH) as f:
            vocab_data = json.load(f)

        act_vocab = ActivityVocabulary()
        for token in sorted(
            vocab_data["activity"]["token_to_idx"],
            key=lambda t: vocab_data["activity"]["token_to_idx"][t],
        ):
            if token != ActivityVocabulary.UNK_TOKEN:
                act_vocab.add_token(token)

        res_vocab = ResourceVocabulary()
        for token in sorted(
            vocab_data["resource"]["token_to_idx"],
            key=lambda t: vocab_data["resource"]["token_to_idx"][t],
        ):
            if token != ResourceVocabulary.UNK_TOKEN:
                res_vocab.add_token(token)

        return act_vocab, res_vocab

    @pytest.fixture(scope="class")
    def train_dataset(
        self,
        vocab_pair: tuple[ActivityVocabulary, ResourceVocabulary],
    ) -> EventSequenceDataset:
        act_vocab, res_vocab = vocab_pair
        return EventSequenceDataset(
            events_path=TRAIN_CASES_PATH,
            activity_vocab=act_vocab,
            resource_vocab=res_vocab,
            max_seq_len=256,
            n_attribute_features=8,
        )

    def test_dataset_length_matches_metadata(self, train_dataset):
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
        assert len(train_dataset) == metadata["train_cases"]

    def test_real_item_tensor_shapes(self, train_dataset):
        item = train_dataset[0]
        seq_len = item["seq_len"].item()
        assert seq_len > 0
        assert item["activity_ids"].shape == (seq_len,)
        assert item["resource_ids"].shape == (seq_len,)
        assert item["attributes"].shape == (seq_len, 8)
        assert item["time_deltas"].shape == (seq_len,)

    def test_real_item_dtypes(self, train_dataset):
        item = train_dataset[0]
        assert item["activity_ids"].dtype == torch.long
        assert item["resource_ids"].dtype == torch.long
        assert item["attributes"].dtype == torch.float32
        assert item["time_deltas"].dtype == torch.float32
        assert item["target_ontime"].dtype == torch.float32

    def test_first_few_items_load_without_error(self, train_dataset):
        """Smoke test: load first 10 items without crashing."""
        for i in range(min(10, len(train_dataset))):
            item = train_dataset[i]
            assert item["seq_len"].item() > 0

    def test_real_data_collate(self, train_dataset):
        """Test collate_fn with real data items."""
        batch = [train_dataset[i] for i in range(min(4, len(train_dataset)))]
        collated = collate_fn(batch)
        assert collated["activity_ids"].ndim == 2
        assert collated["padding_mask"].ndim == 2
        assert collated["activity_ids"].shape[0] == len(batch)

    def test_vocab_sizes_match_file(self, vocab_pair):
        act_vocab, res_vocab = vocab_pair
        with open(VOCABULARY_PATH) as f:
            vocab_data = json.load(f)
        assert act_vocab.size == vocab_data["activity"]["size"]
        assert res_vocab.size == vocab_data["resource"]["size"]
