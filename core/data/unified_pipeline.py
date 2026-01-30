"""Unified data pipeline combining all SAP data sources for AETHER training.

Orchestrates loading from multiple heterogeneous sources, normalizes activity
and resource names, builds shared vocabularies, and produces train/validation
splits as EventSequenceDataset instances ready for PyTorch DataLoaders.

Supported data sources:
    1. SAP IDES SQLite (sap.sqlite) - extracted via sap-extractor
    2. BPI Challenge 2019 JSON - real P2P event log (251K cases)
    3. CSV event logs - O2C and P2P from process mining datasets
    4. OCEL 2.0 P2P SQLite - simulated SAP procurement process (Zenodo)
"""

from __future__ import annotations

import json
import logging
import random
import re
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ..encoder.vocabulary import ActivityVocabulary, ResourceVocabulary
    from ..training.data_loader import EventSequenceDataset

logger = logging.getLogger(__name__)

# Project root: two levels up from this file (core/data/ -> core/ -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default paths for all data sources (relative to project root)
DEFAULT_PATHS: Dict[str, Path] = {
    "sap_sqlite": _PROJECT_ROOT / "data" / "external" / "sap.sqlite",
    "bpi2019_json": _PROJECT_ROOT / "data" / "external" / "bpi_2019.json",
    "o2c_csv": _PROJECT_ROOT / "data" / "external" / "o2c_eventlog.csv",
    "p2p_csv": _PROJECT_ROOT / "data" / "external" / "p2p_eventlog.csv",
    "ocel_p2p": _PROJECT_ROOT / "data" / "external" / "ocel2-p2p.sqlite",
    "output_dir": _PROJECT_ROOT / "data" / "events",
}

# Canonical activity name mapping for cross-source normalization.
# Keys are lowercased, underscored forms; values are the canonical token.
_ACTIVITY_ALIASES: Dict[str, str] = {
    "create sales order": "create_sales_order",
    "create_sales_order": "create_sales_order",
    "create order": "create_sales_order",
    "create delivery": "create_delivery",
    "create_delivery": "create_delivery",
    "create invoice": "create_invoice",
    "create_invoice": "create_invoice",
    "create billing document": "create_invoice",
    "create purchase order": "create_purchase_order",
    "create_purchase_order": "create_purchase_order",
    "create purchase requisition": "create_purchase_requisition",
    "create_purchase_requisition": "create_purchase_requisition",
    "approve purchase order": "approve_purchase_order",
    "approve_purchase_order": "approve_purchase_order",
    "approve purchase requisition": "approve_purchase_requisition",
    "approve_purchase_requisition": "approve_purchase_requisition",
    "create goods receipt": "create_goods_receipt",
    "create_goods_receipt": "create_goods_receipt",
    "goods receipt": "create_goods_receipt",
    "create invoice receipt": "create_invoice_receipt",
    "create_invoice_receipt": "create_invoice_receipt",
    "execute payment": "execute_payment",
    "execute_payment": "execute_payment",
    "payment": "execute_payment",
    "perform two way match": "two_way_match",
    "perform_two_way_match": "two_way_match",
    "two-way match": "two_way_match",
    "delegate purchase requisition approval": "delegate_pr_approval",
    "delegate_purchase_requisition_approval": "delegate_pr_approval",
    "create request for quotation": "create_rfq",
    "create_request_for_quotation": "create_rfq",
    "post goods issue": "post_goods_issue",
    "post_goods_issue": "post_goods_issue",
    "confirm delivery": "confirm_delivery",
    "confirm_delivery": "confirm_delivery",
    "release order": "release_order",
    "release_order": "release_order",
    "credit check": "credit_check",
    "credit_check": "credit_check",
}


def normalize_activity(raw: str) -> str:
    """Normalize an activity name to a canonical underscore-delimited token.

    Applies alias lookup first, then falls back to lowercasing and
    replacing whitespace/hyphens with underscores.

    Args:
        raw: Raw activity string from any data source.

    Returns:
        Canonical activity token (lowercase, underscored).
    """
    if not raw:
        return "unknown_activity"

    lookup_key = raw.strip().lower()
    if lookup_key in _ACTIVITY_ALIASES:
        return _ACTIVITY_ALIASES[lookup_key]

    # Fallback: lowercase, replace spaces/hyphens/colons with underscores,
    # collapse multiple underscores, strip leading/trailing underscores
    normalized = re.sub(r"[\s\-:/]+", "_", raw.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unknown_activity"


def normalize_resource(raw: str) -> str:
    """Normalize a resource/user identifier.

    Args:
        raw: Raw resource string from any data source.

    Returns:
        Normalized resource token (lowercase, underscored).
    """
    if not raw or raw.strip() == "* TRIAL *":
        return "system"

    normalized = re.sub(r"[\s\-:/]+", "_", raw.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "system"


# Activities that signal a process reached a successful completion point.
# Used to compute the heuristic "onTime" outcome flag.
_COMPLETION_ACTIVITIES: set[str] = {
    "create_invoice",
    "execute_payment",
    "customer_payment",
    "vendor_payment",
    "clear_invoice",
    "create_goods_receipt",
    "record_goods_receipt",
    "create_delivery",
    "goods_issue",
    "accounting_document",
    "g_l_account_document",
}


def compute_outcome_heuristic(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute outcome labels from a case's event sequence.

    Heuristics (consistent across all source parsers):
        - onTime: True if the case contains at least one completion activity
          (invoice, payment, goods receipt, etc.).
        - rework: True if any activity appears more than once in the case.
        - durationHours: Wall-clock hours between first and last event.

    Args:
        events: List of event dicts with 'activity' and 'timestamp' keys.

    Returns:
        Dict with onTime (bool), rework (bool), durationHours (float).
    """
    if not events:
        return {"onTime": False, "rework": False, "durationHours": 0.0}

    activities = [e.get("activity", "") for e in events]

    # onTime: contains a completion activity
    on_time = any(a in _COMPLETION_ACTIVITIES for a in activities)

    # rework: any activity repeated
    activity_counts: Counter = Counter(activities)
    rework = any(count > 1 for count in activity_counts.values())

    # durationHours: time between first and last event
    duration_hours = 0.0
    timestamps: list[datetime] = []
    for e in events:
        ts_str = e.get("timestamp", "")
        if ts_str:
            try:
                ts = datetime.fromisoformat(
                    ts_str.replace("Z", "+00:00")
                    if isinstance(ts_str, str)
                    else str(ts_str)
                )
                timestamps.append(ts)
            except (ValueError, TypeError):
                pass

    if len(timestamps) >= 2:
        timestamps.sort()
        delta = timestamps[-1] - timestamps[0]
        duration_hours = round(delta.total_seconds() / 3600.0, 2)

    return {
        "onTime": on_time,
        "rework": rework,
        "durationHours": duration_hours,
    }


class AetherDataPipeline:
    """Unified data pipeline combining all SAP data sources.

    Loads, normalizes, and combines data from:
    - SAP IDES SQLite extraction (sap.sqlite)
    - BPI Challenge 2019 (real P2P event log)
    - CSV event logs (O2C + P2P from process mining)
    - OCEL 2.0 P2P (Zenodo simulated SAP process)

    Produces train/val splits with shared vocabularies.
    """

    def __init__(
        self,
        paths: Optional[Dict[str, Path]] = None,
        train_ratio: float = 0.8,
        max_bpi_cases: Optional[int] = 50000,
        max_seq_len: int = 256,
        n_attribute_features: int = 8,
        seed: int = 42,
    ) -> None:
        """Initialize pipeline.

        Args:
            paths: Override default paths. Keys match DEFAULT_PATHS.
            train_ratio: Fraction of data for training (0.0-1.0).
            max_bpi_cases: Limit BPI 2019 cases (None = all 251K).
            max_seq_len: Maximum events per case sequence.
            n_attribute_features: Number of numerical attribute features.
            seed: Random seed for reproducible splits.
        """
        resolved = dict(DEFAULT_PATHS)
        if paths:
            resolved.update(paths)
        self.paths: Dict[str, Path] = resolved

        self.train_ratio = train_ratio
        self.max_bpi_cases = max_bpi_cases
        self.max_seq_len = max_seq_len
        self.n_attribute_features = n_attribute_features
        self.seed = seed

        # Filled during load_all_sources
        self._source_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all_sources(self) -> List[Dict[str, Any]]:
        """Load data from all available sources.

        Tries each source independently, logs success/failure, and
        continues on errors so partial data is still usable.

        Returns:
            Combined list of cases in AETHER format. Each case is a dict
            with keys: caseId, events (list of event dicts), source,
            and optionally outcome.
        """
        all_cases: List[Dict[str, Any]] = []
        loaders: List[Tuple[str, Any]] = [
            ("sap_sqlite", self._load_sap_sqlite),
            ("bpi2019", self._load_bpi2019),
            ("o2c_csv", self._load_csv_o2c),
            ("p2p_csv", self._load_csv_p2p),
            ("ocel_p2p", self._load_ocel_p2p),
        ]

        for source_name, loader_fn in loaders:
            try:
                cases = loader_fn()
                count = len(cases)
                self._source_counts[source_name] = count
                all_cases.extend(cases)
                logger.info(
                    "Loaded %d cases from %s (%d total events)",
                    count,
                    source_name,
                    sum(len(c.get("events", [])) for c in cases),
                )
            except FileNotFoundError:
                self._source_counts[source_name] = 0
                logger.warning(
                    "Source %s not found, skipping.", source_name
                )
            except Exception:
                self._source_counts[source_name] = 0
                logger.warning(
                    "Failed to load %s, skipping.", source_name, exc_info=True
                )

        # Compute outcome heuristics for cases that don't already have them
        n_added = 0
        for case in all_cases:
            outcome = case.get("outcome", {})
            if "onTime" not in outcome:
                computed = compute_outcome_heuristic(
                    case.get("events", [])
                )
                case["outcome"] = {**outcome, **computed}
                n_added += 1

        logger.info(
            "Loaded %d total cases from %d sources "
            "(computed outcomes for %d cases).",
            len(all_cases),
            sum(1 for v in self._source_counts.values() if v > 0),
            n_added,
        )
        return all_cases

    def build_vocabularies(
        self, cases: List[Dict[str, Any]]
    ) -> Tuple["ActivityVocabulary", "ResourceVocabulary"]:
        """Build activity and resource vocabularies from all cases.

        Collects every unique normalized activity and resource across
        all cases, then populates vocabulary objects.

        Args:
            cases: List of case dicts with 'events' containing
                   'activity' and 'resource' keys.

        Returns:
            Tuple of (ActivityVocabulary, ResourceVocabulary).
        """
        from ..encoder.vocabulary import ActivityVocabulary, ResourceVocabulary

        all_events: List[Dict[str, str]] = []
        for case in cases:
            all_events.extend(case.get("events", []))

        activity_vocab = ActivityVocabulary()
        activity_vocab.build_from_events(all_events)

        resource_vocab = ResourceVocabulary()
        resource_vocab.build_from_events(all_events)

        logger.info(
            "Built vocabularies: %d activities, %d resources.",
            activity_vocab.size,
            resource_vocab.size,
        )
        return activity_vocab, resource_vocab

    def split_train_val(
        self, cases: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split cases into train and validation sets.

        Uses seeded random shuffle for reproducibility. The split is
        performed at the case level (not event level) to prevent
        data leakage within process instances.

        Args:
            cases: Full list of case dicts.

        Returns:
            Tuple of (train_cases, val_cases).
        """
        rng = random.Random(self.seed)
        indices = list(range(len(cases)))
        rng.shuffle(indices)

        split_idx = int(len(cases) * self.train_ratio)
        train_indices = sorted(indices[:split_idx])
        val_indices = sorted(indices[split_idx:])

        train_cases = [cases[i] for i in train_indices]
        val_cases = [cases[i] for i in val_indices]

        logger.info(
            "Split: %d train cases, %d val cases (ratio=%.2f).",
            len(train_cases),
            len(val_cases),
            self.train_ratio,
        )
        return train_cases, val_cases

    def save_processed_data(
        self, cases: List[Dict[str, Any]], output_dir: Path
    ) -> Dict[str, Any]:
        """Save processed data to disk.

        Writes:
            - train_cases.json: Training case data.
            - val_cases.json: Validation case data.
            - vocabulary.json: Activity and resource mappings.
            - metadata.json: Source counts, total events, statistics.

        Args:
            cases: Full list of case dicts (will be split internally).
            output_dir: Directory to write output files.

        Returns:
            Metadata dict with statistics.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build vocabularies
        activity_vocab, resource_vocab = self.build_vocabularies(cases)

        # Split
        train_cases, val_cases = self.split_train_val(cases)

        # Save cases
        train_path = output_dir / "train_cases.json"
        val_path = output_dir / "val_cases.json"
        with open(train_path, "w") as f:
            json.dump(train_cases, f, indent=None, default=str)
        with open(val_path, "w") as f:
            json.dump(val_cases, f, indent=None, default=str)
        logger.info("Saved %s and %s.", train_path, val_path)

        # Save vocabulary
        vocab_data = {
            "activity": {
                "token_to_idx": activity_vocab._token_to_idx,
                "size": activity_vocab.size,
            },
            "resource": {
                "token_to_idx": resource_vocab._token_to_idx,
                "size": resource_vocab.size,
            },
        }
        vocab_path = output_dir / "vocabulary.json"
        with open(vocab_path, "w") as f:
            json.dump(vocab_data, f, indent=2)
        logger.info("Saved %s.", vocab_path)

        # Compute metadata
        total_events = sum(len(c.get("events", [])) for c in cases)
        train_events = sum(len(c.get("events", [])) for c in train_cases)
        val_events = sum(len(c.get("events", [])) for c in val_cases)

        # Event length distribution
        event_lengths = [len(c.get("events", [])) for c in cases]
        event_lengths.sort()
        n = len(event_lengths)

        # Source distribution
        source_dist: Dict[str, int] = Counter()
        for c in cases:
            source_dist[c.get("source", "unknown")] += 1

        # Activity frequency
        activity_counts: Counter = Counter()
        for c in cases:
            for e in c.get("events", []):
                activity_counts[e.get("activity", "unknown")] += 1
        top_activities = activity_counts.most_common(20)

        # Outcome statistics
        n_with_outcome = sum(
            1 for c in cases if "onTime" in c.get("outcome", {})
        )
        n_ontime = sum(
            1 for c in cases if c.get("outcome", {}).get("onTime", False)
        )
        n_rework = sum(
            1 for c in cases if c.get("outcome", {}).get("rework", False)
        )

        metadata: Dict[str, Any] = {
            "total_cases": len(cases),
            "train_cases": len(train_cases),
            "val_cases": len(val_cases),
            "total_events": total_events,
            "train_events": train_events,
            "val_events": val_events,
            "activity_vocab_size": activity_vocab.size,
            "resource_vocab_size": resource_vocab.size,
            "outcome_stats": {
                "cases_with_outcome": n_with_outcome,
                "ontime_rate": round(n_ontime / max(n_with_outcome, 1), 4),
                "rework_rate": round(n_rework / max(n_with_outcome, 1), 4),
            },
            "source_counts": dict(self._source_counts),
            "source_distribution": dict(source_dist),
            "event_length_stats": {
                "min": event_lengths[0] if event_lengths else 0,
                "max": event_lengths[-1] if event_lengths else 0,
                "median": event_lengths[n // 2] if event_lengths else 0,
                "p25": event_lengths[n // 4] if event_lengths else 0,
                "p75": event_lengths[3 * n // 4] if event_lengths else 0,
                "mean": round(total_events / max(len(cases), 1), 2),
            },
            "top_activities": top_activities,
            "seed": self.seed,
            "train_ratio": self.train_ratio,
            "max_seq_len": self.max_seq_len,
            "max_bpi_cases": self.max_bpi_cases,
            "generated_at": datetime.now().isoformat(),
        }

        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info("Saved %s.", meta_path)

        return metadata

    def create_datasets(
        self,
    ) -> Tuple[
        "EventSequenceDataset",
        "EventSequenceDataset",
        "ActivityVocabulary",
        "ResourceVocabulary",
        Dict[str, Any],
    ]:
        """Full pipeline: load -> build vocab -> split -> create datasets.

        Runs the complete data preparation pipeline end-to-end. Saves
        intermediate JSON files, then wraps them in EventSequenceDataset
        instances ready for PyTorch DataLoaders.

        Returns:
            Tuple of:
                - train_dataset (EventSequenceDataset)
                - val_dataset (EventSequenceDataset)
                - activity_vocab (ActivityVocabulary)
                - resource_vocab (ResourceVocabulary)
                - metadata (dict with statistics)
        """
        from ..encoder.vocabulary import ActivityVocabulary, ResourceVocabulary
        from ..training.data_loader import EventSequenceDataset

        # Load all sources
        cases = self.load_all_sources()
        if not cases:
            raise RuntimeError(
                "No data loaded from any source. Check paths and data files."
            )

        # Save processed data (builds vocab, splits, writes files)
        output_dir = self.paths["output_dir"]
        metadata = self.save_processed_data(cases, output_dir)

        # Rebuild vocabularies from the saved vocab file for consistency
        vocab_path = output_dir / "vocabulary.json"
        with open(vocab_path) as f:
            vocab_data = json.load(f)

        activity_vocab = ActivityVocabulary()
        for token in sorted(
            vocab_data["activity"]["token_to_idx"],
            key=lambda t: vocab_data["activity"]["token_to_idx"][t],
        ):
            if token != ActivityVocabulary.UNK_TOKEN:
                activity_vocab.add_token(token)

        resource_vocab = ResourceVocabulary()
        for token in sorted(
            vocab_data["resource"]["token_to_idx"],
            key=lambda t: vocab_data["resource"]["token_to_idx"][t],
        ):
            if token != ResourceVocabulary.UNK_TOKEN:
                resource_vocab.add_token(token)

        # Create datasets from saved JSON files
        train_path = output_dir / "train_cases.json"
        val_path = output_dir / "val_cases.json"

        train_dataset = EventSequenceDataset(
            events_path=train_path,
            activity_vocab=activity_vocab,
            resource_vocab=resource_vocab,
            max_seq_len=self.max_seq_len,
            n_attribute_features=self.n_attribute_features,
        )

        val_dataset = EventSequenceDataset(
            events_path=val_path,
            activity_vocab=activity_vocab,
            resource_vocab=resource_vocab,
            max_seq_len=self.max_seq_len,
            n_attribute_features=self.n_attribute_features,
        )

        logger.info(
            "Created datasets: train=%d cases, val=%d cases.",
            len(train_dataset),
            len(val_dataset),
        )

        return (
            train_dataset,
            val_dataset,
            activity_vocab,
            resource_vocab,
            metadata,
        )

    # ------------------------------------------------------------------
    # Source-specific loaders (private)
    # ------------------------------------------------------------------

    def _load_sap_sqlite(self) -> List[Dict[str, Any]]:
        """Load cases from SAP IDES SQLite extraction.

        Reconstructs process cases from SAP document flow (VBFA) and
        change documents (CDHDR). Produces O2C-style cases from sales
        order -> delivery -> billing -> payment chains.

        Returns:
            List of case dicts in AETHER format.
        """
        db_path = self.paths["sap_sqlite"]
        if not db_path.exists():
            raise FileNotFoundError(f"SAP SQLite not found: {db_path}")

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cases: List[Dict[str, Any]] = []

        try:
            # Build cases from document flow (VBFA) - O2C process chain
            # Each sales order (vbeln from vbak) becomes a case.
            # Follow the document flow for subsequent events.
            cursor.execute(
                "SELECT vbeln, erdat, erzet, ernam, vbtyp, netwr, kunnr "
                "FROM vbak ORDER BY erdat, erzet"
            )
            sales_orders = cursor.fetchall()

            # Document type descriptions (vbtyp -> activity name)
            doc_type_map = {
                "A": "create_inquiry",
                "B": "create_quotation",
                "C": "create_sales_order",
                "H": "create_return",
                "J": "create_delivery",
                "K": "create_credit_memo",
                "L": "create_debit_memo",
                "M": "create_invoice",
                "N": "create_invoice_cancellation",
                "O": "create_goods_movement",
                "P": "create_wbs_element",
                "R": "create_goods_receipt",
                "T": "create_return_delivery",
                "U": "create_pro_forma_invoice",
                "0": "create_master_contract",
            }

            for so in sales_orders:
                case_id = f"sap_o2c_{so['vbeln']}"
                events: List[Dict[str, Any]] = []

                # First event: sales order creation
                ts = self._sap_datetime(so["erdat"], so["erzet"])
                events.append(
                    {
                        "activity": "create_sales_order",
                        "resource": normalize_resource(so["ernam"] or ""),
                        "timestamp": ts,
                        "attributes": {
                            "net_value": float(so["netwr"] or 0),
                            "customer": so["kunnr"] or "",
                        },
                    }
                )

                # Follow document flow from this sales order
                cursor.execute(
                    "SELECT vbeln, vbelv, vbtyp_n, vbtyp_v, erdat, erzet "
                    "FROM vbfa WHERE vbelv = ? ORDER BY erdat, erzet",
                    (so["vbeln"],),
                )
                flow_rows = cursor.fetchall()

                for flow in flow_rows:
                    vbtyp = flow["vbtyp_n"] or ""
                    activity = doc_type_map.get(vbtyp, f"doc_type_{vbtyp}")
                    flow_ts = self._sap_datetime(
                        flow["erdat"], flow["erzet"]
                    )
                    events.append(
                        {
                            "activity": activity,
                            "resource": "system",
                            "timestamp": flow_ts,
                            "attributes": {
                                "doc_number": flow["vbeln"] or "",
                            },
                        }
                    )

                # Also pull change document events for this sales order
                cursor.execute(
                    "SELECT objectclas, objectid, username, udate, utime, "
                    "tcode, change_ind "
                    "FROM cdhdr WHERE objectid LIKE ? "
                    "ORDER BY udate, utime",
                    (f"%{so['vbeln']}%",),
                )
                changes = cursor.fetchall()

                for chg in changes:
                    tcode = chg["tcode"] or "unknown"
                    change_ind = chg["change_ind"] or "U"
                    action_map = {"I": "insert", "U": "update", "D": "delete"}
                    action = action_map.get(change_ind, "change")
                    activity = normalize_activity(
                        f"{action}_{chg['objectclas'] or 'document'}"
                    )
                    chg_ts = self._sap_datetime(chg["udate"], chg["utime"])

                    events.append(
                        {
                            "activity": activity,
                            "resource": normalize_resource(
                                chg["username"] or ""
                            ),
                            "timestamp": chg_ts,
                            "attributes": {
                                "tcode": tcode,
                                "object_class": chg["objectclas"] or "",
                            },
                        }
                    )

                # Sort events by timestamp
                events.sort(key=lambda e: e.get("timestamp", ""))

                if events:
                    cases.append(
                        {
                            "caseId": case_id,
                            "events": events,
                            "source": "sap_sqlite",
                        }
                    )

        finally:
            conn.close()

        return cases

    def _load_bpi2019(self) -> List[Dict[str, Any]]:
        """Load BPI Challenge 2019 data from JSON.

        The BPI 2019 dataset contains 251K procurement cases from a
        real multinational coatings company. We limit loading to
        max_bpi_cases for memory management.

        Returns:
            List of case dicts in AETHER format.
        """
        json_path = self.paths["bpi2019_json"]
        if not json_path.exists():
            raise FileNotFoundError(f"BPI 2019 JSON not found: {json_path}")

        with open(json_path) as f:
            data = json.load(f)

        if isinstance(data, dict) and "traces" in data:
            traces = data["traces"]
        elif isinstance(data, list):
            traces = data
        else:
            raise ValueError(
                f"Unexpected BPI 2019 format: {type(data)}, "
                f"keys={list(data.keys()) if isinstance(data, dict) else 'N/A'}"
            )

        # Limit cases if configured
        if self.max_bpi_cases is not None and len(traces) > self.max_bpi_cases:
            rng = random.Random(self.seed)
            traces = rng.sample(traces, self.max_bpi_cases)

        cases: List[Dict[str, Any]] = []
        for trace in traces:
            case_id = trace.get("case_id", trace.get("concept:name", ""))
            raw_events = trace.get("events", [])

            events: List[Dict[str, Any]] = []
            for raw_event in raw_events:
                # BPI 2019 uses "concept:name" for activity, "org:resource"
                # for resource, "time:timestamp" for timestamp
                raw_activity = raw_event.get(
                    "concept:name", raw_event.get("activity", "")
                )
                raw_resource = raw_event.get(
                    "org:resource",
                    raw_event.get("User", raw_event.get("resource", "")),
                )
                timestamp = raw_event.get(
                    "time:timestamp", raw_event.get("timestamp", "")
                )

                # Extract numerical attributes
                attributes: Dict[str, Any] = {}
                for key, val in raw_event.items():
                    if key in (
                        "concept:name",
                        "org:resource",
                        "time:timestamp",
                        "User",
                    ):
                        continue
                    # Try to parse numeric values
                    if isinstance(val, (int, float)):
                        attributes[key] = val
                    elif isinstance(val, str):
                        try:
                            attributes[key] = float(val)
                        except ValueError:
                            # Keep string attributes for metadata
                            attributes[key] = val

                events.append(
                    {
                        "activity": normalize_activity(raw_activity),
                        "resource": normalize_resource(raw_resource),
                        "timestamp": timestamp,
                        "attributes": attributes,
                    }
                )

            if events:
                # Extract case-level attributes for outcome
                case_attrs = trace.get("attributes", {})
                outcome: Dict[str, Any] = {}
                if case_attrs:
                    # BPI 2019 has procurement-specific attributes
                    doc_type = case_attrs.get("Document Type", "")
                    outcome["document_type"] = doc_type
                    goods_receipt = case_attrs.get("Goods Receipt", "")
                    outcome["goods_receipt"] = goods_receipt == "True"

                case_dict: Dict[str, Any] = {
                    "caseId": f"bpi2019_{case_id}",
                    "events": events,
                    "source": "bpi2019",
                }
                if outcome:
                    case_dict["outcome"] = outcome
                cases.append(case_dict)

        return cases

    def _load_csv_o2c(self) -> List[Dict[str, Any]]:
        """Load Order-to-Cash CSV event log.

        CSV columns: case:concept:name, concept:name, time:timestamp,
                     VBELN, VBELV, VBTYP_N_DESC, VBTYP_V_DESC

        Returns:
            List of case dicts in AETHER format.
        """
        return self._load_csv_generic(
            csv_path=self.paths["o2c_csv"],
            source_name="o2c_csv",
            case_id_col="case:concept:name",
            activity_col="concept:name",
            timestamp_col="time:timestamp",
            resource_col=None,  # O2C CSV has no resource column
            attribute_cols=["VBELN", "VBELV", "VBTYP_N_DESC", "VBTYP_V_DESC"],
        )

    def _load_csv_p2p(self) -> List[Dict[str, Any]]:
        """Load Purchase-to-Pay CSV event log.

        CSV columns: case:concept:name, concept:name, time:timestamp,
                     org:resource, document_id, document_type

        Returns:
            List of case dicts in AETHER format.
        """
        return self._load_csv_generic(
            csv_path=self.paths["p2p_csv"],
            source_name="p2p_csv",
            case_id_col="case:concept:name",
            activity_col="concept:name",
            timestamp_col="time:timestamp",
            resource_col="org:resource",
            attribute_cols=["document_id", "document_type"],
        )

    def _load_csv_generic(
        self,
        csv_path: Path,
        source_name: str,
        case_id_col: str,
        activity_col: str,
        timestamp_col: str,
        resource_col: Optional[str],
        attribute_cols: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Generic CSV event log loader.

        Reads a CSV file with XES-style column names, groups events
        by case ID, and normalizes to AETHER format.

        Args:
            csv_path: Path to the CSV file.
            source_name: Source tag for metadata.
            case_id_col: Column name for case ID.
            activity_col: Column name for activity.
            timestamp_col: Column name for timestamp.
            resource_col: Column name for resource (None if absent).
            attribute_cols: Additional columns to include as attributes.

        Returns:
            List of case dicts in AETHER format.
        """
        import csv

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        cases_dict: Dict[str, List[Dict[str, Any]]] = {}

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                case_id = row.get(case_id_col, "").strip()
                if not case_id:
                    continue

                raw_activity = row.get(activity_col, "")
                raw_resource = row.get(resource_col, "") if resource_col else ""
                timestamp = row.get(timestamp_col, "")

                # Collect attribute columns
                attributes: Dict[str, Any] = {}
                for col in attribute_cols or []:
                    val = row.get(col, "")
                    if val and val != "* TRIAL *":
                        try:
                            attributes[col] = float(val)
                        except ValueError:
                            attributes[col] = val

                event = {
                    "activity": normalize_activity(raw_activity),
                    "resource": normalize_resource(raw_resource),
                    "timestamp": timestamp,
                    "attributes": attributes,
                }

                if case_id not in cases_dict:
                    cases_dict[case_id] = []
                cases_dict[case_id].append(event)

        # Convert to case list
        cases: List[Dict[str, Any]] = []
        for case_id, events in cases_dict.items():
            # Sort events by timestamp
            events.sort(key=lambda e: e.get("timestamp", ""))
            cases.append(
                {
                    "caseId": f"{source_name}_{case_id}",
                    "events": events,
                    "source": source_name,
                }
            )

        return cases

    def _load_ocel_p2p(self) -> List[Dict[str, Any]]:
        """Load OCEL 2.0 P2P data from SQLite.

        OCEL 2.0 stores events in a normalized schema with separate
        tables per event type. Events are linked to objects via
        event_object. We reconstruct cases by grouping events that
        share a purchase_requisition or purchase_order object.

        Returns:
            List of case dicts in AETHER format.
        """
        db_path = self.paths["ocel_p2p"]
        if not db_path.exists():
            raise FileNotFoundError(f"OCEL P2P SQLite not found: {db_path}")

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cases: List[Dict[str, Any]] = []

        try:
            # Get event type mapping
            cursor.execute("SELECT * FROM event_map_type")
            type_map: Dict[str, str] = {}
            for row in cursor.fetchall():
                # row[0] = display name, row[1] = table suffix
                type_map[row[0]] = row[1]

            # Load all events with their types and timestamps
            # Each event type has its own table with ocel_id, ocel_time,
            # lifecycle, resource
            all_events: Dict[str, Dict[str, Any]] = {}

            cursor.execute(
                "SELECT ocel_id, ocel_type FROM event"
            )
            _event_types = {  # noqa: F841 - kept for future expansion
                row["ocel_id"]: row["ocel_type"] for row in cursor.fetchall()
            }

            for display_name, table_suffix in type_map.items():
                table_name = f"event_{table_suffix}"
                try:
                    cursor.execute(
                        f'SELECT ocel_id, ocel_time, resource FROM "{table_name}"'
                    )
                    for row in cursor.fetchall():
                        event_id = row["ocel_id"]
                        all_events[event_id] = {
                            "event_id": event_id,
                            "activity": normalize_activity(display_name),
                            "resource": normalize_resource(
                                row["resource"] or ""
                            ),
                            "timestamp": row["ocel_time"] or "",
                        }
                except sqlite3.OperationalError:
                    logger.debug(
                        "OCEL table %s not found, skipping.", table_name
                    )

            # Link events to objects to form cases.
            # Use purchase_requisition as primary case object; fall back to
            # purchase_order if no requisition link exists.
            cursor.execute(
                "SELECT ocel_event_id, ocel_object_id, ocel_qualifier "
                "FROM event_object"
            )
            event_to_objects: Dict[str, List[str]] = {}
            for row in cursor.fetchall():
                eid = row["ocel_event_id"]
                oid = row["ocel_object_id"]
                if eid not in event_to_objects:
                    event_to_objects[eid] = []
                event_to_objects[eid].append(oid)

            # Get object types
            cursor.execute("SELECT ocel_id, ocel_type FROM object")
            object_types: Dict[str, str] = {
                row["ocel_id"]: row["ocel_type"]
                for row in cursor.fetchall()
            }

            # Group events by their primary case object
            # Priority: purchase_requisition > purchase_order
            case_events: Dict[str, List[Dict[str, Any]]] = {}

            for event_id, event_data in all_events.items():
                objects = event_to_objects.get(event_id, [])
                case_object = None

                # Find primary case object
                for oid in objects:
                    otype = object_types.get(oid, "")
                    if otype == "purchase_requisition":
                        case_object = oid
                        break
                if case_object is None:
                    for oid in objects:
                        otype = object_types.get(oid, "")
                        if otype == "purchase_order":
                            case_object = oid
                            break
                if case_object is None and objects:
                    case_object = objects[0]

                if case_object is not None:
                    if case_object not in case_events:
                        case_events[case_object] = []
                    # Add linked objects as attributes
                    event_with_attrs = dict(event_data)
                    event_with_attrs["attributes"] = {
                        "linked_objects": len(objects),
                    }
                    case_events[case_object].append(event_with_attrs)

            # Build cases
            for obj_id, events in case_events.items():
                events.sort(key=lambda e: e.get("timestamp", ""))
                # Remove internal event_id from output
                clean_events = []
                for ev in events:
                    clean_events.append(
                        {
                            "activity": ev["activity"],
                            "resource": ev["resource"],
                            "timestamp": ev["timestamp"],
                            "attributes": ev.get("attributes", {}),
                        }
                    )

                cases.append(
                    {
                        "caseId": f"ocel_p2p_{obj_id}",
                        "events": clean_events,
                        "source": "ocel_p2p",
                    }
                )

        finally:
            conn.close()

        return cases

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sap_datetime(date_str: Optional[str], time_str: Optional[str]) -> str:
        """Convert SAP date (YYYY-MM-DD or YYYYMMDD) and time to ISO format.

        SAP stores dates as 'YYYY-MM-DDT00:00:00' or 'YYYYMMDD' and
        times as 'HH:MM:SS' or 'HHMMSS'. Handles various edge cases.

        Args:
            date_str: Date string from SAP.
            time_str: Time string from SAP.

        Returns:
            ISO 8601 datetime string, or empty string on failure.
        """
        if not date_str:
            return ""

        try:
            # Handle ISO-style dates with T separator
            if "T" in date_str:
                # Already ISO-ish, extract date part
                date_part = date_str.split("T")[0]
            else:
                date_part = date_str.strip()

            # Handle YYYYMMDD vs YYYY-MM-DD
            if len(date_part) == 8 and "-" not in date_part:
                date_part = (
                    f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                )

            # Handle time
            if time_str:
                time_clean = time_str.strip()
                if "T" in time_clean:
                    time_clean = time_clean.split("T")[-1]
                # Handle HHMMSS vs HH:MM:SS
                if len(time_clean) == 6 and ":" not in time_clean:
                    time_clean = (
                        f"{time_clean[:2]}:{time_clean[2:4]}:{time_clean[4:6]}"
                    )
            else:
                time_clean = "00:00:00"

            return f"{date_part}T{time_clean}"
        except (ValueError, IndexError, AttributeError):
            return ""

    @property
    def source_counts(self) -> Dict[str, int]:
        """Number of cases loaded from each source (after load_all_sources)."""
        return dict(self._source_counts)
