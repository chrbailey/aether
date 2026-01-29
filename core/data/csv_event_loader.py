"""CSV event log parsers for AETHER training pipeline.

Parses Order-to-Cash (O2C) and Purchase-to-Pay (P2P) event log CSVs
from SAP systems into AETHER's standard event sequence format.

Data sources:
- O2C: /Volumes/OWC drive/Datasets/process-mining-logs/o2c_eventlog.csv
       646 cases, 5,708 events, 8 activities (sales document flow)
- P2P: /Volumes/OWC drive/Datasets/process-mining-logs/p2p_eventlog.csv
       2,486 cases, 7,420 events, 20 activities (procurement flow)
"""

from __future__ import annotations

import csv
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default paths on the OWC drive
DEFAULT_O2C_PATH = Path(
    "/Volumes/OWC drive/Datasets/process-mining-logs/o2c_eventlog.csv"
)
DEFAULT_P2P_PATH = Path(
    "/Volumes/OWC drive/Datasets/process-mining-logs/p2p_eventlog.csv"
)

# O2C completion activities (invoice created = order fulfilled)
O2C_COMPLETION_ACTIVITIES = frozenset({
    "Create Invoice",
    "Create Invoice cancellation",  # Still completed, just reversed
})

# P2P completion activities (payment or final accounting step)
P2P_COMPLETION_ACTIVITIES = frozenset({
    "Customer Payment",
    "Vendor Payment",
    "G/L Account Document",
    "Accounting Document",
})

# SAP document type descriptions
SAP_DOC_TYPES = {
    "EKKO": "Purchase Order",
    "EBAN": "Purchase Requisition",
}


def _parse_timestamp(ts_str: str) -> datetime | None:
    """Parse a timestamp string from CSV, returning None on failure."""
    if not ts_str or ts_str.strip() == "":
        return None

    ts_str = ts_str.strip()

    # Try common formats from the CSV files
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S%z",
        "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(ts_str, fmt)
        except (ValueError, TypeError):
            continue

    # Fallback: try fromisoformat
    try:
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        pass

    logger.debug(f"Could not parse timestamp: {ts_str!r}")
    return None


def _format_timestamp(dt: datetime | None) -> str:
    """Format a datetime to ISO 8601 without timezone, or return empty string."""
    if dt is None:
        return ""
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _compute_duration_hours(first_ts: datetime | None, last_ts: datetime | None) -> float:
    """Compute duration in hours between two timestamps."""
    if first_ts is None or last_ts is None:
        return 0.0
    delta = last_ts - first_ts
    return max(delta.total_seconds() / 3600.0, 0.0)


def _has_rework(activities: list[str]) -> bool:
    """Check if any activity appears more than once (rework)."""
    counts = Counter(activities)
    return any(c > 1 for c in counts.values())


def _is_trial_placeholder(value: str) -> bool:
    """Check if a CSV value is the TRIAL placeholder (anonymized data)."""
    return "TRIAL" in value.upper() if value else False


def _clean_value(value: str, strip_trial: bool = True) -> str:
    """Clean a CSV field value.

    Args:
        value: Raw CSV field value.
        strip_trial: If True, replace TRIAL placeholders with empty string.
            Set to False for case IDs where TRIAL is a valid anonymized ID.
    """
    if not value:
        return ""
    value = value.strip()
    if strip_trial and _is_trial_placeholder(value):
        return ""
    return value


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read all rows from a CSV file, handling encoding issues.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of row dicts from csv.DictReader.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Try UTF-8 first, fall back to latin-1
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            with open(csv_path, newline="", encoding=encoding) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            logger.debug(f"Read {len(rows)} rows from {csv_path} (encoding={encoding})")
            return rows
        except (UnicodeDecodeError, UnicodeError):
            continue

    raise ValueError(f"Could not decode CSV file with any supported encoding: {csv_path}")


def _group_by_case(
    rows: list[dict[str, str]],
    case_col: str = "case:concept:name",
) -> dict[str, list[dict[str, str]]]:
    """Group CSV rows by case ID column.

    Args:
        rows: List of row dicts from csv.DictReader.
        case_col: Column name containing the case identifier.

    Returns:
        Dict mapping case ID to list of row dicts for that case.
    """
    cases: dict[str, list[dict[str, str]]] = defaultdict(list)
    unknown_counter = 0
    for row in rows:
        # Keep TRIAL placeholders as valid case IDs (anonymized data)
        case_id = _clean_value(row.get(case_col, ""), strip_trial=False)
        if not case_id:
            case_id = f"UNKNOWN_{unknown_counter}"
            unknown_counter += 1
        cases[case_id].append(row)
    return dict(cases)


def parse_o2c_csv(csv_path: str | Path | None = None) -> list[dict]:
    """Parse O2C event log CSV into AETHER format.

    The O2C (Order-to-Cash) log contains sales document flow events from SAP.

    Columns: case:concept:name, concept:name, time:timestamp,
             VBELN, VBELV, VBTYP_N_DESC, VBTYP_V_DESC

    Args:
        csv_path: Path to the O2C CSV file. Defaults to the standard location.

    Returns:
        List of dicts in AETHER format:
        {
            "caseId": str,
            "events": [{"activity": str, "resource": str, "timestamp": str, "attributes": dict}],
            "outcome": {"onTime": bool, "rework": bool, "durationHours": float}
        }
    """
    path = Path(csv_path) if csv_path is not None else DEFAULT_O2C_PATH
    logger.info(f"Loading O2C event log from {path}")

    rows = _read_csv_rows(path)
    grouped = _group_by_case(rows)

    cases: list[dict] = []
    skipped = 0

    for case_id, case_rows in grouped.items():
        try:
            case = _convert_o2c_case(case_id, case_rows)
            if case["events"]:
                cases.append(case)
            else:
                skipped += 1
        except Exception as e:
            logger.warning(f"Skipping O2C case {case_id}: {e}")
            skipped += 1

    logger.info(
        f"Parsed {len(cases)} O2C cases "
        f"({skipped} skipped, "
        f"{sum(len(c['events']) for c in cases)} total events)"
    )
    return cases


def _convert_o2c_case(case_id: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    """Convert a group of O2C CSV rows into a single AETHER case."""
    events: list[dict[str, Any]] = []
    timestamps: list[datetime | None] = []
    activity_names: list[str] = []

    for row in rows:
        activity = _clean_value(row.get("concept:name", ""))
        if not activity:
            continue

        ts_str = _clean_value(row.get("time:timestamp", ""))
        dt = _parse_timestamp(ts_str)

        # O2C has no resource column; use SYSTEM
        # Extract document type info as attributes
        vbtyp_n = _clean_value(row.get("VBTYP_N_DESC", ""))
        vbtyp_v = _clean_value(row.get("VBTYP_V_DESC", ""))
        vbeln = _clean_value(row.get("VBELN", ""))

        events.append({
            "activity": activity,
            "resource": "SYSTEM",
            "timestamp": _format_timestamp(dt),
            "attributes": {
                "amount": 0.0,  # O2C CSV has no amount column
                "document_type": vbtyp_n or vbtyp_v or "O2C",
                "document_number": vbeln,
                "source_document_type": vbtyp_v,
            },
        })
        timestamps.append(dt)
        activity_names.append(activity)

    # Sort events by timestamp if parseable
    _sort_events_by_timestamp(events, timestamps)

    # Compute outcomes
    valid_ts = [ts for ts in timestamps if ts is not None]
    first_ts = min(valid_ts) if valid_ts else None
    last_ts = max(valid_ts) if valid_ts else None

    return {
        "caseId": case_id,
        "events": events,
        "outcome": {
            "onTime": bool(set(activity_names) & O2C_COMPLETION_ACTIVITIES),
            "rework": _has_rework(activity_names),
            "durationHours": _compute_duration_hours(first_ts, last_ts),
        },
    }


def parse_p2p_csv(csv_path: str | Path | None = None) -> list[dict]:
    """Parse P2P event log CSV into AETHER format.

    The P2P (Purchase-to-Pay) log contains procurement events from SAP.

    Columns: case:concept:name, concept:name, time:timestamp,
             org:resource, document_id, document_type

    Args:
        csv_path: Path to the P2P CSV file. Defaults to the standard location.

    Returns:
        List of dicts in AETHER format:
        {
            "caseId": str,
            "events": [{"activity": str, "resource": str, "timestamp": str, "attributes": dict}],
            "outcome": {"onTime": bool, "rework": bool, "durationHours": float}
        }
    """
    path = Path(csv_path) if csv_path is not None else DEFAULT_P2P_PATH
    logger.info(f"Loading P2P event log from {path}")

    rows = _read_csv_rows(path)
    grouped = _group_by_case(rows)

    cases: list[dict] = []
    skipped = 0

    for case_id, case_rows in grouped.items():
        try:
            case = _convert_p2p_case(case_id, case_rows)
            if case["events"]:
                cases.append(case)
            else:
                skipped += 1
        except Exception as e:
            logger.warning(f"Skipping P2P case {case_id}: {e}")
            skipped += 1

    logger.info(
        f"Parsed {len(cases)} P2P cases "
        f"({skipped} skipped, "
        f"{sum(len(c['events']) for c in cases)} total events)"
    )
    return cases


def _convert_p2p_case(case_id: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    """Convert a group of P2P CSV rows into a single AETHER case."""
    events: list[dict[str, Any]] = []
    timestamps: list[datetime | None] = []
    activity_names: list[str] = []

    for row in rows:
        activity = _clean_value(row.get("concept:name", ""))
        if not activity:
            continue

        ts_str = _clean_value(row.get("time:timestamp", ""))
        dt = _parse_timestamp(ts_str)

        resource = _clean_value(row.get("org:resource", ""))
        doc_id = _clean_value(row.get("document_id", ""))
        doc_type = _clean_value(row.get("document_type", ""))

        # Map SAP doc type codes to readable names for metadata
        doc_type_desc = SAP_DOC_TYPES.get(doc_type, doc_type)

        events.append({
            "activity": activity,
            "resource": resource or "SYSTEM",
            "timestamp": _format_timestamp(dt),
            "attributes": {
                "amount": 0.0,  # P2P CSV has no amount column
                "document_type": doc_type_desc or "P2P",
                "document_id": doc_id,
                "sap_doc_type_code": doc_type,
            },
        })
        timestamps.append(dt)
        activity_names.append(activity)

    # Sort events by timestamp if parseable
    _sort_events_by_timestamp(events, timestamps)

    # Compute outcomes
    valid_ts = [ts for ts in timestamps if ts is not None]
    first_ts = min(valid_ts) if valid_ts else None
    last_ts = max(valid_ts) if valid_ts else None

    return {
        "caseId": case_id,
        "events": events,
        "outcome": {
            "onTime": bool(set(activity_names) & P2P_COMPLETION_ACTIVITIES),
            "rework": _has_rework(activity_names),
            "durationHours": _compute_duration_hours(first_ts, last_ts),
        },
    }


def _sort_events_by_timestamp(
    events: list[dict[str, Any]],
    timestamps: list[datetime | None],
) -> None:
    """Sort events list in-place by parsed timestamps.

    Events with unparseable timestamps are kept in their original order
    at the end.
    """
    if not events:
        return

    # Pair events with their parsed timestamps and original index
    indexed = list(enumerate(zip(events, timestamps)))

    # Separate parseable from unparseable
    parseable = [(i, e, ts) for i, (e, ts) in indexed if ts is not None]
    unparseable = [(i, e, ts) for i, (e, ts) in indexed if ts is None]

    # Sort parseable by timestamp
    parseable.sort(key=lambda x: x[2])

    # Rebuild in sorted order
    sorted_events = [e for _, e, _ in parseable] + [e for _, e, _ in unparseable]
    sorted_timestamps = [ts for _, _, ts in parseable] + [ts for _, _, ts in unparseable]

    # Replace in-place
    events[:] = sorted_events
    timestamps[:] = sorted_timestamps


def parse_csv_eventlogs(
    o2c_path: str | Path | None = None,
    p2p_path: str | Path | None = None,
) -> list[dict]:
    """Parse both CSV event logs and return combined cases.

    Case IDs are prefixed with "O2C_" or "P2P_" to avoid collisions
    between the two datasets.

    Args:
        o2c_path: Path to O2C CSV. Defaults to standard location.
        p2p_path: Path to P2P CSV. Defaults to standard location.

    Returns:
        Combined list of AETHER-format case dicts from both event logs.
    """
    o2c_cases = parse_o2c_csv(o2c_path)
    p2p_cases = parse_p2p_csv(p2p_path)

    # Prefix case IDs to avoid collisions
    for case in o2c_cases:
        case["caseId"] = f"O2C_{case['caseId']}"
    for case in p2p_cases:
        case["caseId"] = f"P2P_{case['caseId']}"

    combined = o2c_cases + p2p_cases

    logger.info(
        f"Combined CSV event logs: {len(combined)} cases "
        f"({len(o2c_cases)} O2C + {len(p2p_cases)} P2P, "
        f"{sum(len(c['events']) for c in combined)} total events)"
    )

    return combined


def get_csv_stats(cases: list[dict]) -> dict[str, Any]:
    """Compute summary statistics for parsed CSV event data.

    Args:
        cases: List of AETHER-format case dicts.

    Returns:
        Dict with summary statistics.
    """
    if not cases:
        return {
            "total_cases": 0,
            "total_events": 0,
            "unique_activities": 0,
            "unique_resources": 0,
            "on_time_rate": 0.0,
            "rework_rate": 0.0,
            "avg_duration_hours": 0.0,
            "median_case_length": 0,
        }

    activities: set[str] = set()
    resources: set[str] = set()
    case_lengths: list[int] = []
    on_time_count = 0
    rework_count = 0
    total_duration = 0.0

    for case in cases:
        events = case["events"]
        case_lengths.append(len(events))
        for event in events:
            activities.add(event["activity"])
            resources.add(event["resource"])

        outcome = case.get("outcome", {})
        if outcome.get("onTime", False):
            on_time_count += 1
        if outcome.get("rework", False):
            rework_count += 1
        total_duration += outcome.get("durationHours", 0.0)

    case_lengths.sort()
    n = len(cases)
    median_len = case_lengths[n // 2] if n % 2 == 1 else (
        (case_lengths[n // 2 - 1] + case_lengths[n // 2]) / 2
    )

    return {
        "total_cases": n,
        "total_events": sum(case_lengths),
        "unique_activities": len(activities),
        "unique_resources": len(resources),
        "on_time_rate": on_time_count / n,
        "rework_rate": rework_count / n,
        "avg_duration_hours": total_duration / n,
        "median_case_length": median_len,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=== O2C Event Log ===")
    o2c_cases = parse_o2c_csv()
    o2c_stats = get_csv_stats(o2c_cases)
    print(f"  Cases:            {o2c_stats['total_cases']}")
    print(f"  Total events:     {o2c_stats['total_events']}")
    print(f"  Unique activities: {o2c_stats['unique_activities']}")
    print(f"  On-time rate:     {o2c_stats['on_time_rate']:.1%}")
    print(f"  Rework rate:      {o2c_stats['rework_rate']:.1%}")
    print(f"  Avg duration:     {o2c_stats['avg_duration_hours']:.1f} hrs")

    print("\n=== P2P Event Log ===")
    p2p_cases = parse_p2p_csv()
    p2p_stats = get_csv_stats(p2p_cases)
    print(f"  Cases:            {p2p_stats['total_cases']}")
    print(f"  Total events:     {p2p_stats['total_events']}")
    print(f"  Unique activities: {p2p_stats['unique_activities']}")
    print(f"  Unique resources:  {p2p_stats['unique_resources']}")
    print(f"  On-time rate:     {p2p_stats['on_time_rate']:.1%}")
    print(f"  Rework rate:      {p2p_stats['rework_rate']:.1%}")
    print(f"  Avg duration:     {p2p_stats['avg_duration_hours']:.1f} hrs")

    print("\n=== Combined ===")
    combined = parse_csv_eventlogs()
    combined_stats = get_csv_stats(combined)
    print(f"  Total cases:      {combined_stats['total_cases']}")
    print(f"  Total events:     {combined_stats['total_events']}")

    if combined:
        print(f"\nSample O2C case: {o2c_cases[0]['caseId']}")
        for e in o2c_cases[0]["events"][:3]:
            print(f"  {e['timestamp']} | {e['activity']} | {e['resource']}")
        print(f"\nSample P2P case: {p2p_cases[0]['caseId']}")
        for e in p2p_cases[0]["events"][:3]:
            print(f"  {e['timestamp']} | {e['activity']} | {e['resource']}")
