"""BPI Challenge 2019 parser for AETHER training pipeline.

Parses the BPI 2019 JSON file (Purchase-to-Pay process from a multinational
coatings company) into AETHER's standard event sequence format.

Dataset: 50,000 processed cases (from 251,734 total), 340,324 events,
39 unique activities, 369 unique users.

Source: https://data.4tu.nl/articles/dataset/BPI_Challenge_2019/12715853
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Activities that indicate successful completion of the P2P cycle
COMPLETION_ACTIVITIES = frozenset({
    "Clear Invoice",
    "Record Invoice Receipt",
    "Record Goods Receipt",
    "SRM: Document Completed",
    "SRM: Transaction Completed",
})

# Default path — set via AETHER_BPI2019_PATH env var or pass explicitly
DEFAULT_BPI2019_PATH: Path | None = (
    Path(os.environ["AETHER_BPI2019_PATH"])
    if os.environ.get("AETHER_BPI2019_PATH")
    else None
)


def _parse_timestamp(ts_str: str) -> datetime | None:
    """Parse an ISO 8601 timestamp string, returning None on failure."""
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        pass
    # Try stripping timezone suffix variants
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts_str, fmt)
        except (ValueError, TypeError):
            continue
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


def _is_on_time(activities: list[str]) -> bool:
    """Check if the case reached a completion activity."""
    return bool(set(activities) & COMPLETION_ACTIVITIES)


def _convert_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Convert a single BPI 2019 trace to AETHER format.

    Args:
        trace: Dict with keys 'case_id', 'attributes', 'events'.

    Returns:
        AETHER-format case dict.
    """
    case_id = trace.get("case_id", "")
    raw_events = trace.get("events", [])

    if not raw_events:
        return {
            "caseId": case_id,
            "events": [],
            "outcome": {"onTime": False, "rework": False, "durationHours": 0.0},
        }

    parsed_events: list[dict[str, Any]] = []
    parsed_timestamps: list[datetime | None] = []
    activity_names: list[str] = []

    for raw_event in raw_events:
        activity = raw_event.get("concept:name", "")
        resource = raw_event.get("org:resource", "") or raw_event.get("User", "")
        ts_str = raw_event.get("time:timestamp", "")
        dt = _parse_timestamp(ts_str)

        # Parse amount from "Cumulative net worth (EUR)"
        amount = 0.0
        raw_amount = raw_event.get("Cumulative net worth (EUR)", "")
        if raw_amount:
            try:
                amount = float(raw_amount)
            except (ValueError, TypeError):
                pass

        parsed_events.append({
            "activity": activity,
            "resource": resource or "SYSTEM",
            "timestamp": _format_timestamp(dt),
            "attributes": {
                "amount": amount,
                "document_type": "PO",  # BPI 2019 is Purchase-to-Pay
            },
        })
        parsed_timestamps.append(dt)
        activity_names.append(activity)

    # Compute outcomes
    valid_timestamps = [ts for ts in parsed_timestamps if ts is not None]
    first_ts = min(valid_timestamps) if valid_timestamps else None
    last_ts = max(valid_timestamps) if valid_timestamps else None

    return {
        "caseId": case_id,
        "events": parsed_events,
        "outcome": {
            "onTime": _is_on_time(activity_names),
            "rework": _has_rework(activity_names),
            "durationHours": _compute_duration_hours(first_ts, last_ts),
        },
    }


def parse_bpi2019(
    json_path: str | Path | None = None,
    max_cases: int | None = None,
) -> list[dict]:
    """Parse BPI 2019 JSON into AETHER format.

    The JSON file has the structure:
        {
            "metadata": {...},
            "stats": {...},
            "traces": [
                {
                    "case_id": "...",
                    "attributes": {...},
                    "events": [
                        {
                            "concept:name": "activity",
                            "org:resource": "user",
                            "time:timestamp": "ISO8601",
                            "Cumulative net worth (EUR)": "123.0",
                            ...
                        }
                    ]
                }
            ]
        }

    Args:
        json_path: Path to the BPI 2019 JSON file. Defaults to the standard
            location on the OWC drive.
        max_cases: Maximum number of cases to parse. None means all cases.

    Returns:
        List of dicts in AETHER format:
        {
            "caseId": str,
            "events": [{"activity": str, "resource": str, "timestamp": str, "attributes": dict}],
            "outcome": {"onTime": bool, "rework": bool, "durationHours": float}
        }

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        ValueError: If the JSON structure is unexpected.
    """
    path = Path(json_path) if json_path is not None else DEFAULT_BPI2019_PATH

    if not path.exists():
        raise FileNotFoundError(f"BPI 2019 JSON not found: {path}")

    logger.info(f"Loading BPI 2019 data from {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Try streaming with ijson for memory efficiency on large files
    traces = _load_traces(path)

    if max_cases is not None:
        traces = traces[:max_cases]

    logger.info(f"Converting {len(traces)} traces to AETHER format")

    cases: list[dict] = []
    skipped = 0
    for trace in traces:
        try:
            case = _convert_trace(trace)
            if case["events"]:  # Skip empty traces
                cases.append(case)
            else:
                skipped += 1
        except Exception as e:
            logger.warning(f"Skipping trace {trace.get('case_id', '?')}: {e}")
            skipped += 1

    logger.info(
        f"Parsed {len(cases)} cases from BPI 2019 "
        f"({skipped} skipped, "
        f"{sum(len(c['events']) for c in cases)} total events)"
    )

    return cases


def _load_traces(path: Path) -> list[dict[str, Any]]:
    """Load traces from the JSON file.

    Attempts ijson streaming first for memory efficiency on 85MB+ files.
    Falls back to json.load() if ijson is unavailable.
    """
    try:
        return _load_traces_ijson(path)
    except ImportError:
        logger.debug("ijson not available, falling back to json.load()")
        return _load_traces_stdlib(path)


def _load_traces_ijson(path: Path) -> list[dict[str, Any]]:
    """Load traces using ijson streaming parser."""
    import ijson

    traces: list[dict[str, Any]] = []
    with open(path, "rb") as f:
        for trace in ijson.items(f, "traces.item"):
            traces.append(trace)

    return traces


def _load_traces_stdlib(path: Path) -> list[dict[str, Any]]:
    """Load traces using standard json.load()."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "traces" in data:
            return data["traces"]
        raise ValueError(
            f"Expected 'traces' key in JSON dict, got keys: {list(data.keys())}"
        )
    elif isinstance(data, list):
        # Direct list of traces (alternative format)
        return data
    else:
        raise ValueError(f"Unexpected JSON root type: {type(data).__name__}")


def get_bpi2019_stats(cases: list[dict]) -> dict[str, Any]:
    """Compute summary statistics for parsed BPI 2019 data.

    Args:
        cases: List of AETHER-format case dicts.

    Returns:
        Dict with keys: total_cases, total_events, unique_activities,
        unique_resources, on_time_rate, rework_rate, avg_duration_hours,
        median_case_length.
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

    # Parse with a small sample for quick testing
    cases = parse_bpi2019(max_cases=100)
    stats = get_bpi2019_stats(cases)

    print(f"\nBPI 2019 Sample Stats ({stats['total_cases']} cases):")
    print(f"  Total events:      {stats['total_events']}")
    print(f"  Unique activities:  {stats['unique_activities']}")
    print(f"  Unique resources:   {stats['unique_resources']}")
    print(f"  On-time rate:       {stats['on_time_rate']:.1%}")
    print(f"  Rework rate:        {stats['rework_rate']:.1%}")
    print(f"  Avg duration (hrs): {stats['avg_duration_hours']:.1f}")
    print(f"  Median case length: {stats['median_case_length']}")

    if cases:
        print(f"\nSample case: {cases[0]['caseId']}")
        print(f"  Events: {len(cases[0]['events'])}")
        for e in cases[0]["events"][:3]:
            print(f"    {e['timestamp']} | {e['activity']} | {e['resource']}")
