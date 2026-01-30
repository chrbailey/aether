"""
Parse Road Traffic Fine Management Process XES file into AETHER event log format.

Dataset: Road Traffic Fine Management Process from 4TU
- 150,370 cases
- 561,470 events
- 11 unique activities
- Process: handling of road traffic fines from creation through payment/appeal

Source: https://data.4tu.nl/articles/dataset/Road_Traffic_Fine_Management_Process/12683249
"""

from __future__ import annotations

import json
import logging
import random
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths
AETHER_ROOT = Path("/Volumes/OWC drive/Dev/aether")
DATA_DIR = AETHER_ROOT / "data" / "external" / "road_traffic_fine"
XES_PATH = DATA_DIR / "Road_Traffic_Fine_Management_Process.xes"
OUTPUT_DIR = DATA_DIR

# XES namespace handling
XES_NS = {"xes": "http://www.xes-standard.org/"}


def parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse XES timestamp format."""
    if not ts_str:
        return None
    try:
        # XES timestamps can be: 2006-07-24T00:00:00.000+02:00
        # Remove milliseconds and timezone for simpler parsing
        ts_clean = ts_str.split(".")[0]
        if "+" in ts_clean:
            ts_clean = ts_clean.split("+")[0]
        elif ts_clean.endswith("Z"):
            ts_clean = ts_clean[:-1]
        return datetime.fromisoformat(ts_clean)
    except (ValueError, TypeError) as e:
        logger.debug(f"Could not parse timestamp {ts_str}: {e}")
        return None


def format_timestamp(dt: Optional[datetime]) -> str:
    """Format datetime to ISO string."""
    if dt is None:
        return ""
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def get_attribute_value(element: ET.Element, key: str) -> Optional[Any]:
    """Extract attribute value from XES element by key."""
    for child in element:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if child.get("key") == key:
            value = child.get("value")
            if tag == "string":
                return value
            elif tag == "int":
                return int(value) if value else 0
            elif tag == "float":
                return float(value) if value else 0.0
            elif tag == "boolean":
                return value.lower() == "true" if value else False
            elif tag == "date":
                return value
            return value
    return None


def parse_xes_file(xes_path: Path, max_cases: Optional[int] = None) -> list[dict]:
    """Parse XES file into AETHER format cases.

    Uses iterparse for memory efficiency on large files.
    """
    logger.info(f"Parsing XES file: {xes_path}")
    logger.info(f"File size: {xes_path.stat().st_size / 1024 / 1024:.1f} MB")

    cases = []
    current_trace = None
    current_events = []
    trace_attributes = {}

    # Use iterparse for memory efficiency
    context = ET.iterparse(str(xes_path), events=("start", "end"))

    case_count = 0
    event_count = 0

    for event_type, elem in context:
        # Remove namespace prefix for easier tag matching
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

        if event_type == "start":
            if tag == "trace":
                current_trace = {}
                current_events = []
                trace_attributes = {}

        elif event_type == "end":
            if tag == "trace":
                if current_events:
                    # Build case from accumulated data
                    case_id = trace_attributes.get("concept:name", f"case_{case_count}")

                    # Sort events by timestamp
                    current_events.sort(key=lambda e: e.get("_ts") or datetime.min)

                    # Clean up events (remove internal timestamp)
                    events = []
                    for e in current_events:
                        clean_event = {
                            "activity": e.get("activity", ""),
                            "resource": e.get("resource", "unknown"),
                            "timestamp": e.get("timestamp", ""),
                            "attributes": e.get("attributes", {}),
                        }
                        events.append(clean_event)

                    if events:
                        # Compute outcomes
                        activities = [e["activity"] for e in events]

                        # Parse timestamps for duration
                        timestamps = []
                        for e in events:
                            ts = parse_timestamp(e["timestamp"])
                            if ts:
                                timestamps.append(ts)

                        duration_hours = 0.0
                        if len(timestamps) >= 2:
                            duration_hours = (max(timestamps) - min(timestamps)).total_seconds() / 3600

                        # Determine outcomes based on activities
                        has_payment = "Payment" in activities
                        has_appeal = any("Appeal" in a for a in activities)
                        has_penalty = "Add penalty" in activities
                        is_complete = has_payment or ("Send for Credit Collection" in activities)

                        case = {
                            "caseId": case_id,
                            "events": events,
                            "source": "road_traffic_fine_4tu",
                            "processType": "traffic_fine_management",
                            "outcome": {
                                "paid": has_payment,
                                "appealed": has_appeal,
                                "penalty": has_penalty,
                                "complete": is_complete,
                                "durationHours": round(duration_hours, 2),
                                # Binary outcomes for training
                                "onTime": has_payment and not has_penalty,
                                "rework": has_appeal,
                            },
                            "caseAttributes": {
                                k: v for k, v in trace_attributes.items()
                                if k not in ("concept:name",)
                            },
                        }
                        cases.append(case)
                        case_count += 1
                        event_count += len(events)

                        if case_count % 10000 == 0:
                            logger.info(f"  Processed {case_count:,} cases, {event_count:,} events...")

                        if max_cases and case_count >= max_cases:
                            logger.info(f"  Reached max_cases limit: {max_cases}")
                            break

                # Clear memory
                current_trace = None
                current_events = []
                trace_attributes = {}
                elem.clear()

            elif tag == "event" and current_trace is not None:
                # Parse event
                event_data = {}
                event_attrs = {}

                for child in elem:
                    child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    key = child.get("key", "")
                    value = child.get("value", "")

                    if key == "concept:name":
                        event_data["activity"] = value
                    elif key == "org:resource":
                        event_data["resource"] = value if value else "unknown"
                    elif key == "time:timestamp":
                        event_data["timestamp"] = format_timestamp(parse_timestamp(value))
                        event_data["_ts"] = parse_timestamp(value)  # Keep for sorting
                    elif key == "lifecycle:transition":
                        event_attrs["lifecycle"] = value
                    elif key not in ("identity:id",):
                        # Store other attributes
                        if child_tag == "int":
                            event_attrs[key] = int(value) if value else 0
                        elif child_tag == "float":
                            event_attrs[key] = float(value) if value else 0.0
                        elif child_tag == "boolean":
                            event_attrs[key] = value.lower() == "true" if value else False
                        else:
                            event_attrs[key] = value

                event_data["attributes"] = event_attrs
                current_events.append(event_data)
                elem.clear()

            elif current_trace is not None and tag in ("string", "int", "float", "boolean", "date"):
                # Trace-level attribute
                key = elem.get("key", "")
                value = elem.get("value", "")
                if key and key not in ("identity:id",):
                    if tag == "int":
                        trace_attributes[key] = int(value) if value else 0
                    elif tag == "float":
                        trace_attributes[key] = float(value) if value else 0.0
                    elif tag == "boolean":
                        trace_attributes[key] = value.lower() == "true" if value else False
                    else:
                        trace_attributes[key] = value

    logger.info(f"Parsed {len(cases):,} cases with {event_count:,} events")
    return cases


def build_vocabulary(cases: list[dict]) -> dict:
    """Build activity and resource vocabularies."""
    activity_tokens = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
    resource_tokens = {"<PAD>": 0, "<UNK>": 1}

    activities = set()
    resources = set()

    for case in cases:
        for event in case["events"]:
            activities.add(event["activity"])
            resources.add(event.get("resource", "unknown"))

    for i, act in enumerate(sorted(activities), start=len(activity_tokens)):
        activity_tokens[act] = i

    for i, res in enumerate(sorted(resources), start=len(resource_tokens)):
        resource_tokens[res] = i

    return {
        "activity": {"token_to_idx": activity_tokens, "size": len(activity_tokens)},
        "resource": {"token_to_idx": resource_tokens, "size": len(resource_tokens)},
    }


def compute_stats(cases: list[dict]) -> dict:
    """Compute dataset statistics."""
    activities = Counter()
    resources = set()
    total_events = 0

    for case in cases:
        for event in case["events"]:
            activities[event["activity"]] += 1
            resources.add(event.get("resource", "unknown"))
            total_events += 1

    durations = [c["outcome"]["durationHours"] for c in cases]
    event_counts = [len(c["events"]) for c in cases]

    paid_count = sum(1 for c in cases if c["outcome"]["paid"])
    appealed_count = sum(1 for c in cases if c["outcome"]["appealed"])
    penalty_count = sum(1 for c in cases if c["outcome"]["penalty"])
    complete_count = sum(1 for c in cases if c["outcome"]["complete"])

    return {
        "total_cases": len(cases),
        "total_events": total_events,
        "activity_vocab_size": len(activities),
        "resource_vocab_size": len(resources),
        "activity_counts": dict(activities.most_common()),
        "outcome_stats": {
            "paid_rate": round(paid_count / len(cases), 4) if cases else 0,
            "appealed_rate": round(appealed_count / len(cases), 4) if cases else 0,
            "penalty_rate": round(penalty_count / len(cases), 4) if cases else 0,
            "complete_rate": round(complete_count / len(cases), 4) if cases else 0,
        },
        "event_length_stats": {
            "min": min(event_counts) if event_counts else 0,
            "max": max(event_counts) if event_counts else 0,
            "mean": round(sum(event_counts) / len(event_counts), 2) if event_counts else 0,
            "median": sorted(event_counts)[len(event_counts) // 2] if event_counts else 0,
        },
        "duration_stats_hours": {
            "min": round(min(durations), 2) if durations else 0,
            "max": round(max(durations), 2) if durations else 0,
            "mean": round(sum(durations) / len(durations), 2) if durations else 0,
        },
        "source": "Road Traffic Fine Management Process (4TU.ResearchData)",
    }


def split_train_val(
    cases: list[dict],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split cases into train and validation sets."""
    random.seed(seed)
    shuffled = cases.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def main():
    print("=" * 60)
    print("ROAD TRAFFIC FINE -> AETHER EVENT LOG PARSER")
    print("=" * 60)

    if not XES_PATH.exists():
        print(f"ERROR: XES file not found: {XES_PATH}")
        return

    # Parse XES file
    cases = parse_xes_file(XES_PATH)

    if not cases:
        print("ERROR: No cases parsed!")
        return

    # Split into train/val
    train_cases, val_cases = split_train_val(cases)

    # Build vocabulary
    vocab = build_vocabulary(cases)

    # Compute statistics
    stats = compute_stats(cases)
    stats["train_cases"] = len(train_cases)
    stats["val_cases"] = len(val_cases)

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "train_cases.json", "w") as f:
        json.dump(train_cases, f)
    print(f"\nSaved {len(train_cases):,} train cases to train_cases.json")

    with open(OUTPUT_DIR / "val_cases.json", "w") as f:
        json.dump(val_cases, f)
    print(f"Saved {len(val_cases):,} val cases to val_cases.json")

    with open(OUTPUT_DIR / "vocabulary.json", "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved vocabulary: {vocab['activity']['size']} activities, {vocab['resource']['size']} resources")

    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("Saved metadata.json")

    # Print summary
    print("\n" + "=" * 60)
    print("ROAD TRAFFIC FINE DATASET SUMMARY")
    print("=" * 60)
    print(f"Cases: {stats['total_cases']:,} (train: {stats['train_cases']:,}, val: {stats['val_cases']:,})")
    print(f"Events: {stats['total_events']:,}")
    print(f"Activities: {stats['activity_vocab_size']}")
    print(f"Resources: {stats['resource_vocab_size']}")

    print(f"\nActivity distribution:")
    for act, count in list(stats["activity_counts"].items())[:15]:
        print(f"  {act}: {count:,}")

    print(f"\nOutcomes:")
    print(f"  Paid: {stats['outcome_stats']['paid_rate']:.1%}")
    print(f"  Appealed: {stats['outcome_stats']['appealed_rate']:.1%}")
    print(f"  Penalty: {stats['outcome_stats']['penalty_rate']:.1%}")
    print(f"  Complete: {stats['outcome_stats']['complete_rate']:.1%}")

    print(f"\nSequence lengths: min={stats['event_length_stats']['min']}, max={stats['event_length_stats']['max']}, mean={stats['event_length_stats']['mean']:.1f}")
    print(f"Duration: mean={stats['duration_stats_hours']['mean']:.1f} hours")


if __name__ == "__main__":
    main()
