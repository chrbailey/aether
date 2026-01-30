"""
Parse Judicial Intelligence case events into AETHER event log format.

Novel legal process mining dataset from Judge Alsup (N.D. Cal) cases.
Spans 2010-2025 across major tech litigation (Oracle v Google, Waymo v Uber, etc.).

Source: /Volumes/OWC drive/Archive AI Projects/judicial-intelligence/
"""

import json
import random
from datetime import datetime
from collections import defaultdict, Counter
from pathlib import Path
from typing import Optional, Dict, List, Any


# Data paths
JUDICIAL_ROOT = Path("/Volumes/OWC drive/Archive AI Projects/judicial-intelligence/judge-evaluation/data")
EVENTS_DIR = JUDICIAL_ROOT / "events"
OUTPUT_DIR = Path("/Volumes/OWC drive/Dev/aether/data/external/judicial")

# Source files
EVENT_FILES = [
    "alsup_ndcal_events.jsonl",
    "alsup_ndcal_events_5w1h.jsonl",
    "alsup_additional_manual.jsonl",
    "alsup_final_push.jsonl",
]


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse date from various formats in the judicial data."""
    if not date_str:
        return None

    date_str = str(date_str).strip()

    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def load_events() -> List[Dict]:
    """Load all events from JSONL files, deduplicating by event_id."""
    events = {}

    for filename in EVENT_FILES:
        path = EVENTS_DIR / filename
        if not path.exists():
            print(f"  Warning: {filename} not found")
            continue

        count = 0
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                event = json.loads(line)
                event_id = event.get("event_id", "")

                if event_id and event_id not in events:
                    events[event_id] = event
                    count += 1

        print(f"  Loaded {count} unique events from {filename}")

    return list(events.values())


def normalize_event(raw_event: Dict) -> Optional[Dict]:
    """
    Normalize a raw event into AETHER event format.

    Handles two source formats:
    1. Simple format: event_id, case_id, decision_date, labels, features
    2. 5W1H format: event_id, case_id, who, what, when, where, why, how
    """
    event_id = raw_event.get("event_id", "")
    case_id = raw_event.get("case_id", "")

    if not case_id:
        return None

    # Extract date - try multiple locations
    date_str = (
        raw_event.get("decision_date") or
        raw_event.get("when", {}).get("date") or
        ""
    )
    timestamp = parse_date(date_str)
    if not timestamp:
        return None

    # Determine activity type from the event structure
    activity = extract_activity(raw_event)
    if not activity:
        return None

    # Extract outcome
    outcome = extract_outcome(raw_event)

    # Extract doctrine/domain
    doctrine = extract_doctrine(raw_event)

    # Extract resource (judge)
    resource = extract_resource(raw_event)

    # Build attributes
    attributes = {
        "event_id": event_id,
        "outcome": outcome,
        "doctrine": doctrine,
    }

    # Add posture if available
    posture = raw_event.get("features", {}).get("posture", "")
    if posture:
        attributes["posture"] = posture

    # Add motion type if available
    motion_type = raw_event.get("features", {}).get("motion_type", "")
    if motion_type:
        attributes["motion_type"] = motion_type

    # Add party types
    plaintiff_type = raw_event.get("features", {}).get("plaintiff_type", "")
    defendant_type = raw_event.get("features", {}).get("defendant_type", "")
    if plaintiff_type:
        attributes["plaintiff_type"] = plaintiff_type
    if defendant_type:
        attributes["defendant_type"] = defendant_type

    # Add beneficiary from labels
    beneficiary = raw_event.get("labels", {}).get("beneficiary", "")
    if beneficiary:
        attributes["beneficiary"] = beneficiary

    return {
        "activity": activity,
        "timestamp": timestamp.isoformat(),
        "resource": resource,
        "attributes": attributes,
    }


def extract_activity(raw_event: Dict) -> Optional[str]:
    """Extract standardized activity name from event."""
    # Check features.posture first
    features = raw_event.get("features", {})
    posture = features.get("posture", "")
    motion_type = features.get("motion_type", "")

    # Check 5W1H format
    what = raw_event.get("what", {})
    event_type = what.get("event_type", "")
    _action = what.get("action", "")  # noqa: F841

    # Standardize activities for legal process
    if posture == "rule_12b6" or "12b6" in motion_type:
        return "motion_to_dismiss_12b6"

    if posture == "summary_judgment" or "sj_" in motion_type:
        return "motion_summary_judgment"

    if posture == "preliminary_injunction" or "pi_" in motion_type:
        return "motion_preliminary_injunction"

    if posture == "settlement_approval" or "settlement" in motion_type:
        return "settlement_approval"

    if posture == "pretrial_motion":
        if "discovery" in motion_type or "sanctions" in motion_type:
            return "discovery_motion"
        return "pretrial_motion"

    if posture == "post_trial_motion":
        if "attorney_fees" in motion_type:
            return "motion_attorney_fees"
        if "jmol" in motion_type or "damages" in motion_type:
            return "post_trial_motion_jmol"
        return "post_trial_motion"

    # Fallback to event_type from 5W1H
    if event_type:
        return event_type.replace(" ", "_").lower()

    # Last resort - use motion_type directly
    if motion_type:
        return motion_type.replace(" ", "_").lower()

    return None


def extract_outcome(raw_event: Dict) -> str:
    """Extract standardized outcome from event."""
    # Check labels first
    labels = raw_event.get("labels", {})
    outcome = labels.get("outcome", "")

    if outcome in ("grant", "granted"):
        return "granted"
    if outcome in ("deny", "denied"):
        return "denied"
    if outcome in ("partial", "granted_in_part"):
        return "partial"

    # Check what.outcome in 5W1H format
    what = raw_event.get("what", {})
    what_outcome = what.get("outcome", "")

    if what_outcome in ("granted", "grant"):
        return "granted"
    if what_outcome in ("denied", "deny"):
        return "denied"
    if what_outcome in ("granted_in_part", "partial"):
        return "partial"

    return "unknown"


def extract_doctrine(raw_event: Dict) -> str:
    """Extract legal doctrine/domain from event."""
    # Check features first
    features = raw_event.get("features", {})
    doctrine = features.get("doctrine", "")

    if doctrine:
        return doctrine

    # Check 5W1H format
    why = raw_event.get("why", {})
    doctrine = why.get("doctrine", "")

    if doctrine:
        return doctrine

    # Check ai_insight
    ai_insight = raw_event.get("ai_insight", {})
    doctrine_tags = ai_insight.get("doctrine_tags", {})
    primary = doctrine_tags.get("primary", "")

    if primary:
        return primary

    return "unknown"


def extract_resource(raw_event: Dict) -> str:
    """Extract resource (judge) from event."""
    # Check judge_id
    judge_id = raw_event.get("judge_id", "")
    if judge_id:
        return judge_id

    # Check who.judge in 5W1H format
    who = raw_event.get("who", {})
    judge = who.get("judge", {})
    judge_name = judge.get("name", "") or judge.get("judge_id", "")

    if judge_name:
        return judge_name

    return "unknown_judge"


def group_by_case(events: List[Dict]) -> Dict[str, List[Dict]]:
    """Group normalized events by case_id."""
    cases = defaultdict(list)

    for raw_event in events:
        normalized = normalize_event(raw_event)
        if normalized:
            case_id = raw_event.get("case_id", "")
            cases[case_id].append(normalized)

    # Sort events within each case by timestamp
    for case_id in cases:
        cases[case_id] = sorted(cases[case_id], key=lambda e: e["timestamp"])

    return dict(cases)


def determine_case_outcome(events: List[Dict], case_id: str) -> Dict[str, Any]:
    """Determine overall case outcome from event sequence."""
    # Check for settlement
    has_settlement = any(
        "settlement" in e["activity"]
        for e in events
    )

    if has_settlement:
        return {
            "disposition": "settlement",
            "plaintiff_prevailed": None,
            "defendant_prevailed": None,
        }

    # Count outcomes by beneficiary
    plaintiff_wins = 0
    defendant_wins = 0

    for event in events:
        attrs = event.get("attributes", {})
        outcome = attrs.get("outcome", "")
        beneficiary = attrs.get("beneficiary", "")

        if beneficiary == "plaintiff":
            if outcome == "granted":
                plaintiff_wins += 1
            elif outcome == "denied":
                defendant_wins += 1
        elif beneficiary == "defendant":
            if outcome == "granted":
                defendant_wins += 1
            elif outcome == "denied":
                plaintiff_wins += 1

    # Determine overall winner
    if plaintiff_wins > defendant_wins:
        return {
            "disposition": "plaintiff_favorable",
            "plaintiff_prevailed": True,
            "defendant_prevailed": False,
        }
    if defendant_wins > plaintiff_wins:
        return {
            "disposition": "defendant_favorable",
            "plaintiff_prevailed": False,
            "defendant_prevailed": True,
        }

    return {
        "disposition": "mixed",
        "plaintiff_prevailed": None,
        "defendant_prevailed": None,
    }


def build_cases(grouped_events: Dict[str, List[Dict]]) -> List[Dict]:
    """Build AETHER case format from grouped events."""
    cases = []

    for case_id, events in grouped_events.items():
        if len(events) < 1:
            continue

        # Calculate duration
        first_ts = datetime.fromisoformat(events[0]["timestamp"])
        last_ts = datetime.fromisoformat(events[-1]["timestamp"])
        duration_days = (last_ts - first_ts).total_seconds() / 86400

        # Get primary doctrine from most common
        doctrines = [e["attributes"].get("doctrine", "unknown") for e in events]
        doctrine_counts = Counter(doctrines)
        primary_doctrine = doctrine_counts.most_common(1)[0][0] if doctrine_counts else "unknown"

        # Determine case outcome
        outcome = determine_case_outcome(events, case_id)
        outcome["primary_doctrine"] = primary_doctrine
        outcome["duration_days"] = round(duration_days, 2)
        outcome["event_count"] = len(events)

        # Extract case attributes
        first_event = events[0]
        case_attrs = {
            "case_id": case_id,
            "first_event_date": first_event["timestamp"],
            "primary_doctrine": primary_doctrine,
        }

        # Add party types from first event
        attrs = first_event.get("attributes", {})
        if attrs.get("plaintiff_type"):
            case_attrs["plaintiff_type"] = attrs["plaintiff_type"]
        if attrs.get("defendant_type"):
            case_attrs["defendant_type"] = attrs["defendant_type"]

        case = {
            "caseId": f"judicial_{case_id}",
            "events": events,
            "source": "judicial_intelligence_alsup",
            "outcome": outcome,
            "caseAttributes": case_attrs,
        }

        cases.append(case)

    return cases


def build_vocabulary(cases: List[Dict]) -> Dict:
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


def compute_stats(cases: List[Dict]) -> Dict:
    """Compute dataset statistics."""
    activities = Counter()
    doctrines = Counter()
    resources = set()
    total_events = 0

    for case in cases:
        for event in case["events"]:
            activities[event["activity"]] += 1
            resources.add(event.get("resource", "unknown"))
            total_events += 1

        primary_doctrine = case["outcome"].get("primary_doctrine", "unknown")
        doctrines[primary_doctrine] += 1

    durations = [c["outcome"]["duration_days"] for c in cases]
    event_counts = [len(c["events"]) for c in cases]

    # Outcome distribution
    settlement_count = sum(1 for c in cases if c["outcome"]["disposition"] == "settlement")
    plaintiff_favorable = sum(1 for c in cases if c["outcome"]["disposition"] == "plaintiff_favorable")
    defendant_favorable = sum(1 for c in cases if c["outcome"]["disposition"] == "defendant_favorable")
    mixed_count = sum(1 for c in cases if c["outcome"]["disposition"] == "mixed")

    return {
        "total_cases": len(cases),
        "total_events": total_events,
        "activity_vocab_size": len(activities),
        "resource_vocab_size": len(resources),
        "activity_counts": dict(activities.most_common()),
        "doctrine_counts": dict(doctrines.most_common()),
        "outcome_stats": {
            "settlement_rate": round(settlement_count / len(cases), 4) if cases else 0,
            "plaintiff_favorable_rate": round(plaintiff_favorable / len(cases), 4) if cases else 0,
            "defendant_favorable_rate": round(defendant_favorable / len(cases), 4) if cases else 0,
            "mixed_rate": round(mixed_count / len(cases), 4) if cases else 0,
        },
        "event_length_stats": {
            "min": min(event_counts) if event_counts else 0,
            "max": max(event_counts) if event_counts else 0,
            "mean": round(sum(event_counts) / len(event_counts), 2) if event_counts else 0,
            "median": sorted(event_counts)[len(event_counts) // 2] if event_counts else 0,
        },
        "duration_stats_days": {
            "min": round(min(durations), 2) if durations else 0,
            "max": round(max(durations), 2) if durations else 0,
            "mean": round(sum(durations) / len(durations), 2) if durations else 0,
        },
        "source": "Judicial Intelligence - Judge Alsup N.D. Cal (2010-2025)",
        "domain": "legal_process_mining",
    }


def split_train_val(cases: List[Dict], train_ratio: float = 0.8, seed: int = 42):
    """Split cases into train and validation sets."""
    random.seed(seed)
    shuffled = cases.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def main():
    print("=" * 60)
    print("JUDICIAL INTELLIGENCE -> AETHER EVENT LOG PARSER")
    print("=" * 60)

    # Load all events
    print("\nLoading events from JSONL files...")
    raw_events = load_events()
    print(f"\n  Total unique events loaded: {len(raw_events)}")

    # Group by case
    print("\nGrouping events by case...")
    grouped = group_by_case(raw_events)
    print(f"  Found {len(grouped)} unique cases")

    # Build cases
    print("\nBuilding AETHER case format...")
    cases = build_cases(grouped)
    print(f"  Built {len(cases)} cases with events")

    if not cases:
        print("ERROR: No cases built!")
        return

    # Split
    train_cases, val_cases = split_train_val(cases)

    # Build vocabulary
    vocab = build_vocabulary(cases)

    # Compute stats
    stats = compute_stats(cases)
    stats["train_cases"] = len(train_cases)
    stats["val_cases"] = len(val_cases)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "train_cases.json", "w") as f:
        json.dump(train_cases, f, indent=2)
    print(f"\nSaved {len(train_cases)} train cases")

    with open(OUTPUT_DIR / "val_cases.json", "w") as f:
        json.dump(val_cases, f, indent=2)
    print(f"Saved {len(val_cases)} val cases")

    with open(OUTPUT_DIR / "vocabulary.json", "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved vocabulary: {vocab['activity']['size']} activities, {vocab['resource']['size']} resources")

    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("Saved metadata.json")

    # Print summary
    print("\n" + "=" * 60)
    print("JUDICIAL PROCESS MINING DATASET SUMMARY")
    print("=" * 60)
    print(f"Cases: {stats['total_cases']} (train: {stats['train_cases']}, val: {stats['val_cases']})")
    print(f"Events: {stats['total_events']}")
    print(f"Unique Activities: {stats['activity_vocab_size']}")
    print(f"Unique Resources: {stats['resource_vocab_size']}")

    print("\nActivity distribution:")
    for act, count in list(stats["activity_counts"].items())[:10]:
        print(f"  {act}: {count}")

    print("\nDoctrine distribution:")
    for doctrine, count in list(stats["doctrine_counts"].items())[:8]:
        print(f"  {doctrine}: {count}")

    print("\nOutcome distribution:")
    print(f"  Settlement: {stats['outcome_stats']['settlement_rate']:.1%}")
    print(f"  Plaintiff favorable: {stats['outcome_stats']['plaintiff_favorable_rate']:.1%}")
    print(f"  Defendant favorable: {stats['outcome_stats']['defendant_favorable_rate']:.1%}")
    print(f"  Mixed: {stats['outcome_stats']['mixed_rate']:.1%}")

    print(f"\nSequence lengths: min={stats['event_length_stats']['min']}, max={stats['event_length_stats']['max']}, mean={stats['event_length_stats']['mean']:.1f}")
    print(f"Duration: mean={stats['duration_stats_days']['mean']:.1f} days")

    print("\n" + "=" * 60)
    print("Output files:")
    print(f"  {OUTPUT_DIR / 'train_cases.json'}")
    print(f"  {OUTPUT_DIR / 'val_cases.json'}")
    print(f"  {OUTPUT_DIR / 'vocabulary.json'}")
    print(f"  {OUTPUT_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()
