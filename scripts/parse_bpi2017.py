"""
Parse BPI Challenge 2017 dataset into AETHER event log format.

Dataset: Loan Application Process from a Dutch Financial Institute
- 31,509 cases (loan applications)
- 1,202,267 events
- 26 unique activities
- Outcomes: Accepted (72%), Selected (63.7%)

This is a REAL prediction task - early events don't trivially determine outcome.

Source: Hugging Face (Modzo18/BPIC2017Iteration)
Original: 4TU.ResearchData BPI Challenge 2017
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths
AETHER_ROOT = Path("/Volumes/OWC drive/Dev/aether")
DATA_DIR = AETHER_ROOT / "data" / "external" / "bpi2017"
PARQUET_PATH = DATA_DIR / "bpi2017_events.parquet"
OUTPUT_DIR = DATA_DIR


def load_and_parse(max_cases: int | None = None) -> list[dict[str, Any]]:
    """Load BPI 2017 parquet and convert to AETHER format."""
    logger.info(f"Loading {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)

    logger.info(f"Total events: {len(df):,}")

    # Group by case
    cases = []
    case_groups = df.groupby("case:concept:name")

    for i, (case_id, case_df) in enumerate(case_groups):
        if max_cases and i >= max_cases:
            break

        # Sort by iteration (implicit ordering in BPI 2017)
        case_df = case_df.sort_values("iteration")

        # Extract events
        events = []
        for _, row in case_df.iterrows():
            event = {
                "activity": row["concept:name"],
                "resource": str(row["org:resource"]) if pd.notna(row["org:resource"]) else "unknown",
                "timestamp": "",  # BPI 2017 uses iterations, not timestamps
                "attributes": {
                    "action": row["Action"] if pd.notna(row["Action"]) else "",
                    "eventOrigin": row["EventOrigin"] if pd.notna(row["EventOrigin"]) else "",
                    "lifecycle": row["lifecycle:transition"] if pd.notna(row["lifecycle:transition"]) else "",
                    "iteration": int(row["iteration"]) if pd.notna(row["iteration"]) else 0,
                }
            }
            events.append(event)

        # Get case-level attributes (same for all events in case)
        first_row = case_df.iloc[0]

        # Determine outcomes
        accepted = bool(first_row["Accepted"]) if pd.notna(first_row["Accepted"]) else False
        selected = bool(first_row["Selected"]) if pd.notna(first_row["Selected"]) else False

        case = {
            "caseId": str(case_id),
            "events": events,
            "source": "bpi2017_huggingface",
            "processType": "loan_application",
            "outcome": {
                "accepted": accepted,
                "selected": selected,
                # For AETHER compatibility, map to standard outcomes
                "onTime": accepted,  # Treat acceptance as "successful" outcome
                "rework": not selected and accepted,  # Accepted but not selected = needs rework
            },
            "caseAttributes": {
                "loanGoal": first_row["case:LoanGoal"] if pd.notna(first_row["case:LoanGoal"]) else "",
                "applicationType": first_row["case:ApplicationType"] if pd.notna(first_row["case:ApplicationType"]) else "",
                "requestedAmount": float(first_row["case:RequestedAmount"]) if pd.notna(first_row["case:RequestedAmount"]) else 0,
                "creditScore": float(first_row["CreditScore"]) if pd.notna(first_row["CreditScore"]) else 0,
                "offeredAmount": float(first_row["OfferedAmount"]) if pd.notna(first_row["OfferedAmount"]) else 0,
            },
        }
        cases.append(case)

        if (i + 1) % 5000 == 0:
            logger.info(f"  Processed {i + 1:,} cases...")

    logger.info(f"Parsed {len(cases):,} cases")
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
    from collections import Counter

    activities = Counter()
    resources = set()
    total_events = 0

    for case in cases:
        for event in case["events"]:
            activities[event["activity"]] += 1
            resources.add(event.get("resource", "unknown"))
            total_events += 1

    event_counts = [len(c["events"]) for c in cases]
    accepted_count = sum(1 for c in cases if c["outcome"]["accepted"])
    selected_count = sum(1 for c in cases if c["outcome"]["selected"])

    return {
        "total_cases": len(cases),
        "total_events": total_events,
        "activity_vocab_size": len(activities),
        "resource_vocab_size": len(resources),
        "activity_counts": dict(activities.most_common()),
        "outcome_stats": {
            "accepted_rate": round(accepted_count / len(cases), 4) if cases else 0,
            "selected_rate": round(selected_count / len(cases), 4) if cases else 0,
        },
        "event_length_stats": {
            "min": min(event_counts) if event_counts else 0,
            "max": max(event_counts) if event_counts else 0,
            "mean": round(sum(event_counts) / len(event_counts), 2) if event_counts else 0,
        },
        "source": "BPI Challenge 2017 (Dutch Financial Institute Loan Applications)",
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
    print("BPI CHALLENGE 2017 -> AETHER EVENT LOG PARSER")
    print("=" * 60)

    if not PARQUET_PATH.exists():
        print(f"ERROR: Parquet file not found: {PARQUET_PATH}")
        print("Please run the Hugging Face download script first.")
        return

    # Parse all cases
    cases = load_and_parse()

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
    print("BPI 2017 DATASET SUMMARY")
    print("=" * 60)
    print(f"Cases: {stats['total_cases']:,} (train: {stats['train_cases']:,}, val: {stats['val_cases']:,})")
    print(f"Events: {stats['total_events']:,}")
    print(f"Activities: {stats['activity_vocab_size']}")
    print(f"Resources: {stats['resource_vocab_size']}")

    print("\nTop activities:")
    for act, count in list(stats["activity_counts"].items())[:10]:
        print(f"  {act}: {count:,}")

    print("\nOutcomes:")
    print(f"  Accepted: {stats['outcome_stats']['accepted_rate']:.1%}")
    print(f"  Selected: {stats['outcome_stats']['selected_rate']:.1%}")

    print(f"\nSequence lengths: min={stats['event_length_stats']['min']}, max={stats['event_length_stats']['max']}, mean={stats['event_length_stats']['mean']:.1f}")


if __name__ == "__main__":
    main()
