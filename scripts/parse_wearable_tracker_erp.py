"""
Parse Wearable Tracker ERP Export (NetSuite → SAP Conversion) into AETHER event log format.

Reconstructs Order-to-Cash process from:
  - Sales Orders (102k orders with types: New, Correction, etc.)
  - RMAs (97k returns linked to customers)

Source: Wearable Tracker NetSuite export (2013-2016), pre-SAP conversion
"""

import csv
import json
import random
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# Data paths
WEARABLE_TRACKER_ROOT = Path("/Volumes/OWC drive/_Archive/Fitbit - working files/Product Earth")
SO_PATH = WEARABLE_TRACKER_ROOT / "BSP 669 SAP Transaction Data Conversion/Extracts/6 - 156 - Sales Orders"

OUTPUT_DIR = Path("/Volumes/OWC drive/Dev/aether/data/external/wearable_tracker")


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse various date formats from NetSuite export."""
    if not date_str or date_str.strip() == "":
        return None

    date_str = date_str.strip()

    formats = [
        "%m/%d/%Y",
        "%m/%d/%Y %I:%M %p",
        "%m/%d/%y",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def read_csv_safe(path: Path, encoding: str = "utf-8") -> List[Dict]:
    """Read CSV with error handling for encoding issues."""
    try:
        with open(path, "r", encoding=encoding, errors="replace") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return []


def load_sales_orders() -> Dict[str, List[Dict]]:
    """Load sales order headers, grouped by customer."""
    print("Loading sales orders...")
    path = SO_PATH / "Sales Order Header - Production.csv"
    rows = read_csv_safe(path)

    # Group orders by customer
    customer_orders = defaultdict(list)

    for row in rows:
        customer = row.get("Customer", "").strip()
        doc_num = row.get("Document Number", "").strip()
        date = parse_date(row.get("Date", ""))

        if not customer or not doc_num or not date:
            continue

        order = {
            "internal_id": row.get("Internal ID", ""),
            "document_number": doc_num,
            "customer": customer,
            "date": date,
            "sales_rep": row.get("Sales Rep", "").strip(),
            "subsidiary": row.get("Subsidiary", "").strip(),
            "location": row.get("Location", "").strip(),
            "order_type": row.get("Sales Order Type", "").strip(),
            "credit_hold": row.get("Customer On Credit Hold", "") == "Yes",
            "po_number": row.get("PO/Check Number", "").strip(),
            "source": row.get("Source", "").strip(),
        }

        customer_orders[customer].append(order)

    total = sum(len(v) for v in customer_orders.values())
    print(f"  Loaded {total:,} sales orders for {len(customer_orders):,} customers")
    return customer_orders


def load_rmas() -> Dict[str, List[Dict]]:
    """Load RMAs (returns), grouped by customer."""
    print("Loading RMAs...")
    path = SO_PATH / "BSP 737 - RMA Extract - QA1.csv"
    rows = read_csv_safe(path)

    customer_rmas = defaultdict(list)

    for row in rows:
        customer = row.get("Customer", "").strip()
        rma_num = row.get("RTN Auth #", "").strip()
        date = parse_date(row.get("Date", ""))

        if not customer or not rma_num or not date:
            continue

        # Skip duplicate rows (RMAs have line items)
        if any(r["rma_number"] == rma_num for r in customer_rmas[customer]):
            continue

        rma = {
            "rma_number": rma_num,
            "customer": customer,
            "date": date,
            "rma_type": row.get("RMA Type", "").strip(),
            "sales_rep": row.get("Sales Rep", "").strip(),
            "status": row.get("Status (RMA)", "").strip(),
            "subsidiary": row.get("Subsidiary (no hierarchy)", "").strip(),
        }

        customer_rmas[customer].append(rma)

    total = sum(len(v) for v in customer_rmas.values())
    print(f"  Loaded {total:,} RMAs for {len(customer_rmas):,} customers")
    return customer_rmas


def reconstruct_customer_journeys(
    customer_orders: Dict[str, List[Dict]],
    customer_rmas: Dict[str, List[Dict]]
) -> List[Dict]:
    """
    Reconstruct customer journey event sequences.

    Each case is a customer's activity over time:
      First Order → Subsequent Orders → RMAs (if any)

    We create meaningful process instances by grouping customer activity.
    """
    print("Reconstructing customer journeys...")

    cases = []

    for customer, orders in customer_orders.items():
        # Skip customers with too few orders
        if len(orders) < 2:
            continue

        # Sort orders by date
        orders = sorted(orders, key=lambda x: x["date"])

        # Get RMAs for this customer
        rmas = customer_rmas.get(customer, [])
        rmas = sorted(rmas, key=lambda x: x["date"])

        # Create events
        events = []

        # Add order events
        for i, order in enumerate(orders):
            if i == 0:
                activity = "first_order"
            elif order["order_type"] == "Correction":
                activity = "order_correction"
            elif order["order_type"] == "New":
                activity = "repeat_order"
            else:
                activity = "order_" + (order["order_type"].lower().replace(" ", "_") or "unknown")

            events.append({
                "activity": activity,
                "timestamp": order["date"].isoformat(),
                "resource": order["sales_rep"] or "unknown",
                "attributes": {
                    "order_type": order["order_type"],
                    "location": order["location"],
                    "credit_hold": order["credit_hold"],
                    "document_number": order["document_number"],
                }
            })

        # Add RMA events
        for rma in rmas:
            rma_type = rma["rma_type"].lower().replace(" ", "_").replace("-", "_") if rma["rma_type"] else "unknown"
            activity = f"rma_{rma_type}"

            events.append({
                "activity": activity,
                "timestamp": rma["date"].isoformat(),
                "resource": rma["sales_rep"] or "unknown",
                "attributes": {
                    "rma_number": rma["rma_number"],
                    "rma_status": rma["status"],
                }
            })

        # Sort all events by timestamp
        events = sorted(events, key=lambda x: x["timestamp"])

        # Skip if still less than 2 events
        if len(events) < 2:
            continue

        # Limit to first 50 events per customer (very long sequences hurt training)
        events = events[:50]

        # Determine outcome
        has_rma = any("rma" in e["activity"] for e in events)
        has_correction = any(e["activity"] == "order_correction" for e in events)
        order_count = sum(1 for e in events if "order" in e["activity"])

        # Calculate duration
        first_ts = datetime.fromisoformat(events[0]["timestamp"])
        last_ts = datetime.fromisoformat(events[-1]["timestamp"])
        duration_hours = (last_ts - first_ts).total_seconds() / 3600

        # Extract customer ID (e.g., "CUS10150" from "CUS10150 ABB Inc.")
        customer_id = customer.split()[0] if customer else "unknown"

        case = {
            "caseId": f"wt_{customer_id}",
            "events": events,
            "source": "wearable_tracker_netsuite",
            "outcome": {
                "onTime": not has_rma and not has_correction,
                "rework": has_correction,
                "hasRma": has_rma,
                "orderCount": order_count,
                "durationHours": round(duration_hours, 2),
            },
            "caseAttributes": {
                "customer": customer,
                "first_order_date": events[0]["timestamp"],
            }
        }
        cases.append(case)

    print(f"  Reconstructed {len(cases):,} customer journey cases")
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
    from collections import Counter

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

    ontime_count = sum(1 for c in cases if c["outcome"]["onTime"])
    rework_count = sum(1 for c in cases if c["outcome"]["rework"])
    rma_count = sum(1 for c in cases if c["outcome"]["hasRma"])

    return {
        "total_cases": len(cases),
        "total_events": total_events,
        "activity_vocab_size": len(activities),
        "resource_vocab_size": len(resources),
        "activity_counts": dict(activities.most_common()),
        "outcome_stats": {
            "ontime_rate": round(ontime_count / len(cases), 4),
            "rework_rate": round(rework_count / len(cases), 4),
            "rma_rate": round(rma_count / len(cases), 4),
        },
        "event_length_stats": {
            "min": min(event_counts),
            "max": max(event_counts),
            "mean": round(sum(event_counts) / len(event_counts), 2),
            "median": sorted(event_counts)[len(event_counts) // 2],
        },
        "duration_stats_hours": {
            "min": round(min(durations), 2),
            "max": round(max(durations), 2),
            "mean": round(sum(durations) / len(durations), 2),
        },
        "source": "Wearable Tracker NetSuite Export (2013-2016)",
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
    print("WEARABLE TRACKER ERP → AETHER EVENT LOG PARSER")
    print("=" * 60)

    # Load transaction data
    customer_orders = load_sales_orders()
    customer_rmas = load_rmas()

    # Reconstruct event sequences
    cases = reconstruct_customer_journeys(customer_orders, customer_rmas)

    if not cases:
        print("ERROR: No cases reconstructed!")
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

    with open(OUTPUT_DIR / "wearable_tracker_all_cases.json", "w") as f:
        json.dump(cases, f)
    print(f"\nSaved {len(cases):,} cases to wearable_tracker_all_cases.json")

    with open(OUTPUT_DIR / "train_cases.json", "w") as f:
        json.dump(train_cases, f)
    print(f"Saved {len(train_cases):,} train cases")

    with open(OUTPUT_DIR / "val_cases.json", "w") as f:
        json.dump(val_cases, f)
    print(f"Saved {len(val_cases):,} val cases")

    with open(OUTPUT_DIR / "vocabulary.json", "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved vocabulary: {vocab['activity']['size']} activities, {vocab['resource']['size']} resources")

    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("WEARABLE TRACKER CUSTOMER JOURNEY DATASET SUMMARY")
    print("=" * 60)
    print(f"Cases: {stats['total_cases']:,} (train: {stats['train_cases']:,}, val: {stats['val_cases']:,})")
    print(f"Events: {stats['total_events']:,}")
    print(f"Activities: {stats['activity_vocab_size']}")
    print(f"Resources: {stats['resource_vocab_size']}")
    print(f"\nActivity distribution:")
    for act, count in list(stats["activity_counts"].items())[:10]:
        print(f"  {act}: {count:,}")
    print(f"\nOutcomes:")
    print(f"  On-time (no RMA/correction): {stats['outcome_stats']['ontime_rate']:.1%}")
    print(f"  Rework (corrections): {stats['outcome_stats']['rework_rate']:.1%}")
    print(f"  Has RMA: {stats['outcome_stats']['rma_rate']:.1%}")
    print(f"\nSequence lengths: min={stats['event_length_stats']['min']}, max={stats['event_length_stats']['max']}, mean={stats['event_length_stats']['mean']:.1f}")
    print(f"Duration: mean={stats['duration_stats_hours']['mean']:.1f} hours")


if __name__ == "__main__":
    main()
