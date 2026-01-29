"""
Parse NetSuite Transaction Export into AETHER event log format.

Processes a 1.1GB Transactions.csv (4.2M rows) into vendor-centric case sequences.
Each case represents a vendor's financial activity (bills, payments, credits, etc.)

Source: NetSuite export (2019-2025)
Output: AETHER event format with train/val split
"""

import csv
import json
import random
from datetime import datetime
from collections import defaultdict, Counter
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Iterator

# Data paths
NETSUITE_ROOT = Path("/Volumes/OWC drive/Datasets/raw/netsuite-export-2025-01-05")
TRANSACTIONS_PATH = NETSUITE_ROOT / "Transactions.csv"
OUTPUT_DIR = Path("/Volumes/OWC drive/Dev/aether/data/external/netsuite_2025")

# Processing config
CHUNK_SIZE = 50000  # Rows to process at a time
MAX_ROWS = None  # Set to int to limit (e.g., 500000), None for full dataset
MIN_EVENTS_PER_CASE = 3  # Skip sparse cases
MAX_EVENTS_PER_CASE = 100  # Truncate very long sequences


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


def parse_amount(amount_str: str) -> float:
    """Parse amount string to float."""
    if not amount_str or amount_str.strip() == "":
        return 0.0

    cleaned = amount_str.strip().replace(",", "").replace("$", "")
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]

    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def normalize_transaction_type(tx_type: str) -> str:
    """Convert transaction type to standardized activity name."""
    tx_type = tx_type.strip().lower()

    mapping = {
        "bill": "bill_received",
        "bill credit": "bill_credit_issued",
        "bill payment": "bill_payment_made",
        "invoice": "invoice_sent",
        "payment": "payment_received",
        "credit memo": "credit_memo_issued",
        "journal": "journal_entry",
        "paycheck": "payroll_check",
        "expense report": "expense_submitted",
        "deposit": "deposit_made",
        "check": "check_issued",
        "payroll liability check": "payroll_liability_paid",
        "payroll adjustment": "payroll_adjusted",
        "sales order": "sales_order_created",
        "opportunity": "opportunity_logged",
        "proposal": "proposal_sent",
    }

    return mapping.get(tx_type, f"other_{tx_type.replace(' ', '_')}")


def extract_account_category(account: str) -> str:
    """Extract high-level account category from full account path."""
    if not account:
        return "unknown"

    account = account.strip().lower()

    if "accounts payable" in account:
        return "accounts_payable"
    if "accounts receivable" in account:
        return "accounts_receivable"
    if "personnel" in account or "payroll" in account:
        return "personnel"
    if "fixed assets" in account or "software" in account:
        return "fixed_assets"
    if "rent" in account or "occupancy" in account:
        return "occupancy"
    if "legal" in account:
        return "legal"
    if "recruiting" in account:
        return "recruiting"
    if "marketing" in account:
        return "marketing"
    if "undeposited" in account:
        return "undeposited_funds"
    if "prepaid" in account:
        return "prepaid"
    if "sales tax" in account:
        return "sales_tax"
    if "capital" in account:
        return "capital"
    if "expense" in account:
        return "expense"

    return "other"


def stream_transactions(path: Path, max_rows: Optional[int] = None) -> Iterator[Dict]:
    """
    Stream transactions from CSV, yielding header rows only (rows with '*').

    NetSuite exports have multiple rows per transaction - the header row has '*'
    in column 3 and contains the key details. Line items follow without '*'.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)

        row_count = 0
        yielded_count = 0

        for row in reader:
            row_count += 1

            if max_rows and row_count > max_rows:
                break

            # Only process header rows (marked with *)
            marker = row.get("*", "").strip()
            if marker != "*":
                continue

            internal_id = row.get("Internal ID", "").strip()
            tx_type = row.get("Type", "").strip()
            date = parse_date(row.get("Date", ""))
            name = row.get("Name", "").strip()
            doc_num = row.get("Document Number", "").strip()
            amount = parse_amount(row.get("Amount", ""))
            account = row.get("Account", "").strip()
            memo = row.get("Memo", "").strip()
            vendor_category = row.get("Vendor Category", "").strip()
            created_from = row.get("Created From", "").strip()

            if not tx_type or not date or not name:
                continue

            yielded_count += 1

            yield {
                "internal_id": internal_id,
                "type": tx_type,
                "date": date,
                "name": name,
                "document_number": doc_num,
                "amount": amount,
                "account": account,
                "account_category": extract_account_category(account),
                "memo": memo,
                "vendor_category": vendor_category or "unknown",
                "created_from": created_from,
            }

            if yielded_count % 100000 == 0:
                print(f"  Processed {yielded_count:,} header transactions...")

        print(f"  Total: {row_count:,} rows, {yielded_count:,} header transactions")


def group_by_entity(transactions: Iterator[Dict]) -> Dict[str, List[Dict]]:
    """Group transactions by entity name (vendor/customer)."""
    print("Grouping transactions by entity...")
    entity_transactions = defaultdict(list)

    for tx in transactions:
        entity = tx["name"]
        entity_transactions[entity].append(tx)

    print(f"  Found {len(entity_transactions):,} unique entities")
    return entity_transactions


def build_entity_cases(entity_transactions: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Build cases from entity transaction histories.

    Each case represents an entity's financial lifecycle - all their
    bills, payments, credits, etc. ordered chronologically.
    """
    print("Building entity cases...")

    cases = []
    skipped_sparse = 0
    skipped_no_activity = 0

    for entity, transactions in entity_transactions.items():
        # Sort by date
        transactions = sorted(transactions, key=lambda x: x["date"])

        # Skip entities with too few transactions
        if len(transactions) < MIN_EVENTS_PER_CASE:
            skipped_sparse += 1
            continue

        # Build events
        events = []
        total_billed = 0.0
        total_paid = 0.0
        tx_types_seen = set()

        for tx in transactions[:MAX_EVENTS_PER_CASE]:
            activity = normalize_transaction_type(tx["type"])
            tx_types_seen.add(tx["type"])

            # Track financial summary
            if tx["type"] in ["Bill", "Invoice"]:
                total_billed += abs(tx["amount"])
            elif tx["type"] in ["Bill Payment", "Payment"]:
                total_paid += abs(tx["amount"])

            event = {
                "activity": activity,
                "timestamp": tx["date"].isoformat(),
                "resource": tx["vendor_category"],
                "attributes": {
                    "amount": round(tx["amount"], 2),
                    "account_category": tx["account_category"],
                    "document_number": tx["document_number"],
                }
            }

            # Add memo if meaningful
            if tx["memo"] and len(tx["memo"]) < 100:
                event["attributes"]["memo"] = tx["memo"]

            events.append(event)

        # Skip if only one type of activity (not interesting process)
        if len(tx_types_seen) < 2:
            skipped_no_activity += 1
            continue

        # Calculate duration
        first_ts = datetime.fromisoformat(events[0]["timestamp"])
        last_ts = datetime.fromisoformat(events[-1]["timestamp"])
        duration_days = (last_ts - first_ts).total_seconds() / 86400

        # Determine outcomes
        payment_ratio = total_paid / total_billed if total_billed > 0 else 0.0
        has_credit = any("credit" in e["activity"] for e in events)
        has_journal = any("journal" in e["activity"] for e in events)

        # Create clean entity ID
        entity_id = "".join(c if c.isalnum() else "_" for c in entity[:30])

        case = {
            "caseId": f"ns_{entity_id}",
            "events": events,
            "source": "netsuite_2025",
            "outcome": {
                "paymentRatio": round(payment_ratio, 4),
                "hasCredit": has_credit,
                "hasJournalAdjustment": has_journal,
                "totalBilled": round(total_billed, 2),
                "totalPaid": round(total_paid, 2),
                "durationDays": round(duration_days, 2),
                "eventCount": len(events),
            },
            "caseAttributes": {
                "entity": entity,
                "firstTxDate": events[0]["timestamp"],
                "lastTxDate": events[-1]["timestamp"],
                "transactionTypes": list(tx_types_seen),
            }
        }

        cases.append(case)

    print(f"  Built {len(cases):,} cases")
    print(f"  Skipped: {skipped_sparse:,} sparse, {skipped_no_activity:,} single-activity")

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
    resources = Counter()
    total_events = 0

    for case in cases:
        for event in case["events"]:
            activities[event["activity"]] += 1
            resources[event.get("resource", "unknown")] += 1
            total_events += 1

    durations = [c["outcome"]["durationDays"] for c in cases]
    event_counts = [len(c["events"]) for c in cases]
    payment_ratios = [c["outcome"]["paymentRatio"] for c in cases]

    credit_count = sum(1 for c in cases if c["outcome"]["hasCredit"])
    journal_count = sum(1 for c in cases if c["outcome"]["hasJournalAdjustment"])

    return {
        "total_cases": len(cases),
        "total_events": total_events,
        "activity_vocab_size": len(activities),
        "resource_vocab_size": len(resources),
        "activity_counts": dict(activities.most_common()),
        "resource_counts": dict(resources.most_common(20)),
        "outcome_stats": {
            "credit_rate": round(credit_count / len(cases), 4) if cases else 0,
            "journal_rate": round(journal_count / len(cases), 4) if cases else 0,
            "avg_payment_ratio": round(sum(payment_ratios) / len(payment_ratios), 4) if payment_ratios else 0,
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
        "source": "NetSuite Transaction Export (2019-2025)",
        "processing_config": {
            "min_events_per_case": MIN_EVENTS_PER_CASE,
            "max_events_per_case": MAX_EVENTS_PER_CASE,
            "max_rows_processed": MAX_ROWS,
        },
    }


def split_train_val(cases: List[Dict], train_ratio: float = 0.8, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Split cases into train and validation sets."""
    random.seed(seed)
    shuffled = cases.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def main():
    print("=" * 70)
    print("NETSUITE TRANSACTIONS -> AETHER EVENT LOG PARSER")
    print("=" * 70)

    # Check input file
    if not TRANSACTIONS_PATH.exists():
        print(f"ERROR: Input file not found: {TRANSACTIONS_PATH}")
        return

    file_size_mb = TRANSACTIONS_PATH.stat().st_size / (1024 * 1024)
    print(f"\nInput: {TRANSACTIONS_PATH}")
    print(f"Size: {file_size_mb:.1f} MB")
    if MAX_ROWS:
        print(f"Processing limit: {MAX_ROWS:,} rows")
    else:
        print("Processing: ALL rows")

    # Stream and group transactions
    print("\n" + "-" * 50)
    transactions = stream_transactions(TRANSACTIONS_PATH, MAX_ROWS)
    entity_transactions = group_by_entity(transactions)

    # Build cases
    print("\n" + "-" * 50)
    cases = build_entity_cases(entity_transactions)

    if not cases:
        print("ERROR: No cases generated!")
        return

    # Split
    train_cases, val_cases = split_train_val(cases)

    # Build vocabulary
    vocab = build_vocabulary(cases)

    # Compute stats
    stats = compute_stats(cases)
    stats["train_cases"] = len(train_cases)
    stats["val_cases"] = len(val_cases)

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "-" * 50)
    print("Saving outputs...")

    with open(OUTPUT_DIR / "train_cases.json", "w") as f:
        json.dump(train_cases, f)
    print(f"  train_cases.json: {len(train_cases):,} cases")

    with open(OUTPUT_DIR / "val_cases.json", "w") as f:
        json.dump(val_cases, f)
    print(f"  val_cases.json: {len(val_cases):,} cases")

    with open(OUTPUT_DIR / "vocabulary.json", "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"  vocabulary.json: {vocab['activity']['size']} activities, {vocab['resource']['size']} resources")

    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("  metadata.json: dataset statistics")

    # Print summary
    print("\n" + "=" * 70)
    print("NETSUITE VENDOR TRANSACTION DATASET SUMMARY")
    print("=" * 70)
    print(f"Cases: {stats['total_cases']:,} (train: {stats['train_cases']:,}, val: {stats['val_cases']:,})")
    print(f"Events: {stats['total_events']:,}")
    print(f"Activities: {stats['activity_vocab_size']}")
    print(f"Resources: {stats['resource_vocab_size']}")
    print(f"\nActivity distribution:")
    for act, count in list(stats["activity_counts"].items())[:12]:
        print(f"  {act}: {count:,}")
    print(f"\nOutcomes:")
    print(f"  Credit rate: {stats['outcome_stats']['credit_rate']:.1%}")
    print(f"  Journal adjustment rate: {stats['outcome_stats']['journal_rate']:.1%}")
    print(f"  Avg payment ratio: {stats['outcome_stats']['avg_payment_ratio']:.2f}")
    print(f"\nSequence lengths: min={stats['event_length_stats']['min']}, max={stats['event_length_stats']['max']}, mean={stats['event_length_stats']['mean']:.1f}")
    print(f"Duration: mean={stats['duration_stats_days']['mean']:.1f} days")

    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
