"""
Parse SAP BSP 669 Transaction Data Conversion extracts into AETHER event log format.

Reconstructs two process types:
  1. Order-to-Cash (O2C): Sales Orders -> Invoices -> Credit Memos/Payments
  2. Procure-to-Pay (P2P): Bills -> Payments (by vendor)

Source: BSP 669 SAP Transaction Data Conversion (NetSuite export, 2013-2017)
Files:
  - Sales Order Header (102K records)
  - DFI Invoice (166K records)
  - Clearing Credit Memo (164K records)
  - Open Bills and Credits (120K records)
  - RMA Extract (97K records)
  - Customer Deductions (7.6K records)
"""

import csv
import json
import random
import re
from datetime import datetime
from collections import defaultdict, Counter
from pathlib import Path
from typing import Optional, Dict, List, Tuple


# Data paths
EXTRACTS_ROOT = Path("/Volumes/OWC drive/Consulant Administration/Product Earth/BSP 669 SAP Transaction Data Conversion/Extracts")
SALES_ORDERS_DIR = EXTRACTS_ROOT / "6 - 156 - Sales Orders"
AR_DIR = EXTRACTS_ROOT / "Production Extracts/7 - Accounts Receivable"
AP_DIR = EXTRACTS_ROOT / "Production Extracts/8 - Accounts Payable"
DEDUCTIONS_DIR = EXTRACTS_ROOT / "Production Extracts/6 - Sales Orders"

OUTPUT_DIR = Path("/Volumes/OWC drive/Dev/aether/data/external/sap_bsp669")


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
        "%d/%m/%Y",
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
        print(f"  ERROR reading {path.name}: {e}")
        return []


def extract_customer_id(customer_str: str) -> str:
    """Extract customer ID from strings like 'CUS10150 ABB Inc.'"""
    if not customer_str:
        return ""
    match = re.match(r"(CUS\d+)", customer_str)
    return match.group(1) if match else customer_str.split()[0] if customer_str else ""


def extract_vendor_id(vendor_str: str) -> str:
    """Extract vendor ID or use first significant word."""
    if not vendor_str:
        return ""
    return vendor_str.strip()[:50]


def extract_doc_number(ref_str: str) -> Optional[str]:
    """Extract document numbers from reference strings like 'Invoice #IN130671'."""
    if not ref_str:
        return None
    # Try to find invoice/SO/PO numbers
    patterns = [
        r"(IN\d+)",
        r"(SO\d+)",
        r"(PO\d+)",
        r"(CM\d+)",
        r"#(\w+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, ref_str)
        if match:
            return match.group(1)
    return None


def load_sales_orders() -> Dict[str, List[Dict]]:
    """Load sales order headers, grouped by customer."""
    print("Loading sales orders...")
    path = SALES_ORDERS_DIR / "Sales Order Header - Production.csv"
    rows = read_csv_safe(path)

    customer_orders = defaultdict(list)
    skipped = 0

    for row in rows:
        customer = row.get("Customer", "").strip()
        doc_num = row.get("Document Number", "").strip()
        date = parse_date(row.get("Date", ""))

        if not customer or not doc_num or not date:
            skipped += 1
            continue

        customer_id = extract_customer_id(customer)

        order = {
            "internal_id": row.get("Internal ID", ""),
            "document_number": doc_num,
            "customer": customer,
            "customer_id": customer_id,
            "date": date,
            "sales_rep": row.get("Sales Rep", "").strip(),
            "subsidiary": row.get("Subsidiary", "").strip(),
            "location": row.get("Location", "").strip(),
            "order_type": row.get("Sales Order Type", "").strip(),
            "credit_hold": row.get("Customer On Credit Hold", "") == "Yes",
            "po_number": row.get("PO/Check Number", "").strip(),
            "source": row.get("Source", "").strip(),
        }

        customer_orders[customer_id].append(order)

    total = sum(len(v) for v in customer_orders.values())
    print(f"  Loaded {total:,} sales orders for {len(customer_orders):,} customers (skipped {skipped:,})")
    return customer_orders


def load_invoices() -> Dict[str, List[Dict]]:
    """Load DFI invoices, grouped by customer."""
    print("Loading invoices (DFI)...")
    path = AR_DIR / "DFI Invoice.csv"
    rows = read_csv_safe(path)

    customer_invoices = defaultdict(list)
    seen_invoices = set()
    skipped = 0

    for row in rows:
        customer = row.get("Customer Companyname", "").strip()
        date = parse_date(row.get("Date", ""))
        customer_ref = row.get("Customer Reference Number", "").strip()

        if not customer or not date:
            skipped += 1
            continue

        # Use customer+date+ref as dedup key (multiple line items per invoice)
        dedup_key = f"{customer}|{date.isoformat()}|{customer_ref}"
        if dedup_key in seen_invoices:
            continue
        seen_invoices.add(dedup_key)

        customer_id = extract_customer_id(customer) or customer[:30]

        invoice = {
            "customer": customer,
            "customer_id": customer_id,
            "date": date,
            "reason_code": row.get("Reason Code", "").strip(),
            "customer_ref": customer_ref,
            "department": row.get("Department (no hierarchy)", "").strip(),
            "linked_credit_memo": row.get("Linked Clearing Credit Memo", "").strip(),
            "terms": row.get("Terms", "").strip(),
        }

        customer_invoices[customer_id].append(invoice)

    total = sum(len(v) for v in customer_invoices.values())
    print(f"  Loaded {total:,} unique invoices for {len(customer_invoices):,} customers (skipped {skipped:,})")
    return customer_invoices


def load_credit_memos() -> Dict[str, List[Dict]]:
    """Load clearing credit memos, grouped by customer."""
    print("Loading credit memos...")
    path = AR_DIR / "Clearing Credit Memo.csv"
    rows = read_csv_safe(path)

    customer_credits = defaultdict(list)
    seen_credits = set()
    skipped = 0

    for row in rows:
        customer = row.get("Customer Companyname", "").strip()
        date = parse_date(row.get("Date", ""))
        customer_ref = row.get("Customer Reference Number", "").strip()

        if not customer or not date:
            skipped += 1
            continue

        # Dedup by customer+date+ref
        dedup_key = f"{customer}|{date.isoformat()}|{customer_ref}"
        if dedup_key in seen_credits:
            continue
        seen_credits.add(dedup_key)

        customer_id = extract_customer_id(customer) or customer[:30]

        credit = {
            "customer": customer,
            "customer_id": customer_id,
            "date": date,
            "reason_code": row.get("Reason Code", "").strip(),
            "customer_ref": customer_ref,
            "linked_dfi": row.get("Linked DFI Invoice", "").strip(),
            "rma_number": row.get("RMA Number", "").strip(),
            "cd_number": row.get("CD Number", "").strip(),
        }

        customer_credits[customer_id].append(credit)

    total = sum(len(v) for v in customer_credits.values())
    print(f"  Loaded {total:,} unique credit memos for {len(customer_credits):,} customers (skipped {skipped:,})")
    return customer_credits


def load_rmas() -> Dict[str, List[Dict]]:
    """Load RMAs (returns), grouped by customer."""
    print("Loading RMAs...")
    path = SALES_ORDERS_DIR / "BSP 737 - RMA Extract - QA1.csv"
    rows = read_csv_safe(path)

    customer_rmas = defaultdict(list)
    seen_rmas = set()
    skipped = 0

    for row in rows:
        customer = row.get("Customer", "").strip()
        rma_num = row.get("RTN Auth #", "").strip()
        date = parse_date(row.get("Date", ""))

        if not customer or not rma_num or not date:
            skipped += 1
            continue

        # Skip duplicate RMA headers (multiple line items)
        dedup_key = f"{customer}|{rma_num}"
        if dedup_key in seen_rmas:
            continue
        seen_rmas.add(dedup_key)

        customer_id = extract_customer_id(customer)

        rma = {
            "rma_number": rma_num,
            "customer": customer,
            "customer_id": customer_id,
            "date": date,
            "rma_type": row.get("RMA Type", "").strip(),
            "sales_rep": row.get("Sales Rep", "").strip(),
            "status": row.get("Status (RMA)", "").strip(),
            "subsidiary": row.get("Subsidiary (no hierarchy)", "").strip(),
            "revenue_status": row.get("Revenue Status", "").strip(),
        }

        customer_rmas[customer_id].append(rma)

    total = sum(len(v) for v in customer_rmas.values())
    print(f"  Loaded {total:,} unique RMAs for {len(customer_rmas):,} customers (skipped {skipped:,})")
    return customer_rmas


def load_deductions() -> Dict[str, List[Dict]]:
    """Load customer deductions, grouped by customer."""
    print("Loading customer deductions...")
    path = DEDUCTIONS_DIR / "Customer Deductions.csv"
    rows = read_csv_safe(path)

    customer_deductions = defaultdict(list)
    seen_deductions = set()
    skipped = 0

    for row in rows:
        customer = row.get("Customer", "").strip()
        cd_num = row.get("CD#", "").strip()
        date = parse_date(row.get("Date", ""))

        if not customer or not date:
            skipped += 1
            continue

        # Dedup
        dedup_key = f"{customer}|{cd_num}"
        if dedup_key in seen_deductions:
            continue
        seen_deductions.add(dedup_key)

        customer_id = extract_customer_id(customer)

        deduction = {
            "cd_number": cd_num,
            "customer": customer,
            "customer_id": customer_id,
            "date": date,
            "deduction_type": row.get("Deduction Order Type", "").strip(),
            "reason_code": row.get("Reason Code", "").strip(),
            "status": row.get("Status", "").strip(),
            "related_dfi": row.get("Related DFI #", "").strip(),
        }

        customer_deductions[customer_id].append(deduction)

    total = sum(len(v) for v in customer_deductions.values())
    print(f"  Loaded {total:,} unique deductions for {len(customer_deductions):,} customers (skipped {skipped:,})")
    return customer_deductions


def load_ap_bills() -> Dict[str, List[Dict]]:
    """Load AP bills and credits, grouped by vendor."""
    print("Loading AP bills...")
    path = AP_DIR / "Open Bills and Credits.csv"
    rows = read_csv_safe(path)

    vendor_bills = defaultdict(list)
    seen_bills = set()
    skipped = 0

    for row in rows:
        vendor = row.get("Name", "").strip()
        doc_num = row.get("Document Number", "").strip()
        date = parse_date(row.get("Date", ""))
        doc_type = row.get("Type", "").strip()

        if not vendor or not date:
            skipped += 1
            continue

        # Dedup
        dedup_key = f"{vendor}|{doc_num}|{date.isoformat()}"
        if dedup_key in seen_bills:
            continue
        seen_bills.add(dedup_key)

        vendor_id = extract_vendor_id(vendor)

        bill = {
            "document_number": doc_num,
            "vendor": vendor,
            "vendor_id": vendor_id,
            "date": date,
            "type": doc_type,
            "status": row.get("Status", "").strip(),
            "subsidiary": row.get("Subsidiary", "").strip(),
            "department": row.get("Department", "").strip(),
            "currency": row.get("Currency", "").strip(),
        }

        vendor_bills[vendor_id].append(bill)

    total = sum(len(v) for v in vendor_bills.values())
    print(f"  Loaded {total:,} unique bills for {len(vendor_bills):,} vendors (skipped {skipped:,})")
    return vendor_bills


def reconstruct_o2c_cases(
    customer_orders: Dict[str, List[Dict]],
    customer_invoices: Dict[str, List[Dict]],
    customer_credits: Dict[str, List[Dict]],
    customer_rmas: Dict[str, List[Dict]],
    customer_deductions: Dict[str, List[Dict]]
) -> List[Dict]:
    """
    Reconstruct Order-to-Cash cases by customer.

    Case flow: Sales Order -> Invoice -> (Credit Memo / RMA / Deduction)
    """
    print("Reconstructing Order-to-Cash cases...")

    cases = []
    all_customers = set(customer_orders.keys()) | set(customer_invoices.keys())

    for customer_id in all_customers:
        orders = customer_orders.get(customer_id, [])
        invoices = customer_invoices.get(customer_id, [])
        credits = customer_credits.get(customer_id, [])
        rmas = customer_rmas.get(customer_id, [])
        deductions = customer_deductions.get(customer_id, [])

        # Need at least 2 events to form a case
        total_events = len(orders) + len(invoices) + len(credits) + len(rmas) + len(deductions)
        if total_events < 2:
            continue

        events = []

        # Add order events
        for order in orders:
            order_type = order["order_type"]
            if order_type == "Correction":
                activity = "order_correction"
            elif order_type == "New":
                activity = "sales_order_created"
            else:
                activity = "sales_order_" + (order_type.lower().replace(" ", "_") if order_type else "other")

            events.append({
                "activity": activity,
                "timestamp": order["date"].isoformat(),
                "resource": order["sales_rep"] or "system",
                "attributes": {
                    "document_number": order["document_number"],
                    "order_type": order_type,
                    "location": order["location"],
                    "credit_hold": order["credit_hold"],
                }
            })

        # Add invoice events
        for inv in invoices:
            reason = inv["reason_code"].lower().replace(" ", "_") if inv["reason_code"] else "standard"
            activity = f"invoice_issued_{reason}" if reason != "standard" else "invoice_issued"

            events.append({
                "activity": activity,
                "timestamp": inv["date"].isoformat(),
                "resource": "ar_system",
                "attributes": {
                    "reason_code": inv["reason_code"],
                    "terms": inv["terms"],
                }
            })

        # Add credit memo events
        for credit in credits:
            reason = credit["reason_code"].lower().replace("/", "_").replace(" ", "_") if credit["reason_code"] else "other"
            activity = f"credit_memo_{reason}" if len(reason) < 30 else "credit_memo_issued"

            events.append({
                "activity": activity,
                "timestamp": credit["date"].isoformat(),
                "resource": "ar_system",
                "attributes": {
                    "reason_code": credit["reason_code"],
                    "has_rma": bool(credit["rma_number"]),
                }
            })

        # Add RMA events
        for rma in rmas:
            rma_type = rma["rma_type"].lower().replace(" ", "_").replace("-", "_") if rma["rma_type"] else "standard"
            activity = f"rma_{rma_type}"

            events.append({
                "activity": activity,
                "timestamp": rma["date"].isoformat(),
                "resource": rma["sales_rep"] or "returns_team",
                "attributes": {
                    "rma_number": rma["rma_number"],
                    "rma_status": rma["status"],
                    "revenue_status": rma["revenue_status"],
                }
            })

        # Add deduction events
        for ded in deductions:
            ded_type = ded["reason_code"].lower().replace(" ", "_").replace("(", "").replace(")", "") if ded["reason_code"] else "other"
            activity = f"deduction_{ded_type}" if len(ded_type) < 30 else "deduction_claimed"

            events.append({
                "activity": activity,
                "timestamp": ded["date"].isoformat(),
                "resource": "deductions_team",
                "attributes": {
                    "cd_number": ded["cd_number"],
                    "deduction_status": ded["status"],
                }
            })

        # Sort events by timestamp
        events = sorted(events, key=lambda x: x["timestamp"])

        # Limit to first 100 events (very long sequences hurt training)
        events = events[:100]

        if len(events) < 2:
            continue

        # Determine outcomes
        has_rma = any("rma_" in e["activity"] for e in events)
        has_credit = any("credit_memo" in e["activity"] for e in events)
        has_deduction = any("deduction" in e["activity"] for e in events)
        has_correction = any(e["activity"] == "order_correction" for e in events)

        # Calculate duration
        first_ts = datetime.fromisoformat(events[0]["timestamp"])
        last_ts = datetime.fromisoformat(events[-1]["timestamp"])
        duration_hours = (last_ts - first_ts).total_seconds() / 3600

        # Get customer name from first available record
        customer_name = ""
        if orders:
            customer_name = orders[0]["customer"]
        elif invoices:
            customer_name = invoices[0]["customer"]

        case = {
            "caseId": f"o2c_{customer_id}",
            "events": events,
            "source": "sap_bsp669_o2c",
            "outcome": {
                "clean": not (has_rma or has_credit or has_deduction or has_correction),
                "hasRma": has_rma,
                "hasCredit": has_credit,
                "hasDeduction": has_deduction,
                "hasCorrection": has_correction,
                "eventCount": len(events),
                "durationHours": round(duration_hours, 2),
            },
            "caseAttributes": {
                "customer_id": customer_id,
                "customer_name": customer_name,
                "process_type": "order_to_cash",
            }
        }
        cases.append(case)

    print(f"  Reconstructed {len(cases):,} Order-to-Cash cases")
    return cases


def reconstruct_p2p_cases(vendor_bills: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Reconstruct Procure-to-Pay cases by vendor.

    Case flow: Bill Received -> (Payment / Credit)
    """
    print("Reconstructing Procure-to-Pay cases...")

    cases = []

    for vendor_id, bills in vendor_bills.items():
        # Need at least 2 bills to form interesting case
        if len(bills) < 2:
            continue

        events = []

        for bill in bills:
            bill_type = bill["type"].lower().replace(" ", "_") if bill["type"] else "bill"
            status = bill["status"].lower().replace(" ", "_") if bill["status"] else "open"

            # Determine activity based on type and status
            if bill_type == "bill":
                if "paid" in status:
                    activity = "bill_paid"
                else:
                    activity = "bill_received"
            elif "credit" in bill_type:
                activity = "vendor_credit_received"
            else:
                activity = f"ap_{bill_type}"

            events.append({
                "activity": activity,
                "timestamp": bill["date"].isoformat(),
                "resource": "ap_system",
                "attributes": {
                    "document_number": bill["document_number"],
                    "type": bill["type"],
                    "status": bill["status"],
                    "currency": bill["currency"],
                }
            })

        # Sort by timestamp
        events = sorted(events, key=lambda x: x["timestamp"])

        # Limit length
        events = events[:100]

        if len(events) < 2:
            continue

        # Calculate metrics
        paid_count = sum(1 for e in events if "paid" in e["activity"])
        credit_count = sum(1 for e in events if "credit" in e["activity"])

        first_ts = datetime.fromisoformat(events[0]["timestamp"])
        last_ts = datetime.fromisoformat(events[-1]["timestamp"])
        duration_hours = (last_ts - first_ts).total_seconds() / 3600

        case = {
            "caseId": f"p2p_{vendor_id.replace(' ', '_')[:30]}",
            "events": events,
            "source": "sap_bsp669_p2p",
            "outcome": {
                "allPaid": paid_count == len(events),
                "hasCredit": credit_count > 0,
                "paidRatio": round(paid_count / len(events), 4) if events else 0,
                "eventCount": len(events),
                "durationHours": round(duration_hours, 2),
            },
            "caseAttributes": {
                "vendor_id": vendor_id,
                "process_type": "procure_to_pay",
            }
        }
        cases.append(case)

    print(f"  Reconstructed {len(cases):,} Procure-to-Pay cases")
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
    resources = set()
    total_events = 0

    for case in cases:
        for event in case["events"]:
            activities[event["activity"]] += 1
            resources.add(event.get("resource", "unknown"))
            total_events += 1

    # Separate O2C and P2P stats
    o2c_cases = [c for c in cases if c["caseAttributes"]["process_type"] == "order_to_cash"]
    p2p_cases = [c for c in cases if c["caseAttributes"]["process_type"] == "procure_to_pay"]

    event_counts = [len(c["events"]) for c in cases]
    durations = [c["outcome"]["durationHours"] for c in cases]

    # O2C outcome stats
    o2c_clean = sum(1 for c in o2c_cases if c["outcome"].get("clean", False))
    o2c_rma = sum(1 for c in o2c_cases if c["outcome"].get("hasRma", False))
    o2c_credit = sum(1 for c in o2c_cases if c["outcome"].get("hasCredit", False))

    return {
        "total_cases": len(cases),
        "total_events": total_events,
        "o2c_cases": len(o2c_cases),
        "p2p_cases": len(p2p_cases),
        "activity_vocab_size": len(activities),
        "resource_vocab_size": len(resources),
        "activity_counts": dict(activities.most_common()),
        "o2c_outcome_stats": {
            "clean_rate": round(o2c_clean / len(o2c_cases), 4) if o2c_cases else 0,
            "rma_rate": round(o2c_rma / len(o2c_cases), 4) if o2c_cases else 0,
            "credit_rate": round(o2c_credit / len(o2c_cases), 4) if o2c_cases else 0,
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
        "source": "SAP BSP 669 Transaction Data Conversion (NetSuite 2013-2017)",
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
    print("SAP BSP 669 TRANSACTION DATA -> AETHER EVENT LOG PARSER")
    print("=" * 70)
    print()

    # Load all transaction data
    customer_orders = load_sales_orders()
    customer_invoices = load_invoices()
    customer_credits = load_credit_memos()
    customer_rmas = load_rmas()
    customer_deductions = load_deductions()
    vendor_bills = load_ap_bills()

    print()

    # Reconstruct cases
    o2c_cases = reconstruct_o2c_cases(
        customer_orders, customer_invoices, customer_credits, customer_rmas, customer_deductions
    )
    p2p_cases = reconstruct_p2p_cases(vendor_bills)

    # Combine all cases
    all_cases = o2c_cases + p2p_cases

    if not all_cases:
        print("ERROR: No cases reconstructed!")
        return

    # Split
    train_cases, val_cases = split_train_val(all_cases)

    # Build vocabulary
    vocab = build_vocabulary(all_cases)

    # Compute stats
    stats = compute_stats(all_cases)
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
    print(f"Saved metadata to metadata.json")

    # Print summary
    print()
    print("=" * 70)
    print("SAP BSP 669 DATASET SUMMARY")
    print("=" * 70)
    print(f"Total Cases: {stats['total_cases']:,}")
    print(f"  - Order-to-Cash: {stats['o2c_cases']:,}")
    print(f"  - Procure-to-Pay: {stats['p2p_cases']:,}")
    print(f"Train/Val Split: {stats['train_cases']:,} / {stats['val_cases']:,}")
    print(f"\nTotal Events: {stats['total_events']:,}")
    print(f"Unique Activities: {stats['activity_vocab_size']}")
    print(f"Unique Resources: {stats['resource_vocab_size']}")

    print(f"\nActivity Distribution (top 15):")
    for act, count in list(stats["activity_counts"].items())[:15]:
        print(f"  {act}: {count:,}")

    print(f"\nO2C Outcomes:")
    print(f"  Clean (no issues): {stats['o2c_outcome_stats']['clean_rate']:.1%}")
    print(f"  Has RMA: {stats['o2c_outcome_stats']['rma_rate']:.1%}")
    print(f"  Has Credit: {stats['o2c_outcome_stats']['credit_rate']:.1%}")

    print(f"\nSequence Lengths: min={stats['event_length_stats']['min']}, max={stats['event_length_stats']['max']}, mean={stats['event_length_stats']['mean']:.1f}, median={stats['event_length_stats']['median']}")
    print(f"Duration: mean={stats['duration_stats_hours']['mean']:.1f} hours, max={stats['duration_stats_hours']['max']:.1f} hours")


if __name__ == "__main__":
    main()
