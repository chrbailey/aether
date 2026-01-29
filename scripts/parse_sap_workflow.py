"""
Parse SAP Workflow Mining synthetic dataset into AETHER event log format.

Reconstructs Order-to-Cash and Procure-to-Pay process from:
  - SD (Sales & Distribution): orders.json, deliveries.json, invoices.json, doc_flows.json
  - MM (Materials Management): purchase_orders.json, goods_receipts.json, invoice_receipts.json, mm_doc_flows.json

Source: SAP-workflow-mining synthetic data generator
"""

import json
import random
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

# Data paths
SAP_DATA_ROOT = Path("/Volumes/OWC drive/Dev/SAP-workflow-mining/synthetic-data/sample_output")
OUTPUT_DIR = Path("/Volumes/OWC drive/Dev/aether/data/external/sap_workflow")

# SAP document type mappings
SD_DOC_TYPES = {
    "C": "sales_order",
    "J": "delivery",
    "M": "invoice",
}

MM_DOC_TYPES = {
    "F": "purchase_order",
    "E": "goods_receipt",
    "P": "invoice_receipt",
}


def parse_date(date_str: str, default_time: str = "00:00:00") -> Optional[datetime]:
    """Parse SAP date format (YYYY-MM-DD) with optional default time."""
    if not date_str or date_str.strip() == "":
        return None
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m-%d")
        if default_time:
            parts = default_time.split(":")
            if len(parts) >= 2:
                dt = dt.replace(
                    hour=int(parts[0]),
                    minute=int(parts[1]),
                    second=int(parts[2]) if len(parts) > 2 else 0,
                )
        return dt
    except ValueError:
        return None


def parse_datetime(date_str: str, time_str: str) -> Optional[datetime]:
    """Parse SAP date and time into datetime."""
    if not date_str:
        return None
    dt = parse_date(date_str)
    if dt and time_str:
        try:
            time_parts = time_str.strip().split(":")
            if len(time_parts) >= 2:
                hour = int(time_parts[0])
                minute = int(time_parts[1])
                second = int(time_parts[2]) if len(time_parts) > 2 else 0
                dt = dt.replace(hour=hour, minute=minute, second=second)
        except (ValueError, IndexError):
            pass
    return dt


def load_json(filename: str) -> List[Dict]:
    """Load JSON file from SAP data root."""
    path = SAP_DATA_ROOT / filename
    print(f"  Loading {filename}...")
    with open(path, "r") as f:
        data = json.load(f)
    print(f"    Loaded {len(data):,} records")
    return data


def build_document_index(
    orders: List[Dict],
    deliveries: List[Dict],
    invoices: List[Dict],
) -> Dict[str, Dict]:
    """Build index of all SD documents by document number."""
    index = {}

    for order in orders:
        doc_num = order.get("vbeln", "")
        if doc_num:
            index[doc_num] = {
                "type": "sales_order",
                "data": order,
                "timestamp": parse_datetime(order.get("erdat", ""), order.get("erzet", "")),
                "resource": order.get("ernam", "unknown"),
            }

    for delivery in deliveries:
        doc_num = delivery.get("vbeln", "")
        if doc_num:
            index[doc_num] = {
                "type": "delivery",
                "data": delivery,
                "timestamp": parse_date(delivery.get("erdat", ""), "12:00:00"),
                "resource": "logistics",
            }

    for invoice in invoices:
        doc_num = invoice.get("vbeln", "")
        if doc_num:
            index[doc_num] = {
                "type": "invoice",
                "data": invoice,
                "timestamp": parse_date(invoice.get("erdat", ""), "18:00:00"),
                "resource": "billing",
            }

    return index


def build_mm_document_index(
    purchase_orders: List[Dict],
    goods_receipts: List[Dict],
    invoice_receipts: List[Dict],
) -> Dict[str, Dict]:
    """Build index of all MM documents by document number."""
    index = {}

    for po in purchase_orders:
        doc_num = po.get("ebeln", "")
        if doc_num:
            index[doc_num] = {
                "type": "purchase_order",
                "data": po,
                "timestamp": parse_datetime(po.get("erdat", ""), po.get("erzet", "")),
                "resource": po.get("ernam", "unknown"),
            }

    for gr in goods_receipts:
        doc_num = gr.get("mblnr", "")
        if doc_num:
            index[doc_num] = {
                "type": "goods_receipt",
                "data": gr,
                "timestamp": parse_date(gr.get("budat", ""), "12:00:00"),
                "resource": gr.get("usnam", "unknown"),
            }

    for ir in invoice_receipts:
        doc_num = ir.get("belnr", "")
        if doc_num:
            index[doc_num] = {
                "type": "invoice_receipt",
                "data": ir,
                "timestamp": parse_date(ir.get("budat", ""), "18:00:00"),
                "resource": ir.get("usnam", "unknown"),
            }

    return index


def build_doc_flow_graph(doc_flows: List[Dict]) -> Dict[str, List[Dict]]:
    """Build graph of document flows: preceding_doc -> [subsequent_docs]."""
    graph = defaultdict(list)
    for flow in doc_flows:
        preceding = flow.get("vbelv", "")
        subsequent = flow.get("vbeln", "")
        if preceding and subsequent:
            graph[preceding].append({
                "doc": subsequent,
                "preceding_type": flow.get("vbtyp_v", ""),
                "subsequent_type": flow.get("vbtyp_n", ""),
                "date": flow.get("erdat", ""),
                "quantity": flow.get("rfmng", 0),
            })
    return graph


def trace_document_chain(
    start_doc: str,
    flow_graph: Dict[str, List[Dict]],
    doc_index: Dict[str, Dict],
) -> List[Dict]:
    """Trace the full document chain from a starting document."""
    events = []
    visited = set()
    queue = [start_doc]

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        if current in doc_index:
            doc_info = doc_index[current]
            if doc_info["timestamp"]:
                events.append({
                    "doc_num": current,
                    "activity": f"create_{doc_info['type']}",
                    "timestamp": doc_info["timestamp"],
                    "resource": doc_info["resource"],
                    "doc_type": doc_info["type"],
                    "data": doc_info["data"],
                })

        for next_doc in flow_graph.get(current, []):
            if next_doc["doc"] not in visited:
                queue.append(next_doc["doc"])

    return sorted(events, key=lambda x: x["timestamp"])


def reconstruct_sd_cases(
    orders: List[Dict],
    deliveries: List[Dict],
    invoices: List[Dict],
    doc_flows: List[Dict],
) -> List[Dict]:
    """Reconstruct Order-to-Cash process cases from SD documents."""
    print("Reconstructing Order-to-Cash cases...")

    doc_index = build_document_index(orders, deliveries, invoices)
    flow_graph = build_doc_flow_graph(doc_flows)

    cases = []

    for order in orders:
        order_num = order.get("vbeln", "")
        if not order_num:
            continue

        event_chain = trace_document_chain(order_num, flow_graph, doc_index)
        if len(event_chain) < 2:
            continue

        events = []
        for evt in event_chain:
            event = {
                "activity": evt["activity"],
                "timestamp": evt["timestamp"].isoformat(),
                "resource": evt["resource"],
                "attributes": {
                    "document_number": evt["doc_num"],
                    "document_type": evt["doc_type"],
                },
            }

            data = evt["data"]
            if evt["doc_type"] == "sales_order":
                event["attributes"]["order_type"] = data.get("auart", "")
                event["attributes"]["customer"] = data.get("kunnr", "")
                event["attributes"]["net_value"] = data.get("netwr", 0)
                event["attributes"]["currency"] = data.get("waerk", "")
                event["attributes"]["sales_org"] = data.get("vkorg", "")
            elif evt["doc_type"] == "delivery":
                event["attributes"]["actual_goods_issue"] = data.get("wadat_ist", "")
                event["attributes"]["planned_goods_issue"] = data.get("wadat", "")
            elif evt["doc_type"] == "invoice":
                event["attributes"]["net_value"] = data.get("netwr", 0)
                event["attributes"]["billing_date"] = data.get("fkdat", "")

            events.append(event)

        first_ts = datetime.fromisoformat(events[0]["timestamp"])
        last_ts = datetime.fromisoformat(events[-1]["timestamp"])
        duration_hours = (last_ts - first_ts).total_seconds() / 3600

        order_data = order
        requested_date = parse_date(order_data.get("vdatu", ""))

        has_delivery = any(e["activity"] == "create_delivery" for e in events)
        has_invoice = any(e["activity"] == "create_invoice" for e in events)

        on_time = True
        if requested_date and has_delivery:
            delivery_events = [e for e in events if e["activity"] == "create_delivery"]
            if delivery_events:
                delivery_ts = datetime.fromisoformat(delivery_events[0]["timestamp"])
                on_time = delivery_ts.date() <= requested_date.date()

        case = {
            "caseId": f"sap_sd_{order_num}",
            "events": events,
            "source": "sap_workflow_mining_synthetic",
            "processType": "order_to_cash",
            "outcome": {
                "onTime": on_time,
                "complete": has_delivery and has_invoice,
                "hasDelivery": has_delivery,
                "hasInvoice": has_invoice,
                "durationHours": round(duration_hours, 2),
            },
            "caseAttributes": {
                "customer": order_data.get("kunnr", ""),
                "salesOrg": order_data.get("vkorg", ""),
                "orderType": order_data.get("auart", ""),
                "netValue": order_data.get("netwr", 0),
                "currency": order_data.get("waerk", ""),
                "requestedDate": order_data.get("vdatu", ""),
            },
        }
        cases.append(case)

    print(f"  Reconstructed {len(cases):,} Order-to-Cash cases")
    return cases


def reconstruct_mm_cases(
    purchase_orders: List[Dict],
    goods_receipts: List[Dict],
    invoice_receipts: List[Dict],
    mm_doc_flows: List[Dict],
) -> List[Dict]:
    """Reconstruct Procure-to-Pay process cases from MM documents."""
    print("Reconstructing Procure-to-Pay cases...")

    doc_index = build_mm_document_index(purchase_orders, goods_receipts, invoice_receipts)
    flow_graph = build_doc_flow_graph(mm_doc_flows)

    cases = []

    for po in purchase_orders:
        po_num = po.get("ebeln", "")
        if not po_num:
            continue

        event_chain = trace_document_chain(po_num, flow_graph, doc_index)
        if len(event_chain) < 2:
            continue

        events = []
        for evt in event_chain:
            event = {
                "activity": evt["activity"],
                "timestamp": evt["timestamp"].isoformat(),
                "resource": evt["resource"],
                "attributes": {
                    "document_number": evt["doc_num"],
                    "document_type": evt["doc_type"],
                },
            }

            data = evt["data"]
            if evt["doc_type"] == "purchase_order":
                event["attributes"]["vendor"] = data.get("lifnr", "")
                event["attributes"]["purchasing_org"] = data.get("ekorg", "")
                event["attributes"]["purchasing_group"] = data.get("ekgrp", "")
            elif evt["doc_type"] == "goods_receipt":
                event["attributes"]["plant"] = data.get("werks", "")
            elif evt["doc_type"] == "invoice_receipt":
                event["attributes"]["company_code"] = data.get("bukrs", "")

            events.append(event)

        first_ts = datetime.fromisoformat(events[0]["timestamp"])
        last_ts = datetime.fromisoformat(events[-1]["timestamp"])
        duration_hours = (last_ts - first_ts).total_seconds() / 3600

        po_data = po
        requested_date = parse_date(po_data.get("eindt", ""))

        has_gr = any(e["activity"] == "create_goods_receipt" for e in events)
        has_ir = any(e["activity"] == "create_invoice_receipt" for e in events)

        on_time = True
        if requested_date and has_gr:
            gr_events = [e for e in events if e["activity"] == "create_goods_receipt"]
            if gr_events:
                gr_ts = datetime.fromisoformat(gr_events[0]["timestamp"])
                on_time = gr_ts.date() <= requested_date.date()

        case = {
            "caseId": f"sap_mm_{po_num}",
            "events": events,
            "source": "sap_workflow_mining_synthetic",
            "processType": "procure_to_pay",
            "outcome": {
                "onTime": on_time,
                "complete": has_gr and has_ir,
                "hasGoodsReceipt": has_gr,
                "hasInvoiceReceipt": has_ir,
                "durationHours": round(duration_hours, 2),
            },
            "caseAttributes": {
                "vendor": po_data.get("lifnr", ""),
                "purchasingOrg": po_data.get("ekorg", ""),
                "purchasingGroup": po_data.get("ekgrp", ""),
                "deliveryDate": po_data.get("eindt", ""),
            },
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
    from collections import Counter

    activities = Counter()
    resources = set()
    process_types = Counter()
    total_events = 0

    for case in cases:
        process_types[case.get("processType", "unknown")] += 1
        for event in case["events"]:
            activities[event["activity"]] += 1
            resources.add(event.get("resource", "unknown"))
            total_events += 1

    durations = [c["outcome"]["durationHours"] for c in cases]
    event_counts = [len(c["events"]) for c in cases]

    ontime_count = sum(1 for c in cases if c["outcome"]["onTime"])
    complete_count = sum(1 for c in cases if c["outcome"]["complete"])

    sd_cases = [c for c in cases if c.get("processType") == "order_to_cash"]
    mm_cases = [c for c in cases if c.get("processType") == "procure_to_pay"]

    return {
        "total_cases": len(cases),
        "total_events": total_events,
        "activity_vocab_size": len(activities),
        "resource_vocab_size": len(resources),
        "process_types": dict(process_types),
        "activity_counts": dict(activities.most_common()),
        "outcome_stats": {
            "ontime_rate": round(ontime_count / len(cases), 4) if cases else 0,
            "complete_rate": round(complete_count / len(cases), 4) if cases else 0,
        },
        "sd_stats": {
            "cases": len(sd_cases),
            "with_delivery": sum(1 for c in sd_cases if c["outcome"].get("hasDelivery")),
            "with_invoice": sum(1 for c in sd_cases if c["outcome"].get("hasInvoice")),
        },
        "mm_stats": {
            "cases": len(mm_cases),
            "with_goods_receipt": sum(1 for c in mm_cases if c["outcome"].get("hasGoodsReceipt")),
            "with_invoice_receipt": sum(1 for c in mm_cases if c["outcome"].get("hasInvoiceReceipt")),
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
        "source": "SAP Workflow Mining Synthetic Dataset",
    }


def split_train_val(
    cases: List[Dict],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Split cases into train and validation sets, stratified by process type."""
    random.seed(seed)

    sd_cases = [c for c in cases if c.get("processType") == "order_to_cash"]
    mm_cases = [c for c in cases if c.get("processType") == "procure_to_pay"]

    random.shuffle(sd_cases)
    random.shuffle(mm_cases)

    sd_split = int(len(sd_cases) * train_ratio)
    mm_split = int(len(mm_cases) * train_ratio)

    train_cases = sd_cases[:sd_split] + mm_cases[:mm_split]
    val_cases = sd_cases[sd_split:] + mm_cases[mm_split:]

    random.shuffle(train_cases)
    random.shuffle(val_cases)

    return train_cases, val_cases


def main():
    print("=" * 60)
    print("SAP WORKFLOW MINING -> AETHER EVENT LOG PARSER")
    print("=" * 60)

    # Load SD (Sales & Distribution) data
    print("\nLoading SD (Sales & Distribution) data...")
    orders = load_json("orders.json")
    deliveries = load_json("deliveries.json")
    invoices = load_json("invoices.json")
    doc_flows = load_json("doc_flows.json")

    # Load MM (Materials Management) data
    print("\nLoading MM (Materials Management) data...")
    purchase_orders = load_json("purchase_orders.json")
    goods_receipts = load_json("goods_receipts.json")
    invoice_receipts = load_json("invoice_receipts.json")
    mm_doc_flows = load_json("mm_doc_flows.json")

    # Reconstruct process cases
    print("\nReconstructing process cases...")
    sd_cases = reconstruct_sd_cases(orders, deliveries, invoices, doc_flows)
    mm_cases = reconstruct_mm_cases(purchase_orders, goods_receipts, invoice_receipts, mm_doc_flows)

    all_cases = sd_cases + mm_cases
    print(f"\nTotal cases: {len(all_cases):,}")

    if not all_cases:
        print("ERROR: No cases reconstructed!")
        return

    # Split into train/val
    train_cases, val_cases = split_train_val(all_cases)

    # Build vocabulary
    vocab = build_vocabulary(all_cases)

    # Compute statistics
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
    print("Saved metadata.json")

    # Print summary
    print("\n" + "=" * 60)
    print("SAP WORKFLOW DATASET SUMMARY")
    print("=" * 60)
    print(f"Cases: {stats['total_cases']:,} (train: {stats['train_cases']:,}, val: {stats['val_cases']:,})")
    print(f"Events: {stats['total_events']:,}")
    print(f"Activities: {stats['activity_vocab_size']}")
    print(f"Resources: {stats['resource_vocab_size']}")

    print(f"\nProcess Types:")
    for ptype, count in stats["process_types"].items():
        print(f"  {ptype}: {count:,}")

    print(f"\nSD (Order-to-Cash) Stats:")
    print(f"  Cases: {stats['sd_stats']['cases']:,}")
    print(f"  With delivery: {stats['sd_stats']['with_delivery']:,}")
    print(f"  With invoice: {stats['sd_stats']['with_invoice']:,}")

    print(f"\nMM (Procure-to-Pay) Stats:")
    print(f"  Cases: {stats['mm_stats']['cases']:,}")
    print(f"  With goods receipt: {stats['mm_stats']['with_goods_receipt']:,}")
    print(f"  With invoice receipt: {stats['mm_stats']['with_invoice_receipt']:,}")

    print(f"\nActivity distribution:")
    for act, count in list(stats["activity_counts"].items())[:10]:
        print(f"  {act}: {count:,}")

    print(f"\nOutcomes:")
    print(f"  On-time: {stats['outcome_stats']['ontime_rate']:.1%}")
    print(f"  Complete: {stats['outcome_stats']['complete_rate']:.1%}")

    print(f"\nSequence lengths: min={stats['event_length_stats']['min']}, max={stats['event_length_stats']['max']}, mean={stats['event_length_stats']['mean']:.1f}")
    print(f"Duration: mean={stats['duration_stats_hours']['mean']:.1f} hours")


if __name__ == "__main__":
    main()
