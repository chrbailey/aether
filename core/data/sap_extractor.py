"""
SAP IDES Event Log Extractor for AETHER.

Extracts Order-to-Cash (O2C) and Procure-to-Pay (P2P) event sequences
from a real SAP IDES SQLite database. Reconstructs process flows by
following document chains (VBFA for O2C, EBAN->EKKO->MSEG->BKPF for P2P).

Output format matches the AETHER case/event schema:
  - caseId: unique process instance identifier
  - events: list of activity/resource/timestamp/attributes dicts
  - outcome: onTime, rework, durationHours
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SAP Document Type Constants (VBTYP field in VBFA)
# ---------------------------------------------------------------------------
VBTYP_ORDER = "C"       # Sales Order
VBTYP_DELIVERY = "J"    # Delivery
VBTYP_INVOICE = "M"     # Invoice / Billing Doc
VBTYP_CREDIT_MEMO = "N" # Credit Memo
VBTYP_GOODS_MOVEMENT = "R"  # Material Document (WM transfer / goods issue)
VBTYP_GOODS_ISSUE = "Q"     # Goods Issue (material doc via delivery)
VBTYP_ACCOUNTING = "3"      # Accounting document (from invoice)

# Movement types in MSEG
BWART_GOODS_RECEIPT = "101"   # Goods receipt for PO
BWART_GOODS_ISSUE_PROD = "261"  # Goods issue for production
BWART_GOODS_ISSUE_DELIVERY = "601"  # Goods issue for delivery
BWART_REVERSAL_RECEIPT = "131"  # Reversal of goods receipt

# Accounting document types (BLART in BKPF)
BLART_CUSTOMER_INVOICE = "RV"  # Revenue / billing
BLART_CUSTOMER_PAYMENT = "DZ"  # Customer incoming payment
BLART_VENDOR_INVOICE = "RE"    # Vendor invoice (logistics)
BLART_VENDOR_PAYMENT = "KZ"    # Vendor payment
BLART_AUTO_PAYMENT = "ZP"      # Automatic payment run
BLART_GOODS_RECEIPT_ACCT = "WE"  # Goods receipt accounting
BLART_GOODS_ISSUE_ACCT = "WA"   # Goods issue accounting

# Material document types (BLART in MKPF)
MKPF_GOODS_ISSUE = "WA"
MKPF_GOODS_RECEIPT = "WE"
MKPF_DELIVERY_RELATED = "WL"  # Delivery-related material movement


# ---------------------------------------------------------------------------
# Timestamp Helpers
# ---------------------------------------------------------------------------

def _parse_sap_datetime(date_str: str | None, time_str: str | None) -> datetime | None:
    """Parse SAP date and time strings into a datetime.

    SAP IDES SQLite stores dates as 'YYYY-MM-DDTHH:MM:SS' (ISO-ish)
    and times with an epoch prefix '1970-01-01THH:MM:SS'.
    We extract the date portion from date_str and the time portion from time_str.
    """
    if not date_str or "TRIAL" in str(date_str):
        return None

    try:
        # Extract date part (first 10 chars: YYYY-MM-DD)
        date_part = str(date_str)[:10]
        dt = datetime.strptime(date_part, "%Y-%m-%d")
    except (ValueError, IndexError):
        return None

    if time_str and "TRIAL" not in str(time_str):
        try:
            # Time stored as '1970-01-01THH:MM:SS' -- extract HH:MM:SS
            time_part = str(time_str)
            if "T" in time_part:
                time_part = time_part.split("T")[1]
            parts = time_part.split(":")
            hour = int(parts[0])
            minute = int(parts[1])
            second = int(parts[2]) if len(parts) > 2 else 0
            dt = dt.replace(hour=hour, minute=minute, second=second)
        except (ValueError, IndexError):
            pass  # Keep date-only

    return dt


def _dt_to_iso(dt: datetime | None) -> str:
    """Convert datetime to ISO 8601 string, or return a fallback."""
    if dt is None:
        return "1970-01-01T00:00:00"
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _safe_float(val: Any) -> float:
    """Safely convert a value to float, defaulting to 0.0."""
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _safe_str(val: Any) -> str:
    """Safely convert to non-null string."""
    if val is None:
        return ""
    s = str(val).strip()
    if "TRIAL" in s:
        return ""
    return s


def _is_valid(val: Any) -> bool:
    """Check if a SAP field value is non-null and non-TRIAL."""
    if val is None:
        return False
    s = str(val).strip()
    return len(s) > 0 and "TRIAL" not in s


# ---------------------------------------------------------------------------
# Database Query Helpers
# ---------------------------------------------------------------------------

def _dict_factory(cursor: sqlite3.Cursor, row: tuple) -> dict:
    """sqlite3 row factory that returns dicts."""
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def _connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a read-only connection with dict row factory."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"SAP database not found: {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = _dict_factory
    return conn


# ---------------------------------------------------------------------------
# O2C Extraction
# ---------------------------------------------------------------------------

def extract_o2c_cases(db_path: str | Path) -> list[dict]:
    """Extract Order-to-Cash event sequences from SAP SQLite.

    Reconstruction logic:
    1. Start from each VBAK sales order
    2. Follow VBFA document flow to find:
       - C->J: Sales Order -> Delivery (LIKP)
       - J->R: Delivery -> Material Doc / Goods Issue (MKPF)
       - J->Q: Delivery -> Goods Issue (MKPF, alternate path)
       - C->M / J->M: Order/Delivery -> Invoice (VBRK)
       - M->3: Invoice -> Accounting Doc (BKPF)
       - C->R: Order -> Material Doc (goods movement)
       - C->N / J->N: Credit Memo
    3. For each document, create an event with activity, resource, timestamp
    4. Look for payment via BSEG clearing on the customer account
    """
    conn = _connect(db_path)
    cases: list[dict] = []

    try:
        # Load all sales orders
        orders = conn.execute(
            "SELECT vbeln, erdat, erzet, ernam, netwr, kunnr "
            "FROM vbak WHERE vbeln IS NOT NULL"
        ).fetchall()

        # Pre-load VBFA flow records indexed by predecessor doc
        all_vbfa = conn.execute(
            "SELECT vbelv, posnv, vbeln, posnn, vbtyp_v, vbtyp_n, erdat, erzet "
            "FROM vbfa"
        ).fetchall()

        # Build lookup: vbelv -> list of flow records
        vbfa_by_pred: dict[str, list[dict]] = {}
        for row in all_vbfa:
            key = _safe_str(row["vbelv"])
            if key:
                vbfa_by_pred.setdefault(key, []).append(row)

        # Pre-load reference tables as lookups
        deliveries = {
            _safe_str(r["vbeln"]): r
            for r in conn.execute(
                "SELECT vbeln, ernam, erdat, erzet, netwr, kunnr FROM likp"
            ).fetchall()
            if _is_valid(r["vbeln"])
        }

        invoices = {
            _safe_str(r["vbeln"]): r
            for r in conn.execute(
                "SELECT vbeln, ernam, erdat, erzet, netwr, fksto, vbtyp FROM vbrk"
            ).fetchall()
            if _is_valid(r["vbeln"])
        }

        material_docs = {
            _safe_str(r["mblnr"]): r
            for r in conn.execute(
                "SELECT mblnr, mjahr, cpudt, cputm, usnam, blart FROM mkpf"
            ).fetchall()
            if _is_valid(r["mblnr"])
        }

        # Pre-load accounting docs by document number
        acct_docs = {
            _safe_str(r["belnr"]): r
            for r in conn.execute(
                "SELECT bukrs, belnr, gjahr, blart, cpudt, cputm, usnam, tcode, awtyp "
                "FROM bkpf"
            ).fetchall()
            if _is_valid(r["belnr"])
        }

        # Pre-load customer payment info from BSEG
        # Payment = clearing doc (augbl) on customer line items
        customer_payments: dict[str, dict] = {}
        payment_rows = conn.execute(
            "SELECT bseg.kunnr, bseg.augbl, bseg.augdt, bseg.dmbtr, bseg.vbeln "
            "FROM bseg "
            "WHERE bseg.kunnr IS NOT NULL AND length(bseg.kunnr) > 0 "
            "AND bseg.augbl IS NOT NULL AND length(bseg.augbl) > 0"
        ).fetchall()
        for pr in payment_rows:
            kunnr = _safe_str(pr["kunnr"])
            augbl = _safe_str(pr["augbl"])
            if kunnr and augbl:
                customer_payments.setdefault(kunnr, []).append(pr)

        for order in orders:
            order_num = _safe_str(order["vbeln"])
            if not order_num:
                continue

            events: list[dict] = []
            seen_docs: set[str] = set()
            has_credit_memo = False
            has_reversal = False

            # Event 1: Create Sales Order
            order_dt = _parse_sap_datetime(order["erdat"], order["erzet"])
            events.append({
                "activity": "Create Sales Order",
                "resource": _safe_str(order["ernam"]) or "SYSTEM",
                "timestamp": _dt_to_iso(order_dt),
                "attributes": {
                    "amount": _safe_float(order["netwr"]),
                    "salesOrder": order_num,
                    "customer": _safe_str(order["kunnr"]),
                },
            })

            # Follow VBFA chain from this order
            flow_from_order = vbfa_by_pred.get(order_num, [])

            # Collect delivery numbers
            delivery_nums: list[str] = []
            invoice_nums: list[str] = []

            for flow in flow_from_order:
                if flow["vbtyp_v"] != VBTYP_ORDER:
                    continue
                succ_type = _safe_str(flow["vbtyp_n"])
                succ_num = _safe_str(flow["vbeln"])

                if not succ_num or succ_num in seen_docs:
                    continue

                if succ_type == VBTYP_DELIVERY:
                    # Order -> Delivery
                    seen_docs.add(succ_num)
                    delivery_nums.append(succ_num)
                    dlv = deliveries.get(succ_num)
                    if dlv:
                        dlv_dt = _parse_sap_datetime(dlv["erdat"], dlv["erzet"])
                        events.append({
                            "activity": "Create Delivery",
                            "resource": _safe_str(dlv["ernam"]) or "SYSTEM",
                            "timestamp": _dt_to_iso(dlv_dt),
                            "attributes": {
                                "amount": _safe_float(dlv["netwr"]),
                                "deliveryDoc": succ_num,
                            },
                        })
                    else:
                        # Use VBFA timestamp as fallback
                        flow_dt = _parse_sap_datetime(flow["erdat"], flow["erzet"])
                        events.append({
                            "activity": "Create Delivery",
                            "resource": "SYSTEM",
                            "timestamp": _dt_to_iso(flow_dt),
                            "attributes": {"deliveryDoc": succ_num},
                        })

                elif succ_type == VBTYP_INVOICE:
                    # Order -> Invoice (direct)
                    seen_docs.add(succ_num)
                    invoice_nums.append(succ_num)
                    inv = invoices.get(succ_num)
                    if inv:
                        inv_dt = _parse_sap_datetime(inv["erdat"], inv["erzet"])
                        events.append({
                            "activity": "Create Invoice",
                            "resource": _safe_str(inv["ernam"]) or "SYSTEM",
                            "timestamp": _dt_to_iso(inv_dt),
                            "attributes": {
                                "amount": _safe_float(inv["netwr"]),
                                "billingDoc": succ_num,
                            },
                        })
                        if _safe_str(inv.get("fksto")) == "X":
                            has_reversal = True

                elif succ_type == VBTYP_GOODS_MOVEMENT:
                    # Order -> Material Doc (goods movement)
                    seen_docs.add(succ_num)
                    mdoc = material_docs.get(succ_num)
                    if mdoc:
                        md_dt = _parse_sap_datetime(mdoc["cpudt"], mdoc["cputm"])
                        activity = "Goods Issue" if mdoc.get("blart") in (MKPF_GOODS_ISSUE, MKPF_DELIVERY_RELATED) else "Goods Movement"
                        events.append({
                            "activity": activity,
                            "resource": _safe_str(mdoc["usnam"]) or "SYSTEM",
                            "timestamp": _dt_to_iso(md_dt),
                            "attributes": {"materialDoc": succ_num},
                        })

                elif succ_type == VBTYP_CREDIT_MEMO:
                    # Credit memo -> indicates rework
                    seen_docs.add(succ_num)
                    has_credit_memo = True
                    inv = invoices.get(succ_num)
                    if inv:
                        cm_dt = _parse_sap_datetime(inv["erdat"], inv["erzet"])
                        events.append({
                            "activity": "Create Credit Memo",
                            "resource": _safe_str(inv["ernam"]) or "SYSTEM",
                            "timestamp": _dt_to_iso(cm_dt),
                            "attributes": {
                                "amount": _safe_float(inv["netwr"]),
                                "creditMemoDoc": succ_num,
                            },
                        })

            # Follow from deliveries -> goods issue, invoice, accounting
            for dlv_num in delivery_nums:
                flow_from_dlv = vbfa_by_pred.get(dlv_num, [])
                for flow in flow_from_dlv:
                    if flow["vbtyp_v"] != VBTYP_DELIVERY:
                        continue
                    succ_type = _safe_str(flow["vbtyp_n"])
                    succ_num = _safe_str(flow["vbeln"])

                    if not succ_num or succ_num in seen_docs:
                        continue

                    if succ_type == VBTYP_INVOICE:
                        # Delivery -> Invoice
                        seen_docs.add(succ_num)
                        if succ_num not in invoice_nums:
                            invoice_nums.append(succ_num)
                        inv = invoices.get(succ_num)
                        if inv:
                            inv_dt = _parse_sap_datetime(inv["erdat"], inv["erzet"])
                            events.append({
                                "activity": "Create Invoice",
                                "resource": _safe_str(inv["ernam"]) or "SYSTEM",
                                "timestamp": _dt_to_iso(inv_dt),
                                "attributes": {
                                    "amount": _safe_float(inv["netwr"]),
                                    "billingDoc": succ_num,
                                },
                            })
                            if _safe_str(inv.get("fksto")) == "X":
                                has_reversal = True

                    elif succ_type == VBTYP_GOODS_MOVEMENT:
                        # Delivery -> Material Doc
                        seen_docs.add(succ_num)
                        mdoc = material_docs.get(succ_num)
                        if mdoc:
                            md_dt = _parse_sap_datetime(mdoc["cpudt"], mdoc["cputm"])
                            events.append({
                                "activity": "Goods Issue",
                                "resource": _safe_str(mdoc["usnam"]) or "SYSTEM",
                                "timestamp": _dt_to_iso(md_dt),
                                "attributes": {"materialDoc": succ_num},
                            })

                    elif succ_type == VBTYP_GOODS_ISSUE:
                        # Delivery -> Goods Issue (Q type, ref is mat doc number)
                        seen_docs.add(succ_num)
                        # Q-type vbeln may be a material doc number
                        mdoc = material_docs.get(succ_num)
                        flow_dt = _parse_sap_datetime(flow["erdat"], flow["erzet"])
                        if mdoc:
                            md_dt = _parse_sap_datetime(mdoc["cpudt"], mdoc["cputm"])
                            events.append({
                                "activity": "Goods Issue",
                                "resource": _safe_str(mdoc["usnam"]) or "SYSTEM",
                                "timestamp": _dt_to_iso(md_dt or flow_dt),
                                "attributes": {"materialDoc": succ_num},
                            })
                        else:
                            events.append({
                                "activity": "Goods Issue",
                                "resource": "SYSTEM",
                                "timestamp": _dt_to_iso(flow_dt),
                                "attributes": {"materialDoc": succ_num},
                            })

                    elif succ_type == VBTYP_CREDIT_MEMO:
                        seen_docs.add(succ_num)
                        has_credit_memo = True
                        inv = invoices.get(succ_num)
                        if inv:
                            cm_dt = _parse_sap_datetime(inv["erdat"], inv["erzet"])
                            events.append({
                                "activity": "Create Credit Memo",
                                "resource": _safe_str(inv["ernam"]) or "SYSTEM",
                                "timestamp": _dt_to_iso(cm_dt),
                                "attributes": {
                                    "amount": _safe_float(inv["netwr"]),
                                    "creditMemoDoc": succ_num,
                                },
                            })

            # Follow from invoices -> accounting doc (M->3 in VBFA)
            for inv_num in invoice_nums:
                flow_from_inv = vbfa_by_pred.get(inv_num, [])
                for flow in flow_from_inv:
                    if flow["vbtyp_v"] != VBTYP_INVOICE:
                        continue
                    succ_type = _safe_str(flow["vbtyp_n"])
                    succ_num = _safe_str(flow["vbeln"])

                    if not succ_num or succ_num in seen_docs:
                        continue

                    if succ_type == VBTYP_ACCOUNTING:
                        # Invoice -> Accounting Document
                        seen_docs.add(succ_num)
                        flow_dt = _parse_sap_datetime(flow["erdat"], flow["erzet"])
                        events.append({
                            "activity": "Post Accounting Document",
                            "resource": "SYSTEM",
                            "timestamp": _dt_to_iso(flow_dt),
                            "attributes": {"accountingDoc": succ_num},
                        })

            # Look for accounting docs linked to billing via BKPF.awtyp='VBRK'
            # These are the RV (revenue) postings
            for inv_num in invoice_nums:
                for belnr, adoc in acct_docs.items():
                    if belnr in seen_docs:
                        continue
                    if _safe_str(adoc.get("awtyp")) == "VBRK" and _safe_str(adoc.get("blart")) == BLART_CUSTOMER_INVOICE:
                        # Check if this accounting doc references our invoice
                        # awkey format is typically the billing doc number + company code + year
                        awkey_check = conn.execute(
                            "SELECT belnr FROM bkpf WHERE awtyp='VBRK' AND awkey LIKE ? AND belnr=?",
                            (f"{inv_num}%", belnr)
                        ).fetchone()
                        if awkey_check:
                            seen_docs.add(belnr)
                            ad_dt = _parse_sap_datetime(adoc["cpudt"], adoc["cputm"])
                            events.append({
                                "activity": "Post Accounting Document",
                                "resource": _safe_str(adoc["usnam"]) or "SYSTEM",
                                "timestamp": _dt_to_iso(ad_dt),
                                "attributes": {
                                    "accountingDoc": belnr,
                                    "docType": BLART_CUSTOMER_INVOICE,
                                },
                            })

            # Look for customer payment
            customer_num = _safe_str(order.get("kunnr"))
            if customer_num and customer_num in customer_payments:
                # Find payments for this customer that occurred after the order
                for pay in customer_payments[customer_num]:
                    augbl = _safe_str(pay["augbl"])
                    if not augbl or augbl in seen_docs:
                        continue
                    pay_dt = _parse_sap_datetime(pay["augdt"], None)
                    if pay_dt and order_dt and pay_dt >= order_dt:
                        # Check if the clearing doc is a payment type
                        clearing_doc = acct_docs.get(augbl)
                        if clearing_doc and _safe_str(clearing_doc.get("blart")) in (
                            BLART_CUSTOMER_PAYMENT, BLART_AUTO_PAYMENT
                        ):
                            seen_docs.add(augbl)
                            cd_dt = _parse_sap_datetime(
                                clearing_doc["cpudt"], clearing_doc["cputm"]
                            )
                            events.append({
                                "activity": "Receive Payment",
                                "resource": _safe_str(clearing_doc["usnam"]) or "SYSTEM",
                                "timestamp": _dt_to_iso(cd_dt or pay_dt),
                                "attributes": {
                                    "amount": _safe_float(pay["dmbtr"]),
                                    "paymentDoc": augbl,
                                },
                            })
                            break  # One payment event per case

            # Sort events by timestamp
            events.sort(key=lambda e: e["timestamp"])

            # Skip cases with only the initial order creation (no downstream docs)
            if len(events) < 2:
                continue

            # Compute outcome
            first_ts = _parse_sap_datetime(events[0]["timestamp"], None)
            last_ts = _parse_sap_datetime(events[-1]["timestamp"], None)
            duration_hours = 0.0
            if first_ts and last_ts:
                delta = last_ts - first_ts
                duration_hours = round(delta.total_seconds() / 3600.0, 2)

            # On-time: VBAK doesn't have VDATU in this schema, so we approximate
            # using whether the case completed within 30 days (common SAP benchmark)
            on_time = duration_hours <= 30 * 24  # 30 days

            rework = has_credit_memo or has_reversal

            cases.append({
                "caseId": f"O2C_{order_num}",
                "events": events,
                "outcome": {
                    "onTime": on_time,
                    "rework": rework,
                    "durationHours": duration_hours,
                },
            })

    finally:
        conn.close()

    logger.info("Extracted %d O2C cases from %s", len(cases), db_path)
    return cases


# ---------------------------------------------------------------------------
# P2P Extraction
# ---------------------------------------------------------------------------

def extract_p2p_cases(db_path: str | Path) -> list[dict]:
    """Extract Procure-to-Pay event sequences from SAP SQLite.

    Reconstruction logic (PO-centric approach):
    In IDES data, PR->PO links are sparse. Instead we reconstruct P2P
    from the PO perspective using accounting line items (BSEG.ebeln):

    1. Collect all PO numbers referenced in BSEG (accounting line items)
    2. For each PO, find EKKO header if available (Create Purchase Order)
    3. Find goods receipt events via BSEG/BKPF with blart=WE (goods receipt acct)
    4. Find invoice receipt events via BSEG/BKPF with blart=RE or awtyp=RMRP
    5. Find payment events via clearing docs (BSEG.augbl -> BKPF with blart=ZP/KZ)
    6. Optionally link back to EBAN purchase requisitions via EKPO.banfn
    """
    conn = _connect(db_path)
    cases: list[dict] = []

    try:
        # Load PO headers (may not cover all POs in BSEG)
        po_headers = {
            _safe_str(r["ebeln"]): r
            for r in conn.execute(
                "SELECT ebeln, ernam, bedat, lifnr, ekorg, waers FROM ekko"
            ).fetchall()
            if _is_valid(r["ebeln"])
        }

        # Load PO items
        po_items_by_po: dict[str, list[dict]] = {}
        for item in conn.execute(
            "SELECT ebeln, ebelp, matnr, menge, netwr, banfn, bnfpo FROM ekpo"
        ).fetchall():
            ebeln = _safe_str(item.get("ebeln"))
            if ebeln:
                po_items_by_po.setdefault(ebeln, []).append(item)

        # Load purchase requisitions for back-linking
        pr_by_num: dict[str, dict] = {}
        for item in conn.execute(
            "SELECT banfn, bnfpo, ernam, erdat, matnr, menge, preis "
            "FROM eban WHERE banfn IS NOT NULL"
        ).fetchall():
            banfn = _safe_str(item["banfn"])
            if banfn and banfn not in pr_by_num:
                pr_by_num[banfn] = item

        # Load ALL accounting doc headers
        acct_docs: dict[str, dict] = {}
        for r in conn.execute(
            "SELECT bukrs, belnr, gjahr, blart, cpudt, cputm, usnam, tcode, awtyp "
            "FROM bkpf"
        ).fetchall():
            belnr = _safe_str(r["belnr"])
            if belnr:
                acct_docs[belnr] = r

        # Collect all PO-linked accounting line items from BSEG
        # Group by PO number, with the accounting doc details
        po_acct_items: dict[str, list[dict]] = {}
        bseg_rows = conn.execute(
            "SELECT bseg.bukrs, bseg.belnr, bseg.gjahr, bseg.ebeln, "
            "bseg.lifnr, bseg.dmbtr, bseg.augbl, bseg.augdt "
            "FROM bseg "
            "WHERE bseg.ebeln IS NOT NULL AND length(bseg.ebeln) > 0"
        ).fetchall()

        for row in bseg_rows:
            ebeln = _safe_str(row.get("ebeln"))
            if ebeln:
                po_acct_items.setdefault(ebeln, []).append(row)

        # Get the set of all PO numbers we can build cases for
        all_po_nums = set(po_acct_items.keys())
        logger.info("Found %d distinct POs in accounting line items", len(all_po_nums))

        # Process each PO as a case
        for po_num in sorted(all_po_nums):
            events: list[dict] = []
            seen_docs: set[str] = set()
            has_reversal = False

            # Try to find a linked PR (via EKPO.banfn)
            pr_banfn: str = ""
            for poi in po_items_by_po.get(po_num, []):
                banfn = _safe_str(poi.get("banfn"))
                if banfn and banfn in pr_by_num:
                    pr_banfn = banfn
                    break

            # Event: Create Purchase Requisition (if linked)
            if pr_banfn:
                pr_data = pr_by_num[pr_banfn]
                pr_dt = _parse_sap_datetime(pr_data["erdat"], None)
                events.append({
                    "activity": "Create Purchase Requisition",
                    "resource": _safe_str(pr_data["ernam"]) or "SYSTEM",
                    "timestamp": _dt_to_iso(pr_dt),
                    "attributes": {
                        "amount": _safe_float(pr_data.get("preis", 0)) * _safe_float(pr_data.get("menge", 0)),
                        "quantity": _safe_float(pr_data.get("menge")),
                        "requisition": pr_banfn,
                        "material": _safe_str(pr_data.get("matnr")),
                    },
                })

            # Event: Create Purchase Order (from EKKO if available)
            po_hdr = po_headers.get(po_num)
            if po_hdr:
                po_dt = _parse_sap_datetime(po_hdr["bedat"], None)
                po_line_items = po_items_by_po.get(po_num, [])
                po_value = sum(_safe_float(li.get("netwr", 0)) for li in po_line_items)
                po_qty = sum(_safe_float(li.get("menge", 0)) for li in po_line_items)
                events.append({
                    "activity": "Create Purchase Order",
                    "resource": _safe_str(po_hdr["ernam"]) or "SYSTEM",
                    "timestamp": _dt_to_iso(po_dt),
                    "attributes": {
                        "amount": round(po_value, 2),
                        "quantity": po_qty,
                        "purchaseOrder": po_num,
                        "vendor": _safe_str(po_hdr.get("lifnr")),
                    },
                })

            # Derive events from accounting documents linked to this PO
            # Categorize each BSEG line by its BKPF doc type
            acct_lines = po_acct_items.get(po_num, [])

            # Collect unique accounting docs and their types
            po_belnrs: dict[str, dict] = {}  # belnr -> first BSEG row
            for line in acct_lines:
                belnr = _safe_str(line.get("belnr"))
                if belnr and belnr not in po_belnrs:
                    po_belnrs[belnr] = line

            for belnr, bseg_line in po_belnrs.items():
                if belnr in seen_docs:
                    continue

                adoc = acct_docs.get(belnr)
                if not adoc:
                    continue

                blart = _safe_str(adoc.get("blart"))
                awtyp = _safe_str(adoc.get("awtyp"))
                doc_dt = _parse_sap_datetime(adoc["cpudt"], adoc["cputm"])
                resource = _safe_str(adoc["usnam"]) or "SYSTEM"
                amount = _safe_float(bseg_line.get("dmbtr"))

                # Map accounting doc type to P2P activity
                if blart == BLART_GOODS_RECEIPT_ACCT:
                    # WE = Goods Receipt posting
                    seen_docs.add(belnr)
                    events.append({
                        "activity": "Goods Receipt",
                        "resource": resource,
                        "timestamp": _dt_to_iso(doc_dt),
                        "attributes": {
                            "amount": amount,
                            "accountingDoc": belnr,
                            "docType": blart,
                        },
                    })

                elif blart == BLART_VENDOR_INVOICE or (blart == "RE" and awtyp == "RMRP"):
                    # RE with RMRP = Logistics Invoice Verification
                    seen_docs.add(belnr)
                    events.append({
                        "activity": "Invoice Receipt",
                        "resource": resource,
                        "timestamp": _dt_to_iso(doc_dt),
                        "attributes": {
                            "amount": amount,
                            "accountingDoc": belnr,
                            "docType": blart,
                        },
                    })

                elif blart == "KR":
                    # KR = Vendor Invoice (manual)
                    seen_docs.add(belnr)
                    events.append({
                        "activity": "Invoice Receipt",
                        "resource": resource,
                        "timestamp": _dt_to_iso(doc_dt),
                        "attributes": {
                            "amount": amount,
                            "accountingDoc": belnr,
                            "docType": blart,
                        },
                    })

                elif blart == BLART_GOODS_ISSUE_ACCT:
                    # WA = Goods Issue posting
                    seen_docs.add(belnr)
                    events.append({
                        "activity": "Goods Issue",
                        "resource": resource,
                        "timestamp": _dt_to_iso(doc_dt),
                        "attributes": {
                            "amount": amount,
                            "accountingDoc": belnr,
                            "docType": blart,
                        },
                    })

                elif blart in (BLART_VENDOR_PAYMENT, BLART_AUTO_PAYMENT):
                    # ZP/KZ = Payment
                    seen_docs.add(belnr)
                    events.append({
                        "activity": "Payment",
                        "resource": resource,
                        "timestamp": _dt_to_iso(doc_dt),
                        "attributes": {
                            "amount": amount,
                            "paymentDoc": belnr,
                        },
                    })

                elif blart == "SA":
                    # SA = G/L Account posting (settlement)
                    seen_docs.add(belnr)
                    events.append({
                        "activity": "Post Accounting Document",
                        "resource": resource,
                        "timestamp": _dt_to_iso(doc_dt),
                        "attributes": {
                            "amount": amount,
                            "accountingDoc": belnr,
                            "docType": blart,
                        },
                    })

                elif blart == "WL":
                    # WL = Goods Issue for Delivery
                    seen_docs.add(belnr)
                    events.append({
                        "activity": "Goods Issue",
                        "resource": resource,
                        "timestamp": _dt_to_iso(doc_dt),
                        "attributes": {
                            "amount": amount,
                            "accountingDoc": belnr,
                            "docType": blart,
                        },
                    })

            # Check for payment via clearing (augbl on vendor line items)
            for line in acct_lines:
                augbl = _safe_str(line.get("augbl"))
                if not augbl or augbl in seen_docs:
                    continue
                clearing_doc = acct_docs.get(augbl)
                if clearing_doc and _safe_str(clearing_doc.get("blart")) in (
                    BLART_VENDOR_PAYMENT, BLART_AUTO_PAYMENT
                ):
                    seen_docs.add(augbl)
                    pay_dt = _parse_sap_datetime(
                        clearing_doc["cpudt"], clearing_doc["cputm"]
                    )
                    events.append({
                        "activity": "Payment",
                        "resource": _safe_str(clearing_doc["usnam"]) or "SYSTEM",
                        "timestamp": _dt_to_iso(pay_dt),
                        "attributes": {
                            "amount": _safe_float(line.get("dmbtr")),
                            "paymentDoc": augbl,
                        },
                    })
                    break  # One payment per case

            # Sort events by timestamp
            events.sort(key=lambda e: e["timestamp"])

            # Skip cases with fewer than 2 events
            if len(events) < 2:
                continue

            # Deduplicate events with same activity + timestamp
            deduped: list[dict] = []
            seen_keys: set[str] = set()
            for evt in events:
                key = f"{evt['activity']}_{evt['timestamp']}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    deduped.append(evt)
            events = deduped

            if len(events) < 2:
                continue

            # Check for rework indicators
            activity_names = [e["activity"] for e in events]
            has_reversal = any("Reverse" in a for a in activity_names)
            # Multiple goods receipts or invoices also indicate rework
            gr_count = sum(1 for a in activity_names if a == "Goods Receipt")
            inv_count = sum(1 for a in activity_names if a == "Invoice Receipt")
            if gr_count > 1 or inv_count > 1:
                has_reversal = True

            # Compute outcome
            first_ts = _parse_sap_datetime(events[0]["timestamp"], None)
            last_ts = _parse_sap_datetime(events[-1]["timestamp"], None)
            duration_hours = 0.0
            if first_ts and last_ts:
                delta = last_ts - first_ts
                duration_hours = round(delta.total_seconds() / 3600.0, 2)

            # On-time: typical P2P benchmark is completion within 45 days
            on_time = duration_hours <= 45 * 24

            cases.append({
                "caseId": f"P2P_{po_num}",
                "events": events,
                "outcome": {
                    "onTime": on_time,
                    "rework": has_reversal,
                    "durationHours": duration_hours,
                },
            })

    finally:
        conn.close()

    logger.info("Extracted %d P2P cases from %s", len(cases), db_path)
    return cases


# ---------------------------------------------------------------------------
# Combined Extraction
# ---------------------------------------------------------------------------

def extract_all_cases(db_path: str | Path) -> list[dict]:
    """Extract both O2C and P2P cases."""
    o2c = extract_o2c_cases(db_path)
    p2p = extract_p2p_cases(db_path)
    all_cases = o2c + p2p
    logger.info(
        "Extracted %d total cases (%d O2C, %d P2P)",
        len(all_cases), len(o2c), len(p2p),
    )
    return all_cases


def save_cases(cases: list[dict], output_path: str | Path) -> None:
    """Save cases to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d cases to %s", len(cases), path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(cases: list[dict]) -> None:
    """Print a summary of extracted cases."""
    o2c_cases = [c for c in cases if c["caseId"].startswith("O2C_")]
    p2p_cases = [c for c in cases if c["caseId"].startswith("P2P_")]

    print(f"\n{'='*60}")
    print(f"SAP IDES Event Log Extraction Summary")
    print(f"{'='*60}")
    print(f"Total cases:    {len(cases)}")
    print(f"  O2C cases:    {len(o2c_cases)}")
    print(f"  P2P cases:    {len(p2p_cases)}")

    total_events = sum(len(c["events"]) for c in cases)
    print(f"Total events:   {total_events}")

    if o2c_cases:
        o2c_events = sum(len(c["events"]) for c in o2c_cases)
        o2c_avg = o2c_events / len(o2c_cases)
        o2c_rework = sum(1 for c in o2c_cases if c["outcome"]["rework"])
        o2c_ontime = sum(1 for c in o2c_cases if c["outcome"]["onTime"])
        print(f"\nO2C Details:")
        print(f"  Avg events/case:  {o2c_avg:.1f}")
        print(f"  On-time:          {o2c_ontime}/{len(o2c_cases)} ({100*o2c_ontime/len(o2c_cases):.0f}%)")
        print(f"  Rework/reversal:  {o2c_rework}/{len(o2c_cases)} ({100*o2c_rework/len(o2c_cases):.0f}%)")

        # Activity distribution
        o2c_activities: dict[str, int] = {}
        for c in o2c_cases:
            for e in c["events"]:
                o2c_activities[e["activity"]] = o2c_activities.get(e["activity"], 0) + 1
        print(f"  Activity counts:")
        for act, cnt in sorted(o2c_activities.items(), key=lambda x: -x[1]):
            print(f"    {act}: {cnt}")

    if p2p_cases:
        p2p_events = sum(len(c["events"]) for c in p2p_cases)
        p2p_avg = p2p_events / len(p2p_cases)
        p2p_rework = sum(1 for c in p2p_cases if c["outcome"]["rework"])
        p2p_ontime = sum(1 for c in p2p_cases if c["outcome"]["onTime"])
        print(f"\nP2P Details:")
        print(f"  Avg events/case:  {p2p_avg:.1f}")
        print(f"  On-time:          {p2p_ontime}/{len(p2p_cases)} ({100*p2p_ontime/len(p2p_cases):.0f}%)")
        print(f"  Rework/reversal:  {p2p_rework}/{len(p2p_cases)} ({100*p2p_rework/len(p2p_cases):.0f}%)")

        p2p_activities: dict[str, int] = {}
        for c in p2p_cases:
            for e in c["events"]:
                p2p_activities[e["activity"]] = p2p_activities.get(e["activity"], 0) + 1
        print(f"  Activity counts:")
        for act, cnt in sorted(p2p_activities.items(), key=lambda x: -x[1]):
            print(f"    {act}: {cnt}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    default_db = Path("/Volumes/OWC drive/Dev/sap-extractor/sap.sqlite")
    db = Path(sys.argv[1]) if len(sys.argv) > 1 else default_db

    if not db.exists():
        print(f"ERROR: Database not found at {db}")
        sys.exit(1)

    print(f"Extracting from: {db}")

    cases = extract_all_cases(db)
    _print_summary(cases)

    # Save output
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else db.parent / "sap_event_log.json"
    save_cases(cases, output)
    print(f"Saved to: {output}")
