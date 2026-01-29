"""OCEL 2.0 parser for AETHER training format.

Parses Object-Centric Event Log (OCEL 2.0) data stored in SQLite or JSON
into flat case-based event sequences compatible with AETHER's data loader.

OCEL 2.0 links events to multiple objects; this parser reconstructs
traditional "cases" by anchoring on a chosen object type and collecting
all events associated with each anchor object instance.

Typical usage:
    >>> schema = discover_ocel_schema("ocel2-p2p.sqlite")
    >>> cases = parse_ocel_sqlite("ocel2-p2p.sqlite", anchor_type="purchase_order")
    >>> # cases is a list[dict] directly loadable by EventSequenceDataset
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Common resource field names across OCEL datasets
_RESOURCE_FIELDS = frozenset({
    "resource",
    "org:resource",
    "user",
    "actor",
    "org:role",
    "responsible",
})

# Event types that signal a completed/successful process outcome
_COMPLETION_ACTIVITIES = frozenset({
    "Execute Payment",
    "ExecutePayment",
    "payment",
    "Pay Invoice",
    "Complete",
    "Close",
    "Completed",
    "payment_complete",
    "process_complete",
    "Perform Two-Way Match",
    "PerformTwoWayMatch",
})


def discover_ocel_schema(db_path: str | Path) -> dict[str, Any]:
    """Discover the OCEL 2.0 schema from a SQLite database.

    Inspects the database tables to determine available event types,
    object types, counts, and the table naming convention used.

    Args:
        db_path: Path to the OCEL 2.0 SQLite database.

    Returns:
        Dict with keys:
            - event_types: list of event type display names
            - object_types: list of object type display names
            - event_count: total number of events
            - object_count: total number of objects
            - event_object_count: total number of event-object links
            - tables: list of all table names in the database
            - table_style: "ocel2" or "ocel2_prefixed" indicating naming convention
            - event_attribute_tables: dict mapping event type to table name
            - object_attribute_tables: dict mapping object type to table name
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"OCEL database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # Discover all tables
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row["name"] for row in cursor.fetchall()]

        # Determine table naming style
        table_style = _detect_table_style(tables)

        # Get event and object table names
        event_table, object_table, eo_table = _core_table_names(table_style)

        # Event types
        event_types: list[str] = []
        try:
            cursor.execute(f"SELECT DISTINCT ocel_type FROM {event_table}")
            event_types = sorted(row["ocel_type"] for row in cursor.fetchall())
        except sqlite3.OperationalError:
            logger.warning("Could not read event types from %s", event_table)

        # Object types
        object_types: list[str] = []
        try:
            cursor.execute(f"SELECT DISTINCT ocel_type FROM {object_table}")
            object_types = sorted(row["ocel_type"] for row in cursor.fetchall())
        except sqlite3.OperationalError:
            logger.warning("Could not read object types from %s", object_table)

        # Counts
        event_count = _safe_count(cursor, event_table)
        object_count = _safe_count(cursor, object_table)
        eo_count = _safe_count(cursor, eo_table)

        # Map event types to their attribute tables
        event_attr_tables = _find_typed_tables(tables, "event_", event_table)
        object_attr_tables = _find_typed_tables(tables, "object_", object_table)

        # Try to resolve display names via map tables
        event_type_map = _load_type_map(cursor, "event_map_type")
        object_type_map = _load_type_map(cursor, "object_map_type")

        return {
            "event_types": event_types,
            "object_types": object_types,
            "event_count": event_count,
            "object_count": object_count,
            "event_object_count": eo_count,
            "tables": tables,
            "table_style": table_style,
            "event_attribute_tables": event_attr_tables,
            "object_attribute_tables": object_attr_tables,
            "event_type_map": event_type_map,
            "object_type_map": object_type_map,
        }
    finally:
        conn.close()


def parse_ocel_sqlite(
    db_path: str | Path,
    anchor_type: str | None = None,
    max_cases: int | None = None,
) -> list[dict[str, Any]]:
    """Parse OCEL 2.0 SQLite into AETHER training format.

    Reconstructs case-based event sequences by anchoring on a chosen
    object type. Each anchor object instance becomes one case, and all
    events linked to that object (via event_object) form the event
    sequence, sorted by timestamp.

    If anchor_type is None, auto-detects the best anchor by selecting
    the object type with the most connected events.

    Args:
        db_path: Path to the OCEL 2.0 SQLite database.
        anchor_type: Object type to use as case anchor (e.g., "purchase_order").
            If None, auto-detected.
        max_cases: Maximum number of cases to return. None for all.

    Returns:
        List of dicts in AETHER format::

            {
                "caseId": "OCEL_purchase_order_po:42",
                "events": [
                    {
                        "activity": "Create Purchase Order",
                        "resource": "Procurement Department",
                        "timestamp": "2022-04-06T07:45:00.000Z",
                        "attributes": {"lifecycle": "complete"}
                    },
                    ...
                ],
                "outcome": {
                    "onTime": True,
                    "rework": False,
                    "durationHours": 72.5
                }
            }
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"OCEL database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        tables = _list_tables(cursor)
        table_style = _detect_table_style(tables)
        event_table, object_table, eo_table = _core_table_names(table_style)

        # Auto-detect anchor type if not specified
        if anchor_type is None:
            anchor_type = _auto_detect_anchor(cursor, object_table, eo_table)
            logger.info("Auto-detected anchor type: %s", anchor_type)

        # Validate anchor type exists
        cursor.execute(
            f"SELECT COUNT(*) as cnt FROM {object_table} WHERE ocel_type = ?",
            (anchor_type,),
        )
        anchor_count = cursor.fetchone()["cnt"]
        if anchor_count == 0:
            available = _get_distinct(cursor, object_table, "ocel_type")
            raise ValueError(
                f"Anchor type '{anchor_type}' not found. "
                f"Available types: {available}"
            )

        logger.info(
            "Parsing OCEL with anchor_type='%s' (%d objects)",
            anchor_type,
            anchor_count,
        )

        # Load all anchor object IDs
        cursor.execute(
            f"SELECT ocel_id FROM {object_table} WHERE ocel_type = ?",
            (anchor_type,),
        )
        anchor_ids = [row["ocel_id"] for row in cursor.fetchall()]

        if max_cases is not None:
            anchor_ids = anchor_ids[:max_cases]

        # Load event type map for display names
        event_type_map = _load_type_map(cursor, "event_map_type")

        # Build event attribute lookup: event_id -> {timestamp, resource, attrs}
        event_attrs = _load_all_event_attributes(cursor, tables, table_style)

        # For each anchor object, find linked events and build the case
        cases: list[dict[str, Any]] = []
        for anchor_id in anchor_ids:
            case = _build_case(
                cursor=cursor,
                anchor_type=anchor_type,
                anchor_id=anchor_id,
                event_table=event_table,
                eo_table=eo_table,
                event_attrs=event_attrs,
                event_type_map=event_type_map,
            )
            if case is not None and len(case["events"]) > 0:
                cases.append(case)

        logger.info(
            "Parsed %d cases from %d anchor objects (%.1f%% with events)",
            len(cases),
            len(anchor_ids),
            100.0 * len(cases) / max(len(anchor_ids), 1),
        )

        return cases

    finally:
        conn.close()


def parse_ocel_json(
    json_path: str | Path,
    anchor_type: str | None = None,
    max_cases: int | None = None,
) -> list[dict[str, Any]]:
    """Parse OCEL 2.0 JSON format into AETHER training format.

    Fallback parser for when the data is in JSON rather than SQLite.
    Supports both OCEL 1.0 and OCEL 2.0 JSON structures.

    Args:
        json_path: Path to the OCEL JSON file.
        anchor_type: Object type to use as case anchor. If None, auto-detected.
        max_cases: Maximum number of cases to return.

    Returns:
        List of dicts in AETHER training format.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"OCEL JSON not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    # Detect OCEL version from structure
    if "ocel:events" in data:
        return _parse_ocel1_json(data, anchor_type, max_cases)
    elif "events" in data or "eventTypes" in data:
        return _parse_ocel2_json(data, anchor_type, max_cases)
    else:
        raise ValueError(
            "Unrecognized OCEL JSON structure. Expected 'ocel:events' (v1) "
            "or 'events'/'eventTypes' (v2) keys."
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_table_style(tables: list[str]) -> str:
    """Detect whether tables use 'event'/'object' or 'ocel_event'/'ocel_object'."""
    table_set = set(tables)
    if "ocel_event" in table_set and "ocel_object" in table_set:
        return "ocel2_prefixed"
    if "event" in table_set and "object" in table_set:
        return "ocel2"
    raise ValueError(
        f"Cannot detect OCEL table style. Tables found: {tables}. "
        f"Expected 'event'+'object' or 'ocel_event'+'ocel_object'."
    )


def _core_table_names(style: str) -> tuple[str, str, str]:
    """Return (event_table, object_table, event_object_table) for the style."""
    if style == "ocel2_prefixed":
        return "ocel_event", "ocel_object", "ocel_event_object"
    return "event", "object", "event_object"


def _list_tables(cursor: sqlite3.Cursor) -> list[str]:
    """List all table names in the database."""
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    return [row["name"] for row in cursor.fetchall()]


def _safe_count(cursor: sqlite3.Cursor, table: str) -> int:
    """Count rows in a table, returning 0 on error."""
    try:
        cursor.execute(f"SELECT COUNT(*) as cnt FROM [{table}]")
        return cursor.fetchone()["cnt"]
    except sqlite3.OperationalError:
        return 0


def _get_distinct(
    cursor: sqlite3.Cursor, table: str, column: str
) -> list[str]:
    """Get distinct values from a column."""
    cursor.execute(f"SELECT DISTINCT [{column}] FROM [{table}]")
    return sorted(row[0] for row in cursor.fetchall())


def _load_type_map(
    cursor: sqlite3.Cursor, table_name: str
) -> dict[str, str]:
    """Load display-name-to-table-suffix mapping from a map table.

    Returns dict mapping display name -> table suffix.
    E.g., {"Create Purchase Order": "CreatePurchaseOrder"}
    """
    mapping: dict[str, str] = {}
    try:
        cursor.execute(f"SELECT ocel_type, ocel_type_map FROM [{table_name}]")
        for row in cursor.fetchall():
            mapping[row["ocel_type"]] = row["ocel_type_map"]
    except sqlite3.OperationalError:
        pass
    return mapping


def _find_typed_tables(
    tables: list[str], prefix: str, exclude: str
) -> dict[str, str]:
    """Find type-specific attribute tables.

    For example, tables starting with "event_" (excluding "event" itself,
    "event_object", and "event_map_type") are event attribute tables.

    Returns dict mapping table suffix -> full table name.
    E.g., {"CreatePurchaseOrder": "event_CreatePurchaseOrder"}
    """
    skip = {exclude, f"{prefix}object", f"{prefix}map_type"}
    result: dict[str, str] = {}
    for t in tables:
        if t.startswith(prefix) and t not in skip:
            suffix = t[len(prefix):]
            result[suffix] = t
    return result


def _auto_detect_anchor(
    cursor: sqlite3.Cursor,
    object_table: str,
    eo_table: str,
) -> str:
    """Auto-detect the best anchor object type.

    Selects the object type whose instances are linked to the most events
    on average, weighted by instance count. This favors types that are
    central to the process (e.g., purchase_order in P2P).
    """
    cursor.execute(f"SELECT DISTINCT ocel_type FROM [{object_table}]")
    obj_types = [row["ocel_type"] for row in cursor.fetchall()]

    if not obj_types:
        raise ValueError("No object types found in the database")

    best_type = obj_types[0]
    best_score = -1.0

    for otype in obj_types:
        # Count how many event-object links this type has
        cursor.execute(
            f"""
            SELECT COUNT(*) as cnt
            FROM [{eo_table}] eo
            JOIN [{object_table}] o ON eo.ocel_object_id = o.ocel_id
            WHERE o.ocel_type = ?
            """,
            (otype,),
        )
        link_count = cursor.fetchone()["cnt"]

        # Count instances of this type
        cursor.execute(
            f"SELECT COUNT(*) as cnt FROM [{object_table}] WHERE ocel_type = ?",
            (otype,),
        )
        instance_count = cursor.fetchone()["cnt"]

        if instance_count == 0:
            continue

        # Score: average events per instance * sqrt(instance_count)
        # This balances coverage (many instances) with richness (many events each)
        avg_events = link_count / instance_count
        score = avg_events * (instance_count ** 0.5)

        logger.debug(
            "Anchor candidate '%s': %d instances, %.1f avg events, score=%.1f",
            otype,
            instance_count,
            avg_events,
            score,
        )

        if score > best_score:
            best_score = score
            best_type = otype

    return best_type


def _load_all_event_attributes(
    cursor: sqlite3.Cursor,
    tables: list[str],
    table_style: str,
) -> dict[str, dict[str, Any]]:
    """Load attributes from all event type-specific tables.

    Returns dict mapping event_id -> {
        "timestamp": str,
        "resource": str,
        "attributes": dict of non-standard columns
    }
    """
    event_table, _, _ = _core_table_names(table_style)
    attr_tables = _find_typed_tables(tables, "event_", event_table)

    result: dict[str, dict[str, Any]] = {}

    for suffix, table_name in attr_tables.items():
        # Get column names for this table
        cursor.execute(f"PRAGMA table_info([{table_name}])")
        columns = [row["name"] for row in cursor.fetchall()]

        # Identify resource column
        resource_col = None
        for col in columns:
            if col.lower() in _RESOURCE_FIELDS:
                resource_col = col
                break

        # Identify timestamp column
        time_col = "ocel_time" if "ocel_time" in columns else None

        # Non-standard attribute columns (exclude id, time, known fields)
        skip_cols = {"ocel_id", "ocel_time", "lifecycle"}
        if resource_col:
            skip_cols.add(resource_col)
        attr_cols = [c for c in columns if c not in skip_cols]

        # Fetch all rows
        cursor.execute(f"SELECT * FROM [{table_name}]")
        for row in cursor.fetchall():
            event_id = row["ocel_id"]
            entry: dict[str, Any] = {
                "timestamp": row[time_col] if time_col else None,
                "resource": row[resource_col] if resource_col else "SYSTEM",
                "attributes": {},
            }

            # Collect extra attributes, keeping only numeric values
            for col in attr_cols:
                val = row[col]
                if val is not None:
                    # Try to parse as number for the attributes dict
                    try:
                        entry["attributes"][col] = float(val)
                    except (ValueError, TypeError):
                        # Keep non-numeric as string attributes for context
                        entry["attributes"][col] = str(val)

            result[event_id] = entry

    return result


def _build_case(
    cursor: sqlite3.Cursor,
    anchor_type: str,
    anchor_id: str,
    event_table: str,
    eo_table: str,
    event_attrs: dict[str, dict[str, Any]],
    event_type_map: dict[str, str],
) -> dict[str, Any] | None:
    """Build one AETHER case from an anchor object.

    Finds all events linked to the anchor object, enriches them with
    attributes from the type-specific tables, sorts by timestamp,
    and computes outcome heuristics.
    """
    # Find all events linked to this anchor object
    cursor.execute(
        f"""
        SELECT e.ocel_id, e.ocel_type
        FROM [{event_table}] e
        JOIN [{eo_table}] eo ON e.ocel_id = eo.ocel_event_id
        WHERE eo.ocel_object_id = ?
        ORDER BY e.ocel_id
        """,
        (anchor_id,),
    )
    event_rows = cursor.fetchall()

    if not event_rows:
        return None

    # Build event list with attributes
    events: list[dict[str, Any]] = []
    for row in event_rows:
        event_id = row["ocel_id"]
        activity = row["ocel_type"]

        # Look up attributes from the type-specific table
        attrs_entry = event_attrs.get(event_id, {})
        timestamp = attrs_entry.get("timestamp", "")
        resource = attrs_entry.get("resource", "SYSTEM")
        raw_attributes = attrs_entry.get("attributes", {})

        # Filter to only numeric attributes for the AETHER format
        numeric_attrs: dict[str, float] = {}
        for k, v in raw_attributes.items():
            if isinstance(v, (int, float)):
                numeric_attrs[k] = v

        events.append({
            "activity": activity,
            "resource": resource or "SYSTEM",
            "timestamp": timestamp or "",
            "attributes": numeric_attrs,
        })

    # Sort events by timestamp
    events = _sort_events_by_time(events)

    # Compute outcome heuristics
    outcome = _compute_outcome(events)

    # Format the case ID
    # Clean anchor_id to be readable (e.g., "purchase_order:42" -> "42")
    clean_id = anchor_id.replace(":", "_") if ":" in anchor_id else anchor_id
    case_id = f"OCEL_{anchor_type}_{clean_id}"

    return {
        "caseId": case_id,
        "events": events,
        "outcome": outcome,
    }


def _sort_events_by_time(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort events by timestamp, handling missing/unparseable timestamps."""

    def parse_ts(ts_str: str) -> datetime:
        if not ts_str:
            return datetime.min
        try:
            # Handle ISO 8601 with Z suffix
            cleaned = ts_str.replace("Z", "+00:00")
            return datetime.fromisoformat(cleaned)
        except ValueError:
            # Try common alternative formats
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
                try:
                    return datetime.strptime(ts_str, fmt)
                except ValueError:
                    continue
            return datetime.min

    return sorted(events, key=lambda e: parse_ts(e["timestamp"]))


def _compute_outcome(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute outcome heuristics from a case's event sequence.

    - onTime: True if the case contains a completion/payment event
    - rework: True if any activity appears more than once
    - durationHours: time between first and last event in hours
    """
    if not events:
        return {"onTime": False, "rework": False, "durationHours": 0.0}

    # Check for completion event
    activities = [e["activity"] for e in events]
    on_time = any(a in _COMPLETION_ACTIVITIES for a in activities)

    # Check for rework (repeated activities)
    activity_counts: dict[str, int] = defaultdict(int)
    for a in activities:
        activity_counts[a] += 1
    rework = any(count > 1 for count in activity_counts.values())

    # Compute duration
    duration_hours = 0.0
    timestamps = _parse_timestamps(events)
    valid_ts = [ts for ts in timestamps if ts is not None]
    if len(valid_ts) >= 2:
        delta = max(valid_ts) - min(valid_ts)
        duration_hours = delta.total_seconds() / 3600.0

    return {
        "onTime": on_time,
        "rework": rework,
        "durationHours": round(duration_hours, 2),
    }


def _parse_timestamps(
    events: list[dict[str, Any]],
) -> list[datetime | None]:
    """Parse timestamps from events, returning None for unparseable values."""
    result: list[datetime | None] = []
    for event in events:
        ts_str = event.get("timestamp", "")
        if not ts_str:
            result.append(None)
            continue
        try:
            cleaned = ts_str.replace("Z", "+00:00")
            result.append(datetime.fromisoformat(cleaned))
        except ValueError:
            result.append(None)
    return result


# ---------------------------------------------------------------------------
# OCEL JSON parsers (fallback formats)
# ---------------------------------------------------------------------------


def _parse_ocel1_json(
    data: dict[str, Any],
    anchor_type: str | None,
    max_cases: int | None,
) -> list[dict[str, Any]]:
    """Parse OCEL 1.0 JSON format.

    OCEL 1.0 structure:
    {
        "ocel:events": {
            "e1": {
                "ocel:activity": "...",
                "ocel:timestamp": "...",
                "ocel:omap": ["obj1", "obj2"],
                "ocel:vmap": {"attr1": val1}
            }
        },
        "ocel:objects": {
            "obj1": {"ocel:type": "order", "ocel:ovmap": {...}}
        }
    }
    """
    raw_events = data.get("ocel:events", {})
    raw_objects = data.get("ocel:objects", {})

    # Build object type lookup
    obj_type_map: dict[str, str] = {}
    for obj_id, obj_data in raw_objects.items():
        obj_type_map[obj_id] = obj_data.get("ocel:type", "unknown")

    # Auto-detect anchor type if needed
    if anchor_type is None:
        # Count events per object type
        type_event_counts: dict[str, int] = defaultdict(int)
        for ev_data in raw_events.values():
            for obj_id in ev_data.get("ocel:omap", []):
                otype = obj_type_map.get(obj_id, "unknown")
                type_event_counts[otype] += 1
        if type_event_counts:
            anchor_type = max(type_event_counts, key=type_event_counts.get)  # type: ignore[arg-type]
        else:
            raise ValueError("No events with object mappings found")

    # Collect anchor object IDs
    anchor_ids = [
        oid for oid, otype in obj_type_map.items() if otype == anchor_type
    ]
    if max_cases is not None:
        anchor_ids = anchor_ids[:max_cases]

    # Build events per anchor object
    obj_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev_id, ev_data in raw_events.items():
        activity = ev_data.get("ocel:activity", "unknown")
        timestamp = ev_data.get("ocel:timestamp", "")
        vmap = ev_data.get("ocel:vmap", {})

        resource = "SYSTEM"
        for key in _RESOURCE_FIELDS:
            if key in vmap:
                resource = str(vmap[key])
                break

        numeric_attrs = {
            k: v for k, v in vmap.items()
            if isinstance(v, (int, float)) and k not in _RESOURCE_FIELDS
        }

        for obj_id in ev_data.get("ocel:omap", []):
            if obj_type_map.get(obj_id) == anchor_type:
                obj_events[obj_id].append({
                    "activity": activity,
                    "resource": resource,
                    "timestamp": timestamp,
                    "attributes": numeric_attrs,
                })

    # Build cases
    cases: list[dict[str, Any]] = []
    for anchor_id in anchor_ids:
        events = obj_events.get(anchor_id, [])
        if not events:
            continue
        events = _sort_events_by_time(events)
        clean_id = anchor_id.replace(":", "_")
        cases.append({
            "caseId": f"OCEL_{anchor_type}_{clean_id}",
            "events": events,
            "outcome": _compute_outcome(events),
        })

    return cases


def _parse_ocel2_json(
    data: dict[str, Any],
    anchor_type: str | None,
    max_cases: int | None,
) -> list[dict[str, Any]]:
    """Parse OCEL 2.0 JSON format.

    OCEL 2.0 JSON structure:
    {
        "objectTypes": [...],
        "eventTypes": [...],
        "objects": [{"id": "...", "type": "...", ...}],
        "events": [{"id": "...", "type": "...", "time": "...",
                     "relationships": [{"objectId": "...", "qualifier": "..."}],
                     "attributes": [...]}]
    }
    """
    raw_events = data.get("events", [])
    raw_objects = data.get("objects", [])

    # Build object type lookup
    obj_type_map: dict[str, str] = {}
    for obj in raw_objects:
        obj_id = obj.get("id", obj.get("ocel_id", ""))
        obj_type = obj.get("type", obj.get("ocel_type", "unknown"))
        obj_type_map[obj_id] = obj_type

    # Auto-detect anchor type
    if anchor_type is None:
        type_event_counts: dict[str, int] = defaultdict(int)
        for ev in raw_events:
            rels = ev.get("relationships", [])
            for rel in rels:
                obj_id = rel.get("objectId", "")
                otype = obj_type_map.get(obj_id, "unknown")
                type_event_counts[otype] += 1
        if type_event_counts:
            anchor_type = max(type_event_counts, key=type_event_counts.get)  # type: ignore[arg-type]
        else:
            raise ValueError("No events with object relationships found")

    # Collect anchor IDs
    anchor_ids = [
        oid for oid, otype in obj_type_map.items() if otype == anchor_type
    ]
    if max_cases is not None:
        anchor_ids = anchor_ids[:max_cases]

    anchor_id_set = set(anchor_ids)

    # Build events per anchor
    obj_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in raw_events:
        activity = ev.get("type", ev.get("ocel_type", "unknown"))
        timestamp = ev.get("time", ev.get("ocel_time", ""))
        ev_attrs = ev.get("attributes", {})

        # Handle attributes as list of dicts or as a dict
        if isinstance(ev_attrs, list):
            attr_dict = {}
            for attr in ev_attrs:
                name = attr.get("name", "")
                value = attr.get("value", "")
                attr_dict[name] = value
            ev_attrs = attr_dict

        resource = "SYSTEM"
        for key in _RESOURCE_FIELDS:
            if key in ev_attrs:
                resource = str(ev_attrs[key])
                break

        numeric_attrs = {
            k: v for k, v in ev_attrs.items()
            if isinstance(v, (int, float)) and k not in _RESOURCE_FIELDS
        }

        rels = ev.get("relationships", [])
        for rel in rels:
            obj_id = rel.get("objectId", "")
            if obj_id in anchor_id_set:
                obj_events[obj_id].append({
                    "activity": activity,
                    "resource": resource,
                    "timestamp": timestamp,
                    "attributes": numeric_attrs,
                })

    # Build cases
    cases: list[dict[str, Any]] = []
    for anchor_id in anchor_ids:
        events = obj_events.get(anchor_id, [])
        if not events:
            continue
        events = _sort_events_by_time(events)
        clean_id = anchor_id.replace(":", "_")
        cases.append({
            "caseId": f"OCEL_{anchor_type}_{clean_id}",
            "events": events,
            "outcome": _compute_outcome(events),
        })

    return cases


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Parse OCEL 2.0 data into AETHER training format"
    )
    parser.add_argument("input", help="Path to OCEL SQLite or JSON file")
    parser.add_argument(
        "--anchor",
        default=None,
        help="Object type to use as case anchor (auto-detected if omitted)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Maximum number of cases to output",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file path (stdout if omitted)",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Only print the discovered schema, then exit",
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    if args.schema_only:
        if input_path.suffix in (".db", ".sqlite", ".sqlite3"):
            schema = discover_ocel_schema(input_path)
            print(json.dumps(schema, indent=2))
        else:
            print("Schema discovery is only supported for SQLite files.", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)

    # Parse based on file type
    if input_path.suffix in (".db", ".sqlite", ".sqlite3"):
        cases = parse_ocel_sqlite(
            input_path, anchor_type=args.anchor, max_cases=args.max_cases
        )
    elif input_path.suffix == ".json":
        cases = parse_ocel_json(
            input_path, anchor_type=args.anchor, max_cases=args.max_cases
        )
    else:
        print(f"Unsupported file format: {input_path.suffix}", file=sys.stderr)
        sys.exit(1)

    # Output
    output_data = json.dumps(cases, indent=2)
    if args.output:
        Path(args.output).write_text(output_data)
        logger.info("Wrote %d cases to %s", len(cases), args.output)
    else:
        print(output_data)
