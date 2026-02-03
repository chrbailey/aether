"""Parse BPI Challenge 2013 Incidents XES file into AETHER format."""
from __future__ import annotations
import json, logging, random
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
AETHER_ROOT = Path("/Volumes/OWC drive/Dev/aether")
DATA_DIR = AETHER_ROOT / "data" / "external" / "bpi2013_incident"
XES_PATH = DATA_DIR / "BPI_Challenge_2013_incidents.xes"
OUTPUT_DIR = DATA_DIR
INCIDENT_ACTIVITIES = frozenset({"Queued", "Accepted", "Completed", "Unmatched"})

def parse_timestamp(ts_str):
    if not ts_str: return None
    try:
        ts_clean = ts_str.split(".")[0]
        if "+" in ts_clean: ts_clean = ts_clean.split("+")[0]
        elif ts_clean.endswith("Z"): ts_clean = ts_clean[:-1]
        return datetime.fromisoformat(ts_clean)
    except: return None

def format_timestamp(dt): return dt.strftime("%Y-%m-%dT%H:%M:%S") if dt else ""

def parse_xes_file(xes_path, max_cases=None):
    logger.info(f"Parsing XES file: {xes_path}")
    cases, current_trace, current_events, trace_attributes = [], None, [], {}
    context = ET.iterparse(str(xes_path), events=("start", "end"))
    case_count, event_count = 0, 0
    for event_type, elem in context:
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if event_type == "start" and tag == "trace":
            current_trace, current_events, trace_attributes = {}, [], {}
        elif event_type == "end":
            if tag == "trace":
                if current_events:
                    case_id = trace_attributes.get("concept:name", f"case_{case_count}")
                    if case_id in ("UNKNOWN", "BPI Challenge 2013, incidents"):
                        current_trace, current_events, trace_attributes = None, [], {}
                        elem.clear(); continue
                    current_events.sort(key=lambda e: e.get("_ts") or datetime.min)
                    events = [{"activity": e["activity"], "resource": e.get("resource", "unknown"), "timestamp": e.get("timestamp", ""), "attributes": e.get("attributes", {})} for e in current_events if e.get("activity") in INCIDENT_ACTIVITIES]
                    if events:
                        activities = [e["activity"] for e in events]
                        timestamps = [parse_timestamp(e["timestamp"]) for e in events if parse_timestamp(e["timestamp"])]
                        duration_hours = (max(timestamps) - min(timestamps)).total_seconds() / 3600 if len(timestamps) >= 2 else 0.0
                        has_completed = "Completed" in activities
                        is_resolved = has_completed and "Unmatched" not in activities
                        case = {"caseId": case_id, "events": events, "outcome": {"completed": has_completed, "resolved": is_resolved, "durationHours": round(duration_hours, 2), "onTime": is_resolved and duration_hours < 24.0, "rework": any(c > 1 for c in Counter(activities).values())}}
                        cases.append(case); case_count += 1; event_count += len(events)
                        if case_count % 1000 == 0: logger.info(f"  Processed {case_count:,} cases")
                        if max_cases and case_count >= max_cases: break
                current_trace, current_events, trace_attributes = None, [], {}; elem.clear()
            elif tag == "event" and current_trace is not None:
                event_data = {"attributes": {}}
                for child in elem:
                    key, value = child.get("key", ""), child.get("value", "")
                    if key == "concept:name": event_data["activity"] = value
                    elif key == "org:resource": event_data["resource"] = value or "unknown"
                    elif key == "time:timestamp": event_data["timestamp"], event_data["_ts"] = format_timestamp(parse_timestamp(value)), parse_timestamp(value)
                current_events.append(event_data); elem.clear()
            elif current_trace is not None and tag in ("string",):
                key, value = elem.get("key", ""), elem.get("value", "")
                if key: trace_attributes[key] = value
    logger.info(f"Parsed {len(cases):,} cases with {event_count:,} events"); return cases

def build_vocabulary(cases):
    activity_tokens = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
    resource_tokens = {"<PAD>": 0, "<UNK>": 1}
    activities, resources = set(), set()
    for case in cases:
        for event in case["events"]: activities.add(event["activity"]); resources.add(event.get("resource", "unknown"))
    for i, act in enumerate(sorted(activities), start=len(activity_tokens)): activity_tokens[act] = i
    for i, res in enumerate(sorted(resources), start=len(resource_tokens)): resource_tokens[res] = i
    return {"activity": {"token_to_idx": activity_tokens, "size": len(activity_tokens)}, "resource": {"token_to_idx": resource_tokens, "size": len(resource_tokens)}}

def split_train_val(cases, train_ratio=0.8, seed=42):
    random.seed(seed); shuffled = cases.copy(); random.shuffle(shuffled)
    return shuffled[:int(len(shuffled) * train_ratio)], shuffled[int(len(shuffled) * train_ratio):]

def main():
    print("BPI 2013 INCIDENTS PARSER")
    if not XES_PATH.exists(): print(f"ERROR: {XES_PATH}"); return
    cases = parse_xes_file(XES_PATH)
    train_cases, val_cases = split_train_val(cases)
    vocab = build_vocabulary(cases)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "train_cases.json", "w") as f: json.dump(train_cases, f)
    with open(OUTPUT_DIR / "val_cases.json", "w") as f: json.dump(val_cases, f)
    with open(OUTPUT_DIR / "vocabulary.json", "w") as f: json.dump(vocab, f, indent=2)
    print(f"Train: {len(train_cases)}, Val: {len(val_cases)}, Activities: {vocab['activity']['size']}")

if __name__ == "__main__": main()
