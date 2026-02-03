#!/usr/bin/env python3
"""Parse BPI Challenge 2015 Building Permits XES file into AETHER format."""
from __future__ import annotations
import json, logging, random, xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

AETHER_ROOT = Path("/Volumes/OWC drive/Dev/aether")
DATA_DIR = AETHER_ROOT / "data" / "external" / "bpi2015_permits"
XES_PATH = DATA_DIR / "BPIC15_1.xes"
OUTPUT_DIR = DATA_DIR

def parse_timestamp(ts_str):
    if not ts_str: return None
    try:
        ts_clean = ts_str.split(".")[0]
        if "+" in ts_clean: ts_clean = ts_clean.split("+")[0]
        elif ts_clean.endswith("Z"): ts_clean = ts_clean[:-1]
        return datetime.fromisoformat(ts_clean)
    except: return None

def format_timestamp(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S") if dt else ""

def parse_xes_file(xes_path, max_cases=None):
    logger.info(f"Parsing XES file: {xes_path}")
    logger.info(f"File size: {xes_path.stat().st_size / 1024 / 1024:.1f} MB")
    cases, current_trace, current_events, trace_attributes = [], None, [], {}
    context = ET.iterparse(str(xes_path), events=("start", "end"))
    case_count, event_count = 0, 0
    for event_type, elem in context:
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if event_type == "start" and tag == "trace":
            current_trace, current_events, trace_attributes = {}, [], {}
        elif event_type == "end":
            if tag == "trace" and current_events:
                case_id = trace_attributes.get("concept:name", f"case_{case_count}")
                current_events.sort(key=lambda e: e.get("_ts") or datetime.min)
                events = [{"activity": e.get("activity", ""), "resource": e.get("resource", "unknown"), "timestamp": e.get("timestamp", ""), "attributes": e.get("attributes", {})} for e in current_events]
                if events:
                    timestamps = [parse_timestamp(e["timestamp"]) for e in events if parse_timestamp(e["timestamp"])]
                    duration_hours = (max(timestamps) - min(timestamps)).total_seconds() / 3600 if len(timestamps) >= 2 else 0.0
                    activities = [e["activity"] for e in events]
                    request_complete = trace_attributes.get("requestComplete", "").upper() == "TRUE"
                    last_phase = trace_attributes.get("last_phase", "")
                    is_approved = "verzonden" in last_phase.lower() or "beschikking" in last_phase.lower()
                    has_objection = any("bezwaar" in a.lower() or "objection" in a.lower() for a in activities)
                    has_appeal = any("beroep" in a.lower() or "appeal" in a.lower() for a in activities)
                    end_date, planned_end = trace_attributes.get("endDate"), trace_attributes.get("endDatePlanned")
                    on_time = True
                    if end_date and planned_end:
                        end_dt, planned_dt = parse_timestamp(end_date), parse_timestamp(planned_end)
                        if end_dt and planned_dt: on_time = end_dt <= planned_dt
                    cases.append({"caseId": case_id, "events": events, "source": "bpi_challenge_2015", "processType": "building_permit", "outcome": {"complete": request_complete, "approved": is_approved, "objection": has_objection, "appeal": has_appeal, "durationHours": round(duration_hours, 2), "onTime": on_time, "rework": has_objection or has_appeal}, "caseAttributes": {k: v for k, v in trace_attributes.items() if k not in ("concept:name", "identity:id")}})
                    case_count += 1
                    event_count += len(events)
                    if case_count % 500 == 0: logger.info(f"  Processed {case_count:,} cases, {event_count:,} events...")
                    if max_cases and case_count >= max_cases: break
                current_trace, current_events, trace_attributes = None, [], {}
                elem.clear()
            elif tag == "event" and current_trace is not None:
                event_data, event_attrs = {}, {}
                for child in elem:
                    child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    key, value = child.get("key", ""), child.get("value", "")
                    if key == "activityNameEN": event_data["activity"] = value if value and value != "UNKNOWN" else None
                    elif key == "concept:name" and not event_data.get("activity"): event_data["activity"] = value
                    elif key == "org:resource": event_data["resource"] = value if value and value != "UNKNOWN" else "unknown"
                    elif key == "time:timestamp": event_data["timestamp"], event_data["_ts"] = format_timestamp(parse_timestamp(value)), parse_timestamp(value)
                    elif key == "action_code": event_attrs["action_code"] = value
                    elif key == "activityNameNL": event_attrs["activityNameNL"] = value
                if not event_data.get("activity"): event_data["activity"] = "unknown"
                event_data["attributes"] = event_attrs
                current_events.append(event_data)
                elem.clear()
            elif current_trace is not None and tag in ("string", "int", "float", "boolean", "date"):
                key, value = elem.get("key", ""), elem.get("value", "")
                if key and key not in ("identity:id",):
                    if tag == "int": trace_attributes[key] = int(value) if value else 0
                    elif tag == "float": trace_attributes[key] = float(value) if value else 0.0
                    else: trace_attributes[key] = value
        if max_cases and case_count >= max_cases: break
    logger.info(f"Parsed {len(cases):,} cases with {event_count:,} events")
    return cases

def build_vocabulary(cases):
    activity_tokens, resource_tokens = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}, {"<PAD>": 0, "<UNK>": 1}
    activities, resources = set(), set()
    for case in cases:
        for event in case["events"]:
            activities.add(event["activity"])
            resources.add(event.get("resource", "unknown"))
    for i, act in enumerate(sorted(activities), start=len(activity_tokens)): activity_tokens[act] = i
    for i, res in enumerate(sorted(resources), start=len(resource_tokens)): resource_tokens[res] = i
    return {"activity": {"token_to_idx": activity_tokens, "size": len(activity_tokens)}, "resource": {"token_to_idx": resource_tokens, "size": len(resource_tokens)}}

def compute_stats(cases):
    activities, resources, total_events = Counter(), set(), 0
    for case in cases:
        for event in case["events"]:
            activities[event["activity"]] += 1
            resources.add(event.get("resource", "unknown"))
            total_events += 1
    durations = [c["outcome"]["durationHours"] for c in cases]
    event_counts = [len(c["events"]) for c in cases]
    return {"total_cases": len(cases), "total_events": total_events, "activity_vocab_size": len(activities), "resource_vocab_size": len(resources), "activity_counts": dict(activities.most_common(50)), "outcome_stats": {"complete_rate": round(sum(1 for c in cases if c["outcome"]["complete"]) / len(cases), 4), "approved_rate": round(sum(1 for c in cases if c["outcome"]["approved"]) / len(cases), 4), "on_time_rate": round(sum(1 for c in cases if c["outcome"]["onTime"]) / len(cases), 4), "rework_rate": round(sum(1 for c in cases if c["outcome"]["rework"]) / len(cases), 4)}, "event_length_stats": {"min": min(event_counts), "max": max(event_counts), "mean": round(sum(event_counts) / len(event_counts), 2), "median": sorted(event_counts)[len(event_counts) // 2]}, "duration_stats_hours": {"min": round(min(durations), 2), "max": round(max(durations), 2), "mean": round(sum(durations) / len(durations), 2)}, "source": "BPI Challenge 2015"}

def split_train_val(cases, train_ratio=0.8, seed=42):
    random.seed(seed)
    shuffled = cases.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]

def main():
    print("=" * 60)
    print("BPI 2015 BUILDING PERMITS -> AETHER EVENT LOG PARSER")
    print("=" * 60)
    if not XES_PATH.exists():
        print(f"ERROR: XES file not found: {XES_PATH}")
        return
    cases = parse_xes_file(XES_PATH)
    if not cases:
        print("ERROR: No cases parsed!")
        return
    train_cases, val_cases = split_train_val(cases)
    vocab = build_vocabulary(cases)
    stats = compute_stats(cases)
    stats["train_cases"], stats["val_cases"] = len(train_cases), len(val_cases)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "train_cases.json", "w") as f: json.dump(train_cases, f)
    print(f"Saved {len(train_cases):,} train cases")
    with open(OUTPUT_DIR / "val_cases.json", "w") as f: json.dump(val_cases, f)
    print(f"Saved {len(val_cases):,} val cases")
    with open(OUTPUT_DIR / "vocabulary.json", "w") as f: json.dump(vocab, f, indent=2)
    act_size = vocab["activity"]["size"]
    res_size = vocab["resource"]["size"]
    print(f"Saved vocabulary: {act_size} activities, {res_size} resources")
    with open(OUTPUT_DIR / "metadata.json", "w") as f: json.dump(stats, f, indent=2)
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_cases = stats["total_cases"]
    train_n = stats["train_cases"]
    val_n = stats["val_cases"]
    total_events = stats["total_events"]
    act_vocab = stats["activity_vocab_size"]
    print(f"Cases: {total_cases:,} (train: {train_n:,}, val: {val_n:,})")
    print(f"Events: {total_events:,}")
    print(f"Activities: {act_vocab} (HIGHEST VOCABULARY)")

if __name__ == "__main__":
    main()
