"""Prepare AETHER training data from all available SAP data sources.

Runs the full unified pipeline: loads data from SAP IDES, BPI 2019,
CSV event logs, and OCEL 2.0, then produces train/val splits with
shared vocabularies saved to disk.

Usage:
    cd "/Volumes/OWC drive/Dev/aether"
    python -m core.data.prepare_training_data
    python -m core.data.prepare_training_data --max-bpi 10000
    python -m core.data.prepare_training_data --output-dir /tmp/aether-data
    python -m core.data.prepare_training_data --train-ratio 0.9 --seed 123
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

from .unified_pipeline import AetherDataPipeline, DEFAULT_PATHS

logger = logging.getLogger(__name__)


def _format_count(n: int) -> str:
    """Format a number with thousands separators."""
    return f"{n:,}"


def _print_header(title: str, width: int = 60) -> None:
    """Print a section header with border."""
    border = "=" * width
    print(f"\n{border}")
    print(f"  {title}")
    print(border)


def _print_source_table(source_counts: Dict[str, int]) -> None:
    """Print a formatted table of source statistics."""
    if not source_counts:
        print("  (no sources loaded)")
        return

    # Column widths
    name_width = max(len(name) for name in source_counts) + 2
    count_width = max(len(_format_count(c)) for c in source_counts.values()) + 2

    header = f"  {'Source':<{name_width}} {'Cases':>{count_width}}  Status"
    print(header)
    print(f"  {'-' * (name_width + count_width + 10)}")

    for source, count in sorted(source_counts.items()):
        status = "OK" if count > 0 else "SKIPPED"
        marker = "[+]" if count > 0 else "[-]"
        print(
            f"  {source:<{name_width}} "
            f"{_format_count(count):>{count_width}}  "
            f"{marker} {status}"
        )


def _print_vocabulary_summary(metadata: Dict[str, Any]) -> None:
    """Print vocabulary size information."""
    act_size = metadata.get("activity_vocab_size", 0)
    res_size = metadata.get("resource_vocab_size", 0)
    print(f"  Activity vocabulary:  {_format_count(act_size)} tokens")
    print(f"  Resource vocabulary:  {_format_count(res_size)} tokens")


def _print_event_stats(metadata: Dict[str, Any]) -> None:
    """Print event distribution statistics."""
    stats = metadata.get("event_length_stats", {})
    print(f"  Total events:    {_format_count(metadata.get('total_events', 0))}")
    print(f"  Train events:    {_format_count(metadata.get('train_events', 0))}")
    print(f"  Val events:      {_format_count(metadata.get('val_events', 0))}")
    print()
    print("  Events per case:")
    print(f"    Min:     {stats.get('min', 0)}")
    print(f"    P25:     {stats.get('p25', 0)}")
    print(f"    Median:  {stats.get('median', 0)}")
    print(f"    Mean:    {stats.get('mean', 0):.1f}")
    print(f"    P75:     {stats.get('p75', 0)}")
    print(f"    Max:     {stats.get('max', 0)}")


def _print_top_activities(metadata: Dict[str, Any]) -> None:
    """Print top activity frequencies."""
    top = metadata.get("top_activities", [])
    if not top:
        print("  (no activity data)")
        return

    max_name_len = max(len(name) for name, _ in top)
    max_count = top[0][1] if top else 1

    for name, count in top:
        bar_len = int(30 * count / max(max_count, 1))
        bar = "#" * bar_len
        print(
            f"  {name:<{max_name_len + 2}} "
            f"{_format_count(count):>8}  {bar}"
        )


def _print_source_distribution(metadata: Dict[str, Any]) -> None:
    """Print case distribution across sources."""
    dist = metadata.get("source_distribution", {})
    if not dist:
        return

    total = sum(dist.values())
    name_width = max(len(n) for n in dist) + 2

    for source, count in sorted(dist.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / max(total, 1)
        bar_len = int(30 * count / max(max(dist.values()), 1))
        bar = "#" * bar_len
        print(
            f"  {source:<{name_width}} "
            f"{_format_count(count):>8} ({pct:5.1f}%)  {bar}"
        )


def main() -> None:
    """Run the full AETHER data preparation pipeline."""
    parser = argparse.ArgumentParser(
        description="Prepare AETHER training data from all SAP data sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python -m core.data.prepare_training_data\n'
            '  python -m core.data.prepare_training_data --max-bpi 10000\n'
            '  python -m core.data.prepare_training_data --output-dir /tmp/data\n'
        ),
    )
    parser.add_argument(
        "--max-bpi",
        type=int,
        default=50000,
        help="Maximum BPI 2019 cases to load (default: 50000, 0=unlimited)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_PATHS["output_dir"]),
        help=f"Output directory (default: {DEFAULT_PATHS['output_dir']})",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train/val split ratio (default: 0.8)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
        help="Maximum events per case (default: 256)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    max_bpi = args.max_bpi if args.max_bpi > 0 else None
    output_dir = Path(args.output_dir)

    _print_header("AETHER Data Preparation Pipeline")
    print(f"  Output:       {output_dir}")
    print(f"  Train ratio:  {args.train_ratio}")
    print(f"  Max BPI:      {max_bpi or 'unlimited'}")
    print(f"  Max seq len:  {args.max_seq_len}")
    print(f"  Seed:         {args.seed}")

    # Check which source files exist
    _print_header("Source File Check")
    source_keys = ["sap_sqlite", "bpi2019_json", "o2c_csv", "p2p_csv", "ocel_p2p"]
    for key in source_keys:
        path = DEFAULT_PATHS[key]
        exists = path.exists()
        marker = "[+]" if exists else "[-]"
        print(f"  {marker} {key}: {path}")

    # Run pipeline
    _print_header("Loading Data Sources")
    start_time = time.time()

    pipeline = AetherDataPipeline(
        paths={"output_dir": output_dir},
        train_ratio=args.train_ratio,
        max_bpi_cases=max_bpi,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )

    cases = pipeline.load_all_sources()

    load_time = time.time() - start_time
    print(f"\n  Loaded {_format_count(len(cases))} cases in {load_time:.1f}s")

    if not cases:
        print("\n  ERROR: No data loaded from any source.")
        print("  Check that data files exist at the expected paths.")
        sys.exit(1)

    # Source statistics
    _print_header("Source Statistics")
    _print_source_table(pipeline.source_counts)

    # Save processed data
    _print_header("Processing and Saving")
    save_start = time.time()
    metadata = pipeline.save_processed_data(cases, output_dir)
    save_time = time.time() - save_start
    print(f"  Saved to {output_dir} in {save_time:.1f}s")

    # Print results
    _print_header("Vocabulary Sizes")
    _print_vocabulary_summary(metadata)

    _print_header("Dataset Split")
    print(f"  Total cases:   {_format_count(metadata.get('total_cases', 0))}")
    print(f"  Train cases:   {_format_count(metadata.get('train_cases', 0))}")
    print(f"  Val cases:     {_format_count(metadata.get('val_cases', 0))}")

    _print_header("Event Distribution")
    _print_event_stats(metadata)

    _print_header("Source Distribution (Cases)")
    _print_source_distribution(metadata)

    _print_header("Top 20 Activities")
    _print_top_activities(metadata)

    # Output files
    _print_header("Output Files")
    for filename in ["train_cases.json", "val_cases.json", "vocabulary.json", "metadata.json"]:
        filepath = output_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  [+] {filepath}  ({size_mb:.1f} MB)")
        else:
            print(f"  [-] {filepath}  (not created)")

    total_time = time.time() - start_time
    _print_header("Done")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Ready for training with: EventSequenceDataset('{output_dir / 'train_cases.json'}')")
    print()


if __name__ == "__main__":
    main()
