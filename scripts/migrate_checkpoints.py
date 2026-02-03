#!/usr/bin/env python3
"""Migrate legacy .pt checkpoints to SafeTensors format.

SafeTensors provides secure model loading by storing only tensor data,
eliminating the risk of arbitrary code execution from pickle deserialization.

Usage:
    # Preview what would be migrated (no changes)
    python scripts/migrate_checkpoints.py --dry-run

    # Migrate all checkpoints in data/models/
    python scripts/migrate_checkpoints.py

    # Migrate a specific checkpoint
    python scripts/migrate_checkpoints.py data/models/best.pt

    # Migrate to a different directory
    python scripts/migrate_checkpoints.py data/models/best.pt --output-dir /path/to/new/

    # Keep original .pt files (default: delete after successful migration)
    python scripts/migrate_checkpoints.py --keep-originals
"""

import argparse
import sys
from pathlib import Path

# Add AETHER root to path
AETHER_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(AETHER_ROOT))

from core.utils.checkpoint import (
    migrate_checkpoint_to_safetensors,
    MODEL_STATE_KEYS,
)


def find_legacy_checkpoints(directory: Path) -> list[Path]:
    """Find all legacy .pt checkpoint files that need migration.

    Skips files that already have a corresponding .safetensors file.
    """
    checkpoints = []
    for ext in [".pt", ".pth"]:
        for pt_file in directory.rglob(f"*{ext}"):
            # Skip training state files
            if str(pt_file).endswith(".training.pt"):
                continue

            # Skip if SafeTensors version already exists
            safetensors_path = pt_file.with_suffix(".safetensors")
            if safetensors_path.exists():
                continue

            checkpoints.append(pt_file)

    return sorted(checkpoints)


def migrate_checkpoint(
    pt_path: Path,
    output_dir: Path | None = None,
    dry_run: bool = False,
    keep_original: bool = False,
) -> tuple[bool, str]:
    """Migrate a single checkpoint file.

    Returns:
        (success, message) tuple
    """
    if dry_run:
        return True, f"Would migrate: {pt_path}"

    try:
        st_path, train_path = migrate_checkpoint_to_safetensors(
            pt_path,
            output_dir=output_dir,
            trusted_source=True,
            model_keys=MODEL_STATE_KEYS,
        )

        msg = f"Migrated: {pt_path} -> {st_path}"
        if train_path:
            msg += f" + {train_path}"

        # Optionally remove original
        if not keep_original:
            pt_path.unlink()
            msg += " (original deleted)"

        return True, msg

    except Exception as e:
        return False, f"FAILED: {pt_path} - {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Migrate legacy .pt checkpoints to SafeTensors format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Specific checkpoint files to migrate. If not provided, scans data/models/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be migrated without making changes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for migrated files (default: same as input)",
    )
    parser.add_argument(
        "--keep-originals",
        action="store_true",
        help="Keep original .pt files after migration (default: delete them)",
    )
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=AETHER_ROOT / "data" / "models",
        help="Directory to scan for checkpoints (default: data/models/)",
    )

    args = parser.parse_args()

    # Determine which files to migrate
    if args.paths:
        checkpoints = [p for p in args.paths if p.exists()]
        missing = [p for p in args.paths if not p.exists()]
        if missing:
            print(f"Warning: Files not found: {missing}")
    else:
        if not args.scan_dir.exists():
            print(f"Checkpoint directory not found: {args.scan_dir}")
            print("No checkpoints to migrate.")
            return 0

        checkpoints = find_legacy_checkpoints(args.scan_dir)

    if not checkpoints:
        print("No legacy checkpoints found that need migration.")
        return 0

    # Preview or migrate
    if args.dry_run:
        print("=== DRY RUN - No changes will be made ===\n")

    print(f"Found {len(checkpoints)} checkpoint(s) to migrate:\n")

    successes = 0
    failures = 0

    for pt_path in checkpoints:
        success, msg = migrate_checkpoint(
            pt_path,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            keep_original=args.keep_originals,
        )
        print(f"  {msg}")
        if success:
            successes += 1
        else:
            failures += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {successes} succeeded, {failures} failed")

    if args.dry_run:
        print("\nTo perform migration, run without --dry-run")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
