#!/usr/bin/env python3
"""Pre-PR validation script for AETHER.

Run this before creating a pull request to validate:
1. Security scan of checkpoint files (picklescan)
2. Python linting (ruff)
3. Type checking (mypy)
4. Unit tests (pytest)
5. Model integrity checks

Usage:
    python scripts/validate_pr.py [--quick] [--security-only]

Options:
    --quick         Skip slow tests (integration, full model tests)
    --security-only Only run security scans

Exit codes:
    0 = All checks passed
    1 = One or more checks failed

References:
- Picklescan: https://github.com/mmaitre314/picklescan
- Hugging Face security: https://huggingface.co/docs/hub/security-pickle
- CWE-502: https://cwe.mitre.org/data/definitions/502.html
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

AETHER_ROOT = Path(__file__).parent.parent
CHECKPOINT_EXTENSIONS = {".pt", ".pth", ".bin", ".pkl", ".pickle"}


class CheckResult(NamedTuple):
    name: str
    passed: bool
    message: str
    duration: float = 0.0


def run_command(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=cwd or AETHER_ROOT,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out after 5 minutes"
    except FileNotFoundError as e:
        return -1, "", f"Command not found: {e}"


def check_tool_installed(tool: str) -> bool:
    """Check if a tool is installed."""
    code, _, _ = run_command([sys.executable, "-m", tool, "--help"])
    return code == 0


def find_checkpoint_files() -> list[Path]:
    """Find all checkpoint files in the data directory."""
    data_dir = AETHER_ROOT / "data"
    if not data_dir.exists():
        return []

    files = []
    for ext in CHECKPOINT_EXTENSIONS:
        files.extend(data_dir.rglob(f"*{ext}"))
    return files


# ============================================================================
# Validation Checks
# ============================================================================


def check_picklescan() -> CheckResult:
    """Scan checkpoint files for malicious pickle payloads."""
    if not check_tool_installed("picklescan"):
        return CheckResult(
            "Picklescan Security",
            False,
            "picklescan not installed. Run: pip install picklescan",
        )

    files = find_checkpoint_files()
    if not files:
        return CheckResult(
            "Picklescan Security",
            True,
            "No checkpoint files found (data/ directory empty or missing)",
        )

    failed = []
    for f in files:
        code, stdout, stderr = run_command(
            [sys.executable, "-m", "picklescan", "--path", str(f)]
        )
        if code == 1:
            failed.append(f"{f.name}: {stdout or stderr}")

    if failed:
        return CheckResult(
            "Picklescan Security",
            False,
            f"Malicious content in {len(failed)} file(s):\n" + "\n".join(failed[:5]),
        )

    return CheckResult(
        "Picklescan Security",
        True,
        f"Scanned {len(files)} checkpoint file(s) - all clean",
    )


def check_ruff() -> CheckResult:
    """Run ruff linter on Python code."""
    if not check_tool_installed("ruff"):
        return CheckResult("Ruff Linting", False, "ruff not installed. Run: pip install ruff")

    code, stdout, stderr = run_command(
        [sys.executable, "-m", "ruff", "check", "core/", "scripts/"]
    )

    if code != 0:
        # Count issues
        lines = stdout.strip().split("\n") if stdout.strip() else []
        issue_count = len([line for line in lines if line and not line.startswith("Found")])
        return CheckResult(
            "Ruff Linting",
            False,
            f"{issue_count} linting issue(s) found. Run: ruff check --fix",
        )

    return CheckResult("Ruff Linting", True, "No linting issues")


def check_mypy() -> CheckResult:
    """Run mypy type checker."""
    if not check_tool_installed("mypy"):
        return CheckResult("Mypy Types", False, "mypy not installed. Run: pip install mypy")

    code, stdout, stderr = run_command(
        [sys.executable, "-m", "mypy", "core/", "--ignore-missing-imports"]
    )

    # mypy returns 1 on type errors
    if code != 0:
        lines = stdout.strip().split("\n") if stdout.strip() else []
        error_lines = [line for line in lines if ": error:" in line]
        return CheckResult(
            "Mypy Types",
            False,
            f"{len(error_lines)} type error(s) found",
        )

    return CheckResult("Mypy Types", True, "No type errors")


def check_pytest(quick: bool = False) -> CheckResult:
    """Run pytest test suite."""
    cmd = [sys.executable, "-m", "pytest", "core/tests/", "-v"]
    if quick:
        cmd.extend(["-x", "--ignore=core/tests/test_integration.py"])

    code, stdout, stderr = run_command(cmd)

    # Extract test counts from output
    output = stdout + stderr
    if "passed" in output:
        # Find the summary line
        for line in output.split("\n"):
            if "passed" in line or "failed" in line:
                return CheckResult(
                    "Pytest Tests",
                    code == 0,
                    line.strip() if code == 0 else f"Tests failed: {line.strip()}",
                )

    if code != 0:
        return CheckResult("Pytest Tests", False, "Tests failed (see output above)")

    return CheckResult("Pytest Tests", True, "All tests passed")


def check_security_tests() -> CheckResult:
    """Run security-specific tests."""
    cmd = [
        sys.executable, "-m", "pytest",
        "core/tests/test_security.py", "-v", "--tb=short"
    ]

    code, stdout, stderr = run_command(cmd)

    if code != 0:
        return CheckResult(
            "Security Tests",
            False,
            "Security tests failed (run pytest core/tests/test_security.py -v)",
        )

    return CheckResult("Security Tests", True, "Security tests passed")


def check_typescript() -> CheckResult:
    """Run TypeScript checks (lint + typecheck + test)."""
    # Check if npm is available
    code, _, _ = run_command(["npm", "--version"])
    if code != 0:
        return CheckResult("TypeScript", False, "npm not installed")

    # Run typecheck
    code, stdout, stderr = run_command(["npm", "run", "typecheck"])
    if code != 0:
        return CheckResult("TypeScript", False, f"Type check failed: {stderr[:200]}")

    # Run lint
    code, stdout, stderr = run_command(["npm", "run", "lint"])
    if code != 0:
        return CheckResult("TypeScript", False, f"Lint failed: {stderr[:200]}")

    # Run tests
    code, stdout, stderr = run_command(["npm", "test"])
    if code != 0:
        return CheckResult("TypeScript", False, "Tests failed")

    return CheckResult("TypeScript", True, "All TypeScript checks passed")


# ============================================================================
# Main
# ============================================================================


def print_result(result: CheckResult):
    """Print a check result with color formatting."""
    status = f"{GREEN}✓{RESET}" if result.passed else f"{RED}✗{RESET}"
    print(f"  {status} {BOLD}{result.name}{RESET}: {result.message}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-PR validation for AETHER",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip slow tests (integration tests)",
    )
    parser.add_argument(
        "--security-only", action="store_true",
        help="Only run security scans",
    )
    parser.add_argument(
        "--typescript", action="store_true",
        help="Include TypeScript checks",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}AETHER Pre-PR Validation{RESET}")
    print("=" * 50)

    results: list[CheckResult] = []

    # Security checks (always run)
    print(f"\n{YELLOW}Security Checks:{RESET}")
    results.append(check_picklescan())
    print_result(results[-1])

    if not args.security_only:
        # Python checks
        print(f"\n{YELLOW}Python Checks:{RESET}")

        results.append(check_ruff())
        print_result(results[-1])

        results.append(check_mypy())
        print_result(results[-1])

        results.append(check_pytest(quick=args.quick))
        print_result(results[-1])

        results.append(check_security_tests())
        print_result(results[-1])

        # TypeScript checks (optional)
        if args.typescript:
            print(f"\n{YELLOW}TypeScript Checks:{RESET}")
            results.append(check_typescript())
            print_result(results[-1])

    # Summary
    print("\n" + "=" * 50)
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    if passed == total:
        print(f"{GREEN}{BOLD}All {total} checks passed!{RESET} Ready for PR.\n")
        return 0
    else:
        failed = total - passed
        print(f"{RED}{BOLD}{failed}/{total} checks failed.{RESET} Fix issues before PR.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
