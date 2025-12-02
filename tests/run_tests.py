#!/usr/bin/env python
"""
Run all SCI tests.

Usage:
    python tests/run_tests.py              # Run all tests
    python tests/run_tests.py --critical   # Run only critical tests
    python tests/run_tests.py --verbose    # Verbose output
"""

import sys
import pytest


def main():
    """Run tests with pytest."""

    args = [
        "tests/",
        "-v",              # Verbose
        "--tb=short",      # Short traceback
        "--strict-markers", # Strict marker checking
    ]

    # Check for arguments
    if "--critical" in sys.argv:
        # Run only critical tests
        args.extend([
            "-m", "critical",
            "-k", "test_data_leakage or test_abstraction_layer or test_pair_generation"
        ])
        print("Running CRITICAL tests only...")
    elif "--quick" in sys.argv:
        # Quick test (skip slow tests)
        args.extend(["-m", "not slow"])
        print("Running quick tests (skipping slow tests)...")
    else:
        print("Running all tests...")

    if "--verbose" in sys.argv or "-vv" in sys.argv:
        args.append("-vv")

    if "--coverage" in sys.argv:
        args.extend(["--cov=sci", "--cov-report=html", "--cov-report=term"])

    # Run pytest
    exit_code = pytest.main(args)

    # Summary
    print("\n" + "=" * 70)
    if exit_code == 0:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 70)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
