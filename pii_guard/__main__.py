"""
Allow running the PII-Guard CLI as a module:
    python3 -m pii_guard <command> [options]

This is equivalent to calling the `piiguard` console script entry point.
"""
import sys

from pii_guard.cli import main

if __name__ == "__main__":
    sys.exit(main())
