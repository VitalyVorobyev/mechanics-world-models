"""Research-hygiene check: ledger provenance, pre-registration, claims, charter.

Usage:
  .venv/bin/python scripts/research/check.py            # errors and warnings
  .venv/bin/python scripts/research/check.py --errors   # errors only

Exits non-zero on any error. The same check runs in the test suite
(``tests/test_research_hygiene.py``).
"""

from __future__ import annotations

import argparse
import sys

from research.check import run_all
from research.provenance import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--errors", action="store_true", help="hide warnings")
    args = parser.parse_args()
    findings = run_all(REPO_ROOT)
    errors = [f for f in findings if f.level == "error"]
    for finding in findings:
        if finding.level == "error" or not args.errors:
            print(finding)
    print(f"{len(errors)} error(s), {len(findings) - len(errors)} warning(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
