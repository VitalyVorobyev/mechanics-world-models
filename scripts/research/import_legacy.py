"""Import result files that predate the ledger as ``legacy`` entries (idempotent).

Usage:
  .venv/bin/python scripts/research/import_legacy.py 'runs/oracles/*/2026*.json'

Legacy entries record what was run and from which (claimed) commit, but they
have no dirty flag and no registered experiment, so claims citing them are
at most grade B.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from research import ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("patterns", nargs="+")
    args = parser.parse_args()
    known = {e["run_id"] for e in ledger.load()}
    added = 0
    for path in sorted({Path(p) for pattern in args.patterns for p in glob.glob(pattern)}):
        if path.name == "latest.json" or ledger.run_id(path) in known:
            continue
        payload = json.loads(path.read_text())
        ledger.append(ledger.entry_for(path, payload, legacy=True))
        known.add(ledger.run_id(path))
        added += 1
    print(f"added {added} legacy entries to {ledger.LEDGER_PATH}")


if __name__ == "__main__":
    main()
