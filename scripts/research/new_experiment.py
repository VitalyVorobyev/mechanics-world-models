"""Scaffold an experiment record from the template.

Usage:
  .venv/bin/python scripts/research/new_experiment.py --slug bracket-sysid \
      --title "Bracket H1b with known-form system ID" --node Q1.1 --hypotheses H1b

Writes docs/experiments/E-NNN-<slug>.md (next free number) with status
``draft``. Fill in the protected sections, then *commit it* (registration)
before asking for approval and running anything.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re

from research.records import EXPERIMENTS_DIR, list_records

TEMPLATE = EXPERIMENTS_DIR / "TEMPLATE.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--node", required=True, help="question node in docs/charter.md, e.g. Q1.1")
    parser.add_argument("--hypotheses", default="", help="comma-separated registry IDs")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9-]+", args.slug):
        parser.error("slug: lowercase letters, digits and dashes")
    numbers = [int(p.name[2:5]) for p in list_records()]
    experiment = f"E-{(max(numbers) + 1) if numbers else 1:03d}"
    text = (
        TEMPLATE.read_text()
        .replace("E-NNN", experiment)
        .replace("<title>", args.title)
        .replace("<node>", args.node)
        .replace("<hypotheses>", args.hypotheses or "—")
        .replace("<date>", dt.date.today().isoformat())
    )
    path = EXPERIMENTS_DIR / f"{experiment}-{args.slug}.md"
    path.write_text(text)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
