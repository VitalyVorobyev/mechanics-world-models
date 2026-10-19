"""Seed-aggregated tables from recorded results (replaces hand-typed tables).

Usage:
  # an experiment: its ledger runs, the metrics on its record's "Report:" line;
  # writes the table into the record and results/E-NNN/summary.json
  .venv/bin/python scripts/research/report.py E-001

  # exploration / audit: any result files, filtered by config values
  .venv/bin/python scripts/research/report.py \
      --glob 'runs/oracles/context_oracle/2026*.json' \
      --where pose_source=null --where 'shifts=["gear_x1.3","gear_x0.5","gear_x2","gear_x3"]' \
      --metric 'mlp_ctx_over_lag_ctx_ratio_k1_h20' --metric '{shift}.k1.mlp_ctx.dev_h20 / {shift}.k1.lag_ctx.dev_h20'

Metric patterns: an exact name; ``{shift}`` expanded over the run's shifts;
``a / b`` a per-seed ratio; ``re:<regex>`` every matching name.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from research import ledger
from research.provenance import REPO_ROOT
from research.records import find_record, replace_report
from research.report import filter_conditions, group_runs, markdown, parse_where, report_patterns, summary_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", nargs="?", help="E-NNN: report this experiment's ledger runs")
    parser.add_argument("--glob", action="append", default=[], help="result files to aggregate instead")
    parser.add_argument("--where", action="append", default=[], help="config filter key=value (JSON values)")
    parser.add_argument("--metric", action="append", default=[], help="metric pattern (repeatable)")
    parser.add_argument("--no-write", action="store_true", help="print only; do not update the record")
    args = parser.parse_args()

    record = None
    if args.experiment:
        record = find_record(args.experiment)
        if record is None:
            parser.error(f"no record for {args.experiment}")
        entries = [e for e in ledger.load() if e.get("experiment") == args.experiment]
        paths = [REPO_ROOT / e["path"] for e in entries]
        missing = [p for p in paths if not p.exists()]
        if missing:
            parser.error(f"{len(missing)} result file(s) missing locally, e.g. {missing[0]}")
        patterns = args.metric or report_patterns(record.read_text())
    else:
        paths = [Path(p) for pattern in args.glob for p in glob.glob(pattern)]
        patterns = args.metric
    if not paths:
        parser.error("no result files")
    if not patterns:
        parser.error("no metric patterns (pass --metric or put a `Report:` line in the record's Primary metric)")

    conditions = filter_conditions(group_runs(paths), parse_where(args.where))
    table = markdown(conditions, patterns)
    print(table)
    if record is not None and not args.no_write:
        record.write_text(replace_report(record.read_text(), table))
        out = REPO_ROOT / "results" / args.experiment / "summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary_json(conditions, patterns), indent=2, sort_keys=True, default=str) + "\n")
        print(f"\nupdated {record.relative_to(REPO_ROOT)} and {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
