"""Experiment records (``docs/experiments/E-NNN-slug.md``): parsing and protected sections.

A record is registered by committing it. Its *protected* sections (prediction,
primary metric, baselines, decision rule) must not change after any run of the
experiment; ``research.check`` compares them with the record as it was at
each run's commit.
"""

from __future__ import annotations

import re
from pathlib import Path

from research.provenance import REPO_ROOT

EXPERIMENTS_DIR = REPO_ROOT / "docs" / "experiments"
PROTECTED_SECTIONS = ("Prediction", "Primary metric", "Baselines", "Decision rule")
REPORT_BEGIN = "<!-- report:begin -->"
REPORT_END = "<!-- report:end -->"
STATUSES = ("draft", "registered", "approved", "running", "done", "abandoned")

_ID = re.compile(r"^E-\d{3}$")
_FIELD = re.compile(r"^- \*\*(?P<key>[^*]+):\*\*\s*(?P<value>.*)$")


def find_record(experiment: str, root: Path = EXPERIMENTS_DIR) -> Path | None:
    if not _ID.match(experiment):
        return None
    matches = sorted(root.glob(f"{experiment}-*.md"))
    return matches[0] if matches else None


def list_records(root: Path = EXPERIMENTS_DIR) -> list[Path]:
    return sorted(p for p in root.glob("E-[0-9][0-9][0-9]-*.md"))


def sections(text: str) -> dict[str, str]:
    """``## Heading`` -> body (stripped), in file order."""

    found: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current is not None:
                found[current] = "\n".join(lines).strip()
            current, lines = line[3:].strip(), []
        elif current is not None:
            lines.append(line)
    if current is not None:
        found[current] = "\n".join(lines).strip()
    return found


def protected_text(text: str) -> dict[str, str]:
    body = sections(text)
    return {name: body.get(name, "") for name in PROTECTED_SECTIONS}


def header_fields(text: str) -> dict[str, str]:
    """The ``- **Key:** value`` lines before the first ``##`` heading."""

    fields: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("## "):
            break
        match = _FIELD.match(line.strip())
        if match:
            fields[match["key"].strip().lower()] = match["value"].strip()
    return fields


def status(text: str) -> str:
    value = header_fields(text).get("status", "").split()[0:1]
    return value[0].lower() if value else ""


def replace_report(text: str, table: str) -> str:
    """Put ``table`` between the report markers (appending them under ``## Result`` if absent)."""

    block = f"{REPORT_BEGIN}\n{table.rstrip()}\n{REPORT_END}"
    if REPORT_BEGIN in text and REPORT_END in text:
        head, _, rest = text.partition(REPORT_BEGIN)
        _, _, tail = rest.partition(REPORT_END)
        return head + block + tail
    if "\n## Result" in text:
        head, _, rest = text.partition("\n## Result")
        heading_line, _, tail = rest.partition("\n")
        return f"{head}\n## Result{heading_line}\n\n{block}\n{tail}"
    return text.rstrip() + f"\n\n## Result\n\n{block}\n"
