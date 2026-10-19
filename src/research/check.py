"""Research-hygiene checks (run by ``scripts/research/check.py`` and the test suite).

Errors break the protocol; warnings are known, accepted gaps (for example
pre-ledger results imported as ``legacy``). See ``docs/research_protocol.md``.
"""

from __future__ import annotations

import datetime as _dt
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from research import ledger as ledger_lib
from research.records import PROTECTED_SECTIONS, list_records, protected_text, sections, status

RUN_ID = re.compile(r"\b[a-z][a-z0-9_]*/\d{8}-\d{6}(?:-\d+)?\b")
CLAIM_HEADING = re.compile(r"^### (C-\d{3})\b")
MAX_STATUS_LINES = 80
APPROVED_STATUSES = ("approved", "running", "done", "abandoned")


@dataclass(frozen=True)
class Finding:
    level: str  # "error" | "warning"
    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.level.upper():7s} {self.where}: {self.message}"


def _git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _commit_exists(repo: Path, commit: str | None) -> bool:
    return bool(commit) and _git(repo, "cat-file", "-e", f"{commit}^{{commit}}") is not None


def check_ledger(repo: Path, entries: list[dict]) -> list[Finding]:
    findings: list[Finding] = []
    legacy_count = sum(1 for e in entries if e.get("legacy"))
    if legacy_count:
        findings.append(Finding(
            "warning", "ledger", f"{legacy_count} legacy entries (pre-ledger: no dirty flag, not registered; grade C/B at most)",
        ))
    seen: set[str] = set()
    for entry in entries:
        where = f"ledger {entry.get('run_id')}"
        if entry["run_id"] in seen:
            findings.append(Finding("error", where, "duplicate run_id"))
        seen.add(entry["run_id"])
        legacy = entry.get("legacy", False)
        if not _commit_exists(repo, entry.get("commit")):
            findings.append(Finding("warning" if legacy else "error", where, f"commit {entry.get('commit')} not in git"))
        if not legacy and entry.get("dirty"):
            findings.append(Finding("warning", where, "run from a dirty tree: ungradeable"))
        path = repo / entry["path"]
        if path.exists():
            if ledger_lib.file_sha256(path) != entry.get("sha256"):
                findings.append(Finding("error", where, "result file changed after it was recorded"))
    return findings


def check_experiment_runs(repo: Path, entries: list[dict], records_dir: Path) -> list[Finding]:
    """Every run of an experiment: registered before it ran, protected sections unchanged since."""

    findings: list[Finding] = []
    records = {p.name.split("-", 2)[0] + "-" + p.name.split("-", 2)[1]: p for p in list_records(records_dir)}
    for entry in entries:
        experiment = entry.get("experiment")
        if not experiment:
            continue
        where = f"ledger {entry['run_id']} ({experiment})"
        record = records.get(experiment)
        if record is None:
            findings.append(Finding("error", where, "no experiment record"))
            continue
        current_text = record.read_text()
        if status(current_text) not in APPROVED_STATUSES:
            findings.append(Finding("error", where, f"record status is {status(current_text)!r}: ran without approval"))
        relative = record.resolve().relative_to(repo.resolve())
        then = _git(repo, "show", f"{entry.get('commit')}:{relative}")
        if then is None:
            findings.append(Finding("error", where, "record was not committed before the run (no pre-registration)"))
            continue
        before, now = protected_text(then), protected_text(current_text)
        for name in PROTECTED_SECTIONS:
            if not before[name]:
                findings.append(Finding("error", where, f"section {name!r} was empty when the run started"))
            elif before[name] != now[name]:
                findings.append(Finding("error", where, f"section {name!r} edited after the run"))
    return findings


def check_records(records_dir: Path) -> list[Finding]:
    findings: list[Finding] = []
    for record in list_records(records_dir):
        text = record.read_text()
        where = str(record.name)
        body = sections(text)
        level = "warning" if status(text) == "draft" else "error"
        for name in PROTECTED_SECTIONS:
            if not body.get(name, "").strip():
                findings.append(Finding(level, where, f"missing section {name!r}"))
        if "stop" not in body.get("Decision rule", "").lower():
            findings.append(Finding(level, where, "decision rule has no stop branch"))
        if "**budget:**" not in text.lower():
            findings.append(Finding(level, where, "no budget"))
        if not status(text):
            findings.append(Finding("error", where, "no status"))
    return findings


def check_claims(claims_path: Path, entries: list[dict]) -> list[Finding]:
    findings: list[Finding] = []
    if not claims_path.exists():
        return [Finding("error", claims_path.name, "missing")]
    index = ledger_lib.by_id(entries)
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for line in claims_path.read_text().splitlines():
        match = CLAIM_HEADING.match(line)
        if match:
            current = match[1]
            if current in blocks:
                findings.append(Finding("error", current, "duplicate claim ID"))
            blocks[current] = []
        elif line.startswith("## "):
            current = None
        elif current is not None:
            blocks[current].append(line)
    for claim, lines in blocks.items():
        text = "\n".join(lines)
        grade_match = re.search(r"\*\*Grade:\*\*\s*([ABC])\b", text)
        if not grade_match:
            findings.append(Finding("error", claim, "no grade (A, B or C)"))
            continue
        grade = grade_match[1]
        evidence = next((l for l in lines if l.strip().startswith("- **Evidence:**")), "")
        cited = RUN_ID.findall(evidence)
        if grade in "AB":
            if not cited:
                findings.append(Finding("error", claim, f"grade {grade} cites no ledger run"))
            if "scratch" in evidence.lower():
                findings.append(Finding("error", claim, f"grade {grade} rests on scratch code"))
        for run in cited:
            entry = index.get(run)
            if entry is None:
                findings.append(Finding("error", claim, f"cites {run}, not in the ledger"))
            elif grade == "A" and (entry.get("legacy") or entry.get("dirty") or not entry.get("experiment")):
                findings.append(Finding("error", claim, f"grade A cites {run}, which is legacy, dirty or unregistered"))
        if grade == "A" and len({index[r].get("seed") for r in cited if r in index}) < 3:
            findings.append(Finding("error", claim, "grade A needs at least 3 seeds"))
    return findings


def check_docs(repo: Path, today: _dt.date | None = None) -> list[Finding]:
    findings: list[Finding] = []
    status_path = repo / "docs" / "current_status.md"
    if status_path.exists() and len(status_path.read_text().splitlines()) > MAX_STATUS_LINES:
        findings.append(Finding("warning", "docs/current_status.md", f"longer than {MAX_STATUS_LINES} lines"))
    charter = repo / "docs" / "charter.md"
    if not charter.exists():
        return findings + [Finding("error", "docs/charter.md", "missing")]
    match = re.search(r"\*\*Stop date:\*\*\s*(\d{4}-\d{2}-\d{2})", charter.read_text())
    if not match:
        findings.append(Finding("error", "docs/charter.md", "no stop date"))
    elif (today or _dt.date.today()) > _dt.date.fromisoformat(match[1]):
        findings.append(Finding("warning", "docs/charter.md", f"stop date {match[1]} has passed: record the decision"))
    return findings


def run_all(repo: Path, *, ledger_path: Path | None = None, today: _dt.date | None = None) -> list[Finding]:
    entries = ledger_lib.load(ledger_path or repo / "results" / "ledger.jsonl")
    records_dir = repo / "docs" / "experiments"
    return (
        check_ledger(repo, entries)
        + check_experiment_runs(repo, entries, records_dir)
        + check_records(records_dir)
        + check_claims(repo / "docs" / "claims.md", entries)
        + check_docs(repo, today)
    )
