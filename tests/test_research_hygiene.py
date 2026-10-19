"""Tests for the research bookkeeping: provenance, ledger, pre-registration checks, reports."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import pytest

from oracles.common import Criterion, OracleResult
from research import ledger, provenance
from research.check import check_claims, check_experiment_runs, check_records, run_all
from research.records import REPORT_BEGIN, REPORT_END, protected_text, replace_report, status
from research.report import group_runs, markdown, report_patterns, summarize

RECORD = """# E-001 — demo

- **Status:** {status}
- **Question node:** Q1
- **Budget:** 1 agent-hour

## Question
Does it work?

## Prediction
Ratio about 2 (60%).

## Primary metric
Report: `ratio`

## Baselines
Floor and ceiling.

## Decision rule
Ratio >= 1.5: continue. Otherwise stop.

## Result
"""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README").write_text("x\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def _register(repo: Path, record_status: str = "approved") -> Path:
    path = repo / "docs" / "experiments" / "E-001-demo.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(RECORD.format(status=record_status))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "register E-001")
    return path


def _entry(commit: str, experiment: str | None = "E-001", **extra) -> dict:
    return {
        "run_id": "demo/20261019-000000", "path": "runs/demo/20261019-000000.json", "sha256": "",
        "experiment": experiment, "commit": commit, "dirty": False, "seed": 0, "legacy": False, **extra,
    }


def test_registered_run_passes(repo: Path) -> None:
    _register(repo)
    findings = check_experiment_runs(repo, [_entry(_git(repo, "rev-parse", "HEAD"))], repo / "docs" / "experiments")
    assert findings == []


def test_prediction_edited_after_run_is_an_error(repo: Path) -> None:
    record = _register(repo)
    commit = _git(repo, "rev-parse", "HEAD")
    record.write_text(record.read_text().replace("Ratio about 2", "Ratio about 3"))
    messages = [f.message for f in check_experiment_runs(repo, [_entry(commit)], repo / "docs" / "experiments")]
    assert any("'Prediction' edited after the run" in m for m in messages)


def test_run_before_registration_is_an_error(repo: Path) -> None:
    before = _git(repo, "rev-parse", "HEAD")
    _register(repo)
    messages = [f.message for f in check_experiment_runs(repo, [_entry(before)], repo / "docs" / "experiments")]
    assert any("not committed before the run" in m for m in messages)


def test_run_without_approval_is_an_error(repo: Path) -> None:
    _register(repo, record_status="registered")
    commit = _git(repo, "rev-parse", "HEAD")
    messages = [f.message for f in check_experiment_runs(repo, [_entry(commit)], repo / "docs" / "experiments")]
    assert any("ran without approval" in m for m in messages)


def test_record_without_stop_branch_is_flagged(tmp_path: Path) -> None:
    records = tmp_path / "experiments"
    records.mkdir()
    (records / "E-002-x.md").write_text(RECORD.format(status="registered").replace("Otherwise stop.", "Otherwise rerun."))
    assert any("no stop branch" in f.message and f.level == "error" for f in check_records(records))


def test_claim_grades_are_enforced(tmp_path: Path) -> None:
    claims = tmp_path / "claims.md"
    claims.write_text(
        "## Claims\n\n"
        "### C-001 — fine\n- **Grade:** B\n- **Evidence:** `demo/20261019-000000`\n\n"
        "### C-002 — no evidence\n- **Grade:** B\n- **Evidence:** a hunch\n\n"
        "### C-003 — scratch\n- **Grade:** B\n- **Evidence:** `demo/20261019-000000`, scratchpad `x.py`\n\n"
        "### C-004 — legacy as A\n- **Grade:** A\n- **Evidence:** `demo/20261019-000000`\n\n"
        "### C-005 — unverified\n- **Grade:** C\n- **Evidence:** scratchpad `y.py`\n"
    )
    entries = [_entry("abc", experiment=None, legacy=True)]
    found = {(f.where, f.message.split(",")[0]) for f in check_claims(claims, entries)}
    wheres = {w for w, _ in found}
    assert "C-001" not in wheres and "C-005" not in wheres
    assert {"C-002", "C-003", "C-004"} <= wheres


def test_oracle_result_records_provenance_and_ledger(tmp_path: Path) -> None:
    run = provenance.begin_run(argparse.Namespace(experiment=None, allow_dirty=True), argv=["demo"])
    result = OracleResult(
        name="demo", hypotheses=["H0"], question="?", config={"seed": 3, "data": str(tmp_path)},
        metrics={"err": 0.1}, criteria=[Criterion("err", "<=", 0.2)],
    )
    ledger_path = tmp_path / "ledger.jsonl"
    path = result.write(root=tmp_path / "runs", run=run, ledger_path=ledger_path)
    payload = json.loads(path.read_text())
    assert payload["provenance"]["argv"] == ["demo"]
    assert payload["provenance"]["code"] and "torch" in payload["provenance"]["versions"]
    assert str(tmp_path) in payload["provenance"]["data"]
    [entry] = ledger.load(ledger_path)
    assert entry["run_id"] == f"demo/{path.stem}" and entry["seed"] == 3 and entry["passed"] is True
    assert entry["sha256"] == ledger.file_sha256(path)


def test_begin_run_refuses_dirty_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provenance, "git_dirty_paths", lambda: ["src/x.py"])
    with pytest.raises(SystemExit, match="uncommitted"):
        provenance.begin_run(argparse.Namespace(experiment=None, allow_dirty=False))
    assert provenance.begin_run(argparse.Namespace(experiment=None, allow_dirty=True)).dirty


def test_report_groups_seeds_supersedes_and_ratios(tmp_path: Path) -> None:
    def write(name: str, seed: int, a: float, b: float, extra: dict | None = None) -> None:
        payload = {"config": {"seed": seed, "shifts": ["s1"], **(extra or {})}, "metrics": {"s1.a": a, "s1.b": b}}
        (tmp_path / name).write_text(json.dumps(payload))

    write("20261019-000001.json", 0, 2.0, 1.0)
    write("20261019-000002.json", 1, 4.0, 1.0)
    write("20261019-000003.json", 1, 6.0, 1.0)  # rerun of seed 1: supersedes the previous file
    write("20261019-000004.json", 0, 1.0, 1.0, {"variant": True})
    conditions = group_runs(sorted(tmp_path.glob("*.json")))
    assert len(conditions) == 2
    base = next(c for c in conditions if "variant" not in c.config)
    assert [p.name for p in base.superseded] == ["20261019-000002.json"]
    stats = summarize(base, ["{shift}.a / {shift}.b"])["s1.a / s1.b"]
    assert stats["mean"] == pytest.approx(4.0) and stats["n"] == 2
    assert "`variant`" in markdown(conditions, ["s1.a"])


def test_record_helpers() -> None:
    text = RECORD.format(status="approved")
    assert status(text) == "approved"
    assert protected_text(text)["Prediction"].startswith("Ratio about 2")
    assert report_patterns(text) == ["ratio"]
    once = replace_report(text, "| a |")
    twice = replace_report(once, "| b |")
    assert twice.count(REPORT_BEGIN) == 1 and "| b |" in twice and "| a |" not in twice and REPORT_END in twice


def test_repository_hygiene_has_no_errors() -> None:
    """The live repo: ledger provenance, registrations, claims and charter (warnings allowed)."""

    errors = [f for f in run_all(provenance.REPO_ROOT, today=dt.date(2026, 10, 19)) if f.level == "error"]
    assert errors == [], "\n".join(map(str, errors))
