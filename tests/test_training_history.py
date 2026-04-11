from __future__ import annotations

import re
from io import StringIO

from rich.console import Console

from train.train_rssm import append_history_record, log_metrics, prepare_history_file
from viz.plot_training_history import load_history, plot_training_history


def test_training_history_jsonl_round_trip(tmp_path) -> None:
    history_path = tmp_path / "history.jsonl"
    prepare_history_file(history_path, resume=False)

    append_history_record(
        history_path=history_path,
        split="train",
        epoch=1,
        global_step=10,
        metrics={
            "total_loss": 0.5,
            "reconstruction_loss": 0.4,
            "kl_loss": 0.2,
            "weighted_kl_loss": 0.1,
        },
    )
    append_history_record(
        history_path=history_path,
        split="val",
        epoch=1,
        global_step=10,
        metrics={
            "total_loss": 0.6,
            "reconstruction_loss": 0.5,
            "kl_loss": 0.3,
            "weighted_kl_loss": 0.15,
        },
    )

    records = load_history(history_path)

    assert len(records) == 2
    assert records[0]["split"] == "train"
    assert records[0]["epoch"] == 1
    assert records[0]["global_step"] == 10
    assert records[0]["total_loss"] == 0.5
    assert records[1]["split"] == "val"


def test_plot_training_history_writes_png(tmp_path) -> None:
    history_path = tmp_path / "history.jsonl"
    prepare_history_file(history_path, resume=False)
    for step in range(1, 4):
        append_history_record(
            history_path=history_path,
            split="train",
            epoch=1,
            global_step=step,
            metrics={
                "total_loss": 1.0 / step,
                "reconstruction_loss": 0.8 / step,
                "kl_loss": 0.2 / step,
                "weighted_kl_loss": 0.1 / step,
            },
        )
    append_history_record(
        history_path=history_path,
        split="val",
        epoch=1,
        global_step=3,
        metrics={
            "total_loss": 0.25,
            "reconstruction_loss": 0.2,
            "kl_loss": 0.1,
            "weighted_kl_loss": 0.05,
        },
    )

    output_path = plot_training_history(
        history_path=history_path,
        output_path=tmp_path / "loss_history.png",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_terminal_log_includes_timestamp_without_changing_history(tmp_path) -> None:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=200)
    history_path = tmp_path / "history.jsonl"
    prepare_history_file(history_path, resume=False)

    log_metrics(
        split="train",
        epoch=2,
        global_step=12,
        metrics={"total_loss": 0.125},
        console=console,
    )
    append_history_record(
        history_path=history_path,
        split="train",
        epoch=2,
        global_step=12,
        metrics={"total_loss": 0.125},
    )

    terminal_text = stream.getvalue()
    records = load_history(history_path)

    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", terminal_text)
    assert "epoch=0002" in terminal_text
    assert "global_step=0000012" in terminal_text
    assert "split=train" in terminal_text
    assert "timestamp" not in records[0]
