"""W&B summary-writer adapter commit semantics.

With an explicit ``step``, W&B defaults to ``commit=False`` and holds the row
until a later step arrives — a finished epoch must instead upload on flush(),
step advance, and close(), each as one committed history row.
"""
import pytest

from agoraforge.conf.training.logging import default as logging_defaults
from agoraforge.training.artifacts.outputs import (
    WandbSummaryWriter, assert_run_dir_available, run_output_dir,
)


def test_wandb_logging_is_opt_in():
    assert logging_defaults().wandb_enabled is False


class StubRun:
    def __init__(self):
        self.logged = []
        self.finished = False

    def log(self, row, step=None, commit=None):
        self.logged.append((dict(row), step, commit))

    def finish(self):
        self.finished = True


def test_run_name_is_one_path_segment(tmp_path):
    assert run_output_dir(str(tmp_path), "run") == str(tmp_path / "run")
    with pytest.raises(ValueError):
        run_output_dir(str(tmp_path), "nested/run")


def test_existing_run_directory_is_rejected(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(FileExistsError):
        assert_run_dir_available(str(run_dir))


def test_flush_commits_one_row_per_step():
    run = StubRun()
    writer = WandbSummaryWriter(run)
    writer.add_scalar("a", 1.0, 0)
    writer.add_scalar("b", 2.0, 0)
    assert run.logged == []
    writer.flush()
    assert run.logged == [({"a": 1.0, "b": 2.0}, 0, True)]
    writer.flush()
    assert len(run.logged) == 1


def test_step_advance_commits_previous_row():
    run = StubRun()
    writer = WandbSummaryWriter(run)
    writer.add_scalar("a", 1.0, 0)
    writer.add_scalar("a", 2.0, 1)
    assert run.logged == [({"a": 1.0}, 0, True)]


def test_close_flushes_pending_row_and_finishes():
    run = StubRun()
    writer = WandbSummaryWriter(run)
    writer.add_scalar("a", 1.0, 5)
    writer.close()
    assert run.logged == [({"a": 1.0}, 5, True)]
    assert run.finished
