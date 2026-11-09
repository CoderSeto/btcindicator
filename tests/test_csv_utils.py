"""
Tests for csv_utils: the shared append path every CSV log goes through.

The race these pin (REVIEW_FINDINGS D7): with two writers on one log -- the dashboard and the
console bot, or two browser tabs -- both could see "needs header" on a new file and both write
one, so pd.read_csv read the second header as a data row and every numeric column came back as
strings. Two writers could also both try to rotate an older-layout file, and the second rename
raised. append_row() now does the whole check-rotate-write under one cross-process lock.

The stress tests use separate Python processes released at a shared start time, not threads,
because the lock has to hold across processes (the console bot and Streamlit are separate ones).
Without the lock they fail most runs (checked by disabling it); with it they pass every run.
"""
import os
import pathlib
import subprocess
import sys
import threading
import time

import pandas as pd

import csv_utils

ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
HEADER = ["timestamp", "writer", "seq", "price"]
WRITERS = 8
ROWS_EACH = 40


# Each writer is a separate Python program, as the real ones are (the console bot and Streamlit):
# it waits for a shared start time, then appends its rows as fast as it can.
_WRITER = """
import sys, time
sys.path.insert(0, sys.argv[1])
import csv_utils
path, writer_id, start_at = sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
header = sys.argv[5].split(",")
while time.time() < start_at:
    pass
for seq in range({rows}):
    csv_utils.append_row(path, header, ["2026-11-07T18:00:00", writer_id, seq, 75700.5])
""".format(rows=ROWS_EACH)


def _run_writers(path):
    start_at = time.time() + 2.0                                           # after every interpreter is up
    procs = [subprocess.Popen([sys.executable, "-c", _WRITER, ROOT, path, str(i), str(start_at),
                               ",".join(HEADER)])
             for i in range(WRITERS)]
    for p in procs:
        assert p.wait(60) == 0


def _assert_consistent(path):
    lines = open(path).read().splitlines()
    assert lines.count(",".join(HEADER)) == 1 and lines[0] == ",".join(HEADER)
    df = pd.read_csv(path)
    assert len(df) == WRITERS * ROWS_EACH
    assert df["price"].dtype.kind == "f"                                   # numeric, not strings
    assert sorted(df.groupby("writer")["seq"].count()) == [ROWS_EACH] * WRITERS
    assert not os.path.exists(path + ".lock")                              # every lock released


def test_concurrent_writers_on_a_new_file_write_one_header_and_every_row(tmp_path):
    path = str(tmp_path / "log.csv")
    _run_writers(path)
    _assert_consistent(path)


def test_concurrent_writers_rotate_an_older_layout_exactly_once(tmp_path):
    path = str(tmp_path / "log.csv")
    with open(path, "w") as f:
        f.write("timestamp,price\n2026-11-06T00:00:00,70000\n")              # last release's layout
    _run_writers(path)
    _assert_consistent(path)
    baks = list(tmp_path.glob("log.csv.*.bak"))
    assert len(baks) == 1
    assert pd.read_csv(baks[0])["price"].tolist() == [70000]


def test_a_writer_waits_while_another_holds_the_lock(tmp_path):
    path = str(tmp_path / "log.csv")
    with csv_utils.file_lock(path):
        t = threading.Thread(target=csv_utils.append_row, args=(path, HEADER, ["t", 1, 0, 1.0]))
        t.start()
        time.sleep(0.3)
        assert not os.path.exists(path)                                    # blocked, not written
    t.join(5)
    assert len(pd.read_csv(path)) == 1


def test_a_stale_lock_left_by_a_crashed_writer_is_cleared(tmp_path):
    path = str(tmp_path / "log.csv")
    lock = path + ".lock"
    open(lock, "w").close()
    old = time.time() - 60
    os.utime(lock, (old, old))
    assert csv_utils.append_row(path, HEADER, ["t", 1, 0, 1.0]) is True
    assert not os.path.exists(lock)
    assert len(pd.read_csv(path)) == 1


def test_a_live_lock_that_never_frees_skips_the_row_and_says_so(tmp_path, capsys):
    # Logging must never hang or crash the dashboard or the console loop.
    path = str(tmp_path / "log.csv")
    lock = path + ".lock"
    open(lock, "w").close()                                                # fresh: someone else's
    assert csv_utils.append_row(path, HEADER, ["t", 1, 0, 1.0], lock_timeout=0.2) is False
    assert not os.path.exists(path)
    assert os.path.exists(lock)                                            # not ours to delete
    assert "row not written" in capsys.readouterr().out


def test_skip_if_is_decided_under_the_lock_and_skips_the_write(tmp_path):
    path = str(tmp_path / "log.csv")
    seen = []

    def skip():
        seen.append(os.path.exists(path + ".lock"))
        return True

    csv_utils.append_row(path, HEADER, ["t", 1, 0, 1.0])
    assert csv_utils.append_row(path, HEADER, ["t", 1, 1, 1.0], skip_if=skip) is False
    assert seen == [True]                                                  # checked while locked
    assert len(pd.read_csv(path)) == 1


def test_skip_if_is_not_consulted_for_a_fresh_file(tmp_path):
    # Nothing to repeat yet; the first row is always written.
    path = str(tmp_path / "log.csv")
    assert csv_utils.append_row(path, HEADER, ["t", 1, 0, 1.0], skip_if=lambda: True) is True
    assert len(pd.read_csv(path)) == 1


# --- Windows: a lock deleted while someone still has it open ------------------------------------
# On Windows, deleting the lock while another process holds a handle on it (a waiter reading its
# age, an antivirus scan of a new file) leaves it "delete pending", and creating it again fails
# with PermissionError, not FileExistsError. Observed on the user's Windows 11 / Python 3.14
# machine: the two stress tests above crashed writers with
#   PermissionError: [Errno 13] Permission denied: '...\\log.csv.lock'
# These tests reproduce that refusal on any OS by patching os.open for the lock path only.

def _refuse_lock_creation(monkeypatch, times, exc=None):
    real_open = os.open
    calls = {"refused": 0}

    def fake_open(path, flags, *args, **kwargs):
        if str(path).endswith(".lock") and (times is None or calls["refused"] < times):
            calls["refused"] += 1
            raise exc or PermissionError(13, "Permission denied", str(path))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", fake_open)
    return calls


def test_a_delete_pending_lock_on_windows_is_waited_for_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(csv_utils, "_PERMISSION_ERROR_MEANS_BUSY", True, raising=False)
    calls = _refuse_lock_creation(monkeypatch, times=3)
    path = str(tmp_path / "log.csv")
    assert csv_utils.append_row(path, HEADER, ["t", 0, 0, 1.5]) is True
    assert calls["refused"] == 3
    assert pd.read_csv(path)["price"].tolist() == [1.5]
    assert not os.path.exists(path + ".lock")


def test_a_lock_that_stays_delete_pending_times_out_and_drops_the_row(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(csv_utils, "_PERMISSION_ERROR_MEANS_BUSY", True, raising=False)
    _refuse_lock_creation(monkeypatch, times=None)
    path = str(tmp_path / "log.csv")
    assert csv_utils.append_row(path, HEADER, ["t", 0, 0, 1.5], lock_timeout=0.1) is False
    assert "stayed busy" in capsys.readouterr().out
    assert not os.path.exists(path)


def test_off_windows_a_permission_error_is_a_real_error_and_is_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(csv_utils, "_PERMISSION_ERROR_MEANS_BUSY", False, raising=False)
    _refuse_lock_creation(monkeypatch, times=None)
    try:
        csv_utils.append_row(str(tmp_path / "log.csv"), HEADER, ["t", 0, 0, 1.5], lock_timeout=5)
    except PermissionError:
        return
    raise AssertionError("a genuine permission problem was hidden as a busy lock")


def test_a_lock_whose_age_cannot_be_read_still_times_out(tmp_path, monkeypatch):
    # The age check used to `continue` straight back to os.open with no sleep and no deadline, so
    # a lock that exists but can't be stat'ed (delete pending on Windows) spun the CPU forever.
    path = str(tmp_path / "log.csv")
    open(path + ".lock", "w").close()
    def unreadable(_):
        raise PermissionError(13, "Permission denied")
    monkeypatch.setattr(os.path, "getmtime", unreadable)
    outcome = {}

    def take():
        try:
            with csv_utils.file_lock(path, timeout=0.2):
                outcome["got"] = True
        except csv_utils.LockTimeout as e:
            outcome["timeout"] = str(e)

    t = threading.Thread(target=take, daemon=True)
    t.start()
    t.join(5)
    assert not t.is_alive(), "file_lock never gave up"
    assert "stayed busy" in outcome.get("timeout", "")


def test_releasing_retries_a_briefly_refused_delete(tmp_path, monkeypatch):
    # If the release gave up on the first refusal, the lock would sit there until it went stale,
    # holding every other writer up for LOCK_STALE_SECONDS.
    real_remove = os.remove
    refused = {"n": 0}

    def flaky_remove(p):
        if str(p).endswith(".lock") and refused["n"] < 2:
            refused["n"] += 1
            raise PermissionError(13, "Permission denied", str(p))
        real_remove(p)

    monkeypatch.setattr(os, "remove", flaky_remove)
    path = str(tmp_path / "log.csv")
    with csv_utils.file_lock(path):
        pass
    assert refused["n"] == 2
    assert not os.path.exists(path + ".lock")


def test_rotate_if_larger_renames_a_big_file_aside_and_the_next_append_restarts_it(tmp_path):
    path = str(tmp_path / "log.csv")
    header = ["a", "b"]
    for i in range(50):
        csv_utils.append_row(path, header, [i, i])
    size = os.path.getsize(path)
    assert csv_utils.rotate_if_larger(path, size + 1) is False and os.path.exists(path)      # under the cap
    assert csv_utils.rotate_if_larger(path, size - 1) is True and not os.path.exists(path)
    (bak,) = list(tmp_path.glob("log.csv.*.bak"))
    assert len(pd.read_csv(bak)) == 50
    csv_utils.append_row(path, header, [1, 2])
    assert len(pd.read_csv(path)) == 1 and list(pd.read_csv(path).columns) == header
    assert csv_utils.rotate_if_larger(str(tmp_path / "missing.csv"), 1) is False
