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
