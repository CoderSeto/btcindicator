"""
Run from the repo root:  python review/csv_writer_race.py [trials=10]   (~1 min)

REVIEW_FINDINGS D7. Eight separate Python processes, released at the same instant, each append
40 rows to one log -- once with the old unlocked "check header, then append" pattern, once with
csv_utils.append_row(). Each against a new file and against a file in an older column layout
(which must be rotated aside exactly once). Prints (header lines, data rows, .bak files) per trial;
correct is (1, 320, 0) for a new file and (1, 320, 1) for an older-layout one.
"""
import glob
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER = "timestamp,writer,seq,price"
WRITERS, ROWS = 8, 40

WRITER = r"""
import csv, sys, time
sys.path.insert(0, sys.argv[1])
import csv_utils
path, wid, start, mode = sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
header = "%s".split(",")
while time.time() < start:
    pass
for seq in range(%d):
    row = ["2026-11-07T18:00:00", wid, seq, 75700.5]
    if mode == "locked":
        csv_utils.append_row(path, header, row)
    else:  # the pre-fix pattern: unlocked check, then append
        try:
            write_header = csv_utils.needs_header(path, header)
        except OSError:
            sys.exit(3)  # lost the rotation race (FileNotFoundError): this writer's rows are gone
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow(row)
""" % (HEADER, ROWS)


def trial(mode, older_layout):
    d = tempfile.mkdtemp()
    path = os.path.join(d, "log.csv")
    if older_layout:
        with open(path, "w") as f:
            f.write("timestamp,price\n2026-11-06T00:00:00,70000\n")
    start = time.time() + 2.0
    procs = [subprocess.Popen([sys.executable, "-c", WRITER, ROOT, path, str(i), str(start), mode],
                              stdout=subprocess.DEVNULL) for i in range(WRITERS)]
    for p in procs:
        p.wait()
    lines = open(path).read().splitlines()
    return lines.count(HEADER), len(lines) - lines.count(HEADER), len(glob.glob(path + ".*.bak"))


if __name__ == "__main__":
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    for mode in ("unlocked", "locked"):
        for older in (False, True):
            label = f"{mode:8s} {'older-layout file' if older else 'new file':17s}"
            print(label, [trial(mode, older) for _ in range(trials)])
