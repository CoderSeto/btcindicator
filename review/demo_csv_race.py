"""
Run from the repo root:  python review/demo_csv_race.py

Demonstrates (does not fix) the concurrent-writer race in csv_utils.needs_header(): two writers
(a Streamlit rerun and main.py, or two browser tabs) that both check before either writes each
write a header. pd.read_csv then reads the second header as a data row, and every numeric
column comes back as strings.
"""
import csv
import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from csv_utils import needs_header  # noqa: E402

path = os.path.join(tempfile.mkdtemp(), "race.csv")
header = ["timestamp", "price"]
first, second = needs_header(path, header), needs_header(path, header)   # both check first
for write_header in (first, second):
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(["2026-11-07T18:00:00", 75700.5])
print(open(path).read())
df = pd.read_csv(path)
print("price dtype:", df["price"].dtype, "| rows:", len(df), "(2 real rows written)")
