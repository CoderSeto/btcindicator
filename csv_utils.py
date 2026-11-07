"""Shared CSV-log helpers (kept dependency-free so any logger can import them)."""
import csv
import os
from datetime import datetime, timezone


def needs_header(path: str, header: list) -> bool:
    """
    Returns True if `header` still has to be written to `path`.

    If the file exists but its header doesn't match -- which happens when a
    new indicator or column is added to a log that already has rows in it --
    the old file is renamed aside and a fresh one is started. Appending
    wider rows under a narrower header is exactly the failure that broke
    pd.read_csv() here once before (see CLAUDE.md, gotcha #1); rotating
    keeps every file internally consistent, and keeps the old data
    readable, instead of corrupting both.
    """
    if not os.path.isfile(path):
        return True
    try:
        with open(path, newline="") as f:
            existing = next(csv.reader(f), None)
    except OSError:
        return True
    if existing == header:
        return False

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = f"{path}.{stamp}.bak"
    os.rename(path, archived)
    print(f"[log] {path} had an older column layout -- moved it to {archived} and started a new file.")
    return True
