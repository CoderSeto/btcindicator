"""
Shared CSV-log helpers (kept dependency-free so any logger can import them).

Every log appends through append_row(), which does the whole header check,
layout rotation, optional duplicate check and write under one cross-process
lock. Without it, two writers on one file -- the dashboard and the console
bot, or two browser tabs -- raced: on a new file both wrote a header (8 of 10
trials with 8 writers), a writer could read a half-written header and rotate a
perfectly good file aside, and on an older-layout file the losers of the
rename crashed and their rows were lost (REVIEW_FINDINGS D7).
"""
import csv
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone

# A writer holds the lock for milliseconds, so a lock file this old was left by
# a writer that died mid-append, and is removed. Waiting is capped a little
# longer than that, so a crashed writer's lock always clears before anyone
# gives up on it.
LOCK_STALE_SECONDS = 10.0
LOCK_TIMEOUT_SECONDS = 15.0

# On Windows, deleting a file that some other process still has open (a waiter
# reading the lock's age, an antivirus scan of a new file) leaves it "delete
# pending" until that handle closes, and creating the same name meanwhile fails
# with access denied rather than "already exists". There, PermissionError means
# "busy". Elsewhere it means a real permission problem and is raised.
_PERMISSION_ERROR_MEANS_BUSY = os.name == "nt"
_RELEASE_RETRY_SECONDS = 0.5


class LockTimeout(Exception):
    pass


@contextmanager
def file_lock(path: str, timeout: float = LOCK_TIMEOUT_SECONDS, stale_after: float = LOCK_STALE_SECONDS,
              poll: float = 0.005):
    """
    Exclusive lock on `path`, held as the lock file `<path>.lock`, which is
    created with O_CREAT | O_EXCL -- atomic on every platform this runs on,
    Windows included, and needing nothing outside the standard library.

    A lock file older than `stale_after` seconds is treated as left by a
    crashed writer and removed. (Best effort: two waiters that both judge the
    same dead lock stale at the same instant could both get in once. Only a
    crash can cause that, and the worst case is one duplicate header.)
    Every wait, including a Windows delete-pending lock (see
    _PERMISSION_ERROR_MEANS_BUSY) and a lock whose age can't be read, sleeps
    `poll` and counts against `timeout`; LockTimeout is raised after it.
    """
    lock = path + ".lock"
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(lock)
            except OSError:
                age = None               # released, or being deleted, between the two calls
            if age is not None and age > stale_after:
                try:
                    os.remove(lock)
                except OSError:
                    pass
                continue
            why = f"has been held for {age:.1f}s" if age is not None else "stayed busy"
        except PermissionError as e:
            if not _PERMISSION_ERROR_MEANS_BUSY:
                raise
            why = f"stayed busy ({e.strerror})"
        if time.monotonic() >= deadline:
            raise LockTimeout(f"{lock} {why}")
        time.sleep(poll)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        _release(lock)


def _release(lock: str):
    # A failed remove would leave the lock in place until it goes stale, holding
    # every other writer up for LOCK_STALE_SECONDS, so a transient refusal (on
    # Windows, another process briefly holding the file open) is retried.
    give_up = time.monotonic() + _RELEASE_RETRY_SECONDS
    while True:
        try:
            os.remove(lock)
            return
        except FileNotFoundError:
            return
        except OSError:
            if time.monotonic() >= give_up:
                return
            time.sleep(0.005)


def append_row(path: str, header: list, row: list, skip_if=None,
               lock_timeout: float = LOCK_TIMEOUT_SECONDS) -> bool:
    """
    Appends `row` to the CSV log at `path`, writing `header` first if the file
    is new and rotating an older-layout file aside (needs_header) -- all under
    file_lock(), so concurrent writers can neither double the header nor race
    the rotation.

    `skip_if`, if given, is called under the same lock once the file is known
    to exist with the right header; returning True skips the write. It's how
    the loggers' duplicate check (main._is_repeat) sees the file's real last
    row rather than one another writer is about to replace.

    Returns True if the row was written. Never raises for a busy lock:
    logging must not hang or crash the dashboard or the console loop, so after
    `lock_timeout` the row is dropped with a printed warning.
    """
    try:
        with file_lock(path, timeout=lock_timeout):
            write_header = needs_header(path, header)
            if not write_header and skip_if is not None and skip_if():
                return False
            with open(path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(header)
                writer.writerow(row)
            return True
    except LockTimeout as e:
        print(f"[log] {e}; row not written to {path}.")
        return False


def needs_header(path: str, header: list) -> bool:
    """
    Returns True if `header` still has to be written to `path`. Call it under
    file_lock() (append_row does); on its own it races with other writers.

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
