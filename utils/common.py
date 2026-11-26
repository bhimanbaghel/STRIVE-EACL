"""
Shared utility functions for the Events task.
"""

import csv
import fcntl
import os
import logging

def load_verbs(path: str) -> list:
    """Load verbs from a text file (one per line, # for comments)."""
    with open(path) as f:
        return [l.strip().lower() for l in f if l.strip() and not l.startswith("#")]

def append_or_replace_csv(path: str, new_rows: list, key_columns: list):
    """
    Append rows to a CSV, replacing any existing rows that match on key_columns.
    This prevents duplicate entries when rerunning experiments.

    Args:
        path: Path to the CSV file
        new_rows: List of dicts to write
        key_columns: Column names that together uniquely identify a row
                     (e.g., ["run_name", "verb"])
    """
    if not new_rows:
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Sharded jobs append the same CSV. Hold an exclusive lock across the
    # read-modify-write so two shards cannot drop each other's rows.
    lock_path = path + ".lock"
    with open(lock_path, "a") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            _append_or_replace_csv_unlocked(path, new_rows, key_columns)
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)


def _append_or_replace_csv_unlocked(path: str, new_rows: list, key_columns: list):
    existing_rows = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            existing_rows = list(csv.DictReader(f))

    # Build set of keys from new rows for fast lookup
    new_keys = set()
    for row in new_rows:
        key = tuple(str(row.get(k, "")) for k in key_columns)
        new_keys.add(key)

    # Keep only existing rows that don't conflict with new rows
    filtered = []
    removed = 0
    for row in existing_rows:
        key = tuple(str(row.get(k, "")) for k in key_columns)
        if key in new_keys:
            removed += 1
        else:
            filtered.append(row)

    # Merge: old (filtered) + new
    all_rows = filtered + new_rows

    # Collect all column names preserving order
    all_keys = []
    seen = set()
    for row in all_rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    if removed:
        logging.info(f"CSV {path}: replaced {removed} existing rows, total {len(all_rows)} rows")
    else:
        logging.info(f"CSV {path}: appended {len(new_rows)} rows, total {len(all_rows)} rows")
