
from pathlib import Path
import csv
import sqlite3

LOG_DIR = Path(__file__).resolve().parent
CSV_PATH = LOG_DIR / "skipped_files.csv"
DB_PATH = LOG_DIR / ".skipped_files.sqlite3"


def _connect():
    connection = sqlite3.connect(DB_PATH, timeout=60)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS skipped (
            file_path TEXT PRIMARY KEY,
            reason TEXT NOT NULL
        )
    """)
    return connection


def initialize_skipped_file_log():
    """Rebuild the CSV from the persistent, deduplicated log."""
    try:
        with _connect() as connection:
            rows = connection.execute(
                "SELECT file_path, reason FROM skipped ORDER BY file_path"
            ).fetchall()

        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file_path", "reason"])
            writer.writerows(rows)
    except Exception:
        pass


def record_skipped_file(file_path, error):
    """Record each failed file once, safely across worker processes."""
    try:
        path = str(Path(file_path).resolve())
        reason = " ".join(str(error).split())[:500] or type(error).__name__

        with _connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            cursor = connection.execute(
                "INSERT OR IGNORE INTO skipped (file_path, reason) "
                "VALUES (?, ?)",
                (path, reason)
            )

            if cursor.rowcount:
                with open(
                    CSV_PATH, "a", newline="", encoding="utf-8"
                ) as f:
                    writer = csv.writer(f)
                    if f.tell() == 0:
                        writer.writerow(["file_path", "reason"])
                    writer.writerow([path, reason])

    except Exception:
        # Logging failures must never interrupt training.
        pass
