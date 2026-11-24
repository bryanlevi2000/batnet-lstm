# input_files.py
"""Find input recordings and write the failed / no-pulse file lists."""

import os
from pathlib import Path

import pandas as pd


def is_wav_file(name):
    """True for WAV recordings."""

    return str(name).lower().endswith(".wav")


def is_zc_file(name):
    """True for Anabat ZC files: names ending in .zc or # (for example H9140021.26#)."""

    name = str(name).lower()

    return name.endswith(".zc") or name.endswith("#")


def is_input_file(name):
    """True for any file type the pipeline can read."""

    return is_wav_file(name) or is_zc_file(name)


def iter_input_files(input_dir):
    """Yield WAV and ZC file paths under input_dir, recursively, in sorted order."""

    for folder, dirs, files in os.walk(input_dir):

        dirs.sort()

        for name in sorted(files):

            if is_input_file(name):
                yield Path(folder) / name


def write_file_lists(failed, no_pulses, failed_path, no_pulses_path):
    """Write the failed-files CSV (file, error) and the no-pulse-files CSV (file).

    Both files are written even when empty.
    """

    for path, rows, columns in (
        (failed_path, failed, ["file", "error"]),
        (no_pulses_path, no_pulses, ["file"]),
    ):

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
