"""Run the inference pipeline on every validation WAV.
Run from the project root: python classifier/validation_inference.py

The validation files are listed in TRAIN_SHARD_DIR/metadata.jsonl, relative to
INPUT_DIR, so INPUT_DIR must still point at the training WAVs.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from dataset import read_manifest
from inference import run


def main():

    shard_dir = Path(config.TRAIN_SHARD_DIR)
    rows, unreadable = [], 0

    # Skip unreadable lines, then check that every validation sample was read.
    with open(shard_dir / "metadata.jsonl", encoding="utf-8") as file:
        for line in file:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                unreadable += 1

    rows = [row for row in rows if row["split"] == "validation"]
    expected = sum(entry["num_samples"] for entry in read_manifest(shard_dir, "validation"))

    if unreadable:
        print(f"Skipped {unreadable:,} unreadable lines in metadata.jsonl")

    if len(rows) != expected:
        raise RuntimeError(
            f"metadata.jsonl has {len(rows):,} readable validation samples, "
            f"but the manifest lists {expected:,}."
        )

    wav_paths = sorted({row["wav_path"] for row in rows})

    run(
        [Path(config.INPUT_DIR) / path for path in wav_paths],
        config.VAL_PREDICTIONS_PATH,
        config.VAL_FAILED_FILES_PATH,
    )


if __name__ == "__main__":
    main()
