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
from inference import run


def main():

    metadata_path = Path(config.TRAIN_SHARD_DIR) / "metadata.jsonl"

    with open(metadata_path, encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]

    wav_paths = sorted({row["wav_path"] for row in rows if row["split"] == "validation"})

    if not wav_paths:
        raise RuntimeError(f"No validation files listed in {metadata_path}")

    run(
        [Path(config.INPUT_DIR) / path for path in wav_paths],
        config.VAL_PREDICTIONS_PATH,
        config.VAL_FAILED_FILES_PATH,
    )


if __name__ == "__main__":
    main()
