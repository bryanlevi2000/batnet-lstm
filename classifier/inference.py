"""Predict a class for every pulse in TEST_DIR. Run from the project root: python classifier/inference.py

Worker processes load each WAV or ZC file (ZC extraction for WAVs, parsing for .zc and # files),
then run pulse cleanup, PCHIP resampling, feature calculation and normalization.
Class-specific pulse filtering is not applied.
The model runs on the CPU, and predictions are appended to the CSV batch by batch.
"""

import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from model import PulseLSTM
from preprocess.features import process_pulse_dataframe
from preprocess.normalize import normalize_dataframe
from preprocess.pchip import pchip_resample_pulses
from zc.input_files import iter_input_files, write_file_lists
from zc.pipeline import process_file


TASKS_PER_WORKER = 4


def find_input_files(root):
    """All WAV and ZC files under root, searched recursively, in sorted order."""

    return list(iter_input_files(root))


def extract_samples(file_path, feature_columns, max_len):
    """Preprocess one WAV or ZC file. Returns (file_path, [(pulse_id, sample)], nan_dropped, error)."""

    try:
        pulses, _ = process_file(file_path)

        if pulses.empty:
            return file_path, [], 0, None

        resampled = pchip_resample_pulses(pulses, config.PCHIP_TIME_INTERVAL)
        df = normalize_dataframe(process_pulse_dataframe(resampled))

    except Exception as error:
        return file_path, [], 0, f"{type(error).__name__}: {error}"

    samples, dropped = [], 0

    for pulse_id, pulse in df.groupby("pulse_id", sort=False):

        sample = pulse[feature_columns].to_numpy(dtype=np.float32)[:max_len]

        if int(pulse_id) <= 0 or len(sample) == 0:
            continue

        if np.isnan(sample).any():
            dropped += 1
        else:
            samples.append((int(pulse_id), sample))

    return file_path, samples, dropped, None


def stream(executor, file_paths, feature_columns, max_len):
    """Yield worker results in file order, keeping a bounded number of files in flight."""

    pending = deque()

    for path in file_paths:

        pending.append(executor.submit(extract_samples, path, feature_columns, max_len))

        if len(pending) >= config.NUM_WORKERS * TASKS_PER_WORKER:
            yield pending.popleft().result()

    while pending:
        yield pending.popleft().result()


def write_batch(model, batch, classes, output_path):
    """Predict a batch of (file_path, pulse_id, sample) and append the rows to the CSV."""

    samples = [torch.from_numpy(sample) for _, _, sample in batch]
    lengths = torch.tensor([len(sample) for sample in samples])

    with torch.no_grad():
        logits = model(pad_sequence(samples, batch_first=True), lengths)

    pd.concat([
        pd.DataFrame({
            "file": [str(path) for path, _, _ in batch],
            "pulse_id": [pulse_id for _, pulse_id, _ in batch],
            "ground_truth": [path.parent.name for path, _, _ in batch],
        }),
        pd.DataFrame(torch.sigmoid(logits).numpy(), columns=classes),
        pd.DataFrame(logits.numpy(), columns=[f"{name}_logit" for name in classes]),
    ], axis=1).to_csv(output_path, mode="a", header=False, index=False)


def run(file_paths, predictions_path, failed_path, no_pulse_path):
    """Predict every pulse in file_paths with the model at CHECKPOINT_PATH.

    Writes the predictions CSV, the failed-files CSV, and the no-pulse-files CSV.
    """

    checkpoint_path = Path(config.CHECKPOINT_PATH)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = PulseLSTM(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    feature_columns = checkpoint["feature_columns"]
    max_len = checkpoint["max_sequence_length"]
    missing = [c for c in feature_columns if c not in config.NORMALIZATION_COLUMNS]

    if missing:
        raise ValueError(f"Checkpoint features {missing} are not in NORMALIZATION_COLUMNS.")

    # Class names ordered by the training label index.
    mapping = checkpoint["class_mapping"]
    classes = sorted(mapping, key=mapping.get)

    output_path = Path(predictions_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        columns=["file", "pulse_id", "ground_truth", *classes, *[f"{c}_logit" for c in classes]]
    ).to_csv(output_path, index=False)

    batch, failed, no_pulses = [], [], []
    predicted = dropped = 0

    with ProcessPoolExecutor(max_workers=config.NUM_WORKERS) as executor:

        results = stream(executor, file_paths, feature_columns, max_len)

        for file_path, samples, n_dropped, error in tqdm(
            results, total=len(file_paths), desc="Inference", unit="file"
        ):

            if error:
                failed.append({"file": str(file_path), "error": error})
            elif not samples and not n_dropped:
                no_pulses.append({"file": str(file_path)})

            dropped += n_dropped
            batch += [(file_path, pulse_id, sample) for pulse_id, sample in samples]

            if len(batch) >= config.BATCH_SIZE:
                write_batch(model, batch, classes, output_path)
                predicted += len(batch)
                batch = []

    if batch:
        write_batch(model, batch, classes, output_path)
        predicted += len(batch)

    write_file_lists(failed, no_pulses, failed_path, no_pulse_path)

    print(f"Saved {predicted:,} predictions to {output_path}")
    print(f"Dropped {dropped:,} samples with NaN values")
    print(f"{len(failed):,} failed files listed in {failed_path}")
    print(f"{len(no_pulses):,} files with no pulses listed in {no_pulse_path}")


def main():

    file_paths = find_input_files(config.TEST_DIR)

    if not file_paths:
        raise FileNotFoundError(f"No WAV or ZC files found in {config.TEST_DIR}")

    run(file_paths, config.PREDICTIONS_PATH, config.FAILED_FILES_PATH, config.NO_PULSE_FILES_PATH)


if __name__ == "__main__":
    main()
