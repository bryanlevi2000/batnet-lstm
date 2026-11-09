import json
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import config


def read_manifest(shard_dir, split):
    """Return the manifest entries for one split, in file order."""

    with open(Path(shard_dir) / "manifest.jsonl", encoding="utf-8") as file:
        entries = [json.loads(line) for line in file if line.strip()]

    return [entry for entry in entries if entry["split"] == split]


def check_ram(needed_bytes):
    """Raise MemoryError if the data does not fit in RAM with the reserve left over."""

    gb = 1024 ** 3
    available = psutil.virtual_memory().available - config.RAM_RESERVE_GB * gb

    if needed_bytes > available:
        raise MemoryError(
            f"Dataset needs {needed_bytes / gb:.2f} GB, but only {available / gb:.2f} GB "
            f"is available after reserving {config.RAM_RESERVE_GB} GB."
        )


def load_split(shard_dir, split, feature_columns, max_len):
    """Load one split into RAM.

    Returns features (total_steps, n_features), offsets, labels, and keep
    (indices of the kept samples in metadata order).
    """

    missing = [c for c in feature_columns if c not in config.NORMALIZATION_COLUMNS]

    if missing:
        raise ValueError(f"Columns not in NORMALIZATION_COLUMNS: {missing}")

    columns = [config.NORMALIZATION_COLUMNS.index(c) for c in feature_columns]

    shard_dir = Path(shard_dir)
    shards = []

    for entry in read_manifest(shard_dir, split):
        files = {key: shard_dir / path for key, path in entry["files"].items()}
        shards.append((
            np.load(files["features"], mmap_mode="r"),
            np.load(files["offsets"]),
            np.load(files["labels"]),
        ))

    total_steps = int(sum(np.minimum(np.diff(o), max_len).sum() for _, o, _ in shards))
    check_ram(total_steps * len(columns) * 4)

    features = np.empty((total_steps, len(columns)), dtype=np.float32)
    offsets, labels, keep = [0], [], []
    position = index = dropped = 0

    for shard_features, shard_offsets, shard_labels in shards:

        data = np.asarray(shard_features[:, columns])

        for i, label in enumerate(shard_labels):

            start = shard_offsets[i]
            sample = data[start:min(shard_offsets[i + 1], start + max_len)]

            if np.isnan(sample).any():
                dropped += 1
            else:
                features[position:position + len(sample)] = sample
                position += len(sample)
                offsets.append(position)
                labels.append(label)
                keep.append(index)

            index += 1

    print(f"{split}: loaded {len(labels):,} samples, dropped {dropped:,} with NaN values")

    return (
        features[:position],
        np.asarray(offsets, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(keep, dtype=np.int64),
    )


class PulseDataset(Dataset):
    """Samples are views into one shared in-memory features array."""

    def __init__(self, features, offsets, labels):
        self.features = torch.from_numpy(features)
        self.offsets = offsets
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        start, end = int(self.offsets[i]), int(self.offsets[i + 1])
        return self.features[start:end], self.labels[i]


def collate_fn(batch):
    """Zero-pad samples to the longest one in the batch."""

    samples, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in samples], dtype=torch.int64)

    return pad_sequence(samples, batch_first=True), lengths, torch.stack(labels)
