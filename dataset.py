# dataset.py
from pathlib import Path
import random

import torch
from torch.utils.data import IterableDataset, get_worker_info
from torch.nn.utils.rnn import pad_sequence

import config
from preprocess import preprocess_wav


# ============================================================
# DISCOVER WAV FILES AND CLASSES
# ============================================================

def discover_wav_files(root_dir, class_to_idx=None):
    """
    Recursively find WAV files.

    The immediate parent folder of each WAV file is treated as
    the class name.

    Parameters
    ----------
    root_dir : str or Path
        Root directory containing class subfolders.
    class_to_idx : dict, optional
        Existing class mapping. If None, a mapping is created
        from the subfolder names found in root_dir.

    Returns
    -------
    samples : list of tuple
        (wav_path, class_index)
    class_to_idx : dict
        Mapping from class name to integer index.
    """

    root_dir = Path(root_dir)

    if class_to_idx is None:

        class_names = sorted(
            [
                path.name
                for path in root_dir.iterdir()
                if path.is_dir()
            ]
        )

        class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(class_names)
        }

    samples = []

    for wav_path in root_dir.rglob("*"):

        if not wav_path.is_file():
            continue

        if wav_path.suffix.lower() != ".wav":
            continue

        class_name = wav_path.parent.name

        if class_name not in class_to_idx:
            continue

        samples.append(
            (
                str(wav_path),
                class_to_idx[class_name]
            )
        )

    samples.sort(key=lambda x: x[0])

    return samples, class_to_idx


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

def split_dataset_samples(
    samples,
    validation_ratio,
    random_seed=7
):
    """
    Split WAV files into train and validation sets.

    Splitting occurs at the WAV-file level so pulses from the
    same WAV never appear in both sets.
    """

    if not samples:
        return [], []

    if validation_ratio <= 0:
        return list(samples), []

    if validation_ratio >= 1:
        return [], list(samples)

    shuffled_samples = list(samples)

    rng = random.Random(random_seed)
    rng.shuffle(shuffled_samples)

    validation_count = int(
        round(
            len(shuffled_samples)
            * validation_ratio
        )
    )

    if (
        validation_count < 1
        and validation_ratio > 0
    ):
        validation_count = 1

    if validation_count >= len(shuffled_samples):
        validation_count = max(
            0,
            len(shuffled_samples) - 1
        )

    validation_samples = (
        shuffled_samples[:validation_count]
    )

    train_samples = (
        shuffled_samples[validation_count:]
    )

    return train_samples, validation_samples


# ============================================================
# ITERABLE PULSE DATASET
# ============================================================

class PulseDataset(IterableDataset):
    """
    Streams WAV files through the complete preprocessing
    pipeline and yields one pulse tensor at a time.

    Each yielded sample is:

        sequence, label, length, source_path
    """

    def __init__(
        self,
        samples,
        normalization_path
    ):
        super().__init__()

        self.samples = list(samples)
        self.normalization_path = normalization_path

    def __iter__(self):

        worker_info = get_worker_info()

        # ----------------------------------------------------
        # Split WAV files between DataLoader workers
        # ----------------------------------------------------

        if worker_info is None:

            worker_samples = self.samples

        else:

            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            worker_samples = self.samples[
                worker_id::num_workers
            ]

        # ----------------------------------------------------
        # Process WAV files
        # ----------------------------------------------------

        for wav_path, label in worker_samples:

            tensors = preprocess_wav(
                wav_path,
                normalization_path=self.normalization_path
            )

            # One tensor = one pulse/sample
            for sequence in tensors:

                # Skip empty sequences
                if sequence.numel() == 0:
                    continue

                # Truncate very long sequences
                if (
                    len(sequence)
                    > config.MAX_SEQUENCE_LENGTH
                ):
                    sequence = sequence[
                        :config.MAX_SEQUENCE_LENGTH
                    ]

                length = len(sequence)

                yield (
                    sequence,
                    label,
                    length,
                    wav_path
                )


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):
    """
    Create padded batches for use with
    pack_padded_sequence().

    Returns
    -------
    padded_sequences : torch.Tensor
        Shape: (batch, max_len, num_features)

    labels : torch.Tensor
        Shape: (batch,)

    lengths : torch.Tensor
        Shape: (batch,)

    file_paths : tuple
        Source WAV path for each pulse.
    """

    sequences, labels, lengths, file_paths = zip(*batch)

    lengths = torch.tensor(
        lengths,
        dtype=torch.long
    )

    labels = torch.tensor(
        labels,
        dtype=torch.long
    )

    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0.0
    )

    return (
        padded_sequences,
        labels,
        lengths,
        file_paths
    )
