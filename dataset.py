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
# STREAMING WAV DATASET
# ============================================================

class PulseDataset(IterableDataset):
    """
    Parallel WAV-level preprocessing dataset.

    Each worker processes WAV files independently, but returns
    ALL pulses from each WAV as one item.

    Pulse-level shuffling is NOT performed here.

    The main process performs the global pulse-level shuffle
    after all worker outputs have been merged by DataLoader.
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
        # Partition WAV files across workers
        # ----------------------------------------------------

        if worker_info is None:

            worker_id = 0
            num_workers = 1

            worker_samples = self.samples

        else:

            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            worker_samples = self.samples[
                worker_id::num_workers
            ]

        # ----------------------------------------------------
        # Process one WAV at a time
        # ----------------------------------------------------

        for wav_path, label in worker_samples:

            tensors = preprocess_wav(
                wav_path,
                normalization_path=self.normalization_path
            )

            pulses = []

            for sequence in tensors:

                if sequence.numel() == 0:
                    continue

                if (
                    len(sequence)
                    > config.MAX_SEQUENCE_LENGTH
                ):
                    sequence = sequence[
                        :config.MAX_SEQUENCE_LENGTH
                    ]

                length = len(sequence)

                pulses.append(
                    (
                        sequence,
                        label,
                        length,
                        wav_path
                    )
                )

            # Return all pulses from this WAV as one item.
            #
            # This substantially reduces inter-process
            # communication compared with sending one pulse
            # at a time.
            if pulses:
                yield pulses


# ============================================================
# FLATTEN WAV OUTPUTS INTO INDIVIDUAL PULSES
# ============================================================

def pulse_stream(wav_loader):

    """
    Flatten the lists of pulses produced by the workers into
    a single stream of individual pulse samples.
    """

    for wav_pulses in wav_loader:

        for pulse in wav_pulses:

            yield pulse


# ============================================================
# GLOBAL PULSE SHUFFLE
# ============================================================

def shuffled_pulse_stream(
    wav_loader,
    shuffle_buffer_size,
    random_seed,
    epoch
):

    """
    Main-process bounded global shuffle.

    IMPORTANT:
    This runs AFTER DataLoader workers have merged their
    outputs, so the buffer contains pulses from ALL workers.

    If shuffle_buffer_size is larger than the entire dataset,
    this produces a full-dataset shuffle.
    """

    if shuffle_buffer_size < 1:
        raise ValueError(
            "shuffle_buffer_size must be >= 1."
        )

    rng = random.Random(
        random_seed + epoch
    )

    generator = pulse_stream(
        wav_loader
    )

    buffer = []

    # --------------------------------------------------------
    # Fill initial buffer
    # --------------------------------------------------------

    for _ in range(shuffle_buffer_size):

        try:

            sample = next(generator)

        except StopIteration:

            break

        buffer.append(sample)

    # --------------------------------------------------------
    # Draw randomly while replacing samples
    # --------------------------------------------------------

    while buffer:

        index = rng.randrange(
            len(buffer)
        )

        selected = buffer[index]

        try:

            replacement = next(generator)

            buffer[index] = replacement

        except StopIteration:

            buffer.pop(index)

        yield selected


# ============================================================
# BATCH GENERATOR
# ============================================================

def iter_batches(
    wav_loader,
    batch_size,
    shuffle=False,
    shuffle_buffer_size=10000,
    random_seed=7,
    epoch=0
):

    """
    Convert the streamed pulse data into padded model batches.
    """

    if shuffle:

        source = shuffled_pulse_stream(
            wav_loader=wav_loader,
            shuffle_buffer_size=shuffle_buffer_size,
            random_seed=random_seed,
            epoch=epoch
        )

    else:

        source = pulse_stream(
            wav_loader
        )

    batch = []

    for sample in source:

        batch.append(sample)

        if len(batch) >= batch_size:

            yield collate_fn(batch)

            batch = []

    # --------------------------------------------------------
    # Final partial batch
    # --------------------------------------------------------

    if batch:

        yield collate_fn(batch)


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):

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
