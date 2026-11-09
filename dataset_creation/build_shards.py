"""Create resumable NumPy shards from preprocess/pipeline.py.

Run from the project root:
    python dataset_creation/build_shards.py

Each positive pulse_id is one sample. All pulses from a WAV stay in the
same train/validation split. Each shard has at most SHARD_NUM_SAMPLES samples.
"""

import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "preprocess"))

import config

import preprocess.pipeline as pipeline


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SPLIT_SEED = 42
TASKS_PER_WORKER = 4


# ---------------------------------------------------------------------
# WAV discovery and splitting
# ---------------------------------------------------------------------

def find_wavs(input_dir):
    """Find WAV files and split them by class."""

    files_by_class = defaultdict(list)

    for root, dirs, files in os.walk(input_dir):

        dirs.sort()

        for name in sorted(files):

            if name.lower().endswith(".wav"):

                path = Path(root) / name

                files_by_class[path.parent.name].append(path)

    train = []
    validation = []

    for class_name, paths in sorted(files_by_class.items()):

        paths.sort()

        random.Random(
            f"{SPLIT_SEED}:{class_name}"
        ).shuffle(paths)

        if config.VAL_RATIO > 0 and len(paths) > 1:

            n_val = min(
                len(paths) - 1,
                max(1, round(len(paths) * config.VAL_RATIO))
            )

        else:

            n_val = 0

        validation.extend(paths[:n_val])
        train.extend(paths[n_val:])

    return sorted(train), sorted(validation), sorted(files_by_class)


# ---------------------------------------------------------------------
# Worker initialization
# ---------------------------------------------------------------------

def initialize_worker():
    """Initialize the preprocessing worker."""

    pipeline.initialize_worker(
        str(pipeline.FILTER_CONFIG_PATH)
    )


def process_wav(path):
    """Process one WAV file."""

    try:

        return str(path), pipeline.process_wav_file(str(path))

    except Exception:

        return str(path), None


# ---------------------------------------------------------------------
# Resume functionality
# ---------------------------------------------------------------------

def load_progress(manifest_path, metadata_path):
    """Load completed shard information."""

    counts = defaultdict(int)

    next_shard = {
        "train": 0,
        "validation": 0,
    }

    metadata_end = 0
    manifest_end = 0

    if manifest_path.exists():

        with manifest_path.open("rb") as manifest:

            for line in manifest:

                try:
                    entry = json.loads(line)

                except json.JSONDecodeError:
                    break

                split = entry["split"]

                next_shard[split] = max(
                    next_shard[split],
                    entry["shard"] + 1
                )

                metadata_end = entry["metadata_end"]

                for wav, n in entry["file_counts"].items():

                    counts[(split, wav)] += n

                manifest_end = manifest.tell()

        # Remove incomplete manifest entries.

        with manifest_path.open("r+b") as manifest:
            manifest.truncate(manifest_end)

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Remove metadata belonging to incomplete shards.

    with metadata_path.open("a+b") as metadata:
        metadata.truncate(metadata_end)

    return counts, next_shard


# ---------------------------------------------------------------------
# Write individual NumPy arrays
# ---------------------------------------------------------------------

def write_array(path, array):
    """Write an array to a temporary file."""

    temp = path.with_name(path.name + ".tmp")

    with temp.open("wb") as handle:

        np.save(
            handle,
            array,
            allow_pickle=False
        )

        handle.flush()
        os.fsync(handle.fileno())

    return temp


# ---------------------------------------------------------------------
# Write a completed shard
# ---------------------------------------------------------------------

def write_shard(
    split,
    shard_index,
    output_dir,
    features,
    labels,
    metadata,
    file_counts,
    metadata_path,
    manifest_path,
):

    if not labels:
        return

    name = f"{split}_{shard_index:06d}"

    shard_dir = output_dir / split

    lengths = np.asarray(
        [len(x) for x in features],
        dtype=np.int64
    )

    offsets = np.empty(
        len(labels) + 1,
        dtype=np.int64
    )

    offsets[0] = 0

    np.cumsum(
        lengths,
        out=offsets[1:]
    )

    arrays = {

        "features": np.concatenate(
            features,
            axis=0
        ).astype(np.float32, copy=False),

        "offsets": offsets,

        "labels": np.asarray(
            labels,
            dtype=np.int64
        ),
    }

    temporary = {}
    final = {}

    try:

        # Write NumPy files to temporary filenames.

        for key, array in arrays.items():

            final[key] = (
                shard_dir /
                f"{name}_{key}.npy"
            )

            temporary[key] = write_array(
                final[key],
                array
            )

        # Append sample metadata to the single metadata file.

        with metadata_path.open("ab") as handle:

            for i, item in enumerate(metadata):

                item["shard"] = name
                item["sample_index"] = i

                handle.write(
                    (
                        json.dumps(
                            item,
                            separators=(",", ":")
                        ) + "\n"
                    ).encode()
                )

            handle.flush()
            os.fsync(handle.fileno())

            metadata_end = handle.tell()

        # Rename temporary files to their final filenames.

        for key in arrays:

            os.replace(
                temporary[key],
                final[key]
            )

        # Register the completed shard.

        entry = {

            "split": split,

            "shard": shard_index,

            "num_samples": len(labels),

            "files": {
                key: str(
                    path.relative_to(output_dir)
                )
                for key, path in final.items()
            },

            "file_counts": dict(file_counts),

            "metadata_end": metadata_end,
        }

        with manifest_path.open("ab") as handle:

            handle.write(
                (
                    json.dumps(
                        entry,
                        separators=(",", ":")
                    ) + "\n"
                ).encode()
            )

            handle.flush()
            os.fsync(handle.fileno())

    except Exception:

        for path in temporary.values():

            path.unlink(missing_ok=True)

        raise


# ---------------------------------------------------------------------
# Process one split
# ---------------------------------------------------------------------

def process_split(
    split,
    wavs,
    input_dir,
    output_dir,
    labels_by_class,
    committed_counts,
    shard_start,
    metadata_path,
    manifest_path,
):

    shard_index = shard_start

    features = []
    labels = []
    metadata = []

    file_counts = defaultdict(int)

    shard_size = config.SHARD_NUM_SAMPLES
    workers = config.NUM_WORKERS

    max_in_flight = max(
        1,
        workers * TASKS_PER_WORKER
    )

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_worker,
    ) as executor:

        with tqdm(
            total=len(wavs),
            desc=f"{split} WAVs",
            unit="file",
        ) as bar:

            # Submit only a bounded number of WAV files at once.

            for start in range(
                0,
                len(wavs),
                max_in_flight
            ):

                batch = wavs[
                    start:start + max_in_flight
                ]

                futures = [

                    executor.submit(
                        process_wav,
                        path
                    )

                    for path in batch

                ]

                # Collect completed WAV files.

                for future in as_completed(futures):

                    bar.update(1)

                    wav_path, result = future.result()

                    if result is None:
                        continue

                    if result["dataframe"] is None or result["dataframe"].empty:
                        continue

                    relative_path = (
                        Path(wav_path)
                        .resolve()
                        .relative_to(input_dir.resolve())
                        .as_posix()
                    )

                    class_name = Path(wav_path).parent.name

                    label = labels_by_class[class_name]

                    # Number of samples already committed for this WAV.

                    skip = committed_counts[
                        (split, relative_path)
                    ]

                    dataframe = result["dataframe"]

                    # Each positive pulse_id represents one sample.

                    for pulse_id, pulse in dataframe.groupby(
                        "pulse_id",
                        sort=False
                    ):

                        if int(pulse_id) <= 0:
                            continue

                        sample = pulse[
                            config.NORMALIZATION_COLUMNS
                        ].to_numpy(
                            dtype=np.float32
                        )

                        if len(sample) == 0:
                            continue

                        # Skip samples already written in previous runs.

                        if skip:

                            skip -= 1
                            continue

                        features.append(sample)

                        labels.append(label)

                        metadata.append({

                            "split": split,

                            "wav_path": relative_path,

                            "pulse_id": int(pulse_id),

                            "label": label,

                            "num_time_steps": len(sample),

                        })

                        file_counts[relative_path] += 1

                        # Write a shard when it reaches the configured size.

                        if len(labels) == shard_size:

                            write_shard(
                                split,
                                shard_index,
                                output_dir,
                                features,
                                labels,
                                metadata,
                                file_counts,
                                metadata_path,
                                manifest_path,
                            )

                            shard_index += 1

                            features = []
                            labels = []
                            metadata = []

                            file_counts = defaultdict(int)

    # Write any remaining samples.

    if labels:

        write_shard(
            split,
            shard_index,
            output_dir,
            features,
            labels,
            metadata,
            file_counts,
            metadata_path,
            manifest_path,
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    input_dir = Path(config.INPUT_DIR)

    output_dir = Path(config.SHARD_PARENT_DIR)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Discover WAV files and create class-stratified splits.

    train_wavs, val_wavs, class_names = find_wavs(
        input_dir
    )

    # Create persistent class-to-integer mapping.

    mapping_path = output_dir / "class_mapping.json"

    if mapping_path.exists():

        with mapping_path.open(
            "r",
            encoding="utf-8"
        ) as handle:

            labels_by_class = json.load(handle)

    else:

        labels_by_class = {}

    next_label = max(
        labels_by_class.values(),
        default=-1
    ) + 1

    for name in class_names:

        if name not in labels_by_class:

            labels_by_class[name] = next_label

            next_label += 1

    # Create output directories.

    (output_dir / "train").mkdir(
        parents=True,
        exist_ok=True
    )

    if config.VAL_RATIO > 0:

        (output_dir / "validation").mkdir(
            parents=True,
            exist_ok=True
        )

    with mapping_path.open(
        "w",
        encoding="utf-8"
    ) as handle:

        json.dump(
            labels_by_class,
            handle,
            indent=2
        )

    # Metadata and completion manifest.

    metadata_path = output_dir / "metadata.jsonl"

    manifest_path = output_dir / "manifest.jsonl"

    committed, next_shard = load_progress(
        manifest_path,
        metadata_path
    )

    print(f"Training WAVs: {len(train_wavs):,}")

    print(f"Validation WAVs: {len(val_wavs):,}")

    print(f"Features: {config.NORMALIZATION_COLUMNS}")

    print(f"Samples per shard: {config.SHARD_NUM_SAMPLES:,}")

    # Process training data.

    process_split(
        "train",
        train_wavs,
        input_dir,
        output_dir,
        labels_by_class,
        committed,
        next_shard["train"],
        metadata_path,
        manifest_path,
    )

    # Process validation data.

    if config.VAL_RATIO > 0:

        process_split(
            "validation",
            val_wavs,
            input_dir,
            output_dir,
            labels_by_class,
            committed,
            next_shard["validation"],
            metadata_path,
            manifest_path,
        )

    print("\nShard creation complete.")


if __name__ == "__main__":

    main()