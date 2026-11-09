# get_normalization_stats.py
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice
from pathlib import Path

import numpy as np
from tqdm import tqdm


# Add project root (batnet-lstm) to Python's module search path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


import config

from zc.pipeline import process_wav

from filtering import (
    load_filter_config,
    filter_file_pulses,
)

from pchip import pchip_resample_pulses

from features import process_pulse_dataframe


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

FILTER_CONFIG_PATH = Path(__file__).with_name(
    "pulse_class_filters.json"
)

OUTPUT_PATH = Path(__file__).with_name(
    "normalization_stats.json"
)

FILES_PER_TASK = 16
TASKS_PER_WORKER = 4

# Initialized separately inside each worker process.
_WORKER_FILTER_CONFIG = None


# ---------------------------------------------------------------------
# File discovery and batching
# ---------------------------------------------------------------------

def iter_wav_files(input_dir):
    """
    Yield WAV file paths recursively without storing all paths in memory.
    """

    for root, dirs, files in os.walk(input_dir):

        dirs.sort()

        for filename in files:

            if filename.lower().endswith(".wav"):
                yield os.path.join(root, filename)


def batched(iterable, batch_size):
    """
    Yield lists containing at most batch_size items.
    """

    iterator = iter(iterable)

    while True:

        batch = list(islice(iterator, batch_size))

        if not batch:
            break

        yield batch


# ---------------------------------------------------------------------
# Worker initialization
# ---------------------------------------------------------------------

def initialize_worker(filter_config_path):
    """
    Load the filtering configuration once per worker process.
    """

    global _WORKER_FILTER_CONFIG

    _WORKER_FILTER_CONFIG = load_filter_config(
        filter_config_path
    )


# ---------------------------------------------------------------------
# Process one WAV file
# ---------------------------------------------------------------------

def process_wav_for_stats(wav_path):
    """
    Process one WAV file and calculate per-column statistics.

    Returns:
        status: "success", "empty", or "failed"
        stats: Dictionary containing count, mean, and M2 per column.
    """

    result = process_wav(wav_path)

    if result is None:
        return "failed", None

    pulses_df, metrics_df = result

    if pulses_df.empty or metrics_df.empty:
        return "empty", None

    # Class label comes from the WAV file's parent directory.
    class_name = Path(wav_path).parent.name

    # Apply class-specific pulse filtering.
    filtered_pulses, _, _ = filter_file_pulses(
        pulses_df,
        metrics_df,
        class_name,
        _WORKER_FILTER_CONFIG,
    )

    if filtered_pulses.empty:
        return "empty", None

    # Resample each pulse onto the fixed time grid.
    resampled_df = pchip_resample_pulses(
        filtered_pulses,
        config.PCHIP_TIME_INTERVAL,
    )

    if resampled_df.empty:
        return "empty", None

    # Calculate slope and position features.
    features_df = process_pulse_dataframe(
        resampled_df
    )

    # Calculate per-column statistics.
    stats = {}

    for column in config.NORMALIZATION_COLUMNS:

        if column not in features_df.columns:
            raise ValueError(
                f"Column '{column}' was not found after preprocessing."
            )

        values = features_df[column].to_numpy(
            dtype=np.float64
        )

        # Ignore NaN and infinite values.
        values = values[np.isfinite(values)]

        if len(values) == 0:

            stats[column] = (0, 0.0, 0.0)
            continue

        mean = float(np.mean(values))

        m2 = float(
            np.sum((values - mean) ** 2)
        )

        stats[column] = (
            len(values),
            mean,
            m2,
        )

    return "success", stats


# ---------------------------------------------------------------------
# Combine statistics
# ---------------------------------------------------------------------

def combine_stats(current, new):
    """
    Combine two sets of count, mean, and M2 statistics.

    M2 is the sum of squared deviations from the mean.
    """

    count_a, mean_a, m2_a = current
    count_b, mean_b, m2_b = new

    if count_a == 0:
        return new

    if count_b == 0:
        return current

    total_count = count_a + count_b

    delta = mean_b - mean_a

    combined_mean = (
        mean_a + delta * count_b / total_count
    )

    combined_m2 = (
        m2_a
        + m2_b
        + delta ** 2 * count_a * count_b / total_count
    )

    return (
        total_count,
        combined_mean,
        combined_m2,
    )


# ---------------------------------------------------------------------
# Process a batch of WAV files
# ---------------------------------------------------------------------

def process_wav_batch(wav_paths):
    """
    Process multiple WAV files inside one worker.

    Statistics are combined inside the worker before being returned,
    reducing the amount of data transferred between processes.
    """

    batch_stats = {
        column: (0, 0.0, 0.0)
        for column in config.NORMALIZATION_COLUMNS
    }

    successful = 0
    empty = 0
    failed = 0

    for wav_path in wav_paths:

        try:

            status, file_stats = process_wav_for_stats(
                wav_path
            )

            if status == "failed":
                failed += 1
                continue

            if status == "empty":
                empty += 1
                continue

            successful += 1

            for column in config.NORMALIZATION_COLUMNS:

                batch_stats[column] = combine_stats(
                    batch_stats[column],
                    file_stats[column],
                )

        except Exception as error:

            failed += 1

            print(
                f"\nError processing {wav_path}: {error}",
                flush=True,
            )

    return (
        successful,
        empty,
        failed,
        batch_stats,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    input_dir = config.INPUT_DIR

    # Number of WAV files processed in each worker task.
    files_per_task = FILES_PER_TASK

    # Maximum number of tasks submitted at one time.
    tasks_per_batch = max(
        1,
        config.NUM_WORKERS * TASKS_PER_WORKER,
    )

    # Initialize global statistics.
    total_stats = {
        column: (0, 0.0, 0.0)
        for column in config.NORMALIZATION_COLUMNS
    }

    successful_files = 0
    empty_files = 0
    failed_files = 0

    # Lazy WAV file discovery.
    wav_files = iter_wav_files(input_dir)

    print(f"Input directory: {input_dir}")
    print(f"Workers: {config.NUM_WORKERS}")
    print(f"Files per task: {files_per_task}")

    # Start worker processes.
    with ProcessPoolExecutor(
        max_workers=config.NUM_WORKERS,
        initializer=initialize_worker,
        initargs=(str(FILTER_CONFIG_PATH),),
    ) as executor:

        # Progress bar tracks individual WAV files.
        with tqdm(
            desc="Processing WAV files",
            unit="file",
        ) as progress:

            # Create a limited number of tasks at a time.
            task_batches = batched(
                batched(wav_files, files_per_task),
                tasks_per_batch,
            )

            for task_batch in task_batches:

                futures = [
                    executor.submit(
                        process_wav_batch,
                        paths,
                    )
                    for paths in task_batch
                ]

                # Collect completed worker tasks.
                for future in as_completed(futures):

                    (
                        successful,
                        empty,
                        failed,
                        batch_stats,
                    ) = future.result()

                    successful_files += successful
                    empty_files += empty
                    failed_files += failed

                    # Combine worker statistics.
                    for column in config.NORMALIZATION_COLUMNS:

                        total_stats[column] = combine_stats(
                            total_stats[column],
                            batch_stats[column],
                        )

                    # Update progress using the number of WAVs
                    # completed by this worker.
                    progress.update(
                        successful + empty + failed
                    )

    # -----------------------------------------------------------------
    # Calculate final normalization statistics
    # -----------------------------------------------------------------

    normalization_stats = {}

    for column, (count, mean, m2) in total_stats.items():

        if count == 0:
            raise RuntimeError(
                f"No valid values found for normalization "
                f"column '{column}'."
            )

        # Population variance and standard deviation.
        variance = m2 / count
        std = float(np.sqrt(variance))

        if std == 0:
            raise RuntimeError(
                f"Standard deviation for '{column}' is zero."
            )

        normalization_stats[column] = {
            "mean": float(mean),
            "std": std,
            "count": int(count),
        }

    # -----------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            normalization_stats,
            file,
            indent=4,
        )

    # -----------------------------------------------------------------
    # Processing summary
    # -----------------------------------------------------------------

    total_files = (
        successful_files
        + empty_files
        + failed_files
    )

    print("\nProcessing complete.")

    print(f"Total processed:   {total_files:,}")
    print(f"Successful files:  {successful_files:,}")
    print(f"Empty files:       {empty_files:,}")
    print(f"Failed files:      {failed_files:,}")

    print(f"\nStatistics saved to: {OUTPUT_PATH}")

    print("\nNormalization statistics:")

    for column, stats in normalization_stats.items():

        print(
            f"{column:12s} "
            f"mean={stats['mean']:.6f}, "
            f"std={stats['std']:.6f}, "
            f"count={stats['count']:,}"
        )


if __name__ == "__main__":
    main()
