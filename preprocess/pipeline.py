
# pipeline.py
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice

from tqdm import tqdm


# Add project root (batnet-lstm) to Python's module search path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


import config

from zc.pipeline import process_wav

from filtering import (
    load_filter_config,
    filter_file_pulses,
    save_rejection_summary,
)

from pchip import pchip_resample_pulses

from features import process_pulse_dataframe

from normalize import normalize_dataframe


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

FILTER_CONFIG_PATH = Path(__file__).with_name(
    "pulse_class_filters.json"
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

def process_wav_file(wav_path):
    """
    Process one WAV file and normalize its resulting features.
    """

    result = process_wav(wav_path)

    if result is None:
        return None

    pulses_df, metrics_df = result

    if pulses_df.empty or metrics_df.empty:
        return None

    class_name = Path(wav_path).parent.name

    # Apply class-specific pulse filtering.
    filtered_pulses, _, rejection_stats = filter_file_pulses(
        pulses_df,
        metrics_df,
        class_name,
        _WORKER_FILTER_CONFIG,
    )

    if filtered_pulses.empty:
        return {
            "wav_path": str(wav_path),
            "dataframe": None,
            "rejection_stats": rejection_stats,
        }

    # Resample each pulse onto the fixed time grid.
    resampled_df = pchip_resample_pulses(
        filtered_pulses,
        config.PCHIP_TIME_INTERVAL,
    )

    if resampled_df.empty:
        return None

    # Calculate slope and position features.
    features_df = process_pulse_dataframe(
        resampled_df
    )

    if features_df.empty:
        return None

    # Apply z-score normalization.
    normalized_df = normalize_dataframe(
        features_df
    )

    if normalized_df.empty:
        return None

    return {
        "wav_path": str(wav_path),
        "dataframe": normalized_df,
        "rejection_stats": rejection_stats,
    }


# ---------------------------------------------------------------------
# Process a batch of WAV files
# ---------------------------------------------------------------------

def process_wav_batch(wav_paths):
    """
    Process multiple WAV files inside one worker.

    Failed files are silently skipped.

    Returns:
        list: Successfully processed WAV file results.
    """

    results = []

    for wav_path in wav_paths:

        try:

            result = process_wav_file(wav_path)

            if result is not None:
                results.append(result)

        except Exception:
            continue

    return results


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def process_folder(input_dir=None):
    """
    Recursively process WAV files using parallel processing.
    """

    if input_dir is None:
        input_dir = config.INPUT_DIR

    input_dir = Path(input_dir)

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input directory does not exist: {input_dir}"
        )

    files_per_task = FILES_PER_TASK

    # Maximum number of tasks submitted at one time.
    tasks_per_batch = max(
        1,
        config.NUM_WORKERS * TASKS_PER_WORKER,
    )

    wav_files = iter_wav_files(input_dir)

    print(f"Input directory: {input_dir}")
    print(f"Workers: {config.NUM_WORKERS}")
    print(f"Files per task: {files_per_task}")

    total_successful = 0

    with ProcessPoolExecutor(
        max_workers=config.NUM_WORKERS,
        initializer=initialize_worker,
        initargs=(str(FILTER_CONFIG_PATH),),
    ) as executor:

        with tqdm(
            desc="Processing WAV files",
            unit="file",
        ) as progress:

            # Submit only a limited number of tasks at a time.
            task_batches = batched(
                batched(wav_files, files_per_task),
                tasks_per_batch,
            )

            for task_batch in task_batches:

                futures = {
                    executor.submit(
                        process_wav_batch,
                        paths,
                    ): len(paths)
                    for paths in task_batch
                }

                # Collect completed tasks.
                for future in as_completed(futures):

                    progress.update(futures[future])

                    try:
                        batch_results = future.result()

                    except Exception:
                        continue

                    for result in batch_results:

                        if result["dataframe"] is not None:
                            total_successful += 1

                        yield result

    print(
        f"\nSuccessfully processed: {total_successful:,}"
    )


# ---------------------------------------------------------------------
# Main - Save CSVs
# ---------------------------------------------------------------------

def main():

    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    rejection_summary = {}
    successful = 0

    for result in process_folder():

        # Accumulate rejection statistics by class.
        stats = result["rejection_stats"]
        class_name = stats["class"]

        if class_name not in rejection_summary:
            rejection_summary[class_name] = {
                key: 0 if key != "class" else class_name
                for key in stats
            }

        for key, value in stats.items():

            if key != "class":
                rejection_summary[class_name][key] += value

        # Skip files with no surviving pulses.
        if result["dataframe"] is None:
            continue

        wav_path = Path(result["wav_path"])
        df = result["dataframe"]

        # Preserve the input folder structure.
        relative_path = wav_path.relative_to(
            Path(config.INPUT_DIR)
        )

        # Create corresponding output directory.
        csv_path = (
            output_dir
            / relative_path.parent
            / f"{wav_path.stem}.csv"
        )

        csv_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Save normalized features.
        df.to_csv(
            csv_path,
            index=False,
        )

        successful += 1

    # Save accumulated rejection statistics.
    rejection_output_path = Path(
        config.PULSE_FILTERED_REJECTION_FILE
    )

    rejection_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_rejection_summary(
        rejection_summary,
        rejection_output_path,
    )

    print("\nPipeline complete.")
    print(f"Successfully processed: {successful:,}")
    print(f"Output directory: {output_dir}")
    print(f"Rejection summary: {rejection_output_path}")


if __name__ == "__main__":
    main()
