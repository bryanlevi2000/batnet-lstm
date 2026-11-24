
# pipeline.py
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice

from tqdm import tqdm


# Add project root (batnet-lstm) to Python's module search path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


import config

from zc.pipeline import process_file
from zc.input_files import iter_input_files, write_file_lists

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
# Batching
# ---------------------------------------------------------------------

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
# Process one input file (WAV or ZC)
# ---------------------------------------------------------------------

def _result(file_path, status, dataframe=None, rejection_stats=None, error=None):
    """
    Build the result of one file.

    status is "processed" (rejection_stats is set, and dataframe is None
    when class filtering removed every pulse), "no_pulses" (the file was
    read but gave zero pulses) or "failed" (error is set).
    """

    return {
        "file_path": str(file_path),
        "status": status,
        "dataframe": dataframe,
        "rejection_stats": rejection_stats,
        "error": error,
    }


def _process_input_file_unsafe(file_path):
    """
    Process one input file and normalize its resulting features.

    Raises an exception if the file cannot be read or processed.
    """

    pulses_df, metrics_df = process_file(file_path)

    if pulses_df.empty or metrics_df.empty:
        return _result(file_path, "no_pulses")

    class_name = Path(file_path).parent.name

    # Apply class-specific pulse filtering.
    filtered_pulses, _, rejection_stats = filter_file_pulses(
        pulses_df,
        metrics_df,
        class_name,
        _WORKER_FILTER_CONFIG,
    )

    if filtered_pulses.empty:
        return _result(
            file_path,
            "processed",
            rejection_stats=rejection_stats,
        )

    # Resample each pulse onto the fixed time grid.
    resampled_df = pchip_resample_pulses(
        filtered_pulses,
        config.PCHIP_TIME_INTERVAL,
    )

    if resampled_df.empty:
        return _result(file_path, "no_pulses")

    # Calculate slope and position features.
    features_df = process_pulse_dataframe(
        resampled_df
    )

    if features_df.empty:
        return _result(file_path, "no_pulses")

    # Apply z-score normalization.
    normalized_df = normalize_dataframe(
        features_df
    )

    if normalized_df.empty:
        return _result(file_path, "no_pulses")

    return _result(
        file_path,
        "processed",
        dataframe=normalized_df,
        rejection_stats=rejection_stats,
    )


def process_input_file(file_path):
    """
    Process one input file. Never raises: a file that cannot be read or
    processed returns status "failed" with the exception type and message.
    """

    try:
        return _process_input_file_unsafe(file_path)

    except Exception as error:
        return _result(
            file_path,
            "failed",
            error=f"{type(error).__name__}: {error}",
        )


# ---------------------------------------------------------------------
# Process a batch of input files
# ---------------------------------------------------------------------

def process_file_batch(file_paths):
    """
    Process multiple input files inside one worker.

    Returns:
        list: One result per file.
    """

    return [process_input_file(path) for path in file_paths]


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def process_folder(input_dir=None):
    """
    Recursively process WAV and ZC files using parallel processing.

    Yields one result per file, including failed and no-pulse files.
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

    input_files = iter_input_files(input_dir)

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
            desc="Processing files",
            unit="file",
        ) as progress:

            # Submit only a limited number of tasks at a time.
            task_batches = batched(
                batched(input_files, files_per_task),
                tasks_per_batch,
            )

            for task_batch in task_batches:

                futures = {
                    executor.submit(
                        process_file_batch,
                        paths,
                    ): len(paths)
                    for paths in task_batch
                }

                # Collect completed tasks.
                for future in as_completed(futures):

                    progress.update(futures[future])

                    for result in future.result():

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
    failed_files = []
    no_pulse_files = []
    successful = 0

    for result in process_folder():

        if result["status"] == "failed":
            failed_files.append(
                {"file": result["file_path"], "error": result["error"]}
            )
            continue

        if result["status"] == "no_pulses":
            no_pulse_files.append({"file": result["file_path"]})
            continue

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

        file_path = Path(result["file_path"])
        df = result["dataframe"]

        # Preserve the input folder structure.
        relative_path = file_path.relative_to(
            Path(config.INPUT_DIR)
        )

        # Create corresponding output directory.
        # The CSV keeps the full input file name, for example
        # recording.wav.csv or H9140021.26#.csv.
        csv_path = (
            output_dir
            / relative_path.parent
            / f"{file_path.name}.csv"
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

    # Save the failed and no-pulse file lists.
    write_file_lists(
        failed_files,
        no_pulse_files,
        config.PREPROCESS_FAILED_FILES_PATH,
        config.PREPROCESS_NO_PULSE_FILES_PATH,
    )

    print("\nPipeline complete.")
    print(f"Successfully processed: {successful:,}")
    print(f"Failed files: {len(failed_files):,} (see {config.PREPROCESS_FAILED_FILES_PATH})")
    print(f"No-pulse files: {len(no_pulse_files):,} (see {config.PREPROCESS_NO_PULSE_FILES_PATH})")
    print(f"Output directory: {output_dir}")
    print(f"Rejection summary: {rejection_output_path}")


if __name__ == "__main__":
    main()
