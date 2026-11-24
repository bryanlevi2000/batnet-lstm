# pipeline.py

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add the parent directory (batnet-lstm) to sys.path so Python can find config.py
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

# Local module imports (since they are now in the same 'zc' folder)
from .denoise import denoise_audio
from .zc_extraction import extract_zc
from .zc_loader import load_zc_points
from .input_files import is_wav_file, iter_input_files
from .pulse_detection import filter_zc_linearity, greedy_walk
from .pulse_cleanup import (
    filter_pulse_tails,
    clean_pulses,
)
from .pulse_metrics import (
    calculate_slopes,
    calculate_fc,
    calculate_knee,
    calculate_duration,
    calculate_fmax,
    calculate_fmin,
    calculate_tstart,
    calculate_tend,
)


METRIC_COLUMNS = [
    "pulse_id",
    "fc_freq_hz",
    "knee_freq_hz",
    "duration_us",
    "fmax_hz",
    "fmin_hz",
    "tstart_us",
    "tend_us",
]


# ============================================================
# SAFE METRIC HELPERS
# ============================================================

def run_metric(func, df, columns, *args, **kwargs):
    """Run one metric function without ever crashing the pipeline.

    First tries the whole file at once (fast). If that fails, retries one
    pulse at a time so a single bad pulse only loses its own value.
    Always returns a DataFrame with the given columns (possibly empty).
    """

    try:
        return func(df, *args, **kwargs)
    except Exception:
        pass

    parts = []

    for _, pulse in df.groupby("pulse_id", sort=False):
        try:
            parts.append(func(pulse, *args, **kwargs))
        except Exception:
            continue

    if parts:
        return pd.concat(parts, ignore_index=True)

    return pd.DataFrame(columns=columns)


def metric_series(result, column):
    """Turn a metric result into a Series indexed by pulse_id."""

    if result is None or result.empty or column not in result.columns:
        return pd.Series(dtype=float)

    return result.drop_duplicates("pulse_id").set_index("pulse_id")[column]


# ============================================================
# CALCULATE METRICS FOR ONE FILE'S PULSES
# ============================================================

def calculate_all_metrics(pulses_df):
    """Return one row per pulse containing all metrics.

    Any metric that fails for a pulse is left empty (NaN).
    """

    if pulses_df.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)

    # Unique row labels are required by the fc and knee calculations.
    # This copy is only used for metrics; the returned pulses dataframe
    # is not changed.
    df = pulses_df.reset_index(drop=True)

    # One row per pulse, in order of appearance
    metrics_df = pd.DataFrame({"pulse_id": df["pulse_id"].unique()})

    # --------------------------------------------------------
    # Slopes (needed by fc and knee)
    # --------------------------------------------------------

    try:
        df = calculate_slopes(df)
    except Exception:
        df = df.copy()
        df["slope"] = np.nan

    # --------------------------------------------------------
    # fc and knee (converted from row labels to frequencies)
    # --------------------------------------------------------

    fc = run_metric(
        calculate_fc,
        df,
        ["pulse_id", "fc_idx"],
        metric_fc_proportion=config.METRIC_FC_PROPORTION,
    )

    knee = run_metric(
        calculate_knee,
        df,
        ["pulse_id", "knee_idx"],
        fc,
        metric_knee_proportion=config.METRIC_KNEE_PROPORTION,
    )

    freq_by_row = df["freq_hz"]

    fc_idx = metric_series(fc, "fc_idx")
    knee_idx = metric_series(knee, "knee_idx")

    fc_freq = fc_idx.map(freq_by_row)
    knee_freq = knee_idx.map(freq_by_row)

    metrics_df["fc_freq_hz"] = metrics_df["pulse_id"].map(fc_freq)
    metrics_df["knee_freq_hz"] = metrics_df["pulse_id"].map(knee_freq)

    # --------------------------------------------------------
    # Simple per-pulse metrics
    # --------------------------------------------------------

    simple_metrics = [
        ("duration_us", calculate_duration),
        ("fmax_hz", calculate_fmax),
        ("fmin_hz", calculate_fmin),
        ("tstart_us", calculate_tstart),
        ("tend_us", calculate_tend),
    ]

    for column, func in simple_metrics:
        result = run_metric(func, df, ["pulse_id", column])
        metrics_df[column] = metrics_df["pulse_id"].map(
            metric_series(result, column)
        )

    return metrics_df[METRIC_COLUMNS]


# ============================================================
# LOAD ZC POINTS FROM ONE FILE
# ============================================================

def load_wav_points(source_path):
    """Read a WAV file and return its ZC points (time_us, freq_hz, amplitude)."""

    source_path = Path(source_path)

    # --------------------------------------------------------
    # Read WAV
    # --------------------------------------------------------

    y, sample_rate = librosa.load(
        source_path,
        sr=None,
        mono=True
    )

    # --------------------------------------------------------
    # Denoising
    # --------------------------------------------------------

    y = denoise_audio(
        y,
        noise_factor=config.NOISE_FACTOR,
        noise_stride=config.NOISE_STRIDE
    )

    # Convert back to the integer scale expected by ZC extraction
    y = (y * 32767).astype("int16")

    # --------------------------------------------------------
    # Zero-crossing extraction
    # --------------------------------------------------------

    zc_data = extract_zc(
        y,
        sample_rate,
        divrat=config.DIVRAT,
        hysteresis_percentile=config.HYSTERESIS_PERCENTILE
    )

    return pd.DataFrame(
        zc_data,
        columns=[
            "time_us",
            "freq_hz",
            "amplitude"
        ]
    )


def load_points(source_path):
    """Return the ZC points of a WAV or ZC (.zc, #) file.

    Raises an exception if the file cannot be read.
    """

    if is_wav_file(source_path):
        return load_wav_points(source_path)

    return load_zc_points(source_path)


# ============================================================
# CLEAN ZC POINTS INTO PULSES (WAV AND ZC FILES)
# ============================================================

def process_points(df):
    """Group ZC points into pulses, clean them, and calculate their metrics.

    Returns (pulses_df, metrics_df). Both are empty when no pulse is found.
    """

    # --------------------------------------------------------
    # Local linearity
    # --------------------------------------------------------

    df = filter_zc_linearity(
        df,
        lookahead_n=config.LOOKAHEAD_N,
        linearity_cutoff=config.LINEARITY_CUTOFF
    )

    # --------------------------------------------------------
    # Greedy walk
    # --------------------------------------------------------

    df = greedy_walk(
        df,
        time_cutoff=config.TIME_CUTOFF,
        inc_freq_cutoff=config.INC_FREQ_CUTOFF_HZ,
        dec_freq_cutoff=config.DEC_FREQ_CUTOFF_HZ,
        min_points=config.MIN_POINTS,
        buffer_us=config.BUFFER_US
    )

    # --------------------------------------------------------
    # Frequency-decrease cutoff in pulse tails
    # --------------------------------------------------------

    df = filter_pulse_tails(
        df,
        tail_time_fraction=config.TAIL_TIME_FRACTION,
        tail_dec_freq_cutoff=config.TAIL_DEC_FREQ_CUTOFF_HZ
    )

    # --------------------------------------------------------
    # Pulse metric calculation and filtering
    # --------------------------------------------------------

    df = clean_pulses(
        df,
        min_bandwidth_hz=config.MIN_BANDWIDTH_HZ,
        max_bandwidth_hz=config.MAX_BANDWIDTH_HZ,
        min_duration_us=config.MIN_DURATION_US,
        max_duration_us=config.MAX_DURATION_US,
        fmin_ranges=config.FMIN_RANGES,
        fmax_ranges=config.FMAX_RANGES
    )

    # --------------------------------------------------------
    # Keep only accepted pulse points
    # --------------------------------------------------------

    pulses_df = df[df["status"] > 1].copy()

    # --------------------------------------------------------
    # One row per pulse with all metrics
    # --------------------------------------------------------

    metrics_df = calculate_all_metrics(pulses_df)

    return pulses_df, metrics_df


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(source_path):
    """Process one WAV or ZC file into (pulses_df, metrics_df).

    Raises an exception if the file cannot be read or processed. A readable
    file with no pulses returns empty dataframes.
    """

    return process_points(load_points(source_path))


# ============================================================
# MAIN
# ============================================================

def main():

    input_dir = Path(config.INPUT_DIR)

    # --------------------------------------------------------
    # Find all WAV and ZC files
    # --------------------------------------------------------

    input_files = list(iter_input_files(input_dir))

    print(f"Found {len(input_files):,} WAV and ZC files.")

    if not input_files:
        return {}, {}

    # --------------------------------------------------------
    # Process files in parallel
    # --------------------------------------------------------

    pulses_by_file = {}
    metrics_by_file = {}
    failed = 0

    with ProcessPoolExecutor(
        max_workers=config.NUM_WORKERS
    ) as executor:

        future_to_path = {
            executor.submit(process_file, path): path
            for path in input_files
        }

        with tqdm(
            total=len(future_to_path),
            desc="Processing files",
            unit="file"
        ) as progress:

            for future in as_completed(future_to_path):

                path = future_to_path[future]

                try:
                    pulses_df, metrics_df = future.result()
                except Exception as error:
                    # Failed files are left out of the results
                    failed += 1
                    tqdm.write(f"Failed: {path}: {type(error).__name__}: {error}")
                else:
                    # Files with no pulses are kept, with empty dataframes
                    pulses_by_file[path] = pulses_df
                    metrics_by_file[path] = metrics_df

                progress.update(1)

    print(f"Failed files: {failed:,}")

    return pulses_by_file, metrics_by_file

# ============================================================
# OPTIONAL DEBUGGING MAIN (WRITES CSV FILES)
# ============================================================

# def main():

#     input_dir = Path(config.INPUT_DIR)
    
#     # Define an output directory for your CSV test files
#     output_dir = Path("debug_csv_outputs")
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # --------------------------------------------------------
#     # Find all WAV and ZC files
#     # --------------------------------------------------------

#     input_files = list(iter_input_files(input_dir))

#     print(f"Found {len(input_files):,} WAV and ZC files total.")

#     if not input_files:
#         return {}, {}

#     # --- QUICK CHECK MODIFICATION ---
#     # Limit to the first 10 files for testing purposes
#     input_files = input_files[:10]
#     print(f"Test mode: Processing only the first {len(input_files)} file(s).")

#     # --------------------------------------------------------
#     # Process files in parallel
#     # --------------------------------------------------------

#     pulses_by_file = {}
#     metrics_by_file = {}

#     with ProcessPoolExecutor(
#         max_workers=config.NUM_WORKERS
#     ) as executor:

#         future_to_path = {
#             executor.submit(process_file, input_path): input_path
#             for input_path in input_files
#         }

#         with tqdm(
#             total=len(future_to_path),
#             desc="Processing files",
#             unit="file"
#         ) as progress:

#             for future in as_completed(future_to_path):

#                 input_path = future_to_path[future]
#                 pulses_df, metrics_df = future.result()

#                 pulses_by_file[input_path] = pulses_df
#                 metrics_by_file[input_path] = metrics_df

#                 # --- WRITE CSV FILES FOR INSPECTION ---
#                 name = input_path.name
#                 pulses_csv_path = output_dir / f"{name}_pulses.csv"
#                 metrics_csv_path = output_dir / f"{name}_metrics.csv"

#                 pulses_df.to_csv(pulses_csv_path, index=False)
#                 metrics_df.to_csv(metrics_csv_path, index=False)
                
#                 print(f"\nSaved CSVs for {name} to {output_dir}/")

#                 progress.update(1)

#     print(f"\nAll test CSVs successfully written to: {output_dir.resolve()}")
#     return pulses_by_file, metrics_by_file

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    pulses_by_file, metrics_by_file = main()