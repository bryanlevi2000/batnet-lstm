from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json

import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

import config

from zc.denoise import denoise_audio
from zc.zc_extraction import extract_zc
from zc.pulse_detection import filter_zc_linearity, greedy_walk
from zc.pulse_filtering import (
    find_characteristic_frequency,
    apply_frequency_decrease_cutoff,
    filter_pulses,
)

from preprocessing.pchip import pchip_resample_pulses
from preprocessing.features import process_pulse_dataframe
from preprocessing.normalize import (
    update_normalizer,
    finalize_normalizer,
    apply_normalizer,
)


# ============================================================
# CLASS-SPECIFIC TRAINING FILTER
# ============================================================

DEFAULT_CLASS_FILTER_CONFIG = (
    Path(__file__).resolve().parent
    / "pulse_class_filters.json"
)


@lru_cache(maxsize=8)
def load_class_filter_config(config_path):
    """
    Load and cache the class-specific pulse filter JSON.

    JSON format:

    {
        "EPTFUS": {
            "fmin": [12000, 30000],
            "fmax": [35000, 70000],
            "fc":   [20000, 30000]
        }
    }
    """

    config_path = str(Path(config_path).resolve())

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as json_file:

        data = json.load(json_file)

    if not isinstance(data, dict):
        raise ValueError(
            "Class filter configuration must contain "
            "a JSON object at the top level."
        )

    return data


def _validate_range(value, class_name, parameter):
    """
    Validate and convert one [min, max] range.
    """

    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"Class '{class_name}' parameter '{parameter}' "
            "must be a two-element array like [12000, 30000]."
        )

    if len(value) != 2:
        raise ValueError(
            f"Class '{class_name}' parameter '{parameter}' "
            "must contain exactly two values."
        )

    minimum = float(value[0])
    maximum = float(value[1])

    if minimum > maximum:
        raise ValueError(
            f"Class '{class_name}' parameter '{parameter}' "
            f"has minimum > maximum: {value}"
        )

    return minimum, maximum


def _value_in_range(value, value_range):
    """
    Inclusive range check.
    """
    minimum, maximum = value_range

    return (
        minimum <= value <= maximum
    )


def filter_pulses_by_class(
    df,
    class_name,
    config_path=DEFAULT_CLASS_FILTER_CONFIG,
    return_stats=False
):
    """
    Apply class-specific fmin/fmax/fc filtering.

    A pulse is retained only if ALL THREE conditions pass:

        fmin range
        fmax range
        fc range

    fmin = minimum freq_hz value in the pulse
    fmax = maximum freq_hz value in the pulse
    fc   = freq_hz value where status == 3

    Rejected pulses are completely removed.

    If return_stats=True, returns:

        filtered_df, stats
    """

    stats = {
        "class_name": class_name,
        "candidate_pulses": 0,
        "accepted_pulses": 0,
        "rejected_pulses": 0,
        "rejected_fmin": 0,
        "rejected_fmax": 0,
        "rejected_fc_range": 0,
        "rejected_missing_fc": 0,
    }

    if df.empty:

        if return_stats:
            return df, stats

        return df

    class_filters = load_class_filter_config(
        str(Path(config_path).resolve())
    )

    if class_name not in class_filters:
        raise ValueError(
            f"No class-specific pulse filter was defined "
            f"for class '{class_name}' in:\n"
            f"{Path(config_path).resolve()}"
        )

    class_config = class_filters[class_name]

    required_parameters = (
        "fmin",
        "fmax",
        "fc"
    )

    for parameter in required_parameters:

        if parameter not in class_config:
            raise ValueError(
                f"Class '{class_name}' is missing "
                f"the '{parameter}' range."
            )

    fmin_range = _validate_range(
        class_config["fmin"],
        class_name,
        "fmin"
    )

    fmax_range = _validate_range(
        class_config["fmax"],
        class_name,
        "fmax"
    )

    fc_range = _validate_range(
        class_config["fc"],
        class_name,
        "fc"
    )

    accepted_pulse_ids = []

    for pulse_id, pulse_df in df.groupby(
        "pulse_id",
        sort=False
    ):

        stats["candidate_pulses"] += 1

        if pulse_df.empty:
            stats["rejected_pulses"] += 1
            continue

        # ----------------------------------------------------
        # fmin / fmax
        # ----------------------------------------------------

        fmin = float(
            pulse_df["freq_hz"].min()
        )

        fmax = float(
            pulse_df["freq_hz"].max()
        )

        fmin_pass = _value_in_range(
            fmin,
            fmin_range
        )

        fmax_pass = _value_in_range(
            fmax,
            fmax_range
        )

        # ----------------------------------------------------
        # fc
        # ----------------------------------------------------

        characteristic_points = pulse_df[
            pulse_df["status"] == 3
        ]

        if len(characteristic_points) != 1:

            stats["rejected_pulses"] += 1
            stats["rejected_missing_fc"] += 1

            continue

        fc = float(
            characteristic_points.iloc[0]["freq_hz"]
        )

        fc_pass = _value_in_range(
            fc,
            fc_range
        )

        # ----------------------------------------------------
        # Record individual criterion failures
        # ----------------------------------------------------

        if not fmin_pass:
            stats["rejected_fmin"] += 1

        if not fmax_pass:
            stats["rejected_fmax"] += 1

        if not fc_pass:
            stats["rejected_fc_range"] += 1

        # ----------------------------------------------------
        # Require ALL conditions to pass
        # ----------------------------------------------------

        if (
            fmin_pass
            and fmax_pass
            and fc_pass
        ):

            accepted_pulse_ids.append(
                pulse_id
            )

            stats["accepted_pulses"] += 1

        else:

            stats["rejected_pulses"] += 1

    # --------------------------------------------------------
    # Remove rejected pulses
    # --------------------------------------------------------

    if not accepted_pulse_ids:

        filtered_df = df.iloc[0:0].copy()

    else:

        filtered_df = df[
            df["pulse_id"].isin(
                accepted_pulse_ids
            )
        ].copy()

    if return_stats:
        return filtered_df, stats

    return filtered_df

# ============================================================
# PROCESS ONE WAV THROUGH ZC + PCHIP + FEATURES
# ============================================================

def process_wav(
    source_path,
    apply_class_filter=False,
    class_filter_config_path=DEFAULT_CLASS_FILTER_CONFIG,
    return_filter_stats=False,
    stop_after_class_filter=False
):
    
    """
    Process one WAV through:

        WAV
        -> denoising
        -> zero-crossing extraction
        -> local linearity
        -> greedy walk
        -> characteristic frequency
        -> frequency-decrease cutoff
        -> pulse filtering
        -> optional class-specific pulse filtering
        -> PCHIP interpolation
        -> feature calculation

    The class-specific filter is disabled by default.

    This allows the same function to be used for inference
    without applying the training-only class filter.
    """

    source_path = Path(source_path)

    # --------------------------------------------------------
    # Read WAV
    # --------------------------------------------------------

    y, sample_rate = sf.read(
        source_path,
        dtype="float32",
        always_2d=False
    )

    # Convert to mono
    if y.ndim > 1:
        y = y.mean(axis=1)

    # --------------------------------------------------------
    # Denoising
    # --------------------------------------------------------

    y = denoise_audio(
        y,
        noise_factor=config.NOISE_FACTOR,
        noise_stride=config.NOISE_STRIDE
    )

    # Convert to integer scale expected by ZC extraction
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

    df = pd.DataFrame(
        zc_data,
        columns=[
            "time_us",
            "freq_hz",
            "amplitude"
        ]
    )

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
    # Characteristic frequency
    # --------------------------------------------------------

    candidate_df = df[df["status"] > 1]
    characteristic_indices = []

    for pulse_id, pulse_df in candidate_df.groupby(
        "pulse_id",
        sort=False
    ):

        characteristic_index = find_characteristic_frequency(
            pulse_df,
            min_points=config.CHARACTERISTIC_MIN_POINTS,
            last_fraction=config.CHARACTERISTIC_LAST_FRACTION
        )

        if characteristic_index is not None:
            characteristic_indices.append(
                characteristic_index
            )

    if characteristic_indices:
        df.loc[
            characteristic_indices,
            "status"
        ] = 3

    # --------------------------------------------------------
    # Frequency-decrease cutoff
    # --------------------------------------------------------

    candidate_df = df[df["status"] > 1]
    cutoff_indices = []

    for pulse_id, pulse_df in candidate_df.groupby(
        "pulse_id",
        sort=False
    ):

        characteristic_points = pulse_df[
            pulse_df["status"] == 3
        ]

        if characteristic_points.empty:
            continue

        characteristic_index = (
            characteristic_points.index[0]
        )

        pulse_result = apply_frequency_decrease_cutoff(
            pulse_df,
            characteristic_index=characteristic_index,
            cutoff_hz=(
                config.CHARACTERISTIC_FREQUENCY_DECREASE_CUTOFF_HZ
            )
        )

        cutoff_indices.extend(
            pulse_result.index[
                pulse_result["status"] == 1
            ]
        )

    if cutoff_indices:
        df.loc[
            cutoff_indices,
            "status"
        ] = 1

    # --------------------------------------------------------
    # Existing pulse filtering
    # --------------------------------------------------------

    df = filter_pulses(
        df,
        min_bandwidth_hz=config.MIN_BANDWIDTH_HZ,
        max_bandwidth_hz=config.MAX_BANDWIDTH_HZ,
        min_duration_us=config.MIN_DURATION_US,
        max_duration_us=config.MAX_DURATION_US,
        fmin_ranges=config.FMIN_RANGES,
        fmax_ranges=config.FMAX_RANGES
    )

    # --------------------------------------------------------
    # Keep accepted pulse points
    # --------------------------------------------------------

    df = df[
        (df["status"] > 1) &
        (df["pulse_id"].notna())
    ].copy()

    if df.empty:

        if return_filter_stats:

            return df, {
                "class_name": source_path.parent.name,
                "candidate_pulses": 0,
                "accepted_pulses": 0,
                "rejected_pulses": 0,
                "rejected_fmin": 0,
                "rejected_fmax": 0,
                "rejected_fc_range": 0,
                "rejected_missing_fc": 0,
            }

        return df
    # --------------------------------------------------------
    # NEW CLASS-SPECIFIC TRAINING FILTER
    #
    # This happens AFTER the existing pulse filtering and
    # BEFORE PCHIP interpolation.
    # --------------------------------------------------------

    filter_stats = None

    if apply_class_filter:

        class_name = source_path.parent.name

        if return_filter_stats:

            df, filter_stats = filter_pulses_by_class(
                df,
                class_name=class_name,
                config_path=class_filter_config_path,
                return_stats=True
            )

        else:

            df = filter_pulses_by_class(
                df,
                class_name=class_name,
                config_path=class_filter_config_path
            )

        if stop_after_class_filter:

            if return_filter_stats:
                return df, filter_stats

            return df

        if df.empty:

            if return_filter_stats:
                return df, filter_stats

            return df

    # --------------------------------------------------------
    # PCHIP interpolation
    # --------------------------------------------------------

    df = pchip_resample_pulses(
        df,
        pchip_time_interval=config.PCHIP_TIME_INTERVAL
    )

    # --------------------------------------------------------
    # Feature calculation
    # --------------------------------------------------------

    df = process_pulse_dataframe(df)

    return df


# ============================================================
# TOP-LEVEL WORKER FOR TRAINING NORMALIZATION
# ============================================================

def _process_training_wav_for_normalizer(
    source_path,
    class_filter_config_path
):
    return process_wav(
        source_path,
        apply_class_filter=True,
        class_filter_config_path=class_filter_config_path
    )


# ============================================================
# FIT NORMALIZATION STATISTICS
# ============================================================

def fit_training_normalizer(
    wav_files,
    class_filter_config_path=DEFAULT_CLASS_FILTER_CONFIG
):
    """
    Process all training WAV files and calculate global
    normalization statistics.

    The class-specific training pulse filter IS applied here.
    """

    stats = {}

    worker_args = [
        (
            path,
            class_filter_config_path
        )
        for path in wav_files
    ]

    with ProcessPoolExecutor(
        max_workers=config.NUM_WORKERS
    ) as executor:

        results = executor.map(
            _process_training_wav_for_normalizer,
            [
                path
                for path, _ in worker_args
            ],
            [
                filter_path
                for _, filter_path in worker_args
            ]
        )

        for df in tqdm(
            results,
            total=len(wav_files),
            desc="Calculating normalization statistics",
            unit="file"
        ):

            if df.empty:
                continue

            stats = update_normalizer(
                stats,
                df,
                columns=config.COLUMNS
            )

    return finalize_normalizer(
        stats,
        config.STATS_FILE
    )


# ============================================================
# CONVERT ONE DATAFRAME TO PULSE TENSORS
# ============================================================

def dataframe_to_pulse_tensors(df):
    """
    Convert a processed, normalized dataframe into one tensor
    per pulse.

    Each tensor has shape:

        [sequence_length, number_of_features]
    """

    if df.empty:
        return []

    feature_columns = [
        f"{column}_normalized"
        for column in config.COLUMNS
    ]

    tensors = []

    for pulse_id, pulse_df in df.groupby(
        "pulse_id",
        sort=False
    ):

        values = pulse_df[
            feature_columns
        ].to_numpy(dtype="float32")

        tensor = torch.from_numpy(values)

        tensors.append(tensor)

    return tensors


# ============================================================
# PROCESS ONE WAV INTO PULSE TENSORS
# ============================================================

def preprocess_wav(
    source_path,
    normalization_path=None
):
    """
    Inference-safe preprocessing.

    The class-specific training filter is NOT applied.

    Parameters
    ----------
    source_path : str or Path
        WAV file.

    normalization_path : str or Path, optional
        JSON normalization statistics. If supplied, normalized
        feature columns are added before tensor conversion.
    """

    df = process_wav(
        source_path,
        apply_class_filter=False
    )

    if df.empty:
        return []

    if normalization_path is not None:
        df = apply_normalizer(
            df,
            normalization_path
        )

    return dataframe_to_pulse_tensors(df)


# ============================================================
# PROCESS TRAINING / VALIDATION WAV INTO PULSE TENSORS
# ============================================================

def preprocess_training_wav(
    source_path,
    normalization_path=None,
    class_filter_config_path=DEFAULT_CLASS_FILTER_CONFIG
):
    """
    Training/validation preprocessing.

    The class-specific pulse filter IS applied before PCHIP.
    """

    df = process_wav(
        source_path,
        apply_class_filter=True,
        class_filter_config_path=class_filter_config_path
    )

    if df.empty:
        return []

    if normalization_path is not None:
        df = apply_normalizer(
            df,
            normalization_path
        )

    return dataframe_to_pulse_tensors(df)


# ============================================================
# FIND WAV FILES
# ============================================================

def find_wav_files(input_dir):
    """
    Recursively find all WAV files under a directory.
    """

    input_dir = Path(input_dir)

    return [
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".wav"
    ]
