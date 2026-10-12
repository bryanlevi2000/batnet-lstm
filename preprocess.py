
# preprocess.py

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

from skipped_files import record_skipped_file


# ============================================================
# CLASS-SPECIFIC FILTER CONFIGURATION
# ============================================================

DEFAULT_CLASS_FILTER_CONFIG = (
    Path(__file__).resolve().parent / "pulse_class_filters.json"
)

EMPTY_COLUMNS = [
    "time_us",
    "freq_hz",
    "amplitude",
    "status",
    "pulse_id",
]


def _empty_dataframe():
    return pd.DataFrame(columns=EMPTY_COLUMNS)


def _empty_filter_stats(class_name):
    return {
        "class_name": class_name,
        "candidate_pulses": 0,
        "accepted_pulses": 0,
        "rejected_pulses": 0,
        "rejected_fmin": 0,
        "rejected_fmax": 0,
        "rejected_fc_range": 0,
        "rejected_missing_fc": 0,
    }


@lru_cache(maxsize=8)
def load_class_filter_config(config_path):
    config_path = str(Path(config_path).resolve())

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            "Class filter configuration must contain a JSON object."
        )

    return data


def _validate_range(value, class_name, parameter):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            f"Class '{class_name}' parameter '{parameter}' "
            "must be a two-element array."
        )

    minimum = float(value[0])
    maximum = float(value[1])

    if minimum > maximum:
        raise ValueError(
            f"Class '{class_name}' parameter '{parameter}' "
            f"has minimum greater than maximum: {value}"
        )

    return minimum, maximum


def _value_in_range(value, value_range):
    minimum, maximum = value_range
    return minimum <= value <= maximum


def filter_pulses_by_class(
    df,
    class_name,
    config_path=DEFAULT_CLASS_FILTER_CONFIG,
    return_stats=False,
):
    stats = _empty_filter_stats(class_name)

    if df.empty:
        return (df, stats) if return_stats else df

    class_filters = load_class_filter_config(config_path)

    if class_name not in class_filters:
        raise ValueError(
            f"No class-specific pulse filter was defined for "
            f"class '{class_name}' in {Path(config_path).resolve()}"
        )

    class_config = class_filters[class_name]

    ranges = {}
    for parameter in ("fmin", "fmax", "fc"):
        if parameter not in class_config:
            raise ValueError(
                f"Class '{class_name}' is missing the "
                f"'{parameter}' range."
            )
        ranges[parameter] = _validate_range(
            class_config[parameter],
            class_name,
            parameter,
        )

    accepted_ids = []

    for pulse_id, pulse_df in df.groupby("pulse_id", sort=False):
        stats["candidate_pulses"] += 1

        if pulse_df.empty:
            stats["rejected_pulses"] += 1
            continue

        fmin = float(pulse_df["freq_hz"].min())
        fmax = float(pulse_df["freq_hz"].max())

        fmin_pass = _value_in_range(fmin, ranges["fmin"])
        fmax_pass = _value_in_range(fmax, ranges["fmax"])

        characteristic_points = pulse_df[pulse_df["status"] == 3]

        if len(characteristic_points) != 1:
            stats["rejected_pulses"] += 1
            stats["rejected_missing_fc"] += 1
            continue

        fc = float(characteristic_points.iloc[0]["freq_hz"])
        fc_pass = _value_in_range(fc, ranges["fc"])

        if not fmin_pass:
            stats["rejected_fmin"] += 1

        if not fmax_pass:
            stats["rejected_fmax"] += 1

        if not fc_pass:
            stats["rejected_fc_range"] += 1

        if fmin_pass and fmax_pass and fc_pass:
            accepted_ids.append(pulse_id)
            stats["accepted_pulses"] += 1
        else:
            stats["rejected_pulses"] += 1

    if accepted_ids:
        filtered_df = df[df["pulse_id"].isin(accepted_ids)].copy()
    else:
        filtered_df = df.iloc[0:0].copy()

    return (filtered_df, stats) if return_stats else filtered_df


# ============================================================
# PROCESS ONE WAV
# ============================================================

def _process_wav_impl(
    source_path,
    apply_class_filter=False,
    class_filter_config_path=DEFAULT_CLASS_FILTER_CONFIG,
    return_filter_stats=False,
    stop_after_class_filter=False,
):
    source_path = Path(source_path)

    y, sample_rate = sf.read(
        source_path,
        dtype="float32",
        always_2d=False,
    )

    if y.size == 0:
        raise ValueError("WAV file contains no audio samples.")

    if y.ndim > 1:
        y = y.mean(axis=1)

    y = denoise_audio(
        y,
        noise_factor=config.NOISE_FACTOR,
        noise_stride=config.NOISE_STRIDE,
    )

    y = (y * 32767).astype("int16")

    zc_data = extract_zc(
        y,
        sample_rate,
        divrat=config.DIVRAT,
        hysteresis_percentile=config.HYSTERESIS_PERCENTILE,
    )

    df = pd.DataFrame(
        zc_data,
        columns=["time_us", "freq_hz", "amplitude"],
    )

    df = filter_zc_linearity(
        df,
        lookahead_n=config.LOOKAHEAD_N,
        linearity_cutoff=config.LINEARITY_CUTOFF,
    )

    df = greedy_walk(
        df,
        time_cutoff=config.TIME_CUTOFF,
        inc_freq_cutoff=config.INC_FREQ_CUTOFF_HZ,
        dec_freq_cutoff=config.DEC_FREQ_CUTOFF_HZ,
        min_points=config.MIN_POINTS,
        buffer_us=config.BUFFER_US,
    )

    candidate_df = df[df["status"] > 1].copy()

    for _, pulse_df in candidate_df.groupby("pulse_id", sort=False):
        pulse_result = find_characteristic_frequency(
            pulse_df,
            min_points=config.CHARACTERISTIC_MIN_POINTS,
            last_fraction=config.CHARACTERISTIC_LAST_FRACTION,
        )

        if pulse_result is not None:
            df.loc[pulse_result.index, "status"] = pulse_result["status"]

    df = apply_frequency_decrease_cutoff(
        df,
        cutoff_hz=config.CHARACTERISTIC_FREQUENCY_DECREASE_CUTOFF_HZ,
    )

    df = filter_pulses(
        df,
        min_bandwidth_hz=config.MIN_BANDWIDTH_HZ,
        max_bandwidth_hz=config.MAX_BANDWIDTH_HZ,
        min_duration_us=config.MIN_DURATION_US,
        max_duration_us=config.MAX_DURATION_US,
        fmin_ranges=config.FMIN_RANGES,
        fmax_ranges=config.FMAX_RANGES,
    )

    df = df[
        (df["status"] > 1) & df["pulse_id"].notna()
    ].copy()

    if df.empty:
        if return_filter_stats:
            return df, _empty_filter_stats(source_path.parent.name)
        return df

    filter_stats = None

    if apply_class_filter:
        class_name = source_path.parent.name

        if return_filter_stats:
            df, filter_stats = filter_pulses_by_class(
                df,
                class_name=class_name,
                config_path=class_filter_config_path,
                return_stats=True,
            )
        else:
            df = filter_pulses_by_class(
                df,
                class_name=class_name,
                config_path=class_filter_config_path,
            )

        if stop_after_class_filter or df.empty:
            if return_filter_stats:
                return df, filter_stats
            return df

    df = pchip_resample_pulses(
        df,
        pchip_time_interval=config.PCHIP_TIME_INTERVAL,
    )

    df = process_pulse_dataframe(df)

    if return_filter_stats:
        return df, filter_stats

    return df


def process_wav(
    source_path,
    apply_class_filter=False,
    class_filter_config_path=DEFAULT_CLASS_FILTER_CONFIG,
    return_filter_stats=False,
    stop_after_class_filter=False,
):
    """Process a WAV; log and skip it if processing raises an error."""
    try:
        return _process_wav_impl(
            source_path,
            apply_class_filter=apply_class_filter,
            class_filter_config_path=class_filter_config_path,
            return_filter_stats=return_filter_stats,
            stop_after_class_filter=stop_after_class_filter,
        )
    except Exception as error:
        record_skipped_file(source_path, error)

        if return_filter_stats:
            return (
                _empty_dataframe(),
                _empty_filter_stats(Path(source_path).parent.name),
            )

        return _empty_dataframe()


# ============================================================
# NORMALIZATION
# ============================================================

def _process_training_wav_for_normalizer(
    source_path,
    class_filter_config_path,
):
    """Worker wrapper: one bad file must not stop normalization."""
    try:
        return _process_wav_impl(
            source_path,
            apply_class_filter=True,
            class_filter_config_path=class_filter_config_path,
        )
    except Exception as error:
        record_skipped_file(source_path, error)
        return _empty_dataframe()


def fit_training_normalizer(
    wav_files,
    class_filter_config_path=DEFAULT_CLASS_FILTER_CONFIG,
):
    stats = {}

    wav_files = list(wav_files)

    with ProcessPoolExecutor(
        max_workers=config.NUM_WORKERS
    ) as executor:
        results = executor.map(
            _process_training_wav_for_normalizer,
            wav_files,
            [class_filter_config_path] * len(wav_files),
        )

        for source_path, df in tqdm(
            zip(wav_files, results),
            total=len(wav_files),
            desc="Calculating normalization statistics",
            unit="file",
        ):
            if df.empty:
                continue

            try:
                stats = update_normalizer(
                    stats,
                    df,
                    columns=config.COLUMNS,
                )
            except Exception as error:
                record_skipped_file(source_path, error)

    if not stats:
        raise RuntimeError(
            "No normalization statistics could be calculated. "
            "Check skipped_files.csv and the preprocessing configuration."
        )

    return finalize_normalizer(stats, config.STATS_FILE)


# ============================================================
# DATAFRAME TO PULSE TENSORS
# ============================================================

def dataframe_to_pulse_tensors(df):
    if df.empty:
        return []

    feature_columns = [
        f"{column}_normalized" for column in config.COLUMNS
    ]

    tensors = []

    for _, pulse_df in df.groupby("pulse_id", sort=False):
        try:
            values = pulse_df[feature_columns].to_numpy(
                dtype="float32"
            )
            if len(values) == 0:
                continue

            tensors.append(torch.from_numpy(values))
        except Exception:
            # Skip an invalid pulse without discarding other pulses.
            continue

    return tensors


# ============================================================
# INFERENCE PREPROCESSING
# ============================================================

def preprocess_wav(source_path, normalization_path=None):
    try:
        df = _process_wav_impl(
            source_path,
            apply_class_filter=False,
        )

        if df.empty:
            return []

        if normalization_path is not None:
            df = apply_normalizer(df, normalization_path)

        return dataframe_to_pulse_tensors(df)

    except Exception as error:
        record_skipped_file(source_path, error)
        return []


# ============================================================
# TRAINING / VALIDATION PREPROCESSING
# ============================================================

def preprocess_training_wav(
    source_path,
    normalization_path=None,
    class_filter_config_path=DEFAULT_CLASS_FILTER_CONFIG,
):
    try:
        df = process_wav(
            source_path,
            apply_class_filter=True,
            class_filter_config_path=class_filter_config_path,
        )

        if df.empty:
            return []

        if normalization_path is None:
            normalization_path = config.STATS_FILE

        df = apply_normalizer(df, normalization_path)

        return dataframe_to_pulse_tensors(df)

    except Exception as error:
        record_skipped_file(source_path, error)
        return []

# ============================================================
# FIND WAV FILES
# ============================================================

def find_wav_files(input_dir):
    input_dir = Path(input_dir)

    return [
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    ]
