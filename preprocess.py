# preprocess.py
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

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
# PROCESS ONE WAV THROUGH ZC + PCHIP + FEATURES
# ============================================================

def process_wav(source_path):
    """
    Process one WAV file through:

        WAV
        -> denoising
        -> zero-crossing extraction
        -> local linearity
        -> greedy walk
        -> characteristic frequency
        -> frequency-decrease cutoff
        -> pulse filtering
        -> PCHIP interpolation
        -> feature calculation

    Returns
    -------
    pandas.DataFrame
        Processed dataframe containing accepted pulses.
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
            characteristic_indices.append(characteristic_index)

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

        characteristic_index = characteristic_points.index[0]

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
        df.loc[cutoff_indices, "status"] = 1

    # --------------------------------------------------------
    # Pulse filtering
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
# FIT NORMALIZATION STATISTICS
# ============================================================

def fit_training_normalizer(wav_files):
    """
    Process all training WAV files and calculate global
    normalization statistics.

    Statistics are calculated across all valid rows belonging
    to actual pulses.
    """

    stats = {}

    with ProcessPoolExecutor(
        max_workers=config.NUM_WORKERS
    ) as executor:

        results = executor.map(
            process_wav,
            wav_files
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
    Process one WAV through the entire preprocessing pipeline
    and return one tensor per pulse.

    Parameters
    ----------
    source_path : str or Path
        WAV file.
    normalization_path : str or Path, optional
        JSON normalization statistics. If supplied, normalized
        feature columns are added before tensor conversion.

    Returns
    -------
    list[torch.Tensor]
    """

    df = process_wav(source_path)

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
