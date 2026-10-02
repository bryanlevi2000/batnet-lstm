# pulse_filtering.py
import numpy as np
import pandas as pd


def find_characteristic_frequency(
    pulse_df: pd.DataFrame,
    min_points: int,
    last_fraction: float,
):
    """
    Find the characteristic-frequency point in one pulse.

    The algorithm is:
    1. Sort points by time.
    2. Calculate the actual pulse start and end times.
    3. Keep only the last fraction of the pulse by time.
    4. Calculate slopes between adjacent points.
    5. Ignore zero-frequency-change and positive-slope segments.
    6. Select the least-negative slope.
    7. Return the original dataframe index of the later endpoint.
    """
    time_us = pulse_df["time_us"]
    if not (time_us.is_monotonic_increasing and time_us.is_unique):
        pulse_df = pulse_df.sort_values("time_us")
        time_us = pulse_df["time_us"]

    # Use raw numpy arrays for the rest of the operations
    time_us_arr = time_us.to_numpy(dtype=float)

    start_time = time_us_arr[0]
    end_time = time_us_arr[-1]
    pulse_duration = end_time - start_time

    if pulse_duration <= 0:
        return None

    cutoff_time = start_time + (1.0 - last_fraction) * pulse_duration

    cutoff_idx = np.searchsorted(time_us_arr, cutoff_time)

    # 2. Check min_points before allocating any new arrays
    if len(time_us_arr) - cutoff_idx < min_points:
        return None

    # Extract only the required tail of the arrays (Creates lightweight views)
    x = time_us_arr[cutoff_idx:]
    y = pulse_df["freq_hz"].to_numpy(dtype=float)[cutoff_idx:]
    original_indices = pulse_df.index.to_numpy()[cutoff_idx:]

    delta_time = np.diff(x)
    delta_frequency = np.diff(y)

    # 3. Mathematical Simplification: 
    # A slope is negative ONLY if delta_freq is negative and delta_time > 0. 
    # This completely avoids computing slopes for the whole array.
    valid_segments = (delta_frequency < 0) & (delta_time > 0)

    if not valid_segments.any():
        return None

    # 4. Defer division: calculate division ONLY on the valid segments
    valid_dfreq = delta_frequency[valid_segments]
    valid_dtime = delta_time[valid_segments]
    
    valid_slopes = valid_dfreq / valid_dtime

    # Find the maximum (least negative) slope among the valid ones
    least_negative_idx = np.argmax(valid_slopes)

    # Map back to the original array position
    segment_positions = np.flatnonzero(valid_segments)
    segment_start = segment_positions[least_negative_idx]
    segment_end = segment_start + 1

    return original_indices[segment_end]


def apply_frequency_decrease_cutoff(
    pulse_df: pd.DataFrame,
    characteristic_index,
    cutoff_hz: float,
) -> pd.DataFrame:
    """
    Starting after the characteristic-frequency point, find the first
    adjacent frequency decrease greater than cutoff_hz.

    The point immediately before the excessive decrease is retained.
    The point causing the excessive decrease and all points after it
    within the pulse are assigned status 1.
    """
    if characteristic_index is None:
        return pulse_df

    time_us = pulse_df["time_us"]
    if not (time_us.is_monotonic_increasing and time_us.is_unique):
        pulse_df = pulse_df.sort_values("time_us")

    # 1. Fast index lookup (O(1) average) instead of full array comparison (O(N))
    try:
        char_pos = pulse_df.index.get_loc(characteristic_index)
        # Handle cases where index might not be strictly unique
        if isinstance(char_pos, slice):
            char_pos = char_pos.start
        elif isinstance(char_pos, np.ndarray):
            char_pos = char_pos.argmax()
    except KeyError:
        return pulse_df

    # 2. Extract and diff ONLY the relevant slice again
    freqs = pulse_df["freq_hz"].to_numpy(copy=False)[char_pos:]
    if len(freqs) < 2:
        return pulse_df

    changes = np.diff(freqs)
    excessive = changes < -cutoff_hz

    if not excessive.any():
        return pulse_df

    first_excessive = excessive.argmax()
    cutoff_position = char_pos + first_excessive + 1

    # 4. Defer the DataFrame copy until modification is guaranteed
    pulse_df = pulse_df.copy()

    if "status" in pulse_df.columns:
        status_col_idx = pulse_df.columns.get_loc("status")
        pulse_df.iloc[cutoff_position:, status_col_idx] = 1
    else:
        pulse_df.loc[pulse_df.index[cutoff_position:], "status"] = 1

    return pulse_df


def calculate_pulse_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate summary metrics for each pulse.

    Only points with status > 1 are included.
    """
    # Define columns once to avoid repetition
    out_cols = [
        "pulse_id",
        "n_points",
        "duration_us",
        "bandwidth_hz",
        "fmin",
        "fmax",
        "first_freq_hz",
        "last_freq_hz",
    ]

    if df.empty:
        return pd.DataFrame(columns=out_cols)

    # 1. Evaluate mask without immediately allocating a copied DataFrame
    mask = df["status"] > 1
    if not mask.any():
        return pd.DataFrame(columns=out_cols)

    # 2. Subset ONLY the required columns BEFORE sorting
    # This prevents pandas from moving unneeded data around in memory during the sort
    df_subset = df.loc[mask, ["pulse_id", "time_us", "freq_hz"]]

    # 3. Sort the this DataFrame
    df_sorted = df_subset.sort_values(
        ["pulse_id", "time_us"], 
        kind="stable"
    )

    # No need to sort again here as we just did that 
    grouped = df_sorted.groupby(
        "pulse_id", 
        sort=False
    )

    pulse_metrics = grouped.agg(
        n_points=("freq_hz", "size"),
        first_time=("time_us", "first"),
        last_time=("time_us", "last"),
        fmin=("freq_hz", "min"),
        fmax=("freq_hz", "max"),
        first_freq_hz=("freq_hz", "first"),
        last_freq_hz=("freq_hz", "last"),
    )

    pulse_metrics["duration_us"] = (
        pulse_metrics["last_time"] - pulse_metrics["first_time"]
    )
    pulse_metrics["bandwidth_hz"] = (
        pulse_metrics["fmax"] - pulse_metrics["fmin"]
    )

    pulse_metrics = pulse_metrics.reset_index()

    return pulse_metrics[out_cols]


def filter_pulses(
    df: pd.DataFrame,
    min_bandwidth_hz: float,
    max_bandwidth_hz: float,
    min_duration_us: float,
    max_duration_us: float,
    fmin_ranges: tuple[float, float],
    fmax_ranges: tuple[float, float],
) -> pd.DataFrame:
    """
    Calculate pulse metrics and filter pulses.

    Accepted pulse points receive status 2.
    Rejected pulse points receive status 1 and have their pulse_id
    cleared (set to pd.NA).
    Characteristic-frequency points retain status 3
    within accepted pulses.
    """
    pulse_metrics = calculate_pulse_metrics(df)

    min_fmin_hz, max_fmin_hz = fmin_ranges
    min_fmax_hz, max_fmax_hz = fmax_ranges

    if not pulse_metrics.empty:
        bw = pulse_metrics["bandwidth_hz"].to_numpy()
        dur = pulse_metrics["duration_us"].to_numpy()
        fmin = pulse_metrics["fmin"].to_numpy()
        fmax = pulse_metrics["fmax"].to_numpy()
        last_f = pulse_metrics["last_freq_hz"].to_numpy()
        first_f = pulse_metrics["first_freq_hz"].to_numpy()

        rejected = (
            (bw < min_bandwidth_hz) | (bw > max_bandwidth_hz) |
            (dur < min_duration_us) | (dur > max_duration_us) |
            (fmin < min_fmin_hz) | (fmin > max_fmin_hz) |
            (fmax < min_fmax_hz) | (fmax > max_fmax_hz) |
            (last_f > first_f)
        )
        accepted_ids = pulse_metrics.loc[~rejected, "pulse_id"]
    else:
        accepted_ids = pd.Series([], dtype=object)

    result = df.copy()

    status_arr = result["status"].to_numpy(copy=True)
    pulse_id_series = result["pulse_id"]

    # Masks must be built from the ORIGINAL pulse_id column,
    # before any pulse_ids are cleared.
    is_pulse = pulse_id_series.notna().to_numpy()
    is_accepted = pulse_id_series.isin(accepted_ids).to_numpy()
    is_rejected = is_pulse & ~is_accepted
    was_characteristic = (status_arr == 3)

    # Status updates (unchanged)
    status_arr[is_pulse] = 1
    status_arr[is_accepted] = 2
    status_arr[is_accepted & was_characteristic] = 3

    # Clear pulse_id on rejected pulse points
    pulse_id_arr = pulse_id_series.to_numpy(dtype=object, copy=True)
    pulse_id_arr[is_rejected] = pd.NA

    # Write back to the DataFrame
    result["status"] = status_arr
    result["pulse_id"] = pulse_id_arr

    return result

