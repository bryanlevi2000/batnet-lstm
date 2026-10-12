# pulse_filtering.py
import numpy as np
import pandas as pd


def find_characteristic_frequency(
    pulse_df: pd.DataFrame,
    min_points: int,
    last_fraction: float,
):
    """
    Find and mark the characteristic-frequency point in one pulse.

    The algorithm is:
    1. Sort points by time.
    2. Calculate the actual pulse start and end times.
    3. Keep only the last fraction of the pulse by time.
    4. Calculate slopes between adjacent points.
    5. Ignore zero-frequency-change and positive-slope segments.
    6. Select the least-negative slope.
    7. Set the characteristic-frequency point to status = 3.
    8. Return the modified dataframe.

    Returns None if the pulse does not contain enough points or
    does not contain a valid negative-frequency slope.
    """
    time_us = pulse_df["time_us"]

    if not (time_us.is_monotonic_increasing and time_us.is_unique):
        pulse_df = pulse_df.sort_values("time_us")
        time_us = pulse_df["time_us"]

    # Use raw numpy arrays for the rest of the operations.
    time_us_arr = time_us.to_numpy(dtype=float)

    start_time = time_us_arr[0]
    end_time = time_us_arr[-1]
    pulse_duration = end_time - start_time

    if pulse_duration <= 0:
        return None

    cutoff_time = (
        start_time
        + (1.0 - last_fraction) * pulse_duration
    )

    cutoff_idx = np.searchsorted(
        time_us_arr,
        cutoff_time
    )

    # Check min_points before allocating additional arrays.
    if len(time_us_arr) - cutoff_idx < min_points:
        return None

    # Extract only the required tail of the arrays.
    x = time_us_arr[cutoff_idx:]
    y = pulse_df["freq_hz"].to_numpy(dtype=float)[cutoff_idx:]
    original_indices = pulse_df.index.to_numpy()[cutoff_idx:]

    delta_time = np.diff(x)
    delta_frequency = np.diff(y)

    # A slope is negative only if frequency decreases
    # and time increases.
    valid_segments = (
        (delta_frequency < 0)
        & (delta_time > 0)
    )

    if not valid_segments.any():
        return None

    valid_dfreq = delta_frequency[valid_segments]
    valid_dtime = delta_time[valid_segments]

    valid_slopes = valid_dfreq / valid_dtime

    # Find the least-negative slope.
    least_negative_idx = np.argmax(valid_slopes)

    # Map back to the original array position.
    segment_positions = np.flatnonzero(valid_segments)
    segment_start = segment_positions[least_negative_idx]
    segment_end = segment_start + 1

    characteristic_index = original_indices[segment_end]

    # Mark the characteristic-frequency point.
    pulse_df.loc[characteristic_index, "status"] = 3

    return pulse_df


def apply_frequency_decrease_cutoff(
    df: pd.DataFrame,
    cutoff_hz: float,
) -> pd.DataFrame:
    """
    For each pulse, starting after the characteristic-frequency point
    (status == 3), find the first adjacent frequency decrease greater
    than cutoff_hz.

    The point immediately before the excessive decrease is retained.
    The point causing the excessive decrease and all subsequent points
    within that pulse are assigned status 1 and their pulse_id is
    cleared.

    The input dataframe is modified in place and returned.
    No copy is created.
    """
    if df.empty:
        return df

    # Only process rows belonging to an actual pulse.
    pulse_mask = df["pulse_id"].notna()

    if not pulse_mask.any():
        return df

    # Group only rows with valid pulse IDs.
    for pulse_id, pulse_df in df.loc[pulse_mask].groupby(
        "pulse_id",
        sort=False
    ):
        # Find the characteristic-frequency point.
        characteristic_rows = pulse_df.index[
            pulse_df["status"] == 3
        ]

        if len(characteristic_rows) == 0:
            continue

        characteristic_index = characteristic_rows[0]

        # Position of the characteristic point within this pulse.
        char_pos = pulse_df.index.get_loc(characteristic_index)

        if isinstance(char_pos, slice):
            char_pos = char_pos.start
        elif isinstance(char_pos, np.ndarray):
            char_pos = np.flatnonzero(char_pos)[0]

        # Need at least one point after the characteristic point.
        if char_pos >= len(pulse_df) - 1:
            continue

        # Frequency values beginning at the characteristic point.
        freqs = pulse_df["freq_hz"].to_numpy(copy=False)[char_pos:]

        # Frequency changes between adjacent points.
        changes = np.diff(freqs)

        # Look for the first decrease greater than cutoff_hz.
        excessive = changes < -cutoff_hz

        if not excessive.any():
            continue

        first_excessive = np.flatnonzero(excessive)[0]

        # +1 because the excessive change occurs between two points;
        # this identifies the point causing the excessive decrease.
        cutoff_position = char_pos + first_excessive + 1

        # Get the original dataframe indices for the rejected tail.
        rejected_indices = pulse_df.index[cutoff_position:]

        # Reject the tail.
        df.loc[rejected_indices, "status"] = 1
        df.loc[rejected_indices, "pulse_id"] = pd.NA

    return df

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

    Accepted pulse points retain status 2, except characteristic-
    frequency points, which retain status 3.

    Rejected pulse points receive status 1 and have their pulse_id
    cleared (set to pd.NA).

    The input dataframe is modified in place and returned.
    """
    pulse_metrics = calculate_pulse_metrics(df)

    min_fmin_hz, max_fmin_hz = fmin_ranges
    min_fmax_hz, max_fmax_hz = fmax_ranges

    if pulse_metrics.empty:
        return df

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

    rejected_ids = set(
        pulse_metrics.loc[rejected, "pulse_id"]
    )

    if rejected_ids:
        rejected_mask = df["pulse_id"].isin(rejected_ids)

        df.loc[rejected_mask, "status"] = 1
        df.loc[rejected_mask, "pulse_id"] = pd.NA

    return df
