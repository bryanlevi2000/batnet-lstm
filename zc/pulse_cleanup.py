# pulse_cleanup.py
import numpy as np
import pandas as pd

def filter_pulse_tails(
    df,
    tail_time_fraction,
    tail_dec_freq_cutoff
):
    """Turn off pulse points following an excessive frequency drop
    within the final fraction of each pulse's duration.
    """

    df_out = df.copy()

    t = df_out["time_us"].to_numpy()
    f = df_out["freq_hz"].to_numpy()
    turn_off = np.zeros(len(df_out), dtype=bool)

    for pulse_id, idx in df_out.groupby("pulse_id", sort=False).indices.items():
        if pulse_id == 0 or len(idx) < 2:
            continue

        tail_start_time = (
            t[idx[-1]]
            - tail_time_fraction * (t[idx[-1]] - t[idx[0]])
        )
        tail_start = np.searchsorted(t[idx], tail_start_time)

        first_diff = max(0, tail_start - 1)
        bad = np.flatnonzero(
            np.diff(f[idx])[first_diff:] < -tail_dec_freq_cutoff
        )

        if bad.size:
            start = first_diff + bad[0] + 1
            turn_off[idx[start:]] = True

    df_out.loc[turn_off, "status"] = 1
    df_out.loc[turn_off, "pulse_id"] = 0

    return df_out


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


def clean_pulses(
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
    Rejected pulse points receive status 1 and have pulse_id = 0.
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

    pulse_id_series = result["pulse_id"]

    # Build masks from the original pulse_id column.
    is_pulse = (pulse_id_series.notna() & (pulse_id_series != 0)).to_numpy()
    is_accepted = pulse_id_series.isin(accepted_ids).to_numpy()
    is_rejected = is_pulse & ~is_accepted

    # Update statuses.
    status_arr = result["status"].to_numpy(copy=True)
    status_arr[is_pulse] = 1
    status_arr[is_accepted] = 2

    # Assign pulse_id = 0 to rejected pulse points.
    pulse_id_arr = pulse_id_series.to_numpy(dtype=object, copy=True)
    pulse_id_arr[is_rejected] = 0

    result["status"] = status_arr
    result["pulse_id"] = pulse_id_arr

    return result
