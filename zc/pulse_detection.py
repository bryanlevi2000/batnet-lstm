# pulse_detection.py
import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def filter_zc_linearity(
    df,
    lookahead_n=10,
    linearity_cutoff=0.9
):
    """Calculate local point linearity every lookahead_n / 2 points,
    plus a final backward-looking window ending at the last point.
    """

    df_out = (
        df
        if df["time_us"].is_monotonic_increasing
        else df.sort_values("time_us")
    ).copy()

    t = df_out["time_us"].to_numpy(dtype=np.float64)
    f = df_out["freq_hz"].to_numpy(dtype=np.float64)

    n = len(t)
    step = max(1, lookahead_n // 2)
    window = lookahead_n + 1

    if n <= lookahead_n:
        df_out["linearity"] = np.nan
        df_out["status"] = 1
        return df_out

    # Starting points for the forward-looking windows.
    indices = np.arange(
        0,
        n - lookahead_n,
        step
    )

    # The final window ends at the last point.
    final_start = n - window

    # If the final window is not already included, add it.
    if indices[-1] != final_start:
        indices = np.append(indices, final_start)

    # Generate all possible windows as a view, then select
    # only the windows we actually need.
    t_win = sliding_window_view(t, window)[indices]
    f_win = sliding_window_view(f, window)[indices]

    # Center each window.
    t_mean = t_win.mean(axis=1, keepdims=True)
    f_mean = f_win.mean(axis=1, keepdims=True)

    dt = t_win - t_mean
    df_val = f_win - f_mean

    # Calculate R2.
    cov = np.sum(dt * df_val, axis=1)
    var_t = np.sum(dt * dt, axis=1)
    var_f = np.sum(df_val * df_val, axis=1)

    denom = var_t * var_f

    r2 = np.divide(
        cov * cov,
        denom,
        out=np.zeros_like(cov),
        where=denom > 0
    )

    # Initialize all points as having no linearity score.
    linearity = np.full(n, np.nan, dtype=np.float64)

    # All regular windows get their score at their starting point.
    regular_mask = indices != final_start
    linearity[indices[regular_mask]] = r2[regular_mask]

    # The final backward-looking window gets its score
    # assigned specifically to the final point.
    final_mask = indices == final_start
    linearity[-1] = r2[final_mask][0]

    df_out["linearity"] = linearity

    df_out["status"] = np.where(
        (linearity >= linearity_cutoff) & np.isfinite(linearity),
        2,
        1
    )

    return df_out.drop(columns=["linearity"])


def greedy_walk(
    df,
    time_cutoff,
    inc_freq_cutoff,
    dec_freq_cutoff,
    min_points,
    buffer_us=10000
):
    """Greedy walk algorithm to expand high-linearity points into candidate pulse segments.

    After each walk, the next walk cannot begin until after the specified
    time buffer following the end of the previous walk.

    Accepted walks are assigned sequential pulse_id values starting at 1.
    """

    df_out = (
        df
        if df["time_us"].is_monotonic_increasing
        else df.sort_values("time_us")
    ).copy()

    # Replace any existing pulse_id column.
    df_out["pulse_id"] = pd.NA

    t = df_out["time_us"].to_numpy()
    f = df_out["freq_hz"].to_numpy()
    s = df_out["status"].to_numpy()

    n = len(s)
    pulse_ids = np.full(n, pd.NA, dtype=object)
    i = 0
    pulse_id = 1

    while i < n:
        while i < n and s[i] != 2:
            i += 1

        if i == n:
            break

        base = i

        # Forward walk
        curr = base

        while curr < n - 1:

            dt = t[curr + 1] - t[curr]
            dfreq = f[curr + 1] - f[curr]

            if (
                dt > time_cutoff
                or (dfreq > 0 and dfreq > inc_freq_cutoff)
                or (dfreq <= 0 and -dfreq > dec_freq_cutoff)
            ):
                break

            s[curr + 1] = 2
            curr += 1

        fw_end = curr

        # Backward walk
        curr = base

        while curr > 0:

            dt = t[curr] - t[curr - 1]
            dfreq = f[curr] - f[curr - 1]

            if (
                dt > time_cutoff
                or (dfreq > 0 and dfreq > inc_freq_cutoff)
                or (dfreq <= 0 and -dfreq > dec_freq_cutoff)
            ):
                break

            s[curr - 1] = 2
            curr -= 1

        bw_end = curr

        # Reject segment if it is too short.
        if (fw_end - bw_end + 1) < min_points:
            s[bw_end:fw_end + 1] = 1

        else:
            pulse_ids[bw_end:fw_end + 1] = pulse_id
            pulse_id += 1

        # The next walk must start after the buffer following
        # the end of this walk.
        buffer_end_time = t[fw_end] + buffer_us

        i = np.searchsorted(
            t,
            buffer_end_time,
            side="right"
        )

    df_out["pulse_id"] = pulse_ids
    df_out["status"] = np.where(pd.isna(pulse_ids), 1, 2)

    return df_out


def detect_pulse_points(
    df,
    lookahead_n=10,
    linearity_cutoff=0.9,
    time_cutoff=500,
    inc_freq_cutoff=1000,
    dec_freq_cutoff=5000,
    min_points=10,
    buffer_us=10000
):
    """
    Identify potential bat pulse points from zero-crossing data.

    All input rows are retained. Accepted greedy-walk segments are
    marked with status = 2 and assigned a sequential pulse_id.
    Rows not belonging to an accepted segment retain their status
    and have a blank pulse_id.

    """

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning
        )

        df = filter_zc_linearity(
            df,
            lookahead_n=lookahead_n,
            linearity_cutoff=linearity_cutoff
        )

        df = greedy_walk(
            df,
            time_cutoff=time_cutoff,
            inc_freq_cutoff=inc_freq_cutoff,
            dec_freq_cutoff=dec_freq_cutoff,
            min_points=min_points,
            buffer_us=buffer_us
        )

    return df
