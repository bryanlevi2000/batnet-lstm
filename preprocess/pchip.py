
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


def pchip_resample_pulses(df, pchip_time_interval):
    """
    PCHIP-interpolate each pulse onto a fixed time grid.

    Each pulse is interpolated independently, so interpolation never
    occurs across gaps between pulses.

    The grid starts at the first time point of each pulse and advances
    by pchip_time_interval microseconds. The final time point is included
    only if it falls exactly on the grid.

    Pulses with fewer than two valid time/frequency observations are skipped.
    """

    required = {"time_us", "freq_hz", "pulse_id"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {sorted(missing)}"
        )

    if pchip_time_interval <= 0:
        raise ValueError("pchip_time_interval must be greater than 0.")

    if df.empty:
        return df.copy()

    has_amplitude = "amplitude" in df.columns
    has_status = "status" in df.columns

    resampled_pulses = []

    # Process each pulse independently.
    for pulse_id, pulse_df in df.groupby("pulse_id", sort=False):

        pulse_df = pulse_df.sort_values("time_us").copy()

        # Convert interpolation inputs to numeric arrays.
        t = pulse_df["time_us"].to_numpy(dtype=float)
        f = pulse_df["freq_hz"].to_numpy(dtype=float)

        # Remove invalid time/frequency observations.
        valid = np.isfinite(t) & np.isfinite(f)
        pulse_df = pulse_df.iloc[np.flatnonzero(valid)]

        t = t[valid]
        f = f[valid]

        # Remove duplicate time values.
        unique_times, unique_indices = np.unique(t, return_index=True)

        t = unique_times
        f = f[unique_indices]

        if has_amplitude:
            amp = pulse_df["amplitude"].to_numpy(dtype=float)[unique_indices]

        if has_status:
            status = pulse_df["status"].to_numpy()[unique_indices]

        # PCHIP requires at least two unique time points.
        if len(t) < 2:
            continue

        # Build the fixed time grid.
        n_points = (
            int(np.floor((t[-1] - t[0]) / pchip_time_interval)) + 1
        )

        new_times = (
            t[0] + np.arange(n_points) * pchip_time_interval
        )

        # Interpolate frequency.
        new_freq = PchipInterpolator(t, f)(new_times)

        # Interpolate amplitude if available.
        if has_amplitude:
            valid_amp = np.isfinite(amp)

            if valid_amp.sum() >= 2:
                new_amp = PchipInterpolator(
                    t[valid_amp], amp[valid_amp]
                )(new_times)
            elif valid_amp.sum() == 1:
                new_amp = np.full(n_points, amp[valid_amp][0])
            else:
                new_amp = np.full(n_points, np.nan)

        # Preserve the original status associated with the pulse.
        if has_status:
            pulse_status = status[0]

        # Assemble this pulse's resampled DataFrame.
        pulse_data = {
            "time_us": new_times,
            "freq_hz": new_freq,
            "pulse_id": pulse_id,
        }

        if has_amplitude:
            pulse_data["amplitude"] = new_amp

        if has_status:
            pulse_data["status"] = pulse_status

        resampled_pulses.append(pd.DataFrame(pulse_data))

    if not resampled_pulses:
        return df.iloc[0:0].copy()

    result = pd.concat(resampled_pulses, ignore_index=True)

    return result.reindex(columns=df.columns)
