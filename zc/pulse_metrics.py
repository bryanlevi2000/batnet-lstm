# pulse_metrics.py
import numpy as np
import pandas as pd


def filter_valid_status(df: pd.DataFrame) -> pd.DataFrame:
    """Filter rows where status > 1."""
    return df.loc[df["status"] > 1].copy()


def calculate_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate point-to-point frequency slopes within each pulse.
    Backfill the first point's slope.
    """
    df = df.copy()

    groups = df.groupby("pulse_id", sort=False)
    dt = groups["time_us"].diff()
    dfreq = groups["freq_hz"].diff()

    df["slope"] = (dfreq / dt).groupby(df["pulse_id"], sort=False).bfill()

    return df


def calculate_fc(
    df: pd.DataFrame,
    metric_fc_proportion: float,
) -> pd.DataFrame:
    """Return the selected row index for each pulse."""
    results = []

    for pulse_id, pulse in df.groupby("pulse_id", sort=False):
        if len(pulse) < 2:
            continue

        # Exclude the final point in the pulse.
        candidates = pulse.iloc[:-1]

        times = candidates["time_us"].to_numpy()
        cutoff = times[0] + (
            times[-1] - times[0]
        ) * (1 - metric_fc_proportion)

        candidates = candidates.loc[candidates["time_us"] >= cutoff]

        if candidates.empty:
            continue

        best_idx = candidates["slope"].abs().idxmin()
        results.append((pulse_id, best_idx))

    return pd.DataFrame(results, columns=["pulse_id", "fc_idx"])


def calculate_knee(
    df: pd.DataFrame,
    fc_indices: pd.DataFrame,
    metric_knee_proportion: float,
) -> pd.DataFrame:
    """Return the original DataFrame index of the knee point for each pulse."""
    results = []

    fc_lookup = fc_indices.set_index("pulse_id")["fc_idx"]

    for pulse_id, pulse in df.groupby("pulse_id", sort=False):
        if pulse_id not in fc_lookup.index:
            continue

        fc_idx = fc_lookup[pulse_id]
        fc_position = pulse.index.get_loc(fc_idx)

        # Only consider points before FC.
        candidates = pulse.iloc[:fc_position]

        if len(candidates) < 2:
            continue

        # Exclude the initial proportion of the pulse by duration.
        times = pulse["time_us"].to_numpy()
        cutoff = times[0] + (
            times[-1] - times[0]
        ) * metric_knee_proportion

        candidates = candidates.loc[candidates["time_us"] >= cutoff]

        if len(candidates) < 2:
            continue

        slopes = candidates["slope"].to_numpy()

        # Find the largest absolute change between consecutive slopes.
        slope_changes = np.abs(np.diff(slopes))
        knee_position = np.argmax(slope_changes)

        # Select the first point of the pair with the greatest change.
        knee_idx = candidates.index[knee_position]

        results.append((pulse_id, knee_idx))

    return pd.DataFrame(results, columns=["pulse_id", "knee_idx"])


def calculate_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Return the duration in microseconds for each pulse."""

    durations = df.groupby("pulse_id", sort=False)["time_us"].agg(
        lambda x: x.iloc[-1] - x.iloc[0]
    )

    return durations.reset_index(name="duration_us")


def calculate_fmax(df: pd.DataFrame) -> pd.DataFrame:
    """Return the maximum frequency in Hz for each pulse."""
    fmax = df.groupby("pulse_id", sort=False)["freq_hz"].max()
    return fmax.reset_index(name="fmax_hz")


def calculate_fmin(df: pd.DataFrame) -> pd.DataFrame:
    """Return the minimum frequency in Hz for each pulse."""
    fmin = df.groupby("pulse_id", sort=False)["freq_hz"].min()
    return fmin.reset_index(name="fmin_hz")


def calculate_tstart(df: pd.DataFrame) -> pd.DataFrame:
    """Return the start time in microseconds for each pulse."""
    tstart = df.groupby("pulse_id", sort=False)["time_us"].first()
    return tstart.reset_index(name="tstart_us")


def calculate_tend(df: pd.DataFrame) -> pd.DataFrame:
    """Return the end time in microseconds for each pulse."""
    tend = df.groupby("pulse_id", sort=False)["time_us"].last()
    return tend.reset_index(name="tend_us")
