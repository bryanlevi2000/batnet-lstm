# pchip.py
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


def pchip_resample_pulses(df, pchip_time_interval):
    """
    PCHIP-interpolate pulse data onto a fixed time grid while
    preserving rows between pulses that have no pulse_id

    For each pulse, start_time <= time_us <= end_time is sampled every
    pchip_time_interval microseconds. The final pulse time is only included
    if it falls exactly on the grid. No interpolation is performed across
    gaps between pulses. Rows with missing pulse_id are preserved unchanged.
    """
    required = {"time_us", "freq_hz", "pulse_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {sorted(missing)}"
        )
    if pchip_time_interval <= 0:
        raise ValueError("pchip_time_interval must be greater than 0.")

    columns = df.columns
    pulse_mask = df["pulse_id"].notna().to_numpy()

    # Nothing to interpolate
    if not pulse_mask.any():
        return df.reset_index(drop=True)

    pulse_idx = np.flatnonzero(pulse_mask)
    has_amplitude = "amplitude" in columns
    has_status = "status" in columns

    # ---------------------------------------------------------
    # Pull pulse rows out as flat arrays (no groupby / per-pulse frames)
    # ---------------------------------------------------------
    t = df["time_us"].to_numpy(dtype=float)[pulse_idx]
    f = df["freq_hz"].to_numpy(dtype=float)[pulse_idx]
    codes, uniques = pd.factorize(df["pulse_id"].iloc[pulse_idx])
    amp = df["amplitude"].to_numpy(dtype=float)[pulse_idx] if has_amplitude else None
    status = df["status"].to_numpy()[pulse_idx] if has_status else None

    # Drop rows unusable for frequency interpolation
    valid = ~(np.isnan(t) | np.isnan(f))
    if not valid.all():
        t, f, codes = t[valid], f[valid], codes[valid]
        if has_amplitude:
            amp = amp[valid]
        if has_status:
            status = status[valid]

    # Sort by (pulse, time); lexsort is stable, so "keep first" on
    # duplicate times is deterministic.
    order = np.lexsort((t, codes))
    t, f, codes = t[order], f[order], codes[order]
    if has_amplitude:
        amp = amp[order]
    if has_status:
        status = status[order]

    # Drop duplicate time values within a pulse (keep first)
    if len(t) > 1:
        keep = np.ones(len(t), dtype=bool)
        keep[1:] = (codes[1:] != codes[:-1]) | (t[1:] != t[:-1])
        if not keep.all():
            t, f, codes = t[keep], f[keep], codes[keep]
            if has_amplitude:
                amp = amp[keep]
            if has_status:
                status = status[keep]

    # ---------------------------------------------------------
    # Pulse boundaries; need at least two points per pulse
    # ---------------------------------------------------------
    change = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(t)]))   # exclusive
    ok = (ends - starts) >= 2
    starts, ends = starts[ok], ends[ok]

    if len(starts) == 0:
        non_pulse = df.loc[~pulse_mask]
        return (
            non_pulse.iloc[np.argsort(non_pulse["time_us"].to_numpy(), kind="stable")]
            .reset_index(drop=True)
        )

    # ---------------------------------------------------------
    # Build every pulse's output grid in one vectorized pass
    #   new_times = start + arange(n_intervals + 1) * interval
    # ---------------------------------------------------------
    t_start = t[starts]
    t_end = t[ends - 1]
    n_pts = np.floor((t_end - t_start) / pchip_time_interval).astype(np.int64) + 1

    out_ends = np.cumsum(n_pts)
    out_starts = out_ends - n_pts
    total = int(out_ends[-1])

    grp = np.repeat(np.arange(len(starts)), n_pts)
    new_times = (
        t_start[grp]
        + (np.arange(total) - out_starts[grp]) * pchip_time_interval
    )

    new_freq = np.empty(total)
    new_amp = np.full(total, np.nan) if has_amplitude else None
    amp_has_nan = has_amplitude and np.isnan(amp).any()

    # ---------------------------------------------------------
    # Per-pulse PCHIP (the only Python-level loop)
    # ---------------------------------------------------------
    for s, e, o0, o1 in zip(
        starts.tolist(), ends.tolist(), out_starts.tolist(), out_ends.tolist()
    ):
        tt = t[s:e]
        nt = new_times[o0:o1]
        new_freq[o0:o1] = PchipInterpolator(tt, f[s:e])(nt)

        if has_amplitude:
            a = amp[s:e]
            if amp_has_nan:
                m = ~np.isnan(a)
                n_valid = int(m.sum())
                if n_valid >= 2:
                    new_amp[o0:o1] = PchipInterpolator(tt[m], a[m])(nt)
                elif n_valid == 1:
                    new_amp[o0:o1] = a[m][0]
                # else: stays NaN
            else:
                new_amp[o0:o1] = PchipInterpolator(tt, a)(nt)

    # ---------------------------------------------------------
    # Assemble output
    # ---------------------------------------------------------
    data = {
        "time_us": new_times,
        "freq_hz": new_freq,
        "pulse_id": uniques.take(codes[starts][grp]),
    }
    if has_amplitude:
        data["amplitude"] = new_amp
    if has_status:
        data["status"] = status[starts][grp]

    resampled = pd.DataFrame(data).reindex(columns=columns)
    non_pulse = df.loc[~pulse_mask]

    result = pd.concat([resampled, non_pulse], ignore_index=True, sort=False)

    # Stable time sort (pulse rows first on ties, NaN times last)
    result = result.iloc[np.argsort(result["time_us"].to_numpy(), kind="stable")]
    return result.reset_index(drop=True)
