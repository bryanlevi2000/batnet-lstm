# zc_extraction.py
import numpy as np
from numba import njit


def compute_hysteresis(
    samples: np.ndarray,
    percentile: float = 85.0
) -> int:
    """Calculate the hysteresis threshold from the waveform."""

    if samples.size == 0:
        return 1

    h = int(np.percentile(np.abs(samples), percentile))

    return max(1, min(h, 32767))


@njit
def wav_to_zc_times(
    samples: np.ndarray,
    divrat: int,
    hysteresis: int,
    dt: float
) -> np.ndarray:
    """
    Extract zero-crossing times using a Numba-compiled loop.

    """

    if samples.size == 0:
        return np.empty(0, dtype=np.int64)

    # Maximum possible number of output crossings is approximately
    # half the number of input samples.
    outbuf = np.empty(
        samples.size // 2 + 1,
        dtype=np.int64
    )

    divcounter = 0
    T = 0.0
    n_out = 0

    # Initialize state based on first sample
    y1 = float(samples[0])

    if y1 <= 0:
        new_threshold = float(hysteresis)
        rising = True
    else:
        new_threshold = float(-hysteresis)
        rising = False

    thresh_val = float(hysteresis)

    for i in range(1, len(samples)):

        y2 = float(samples[i])
        T += dt

        if rising:

            if y2 > new_threshold:

                # Sub-sample interpolation
                tz = (
                    (T - dt)
                    + (new_threshold - y1) * dt / (y2 - y1)
                )

                divcounter += 1

                if divcounter >= divrat:

                    outbuf[n_out] = int(tz + 0.5)
                    n_out += 1

                    divcounter = 0

                new_threshold = -thresh_val
                rising = False

        else:

            if y2 < new_threshold:

                # Sub-sample interpolation
                tz = (
                    (T - dt)
                    + (new_threshold - y1) * dt / (y2 - y1)
                )

                divcounter += 1

                if divcounter >= divrat:

                    outbuf[n_out] = int(tz + 0.5)
                    n_out += 1

                    divcounter = 0

                new_threshold = thresh_val
                rising = True

        y1 = y2

    return outbuf[:n_out]


def compute_segment_integral(
    cumulative_abs: np.ndarray,
    start_time_us: float,
    end_time_us: float,
    sample_period_us: float
) -> float:
    """
    Calculate the integral of |samples| between two arbitrary
    times using a cumulative sum.

    Linear interpolation is used at the interval boundaries.
    """

    if not np.isfinite(start_time_us) or not np.isfinite(end_time_us):
        return np.nan

    start = min(start_time_us, end_time_us)
    end = max(start_time_us, end_time_us)

    if end <= start:
        return np.nan

    n_samples = len(cumulative_abs) - 1

    max_time = n_samples * sample_period_us

    start = max(0.0, start)
    end = min(max_time, end)

    if end <= start:
        return np.nan

    start_pos = start / sample_period_us
    end_pos = end / sample_period_us

    # Integral at an arbitrary position.
    #
    # cumulative_abs[k] represents the integral through sample k.
    start_index = int(np.floor(start_pos))
    end_index = int(np.floor(end_pos))

    start_index = max(0, min(start_index, n_samples))
    end_index = max(0, min(end_index, n_samples))

    # Interpolate cumulative integral at the exact boundaries.
    start_fraction = start_pos - start_index
    end_fraction = end_pos - end_index

    if start_index < n_samples:
        start_integral = (
            cumulative_abs[start_index]
            + start_fraction
            * (
                cumulative_abs[start_index + 1]
                - cumulative_abs[start_index]
            )
        )
    else:
        start_integral = cumulative_abs[start_index]

    if end_index < n_samples:
        end_integral = (
            cumulative_abs[end_index]
            + end_fraction
            * (
                cumulative_abs[end_index + 1]
                - cumulative_abs[end_index]
            )
        )
    else:
        end_integral = cumulative_abs[end_index]

    return end_integral - start_integral


def compute_time_frequency_and_amplitude(
    samples: np.ndarray,
    times_us: np.ndarray,
    divrat: int,
    sample_period_us: float
) -> np.ndarray:
    """
    Return columns:

        time_us
        freq_hz
        amplitude

    Frequency is calculated from three consecutive zero crossings.

    Amplitude is the mean absolute waveform amplitude between the
    corresponding first and third zero crossings.
    """

    if len(times_us) < 3:
        return np.empty((0, 3))

    # ------------------------------------------------------------
    # Frequency
    # ------------------------------------------------------------

    dt_us = times_us[2:] - times_us[:-2]

    valid = dt_us > 0

    freqs = np.zeros_like(
        dt_us,
        dtype=float
    )

    freqs[valid] = (
        divrat * 1e6
    ) / dt_us[valid]

    # ------------------------------------------------------------
    # Amplitude
    # ------------------------------------------------------------

    # Compute |samples| once.
    abs_samples = np.abs(
        samples.astype(np.float64)
    )

    # Cumulative integral of |samples|.
    #
    # Each sample represents one sample_period_us interval.
    cumulative_abs = np.empty(
        len(abs_samples) + 1,
        dtype=np.float64
    )

    cumulative_abs[0] = 0.0

    cumulative_abs[1:] = np.cumsum(
        abs_samples
    ) * sample_period_us

    amplitudes = np.full(
        len(dt_us),
        np.nan,
        dtype=float
    )

    if np.any(valid):

        valid_indices = np.flatnonzero(valid)

        start_times = times_us[:-2][valid]
        end_times = times_us[2:][valid]

        for idx, start, end in zip(
            valid_indices,
            start_times,
            end_times
        ):

            duration = end - start

            if duration <= 0:
                continue

            integral = compute_segment_integral(
                cumulative_abs,
                float(start),
                float(end),
                sample_period_us
            )

            if np.isfinite(integral):

                amplitudes[idx] = (
                    integral / duration
                )

    return np.column_stack(
        (
            times_us[2:],
            freqs,
            amplitudes
        )
    )


def extract_zc(
    samples: np.ndarray,
    sample_rate: int,
    divrat: int = 8,
    hysteresis_percentile: float = 85.0
) -> np.ndarray:
    """
    Extract zero-crossing time, frequency, and amplitude data
    from an audio signal.

    """

    if samples.size == 0:
        return np.empty((0, 3))

    # ------------------------------------------------------------
    # Hysteresis
    # ------------------------------------------------------------

    hysteresis = compute_hysteresis(
        samples,
        hysteresis_percentile
    )

    sample_period_us = (
        1e6 / sample_rate
    )

    # ------------------------------------------------------------
    # Zero-crossing times
    # ------------------------------------------------------------

    times_us = wav_to_zc_times(
        samples,
        divrat,
        hysteresis,
        sample_period_us
    )

    # ------------------------------------------------------------
    # Frequency + amplitude
    # ------------------------------------------------------------

    tf_data = compute_time_frequency_and_amplitude(
        samples,
        times_us,
        divrat,
        sample_period_us
    )

    return tf_data
