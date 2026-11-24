# zc_loader.py
"""Load Anabat ZC files (types 129 to 132) as ZC point data.

Follows the loader in the Anabat file format notes. The output has the same
columns as zc_extraction.extract_zc: time_us, freq_hz, amplitude.
Anabat files hold no amplitude, so it is NaN.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd


class ZCFileError(Exception):
    """The file is not a readable Anabat ZC file."""


INFO_POINTER = 0x11A

# File type -> offset of the first data byte.
DATA_START = {129: 0x120, 130: 0x120, 131: 0x120, 132: 0x150}

# Point status values in the file: 0 out of range, 1 switched off, 2 normal, 3 main dot.
KEPT_STATUSES = (2, 3)


def _u16(buf, offset):
    return buf[offset] + 256 * buf[offset + 1]


def _read_header(buf):
    """Return (file_type, data_start, res1, divrat). Raise ZCFileError if the header is bad."""

    if len(buf) < 0x120:
        raise ZCFileError(f"file too short ({len(buf)} bytes)")

    file_type = buf[3]

    if file_type not in DATA_START:
        raise ZCFileError(f"unknown file type {file_type}")

    if _u16(buf, 0) != INFO_POINTER:
        raise ZCFileError("bad data information table pointer")

    res1 = _u16(buf, INFO_POINTER + 2)

    if not 10000 <= res1 <= 60000:
        raise ZCFileError(f"bad time resolution RES1={res1}")

    divrat = buf[INFO_POINTER + 4]

    if not 1 <= divrat <= 64:
        raise ZCFileError(f"bad division ratio {divrat}")

    data_start = _u16(buf, INFO_POINTER)

    if data_start != DATA_START[file_type]:
        raise ZCFileError(f"bad data start 0x{data_start:x} for type {file_type}")

    if len(buf) < data_start:
        raise ZCFileError("header is truncated")

    return file_type, data_start, res1, divrat


def _decode_times(buf, p, file_type, res1):
    """Decode the data bytes into point times (us) and a status per point.

    Point 0 is the time origin. A file that ends mid-entry keeps the points read so far.
    """

    n = len(buf)
    timefactor = 25000.0 / res1
    times = [0]
    status = np.full(n + 2, 2, dtype=np.uint8)
    time = lastdif = 0

    while p < n:

        b = buf[p]

        if b < 0x80:
            # One byte: signed 7-bit change from the previous interval.
            lastdif += b - 128 if b > 63 else b
            p += 1

        elif file_type == 129 and b >= 0xF8:
            # The next (b - 0xF8) points are switched off.
            status[len(times):len(times) + b - 0xF8] = 1
            p += 1
            continue

        elif file_type == 129:
            if p + 2 > n:
                break

            lastdif = (((b & 7) << 8) | buf[p + 1]) << ((b >> 3) & 0x0F)
            p += 2

        elif b >= 0xE0:

            if file_type == 130:
                status[len(times):len(times) + b - 0xE0] = 1
                p += 1

            else:
                if p + 2 > n:
                    break

                status[len(times):len(times) + buf[p + 1]] = b & 3
                p += 2

            continue

        else:
            size = {0x00: 2, 0x20: 3, 0x40: 4}[b & 0x60]

            if p + size > n:
                break

            lastdif = b & 0x1F

            for k in range(1, size):
                lastdif = (lastdif << 8) | buf[p + k]

            p += size

        time += lastdif if res1 == 25000 else math.floor(timefactor * lastdif + 0.5)
        times.append(time)

    return np.asarray(times, dtype=np.int64), status[:len(times)]


def load_zc_points(path):
    """Read one ZC file. Returns a DataFrame with time_us, freq_hz and amplitude (NaN).

    Frequency at each point comes from the time between it and the point two before it,
    as in the WAV extraction. Out-of-range points, switched-off points and points with a
    frequency outside the detector range are dropped. Raises ZCFileError if the file
    header is unreadable.
    """

    buf = Path(path).read_bytes()

    file_type, data_start, res1, divrat = _read_header(buf)

    times, status = _decode_times(buf, data_start, file_type, res1)

    # Same limits as the Anabat loader: 4 kHz to 250 kHz, and at least 48 us.
    t_min = max(48, math.ceil(divrat * 4.0))
    t_max = min(12859, math.floor(divrat * 250.0))

    span = times[2:] - times[:-2]
    keep = (span >= t_min) & (span <= t_max) & np.isin(status[2:], KEPT_STATUSES)

    return pd.DataFrame({
        "time_us": times[2:][keep].astype(np.float64),
        "freq_hz": divrat * 1e6 / span[keep],
        "amplitude": np.full(int(keep.sum()), np.nan),
    })
