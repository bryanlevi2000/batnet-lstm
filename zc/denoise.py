# denoise.py
import numpy as np
import scipy.fft
from scipy.signal.windows import hann


def _stft_manual(y, n_fft, hop, win, fft_workers):
    pad = n_fft // 2
    y_padded = np.pad(y, (pad, pad), mode="constant")

    # Calculate number of frames using integer arithmetic.
    remaining = len(y_padded) - n_fft

    if remaining <= 0:
        n_frames = 1
    else:
        n_frames = 1 + (remaining + hop - 1) // hop

    # Total number of samples needed to contain all frames.
    needed = (n_frames - 1) * hop + n_fft

    extra = int(needed - len(y_padded))

    if extra > 0:
        y_padded = np.pad(y_padded, (0, extra), mode="constant")

    strides = (
        y_padded.strides[0] * hop,
        y_padded.strides[0],
    )

    frames = np.lib.stride_tricks.as_strided(
        y_padded,
        shape=(n_frames, n_fft),
        strides=strides,
        writeable=False,
    )

    frames = frames * win

    S = scipy.fft.rfft(
        frames,
        axis=1,
        workers=fft_workers,
    )

    return S.T, pad


def _istft_manual(S, n_fft, hop, win, pad, out_len, fft_workers):
    frames = scipy.fft.irfft(S.T, n=n_fft, axis=1, workers=fft_workers)  # (n_frames, n_fft)
    frames = frames * win
    n_frames = frames.shape[0]

    win_sq = (win * win).astype(np.float32)

    blocks = np.zeros((n_frames + 1, hop), dtype=np.float32)
    blocks[:n_frames] += frames[:, :hop]
    blocks[1:] += frames[:, hop:]
    y_out = blocks.reshape(-1)

    win_blocks = np.zeros((n_frames + 1, hop), dtype=np.float32)
    win_blocks[:n_frames] += win_sq[:hop]
    win_blocks[1:] += win_sq[hop:]
    win_sum = win_blocks.reshape(-1)
    win_sum[win_sum < 1e-8] = 1e-8

    y_out = y_out / win_sum
    return y_out[pad:pad + out_len]


def denoise_audio(y, noise_factor=2.0, n_fft=512, noise_stride=4, fft_workers=1):
    """
    Denoise audio using spectral gating.

    fft_workers: threads used PER FFT CALL. Default 1 because this is meant to run
    inside the existing file-level parallelism -- multithreading here too
    would oversubscribe cores. Only raise this to run it standalone,
    single-file-at-a-time.
    """
    y = np.asarray(y, dtype=np.float32)
    n = len(y)
    hop = n_fft // 2
    win = hann(n_fft, sym=False).astype(np.float32)

    S, pad = _stft_manual(y, n_fft, hop, win, fft_workers)
    mag = np.abs(S)

    noise_slice = np.ascontiguousarray(mag[:, ::noise_stride])
    noise = np.median(noise_slice, axis=1).astype(np.float32) * np.float32(noise_factor)
    noise_2d = noise[:, None]

    eps = np.float32(1e-12)
    gain = 1.0 - noise_2d / np.maximum(mag, eps)
    gain = np.clip(gain, 0.0, None).astype(np.float32)
    S = S * gain

    y_denoised = _istft_manual(S, n_fft, hop, win, pad, n, fft_workers)
    return y_denoised
