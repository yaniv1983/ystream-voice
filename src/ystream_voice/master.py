"""Broadcast mastering: rumble filter, loudness target, peak ceiling.

Streaming platforms and most online radio aim at about -16 LUFS with peaks
below -1 dBTP; EBU R128 broadcast uses -23 LUFS. Works at any sample rate.
"""
from __future__ import annotations

import numpy as np


def _highpass(wav: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    sos = butter(2, cutoff, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, wav).astype(np.float32)


def true_peak_db(wav: np.ndarray) -> float:
    """Peak level in dBFS measured on a 4x oversampled copy (inter-sample peaks)."""
    from scipy.signal import resample_poly
    if len(wav) == 0:
        return -np.inf
    peak = np.max(np.abs(resample_poly(wav.astype(np.float64), 4, 1)))
    return float(20 * np.log10(max(peak, 1e-12)))


def loudness_lufs(wav: np.ndarray, sr: int) -> float:
    import pyloudnorm
    return float(pyloudnorm.Meter(sr).integrated_loudness(wav.astype(np.float64)))


def _limit(wav: np.ndarray, sr: int, ceiling: float) -> np.ndarray:
    """Look-ahead peak limiter: gain dips smoothly around each peak, no clipping."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    needed = np.minimum(1.0, ceiling / np.maximum(np.abs(wav), 1e-9))
    window = max(3, int(sr * 0.005) | 1)
    # Hold the reduction across the window, then smooth it. The smoothed gain can
    # never exceed the needed gain at a peak, because the hold is as wide as the smoother.
    gain = uniform_filter1d(minimum_filter1d(needed, size=window), size=window)
    return (wav * gain).astype(np.float32)


def master(wav: np.ndarray, sr: int, target_lufs: float = -16.0, peak_dbtp: float = -1.0,
           highpass_hz: float | None = 70.0) -> np.ndarray:
    """Return a mastered copy of mono float audio.

    Clips shorter than 0.4 s cannot be measured for loudness and are only
    filtered and peak-limited.
    """
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if len(wav) == 0:
        return wav
    if highpass_hz:
        wav = _highpass(wav, sr, highpass_hz)
    ceiling = 10 ** (peak_dbtp / 20)
    measurable = len(wav) >= int(0.4 * sr)
    # Limiting lowers loudness a little, so correct and limit twice.
    for _ in range(2 if measurable else 1):
        if measurable:
            current = loudness_lufs(wav, sr)
            if np.isfinite(current):
                wav = wav * 10 ** ((target_lufs - current) / 20)
        # Limit slightly under the ceiling; inter-sample overshoot is trimmed below.
        wav = _limit(wav, sr, ceiling * 0.94)
    overshoot = true_peak_db(wav) - peak_dbtp
    if overshoot > 0:
        wav = wav * 10 ** (-overshoot / 20)
    return wav.astype(np.float32)
