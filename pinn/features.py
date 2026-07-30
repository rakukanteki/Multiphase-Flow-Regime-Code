"""
Feature extraction from raw pressure time-series.

Pure array-in / array-out functions -- no file I/O.
"""

import numpy as np
from scipy.fft import fft, fftfreq

from .config import P_ATM_BAR, PIPE_LENGTH


def extract_pressure_features(series: np.ndarray) -> np.ndarray:
    """
    Extract 8 physics-motivated statistical + spectral features from a
    pressure trace. Works on any length array -- intended to be called on
    a FULL file's raw pressure trace (before it gets resampled for the
    TCN branch).

    Features:
        [0] Mean pressure
        [1] Standard deviation
        [2] Peak-to-peak range
        [3] Mean of gradient  (average slope)
        [4] Std  of gradient  (variability of slope)
        [5] Max |gradient|    (sharpest transition)
        [6] Dominant FFT frequency
        [7] Dominant FFT magnitude
    """
    features = []

    # --- Time-domain statistics ---
    features.append(float(np.mean(series)))
    features.append(float(np.std(series)))
    features.append(float(np.max(series) - np.min(series)))

    grad = np.gradient(series)
    features.append(float(np.mean(grad)))
    features.append(float(np.std(grad)))
    features.append(float(np.max(np.abs(grad))))

    # --- Frequency-domain (FFT) ---
    if len(series) > 4:
        freqs = fftfreq(len(series), d=0.5)
        fft_vals = np.abs(fft(series))
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_fft = fft_vals[pos_mask]
        if len(pos_fft) > 0:
            peak_idx = int(np.argmax(pos_fft))
            features.append(float(pos_freqs[peak_idx]))
            features.append(float(pos_fft[peak_idx]))
        else:
            features.extend([0.0, 0.0])
    else:
        features.extend([0.0, 0.0])

    return np.array(features, dtype=np.float32)   # shape (8,)


def compute_measured_dp_dx(series: np.ndarray) -> float:
    """
    Real, measured physics anchor for the PINN residual.

    Estimates the frictional pressure gradient (bar/m) from the file's
    ACTUAL measured mean pressure (over the FULL run), assuming an
    atmospheric-outlet boundary condition:

        dP/dx_measured = (P_mean_measured - P_atm) / PIPE_LENGTH

    Deterministic transform of a raw sensor reading (no fitting, no
    validation/test statistics involved) -- carries zero data-leakage risk
    and is NOT derived from the model's own predictions.
    """
    p_mean_bar = float(np.mean(series))
    return (p_mean_bar - P_ATM_BAR) / PIPE_LENGTH


def resample_series(series: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample a 1-D pressure trace to a fixed length via linear
    interpolation over a normalized [0, 1] time axis, so the TCN branch
    always sees a consistent input shape regardless of how many samples
    the original trace logged.
    """
    series = np.asarray(series, dtype=np.float32)
    if len(series) == target_len:
        return series.copy()
    x_old = np.linspace(0.0, 1.0, num=len(series))
    x_new = np.linspace(0.0, 1.0, num=target_len)
    return np.interp(x_new, x_old, series).astype(np.float32)
