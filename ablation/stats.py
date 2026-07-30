"""
Confidence-interval helpers for summarizing cross-validation fold results.
"""

import numpy as np
from scipy import stats as sp_stats


def confidence_interval(values, confidence: float = 0.95):
    """Mean + half-width of a (small-sample) t-distribution CI across CV folds."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values)) if n else 0.0
    if n <= 1:
        return mean, 0.0
    std = float(np.std(values, ddof=1))
    sem = std / np.sqrt(n)
    t_crit = float(sp_stats.t.ppf((1 + confidence) / 2.0, n - 1))
    return mean, t_crit * sem


def clipped_asymmetric_error(mean: float, half_width: float, lo: float = 0.0, hi: float = 1.0):
    """
    Asymmetric (lower_err, upper_err) lengths so mean-lower_err and
    mean+upper_err never leave [lo, hi] -- e.g. an accuracy/F1 CI can
    never imply less than 0% or more than 100%.
    """
    lower_err = mean - max(lo, mean - half_width)
    upper_err = min(hi, mean + half_width) - mean
    return max(0.0, lower_err), max(0.0, upper_err)
