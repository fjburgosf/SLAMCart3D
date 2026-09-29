"""Descriptive statistics for repeated-seed experiments."""

from __future__ import annotations

import numpy as np


def summarize(values: list[float] | np.ndarray) -> dict[str, float | int]:
    """Return mean, median, sample SD, IQR, and normal-approximation 95% CI."""

    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or len(data) == 0 or not np.all(np.isfinite(data)):
        raise ValueError("values must be a non-empty finite one-dimensional sequence")
    mean = float(np.mean(data))
    standard_deviation = float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
    half_width = 1.96 * standard_deviation / np.sqrt(len(data))
    q1, q3 = np.percentile(data, [25.0, 75.0])
    return {
        "count": len(data),
        "mean": mean,
        "median": float(np.median(data)),
        "standard_deviation": standard_deviation,
        "iqr": float(q3 - q1),
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }
