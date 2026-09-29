"""Held-out probabilistic diagnostics against repeated binary outcomes."""

from __future__ import annotations

import numpy as np


def probability_diagnostics(
    predicted_failure: np.ndarray, observed_failure_fraction: np.ndarray
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    """Return trial-level Brier score and equal-width-bin calibration error.

    Each condition contributes the same number of independent validation
    seeds. A fraction ``f`` therefore contributes ``p² - 2pf + f`` to the
    Bernoulli Brier score; this retains irreducible trial variability.
    """

    probability = np.asarray(predicted_failure, dtype=float).reshape(-1)
    fraction = np.asarray(observed_failure_fraction, dtype=float).reshape(-1)
    if len(probability) == 0 or probability.shape != fraction.shape:
        raise ValueError("probability and failure fractions must have matching non-empty shapes")
    bin_count = 10
    if (
        not np.isfinite(probability).all()
        or not np.isfinite(fraction).all()
        or np.any((probability < 0.0) | (probability > 1.0))
        or np.any((fraction < 0.0) | (fraction > 1.0))
    ):
        raise ValueError("probabilities and failure fractions must be finite in [0, 1]")
    brier = float(np.mean(probability**2 - 2.0 * probability * fraction + fraction))
    bins = np.minimum((probability * bin_count).astype(int), bin_count - 1)
    rows: list[dict[str, float | int]] = []
    ece = 0.0
    for index in range(bin_count):
        selected = bins == index
        count = int(np.count_nonzero(selected))
        if count == 0:
            continue
        mean_probability = float(np.mean(probability[selected]))
        observed_rate = float(np.mean(fraction[selected]))
        ece += count / len(probability) * abs(mean_probability - observed_rate)
        rows.append(
            {
                "bin_index": index,
                "conditions": count,
                "mean_predicted_failure_probability": mean_probability,
                "observed_failure_fraction": observed_rate,
            }
        )
    return {"brier_score": brier, "ece10": ece}, rows
