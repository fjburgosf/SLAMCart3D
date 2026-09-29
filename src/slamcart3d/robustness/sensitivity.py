"""Exploratory rank-based sensitivity of sampled robustness conditions."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Assign the mean rank to ties, using stable sorting."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float | None:
    """Calculate Spearman's rho on finite pairs; undefined constants return None."""

    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("rank correlation inputs must have equal length")
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 3:
        return None
    x_rank = _average_ranks(x[finite])
    y_rank = _average_ranks(y[finite])
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = math.sqrt(float(x_centered @ x_centered) * float(y_centered @ y_centered))
    if denominator == 0.0:
        return None
    return float((x_centered @ y_centered) / denominator)


def sampled_sensitivity(
    records: tuple[dict[str, Any], ...], parameter_names: tuple[str, str]
) -> list[dict[str, str | float | int | None]]:
    """Relate sampled parameter values to median metric; no causal claim."""

    rows: list[dict[str, str | float | int | None]] = []
    for name in parameter_names:
        x = np.asarray([record[name] for record in records], dtype=float)
        y = np.asarray([record["median"] for record in records], dtype=float)
        rows.append(
            {
                "parameter": name,
                "metric": str(records[0]["metric"]) if records else "",
                "finite_pair_count": int(np.count_nonzero(np.isfinite(x) & np.isfinite(y))),
                "spearman_rho": spearman_rho(x, y),
            }
        )
    return rows
