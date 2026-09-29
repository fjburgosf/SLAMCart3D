"""Rank sensitivity is deterministic, tie-aware, and explicitly limited."""

import numpy as np
import pytest

from slamcart3d.robustness.sensitivity import sampled_sensitivity, spearman_rho


def test_spearman_handles_ties_and_direction() -> None:
    x = np.array([1.0, 2.0, 2.0, 3.0])
    assert spearman_rho(x, np.array([3.0, 2.0, 2.0, 1.0])) == pytest.approx(-1.0)
    assert spearman_rho(x, x) == pytest.approx(1.0)


def test_spearman_excludes_missing_pairs_and_rejects_undefined() -> None:
    assert spearman_rho(np.array([1.0, 2.0, 3.0, np.nan]), np.array([3.0, 2.0, 1.0, 4.0])) == pytest.approx(
        -1.0
    )
    assert spearman_rho(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])) is None
    assert spearman_rho(np.array([1.0, 2.0]), np.array([1.0, 2.0])) is None
    with pytest.raises(ValueError, match="equal length"):
        spearman_rho(np.array([1.0]), np.array([1.0, 2.0]))


def test_sampled_sensitivity_is_one_row_per_axis() -> None:
    records = tuple(
        {"x": value, "y": 4.0 - value, "median": value, "metric": "ate_rmse_m"}
        for value in (1.0, 2.0, 3.0, 4.0)
    )
    rows = sampled_sensitivity(records, ("x", "y"))
    assert rows[0]["spearman_rho"] == pytest.approx(1.0)
    assert rows[1]["spearman_rho"] == pytest.approx(-1.0)
    assert all(row["finite_pair_count"] == 4 for row in rows)
