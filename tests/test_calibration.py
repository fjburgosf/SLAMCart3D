import numpy as np
import pytest

from slamcart3d.robustness.calibration import probability_diagnostics


def test_perfect_deterministic_probabilities_have_zero_brier_and_ece() -> None:
    scores, bins = probability_diagnostics(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert scores == {"brier_score": 0.0, "ece10": 0.0}
    assert sum(int(row["conditions"]) for row in bins) == 2


def test_stochastic_fraction_retains_trial_level_brier_variance() -> None:
    scores, bins = probability_diagnostics(np.array([0.5]), np.array([0.5]))
    assert scores["brier_score"] == pytest.approx(0.25)
    assert scores["ece10"] == pytest.approx(0.0)
    assert bins[0]["observed_failure_fraction"] == pytest.approx(0.5)


def test_overconfident_probability_has_nonzero_calibration_error() -> None:
    scores, _ = probability_diagnostics(np.array([0.8]), np.array([0.2]))
    assert scores["brier_score"] == pytest.approx(0.52)
    assert scores["ece10"] == pytest.approx(0.6)


def test_invalid_probability_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="matching"):
        probability_diagnostics(np.array([0.5]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="finite"):
        probability_diagnostics(np.array([np.nan]), np.array([0.0]))
