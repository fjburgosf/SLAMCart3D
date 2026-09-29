import numpy as np
import pytest

from slamcart3d.robustness.asrf import ASRF, ParameterSpace
from slamcart3d.robustness.benchmark import (
    compare_sampling_methods,
    grid_samples,
    run_asrf_ablation,
    synthetic_margin,
    validate_asrf,
)


def test_asrf_is_reproducible_and_samples_both_classes() -> None:
    space = ParameterSpace(("a", "b"), np.zeros(2), np.ones(2))
    first = ASRF(space, initial_samples=8, candidate_count=256, seed=4).run(synthetic_margin, budget=20)
    second = ASRF(space, initial_samples=8, candidate_count=256, seed=4).run(synthetic_margin, budget=20)
    np.testing.assert_allclose(first.samples, second.samples)
    assert set(first.failed) == {False, True}


def test_boundary_uncertainty_is_higher_away_from_observations() -> None:
    space = ParameterSpace(("a", "b"), np.zeros(2), np.ones(2))
    model = ASRF(space)
    model.samples = np.array([[0.1, 0.1], [0.1, 0.2], [0.2, 0.1], [0.2, 0.2]])
    model.failed = np.array([False, False, False, False])
    near, far = model.predict_failure_uncertainty(np.array([[0.15, 0.15], [0.9, 0.9]]))
    assert near < far


def test_asrf_known_boundary_error_is_bounded() -> None:
    metrics = validate_asrf(budget=40, seed=3)
    assert metrics["boundary_classification_error"] < 0.18
    assert metrics["fnr"] < 0.25


def test_baseline_comparison_and_ablation_report_all_methods() -> None:
    comparison = compare_sampling_methods(budget=16, seed=2)
    assert {row["method"] for row in comparison} == {"grid", "random", "lhs", "asrf"}
    ablation = run_asrf_ablation(budget=16, seed=2)
    assert len(ablation) == 3


def test_grid_uses_exact_nonsquare_budget() -> None:
    points = grid_samples(50)
    assert points.shape == (50, 2)
    assert np.all((0.0 <= points) & (points <= 1.0))
    assert len(np.unique(points, axis=0)) == 50


@pytest.mark.parametrize(
    ("names", "lower", "upper"),
    [
        ((), [], []),
        (("a", "a"), [0.0, 0.0], [1.0, 1.0]),
        (("a",), [0.0], [float("nan")]),
        (("a",), [float("inf")], [1.0]),
        (("a",), [1.0], [1.0]),
    ],
)
def test_parameter_space_rejects_invalid_domains(names, lower, upper) -> None:
    with pytest.raises(ValueError):
        ParameterSpace(names, np.asarray(lower), np.asarray(upper))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_samples": 0},
        {"candidate_count": 0},
        {"bandwidth": 0.0},
        {"bandwidth": float("nan")},
        {"boundary_weight": 0.0, "coverage_weight": 0.0},
        {"coverage_weight": -1.0},
    ],
)
def test_asrf_rejects_invalid_hyperparameters(kwargs) -> None:
    space = ParameterSpace(("a",), np.zeros(1), np.ones(1))
    with pytest.raises(ValueError):
        ASRF(space, **kwargs)


def test_asrf_rejects_nonfinite_and_wrong_sized_objective_values() -> None:
    space = ParameterSpace(("a",), np.zeros(1), np.ones(1))
    model = ASRF(space, initial_samples=2, candidate_count=8)
    with pytest.raises(ValueError, match="initial sample"):
        model.run(lambda points: np.full(len(points), np.nan), budget=3)
    with pytest.raises(ValueError, match="initial sample"):
        model.run(lambda points: np.zeros(1), budget=3)

    def invalid_later(points: np.ndarray) -> np.ndarray:
        return np.zeros(len(points)) if len(points) > 1 else np.array([np.inf])

    with pytest.raises(ValueError, match="selected sample"):
        model.run(invalid_later, budget=3)
