"""Known-boundary synthetic benchmark for validating ASRF numerically."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from slamcart3d.robustness.asrf import ASRF, ParameterSpace, classification_metrics, latin_hypercube


def synthetic_margin(points: np.ndarray) -> np.ndarray:
    """Curved known frontier: positive values are failures."""

    x = points[:, 0]
    y = points[:, 1]
    return x + 0.35 * np.sin(np.pi * y) + 0.30 * y - 0.78


def validate_asrf(budget: int = 50, seed: int = 42) -> dict[str, float | int]:
    """Run ASRF on a dense-grid ground truth and report objective errors."""

    space = ParameterSpace(("noise", "dropout"), np.zeros(2), np.ones(2))
    model = ASRF(space, initial_samples=8, seed=seed)
    result = model.run(synthetic_margin, budget=budget)
    axis = np.linspace(0.0, 1.0, 101)
    xx, yy = np.meshgrid(axis, axis)
    validation = np.column_stack((xx.ravel(), yy.ravel()))
    truth = synthetic_margin(validation) >= 0.0
    prediction = model.predict_failure_probability(validation) >= 0.5
    metrics = classification_metrics(truth, prediction)
    metrics.update({"simulations": len(result.samples), "budget": budget})
    return metrics


def grid_samples(budget: int) -> np.ndarray:
    """Return exactly *budget* near-rectangular grid points in 2D."""

    if budget < 4:
        raise ValueError("grid budget must be at least 4")
    rows = max(2, int(np.floor(np.sqrt(budget))))
    columns, extra = divmod(budget, rows)
    points = [
        (x, y)
        for row, y in enumerate(np.linspace(0.0, 1.0, rows))
        for x in np.linspace(0.0, 1.0, columns + (row < extra))
    ]
    return np.asarray(points, dtype=float)


def _evaluate_fixed_samples(
    name: str, samples: np.ndarray, validation: np.ndarray, truth: np.ndarray, space: ParameterSpace
) -> dict[str, float | int | str]:
    started = perf_counter()
    physical = space.denormalize(samples)
    failed = synthetic_margin(physical) >= 0.0
    surrogate = ASRF(space, initial_samples=max(2, len(samples)), seed=0)
    surrogate.samples = physical
    surrogate.failed = failed
    prediction = surrogate.predict_failure_probability(validation) >= 0.5
    metrics: dict[str, float | int | str] = classification_metrics(truth, prediction)
    metrics.update({"method": name, "simulations": len(samples), "runtime_s": perf_counter() - started})
    return metrics


def compare_sampling_methods(budget: int = 50, seed: int = 42) -> list[dict[str, float | int | str]]:
    """Compare Grid, Random, LHS, and ASRF on the same known frontier."""

    if budget < 8:
        raise ValueError("budget must be at least 8")
    space = ParameterSpace(("noise", "dropout"), np.zeros(2), np.ones(2))
    axis = np.linspace(0.0, 1.0, 151)
    xx, yy = np.meshgrid(axis, axis)
    validation = np.column_stack((xx.ravel(), yy.ravel()))
    truth = synthetic_margin(validation) >= 0.0
    rng = np.random.default_rng(seed)
    sample_sets = {
        "grid": grid_samples(budget),
        "random": rng.random((budget, 2)),
        "lhs": latin_hypercube(budget, 2, np.random.default_rng(seed)),
    }
    rows = [
        _evaluate_fixed_samples(name, samples, validation, truth, space)
        for name, samples in sample_sets.items()
    ]
    started = perf_counter()
    asrf = ASRF(space, initial_samples=8, seed=seed)
    result = asrf.run(synthetic_margin, budget=budget)
    prediction = asrf.predict_failure_probability(validation) >= 0.5
    asrf_metrics: dict[str, float | int | str] = classification_metrics(truth, prediction)
    asrf_metrics.update(
        {"method": "asrf", "simulations": len(result.samples), "runtime_s": perf_counter() - started}
    )
    rows.append(asrf_metrics)
    return rows


def run_asrf_ablation(budget: int = 50, seed: int = 42) -> list[dict[str, float | int | str]]:
    """Measure the contribution of boundary uncertainty and coverage terms."""

    space = ParameterSpace(("noise", "dropout"), np.zeros(2), np.ones(2))
    axis = np.linspace(0.0, 1.0, 151)
    xx, yy = np.meshgrid(axis, axis)
    validation = np.column_stack((xx.ravel(), yy.ravel()))
    truth = synthetic_margin(validation) >= 0.0
    variants = {
        "asrf_full": (1.0, 0.35),
        "asrf_without_boundary_uncertainty": (0.0, 1.0),
        "asrf_without_coverage": (1.0, 0.0),
    }
    rows: list[dict[str, float | int | str]] = []
    for name, (boundary_weight, coverage_weight) in variants.items():
        model = ASRF(
            space,
            initial_samples=8,
            boundary_weight=boundary_weight,
            coverage_weight=coverage_weight,
            seed=seed,
        )
        result = model.run(synthetic_margin, budget=budget)
        metrics: dict[str, float | int | str] = classification_metrics(
            truth, model.predict_failure_probability(validation) >= 0.5
        )
        metrics.update({"method": name, "simulations": len(result.samples)})
        rows.append(metrics)
    return rows
