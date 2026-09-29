"""Adaptive SLAM Robustness Frontier (ASRF) reference implementation.

The current MVP uses a normalized radial-basis probabilistic surrogate. Its
acquisition function balances boundary ambiguity and distance-based coverage.
This formulation is dependency-light, deterministic, and directly ablatable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

ArrayFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ParameterSpace:
    """Named bounded experimental domain."""

    names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        if lower.ndim != 1 or upper.ndim != 1 or not self.names:
            raise ValueError("parameter space needs one-dimensional, nonempty bounds")
        if len(self.names) != len(lower) or lower.shape != upper.shape:
            raise ValueError("parameter names and bounds must have equal dimensions")
        if any(not isinstance(name, str) or not name.strip() for name in self.names):
            raise ValueError("parameter names must be nonempty strings")
        if len(set(self.names)) != len(self.names):
            raise ValueError("parameter names must be unique")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("parameter bounds must be finite")
        if np.any(upper <= lower):
            raise ValueError("each upper bound must exceed its lower bound")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def dimensions(self) -> int:
        return len(self.names)

    def normalize(self, points: np.ndarray) -> np.ndarray:
        return (points - self.lower) / (self.upper - self.lower)

    def denormalize(self, points: np.ndarray) -> np.ndarray:
        return self.lower + points * (self.upper - self.lower)


@dataclass(frozen=True)
class ASRFResult:
    """Observations and convergence trace produced by ASRF."""

    samples: np.ndarray
    performance_margin: np.ndarray
    failed: np.ndarray
    convergence: np.ndarray


def latin_hypercube(count: int, dimensions: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a randomized Latin hypercube in the unit domain."""

    points = np.empty((count, dimensions), dtype=float)
    for dimension in range(dimensions):
        points[:, dimension] = (rng.permutation(count) + rng.random(count)) / count
    return points


class ASRF:
    """Sequentially estimate the transition ``g(theta) = 0``.

    The surrogate probability is a kernel-weighted estimate of failure. The
    acquisition is ``4 p (1-p) + lambda d_nearest``: the first term targets
    the estimated boundary, while the second prevents unobserved regions from
    being ignored. Both terms are dimensionless in normalized coordinates.
    """

    def __init__(
        self,
        space: ParameterSpace,
        initial_samples: int | None = None,
        bandwidth: float = 0.18,
        boundary_weight: float = 1.0,
        coverage_weight: float = 0.35,
        candidate_count: int = 2048,
        seed: int = 42,
    ) -> None:
        self.space = space
        self.initial_samples = max(8, 3 * space.dimensions) if initial_samples is None else initial_samples
        if self.initial_samples < 1 or candidate_count < 1:
            raise ValueError("initial_samples and candidate_count must be positive")
        if not np.isfinite(bandwidth) or bandwidth <= 0.0:
            raise ValueError("bandwidth must be finite and positive")
        if (
            not np.isfinite(boundary_weight)
            or not np.isfinite(coverage_weight)
            or boundary_weight < 0.0
            or coverage_weight < 0.0
            or boundary_weight + coverage_weight <= 0.0
        ):
            raise ValueError("acquisition weights must be finite, nonnegative, and not both zero")
        self.bandwidth = bandwidth
        self.boundary_weight = boundary_weight
        self.coverage_weight = coverage_weight
        self.candidate_count = candidate_count
        self.seed = seed
        self.samples = np.empty((0, space.dimensions), dtype=float)
        self.failed = np.empty(0, dtype=bool)

    def _probability_normalized(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        observed = self.space.normalize(self.samples)
        squared = np.sum((query[:, None, :] - observed[None, :, :]) ** 2, axis=2)
        weights = np.exp(-0.5 * squared / (self.bandwidth**2)) + 1e-12
        probability = (weights @ self.failed.astype(float) + 0.5) / (weights.sum(axis=1) + 1.0)
        nearest = np.sqrt(np.min(squared, axis=1)) / np.sqrt(self.space.dimensions)
        return probability, nearest

    def predict_failure_probability(self, points: np.ndarray) -> np.ndarray:
        """Estimate failure probability at physical-domain points."""

        query = np.atleast_2d(points).astype(float)
        if len(self.samples) == 0:
            return np.full(len(query), 0.5)
        probability, _ = self._probability_normalized(self.space.normalize(query))
        return probability

    def predict_failure_uncertainty(self, points: np.ndarray) -> np.ndarray:
        """Return approximate beta-posterior SD of the kernel-weighted label rate.

        This is an exploratory uncertainty estimate, not a calibrated
        confidence interval for repeated stochastic SLAM trials.
        """

        query = np.atleast_2d(points).astype(float)
        if len(self.samples) == 0:
            return np.full(len(query), np.sqrt(0.125))
        normalized = self.space.normalize(query)
        observed = self.space.normalize(self.samples)
        squared = np.sum((normalized[:, None, :] - observed[None, :, :]) ** 2, axis=2)
        weights = np.exp(-0.5 * squared / (self.bandwidth**2))
        alpha = 0.5 + weights @ self.failed.astype(float)
        beta = 0.5 + weights @ (~self.failed).astype(float)
        total = alpha + beta
        return np.sqrt(alpha * beta / (total**2 * (total + 1.0)))

    def run(
        self, objective: ArrayFunction, budget: int, tolerance: float = 0.01, patience: int = 8
    ) -> ASRFResult:
        """Evaluate ``g`` adaptively up to *budget* observations."""

        if budget < self.initial_samples:
            raise ValueError("budget must be at least initial_samples")
        if not np.isfinite(tolerance) or tolerance < 0.0 or patience < 1:
            raise ValueError("tolerance must be finite and nonnegative; patience must be positive")
        rng = np.random.default_rng(self.seed)
        initial = self.space.denormalize(latin_hypercube(self.initial_samples, self.space.dimensions, rng))
        margins = np.asarray(objective(initial), dtype=float).reshape(-1)
        if len(margins) != self.initial_samples or not np.isfinite(margins).all():
            raise ValueError("objective must return one finite margin per initial sample")
        self.samples = initial
        self.failed = margins >= 0.0
        all_margins = list(margins)
        monitor = latin_hypercube(512, self.space.dimensions, np.random.default_rng(self.seed + 1))
        previous, stable = None, 0
        convergence: list[float] = []
        while len(self.samples) < budget:
            candidates = latin_hypercube(self.candidate_count, self.space.dimensions, rng)
            probability, nearest = self._probability_normalized(candidates)
            acquisition = (
                self.boundary_weight * 4.0 * probability * (1.0 - probability)
                + self.coverage_weight * nearest
            )
            selected = self.space.denormalize(candidates[[int(np.argmax(acquisition))]])
            observation = np.asarray(objective(selected), dtype=float).reshape(-1)
            if len(observation) != 1 or not np.isfinite(observation[0]):
                raise ValueError("objective must return one finite margin per selected sample")
            margin = float(observation[0])
            self.samples = np.vstack((self.samples, selected))
            self.failed = np.append(self.failed, margin >= 0.0)
            all_margins.append(margin)
            current, _ = self._probability_normalized(monitor)
            change = float("inf") if previous is None else float(np.mean(np.abs(current - previous)))
            convergence.append(change)
            stable = stable + 1 if change < tolerance else 0
            previous = current
            if stable >= patience and len(np.unique(self.failed)) == 2:
                break
        return ASRFResult(
            samples=self.samples.copy(),
            performance_margin=np.asarray(all_margins),
            failed=self.failed.copy(),
            convergence=np.asarray(convergence),
        )


def classification_metrics(true_failed: np.ndarray, predicted_failed: np.ndarray) -> dict[str, float]:
    """Return boundary classification error, FNR, and FPR."""

    true_failed = np.asarray(true_failed, dtype=bool)
    predicted_failed = np.asarray(predicted_failed, dtype=bool)
    false_negative = np.count_nonzero(true_failed & ~predicted_failed)
    false_positive = np.count_nonzero(~true_failed & predicted_failed)
    positives = max(1, np.count_nonzero(true_failed))
    negatives = max(1, np.count_nonzero(~true_failed))
    return {
        "boundary_classification_error": float(np.mean(true_failed != predicted_failed)),
        "fnr": false_negative / positives,
        "fpr": false_positive / negatives,
    }
