"""Interface implemented by localization and mapping algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from slamcart3d.core.models import IMUSample, LidarScan, Pose2D


class SLAMAlgorithm(ABC):
    """Stable interface for built-in algorithms and future external adapters."""

    @abstractmethod
    def initialize(self, pose: Pose2D) -> None:
        """Reset state to a known initial pose."""

    @abstractmethod
    def update(
        self,
        timestamp: float,
        left_rate: float,
        right_rate: float,
        dt_s: float,
        scan: LidarScan,
        imu: IMUSample | None = None,
    ) -> Pose2D:
        """Consume one synchronized observation and return the new pose."""

    @abstractmethod
    def reset(self) -> None:
        """Discard all accumulated state."""

    @abstractmethod
    def get_pose(self) -> Pose2D:
        """Return the current pose estimate."""

    @abstractmethod
    def get_map(self) -> np.ndarray:
        """Return map points as an ``N x 2`` array."""

    @abstractmethod
    def get_diagnostics(self) -> dict[str, float | int | str | None]:
        """Return algorithm-specific health information."""
