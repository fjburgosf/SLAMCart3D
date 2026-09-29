"""End-to-end vertical slice for deterministic SLAM robustness experiments."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Callable

import numpy as np

from slamcart3d import __version__
from slamcart3d.core.config import ExperimentConfig
from slamcart3d.core.provenance import source_sha256
from slamcart3d.metrics.map_quality import point_map_metrics
from slamcart3d.metrics.map_quality_3d import wall_point_map_metrics_3d
from slamcart3d.metrics.occupancy_grid import OccupancyGrid
from slamcart3d.metrics.robustness_score import robustness_score_metrics
from slamcart3d.metrics.trajectory import trajectory_metrics
from slamcart3d.metrics.trajectory_geometry import static_wall_contact_steps
from slamcart3d.sensors.lidar3d import MultiLayerLidar
from slamcart3d.simulation.observations import ObservationStream
from slamcart3d.slam.adaptive_hybrid import AdaptiveHybridSLAM
from slamcart3d.slam.dead_reckoning import DeadReckoningSLAM
from slamcart3d.slam.icp_odometry import ICPOdometrySLAM
from slamcart3d.slam.wheel_imu_fusion import WheelIMUFusionSLAM
from slamcart3d.worlds.geometry import World3D


@dataclass(frozen=True)
class RunResult:
    """Numerical outputs and provenance from one simulation run."""

    experiment_id: str
    created_utc: str
    software_version: str
    source_sha256: str
    config: ExperimentConfig
    timestamps: np.ndarray
    ground_truth: np.ndarray
    estimated: np.ndarray
    map_points: np.ndarray
    map_points_3d: np.ndarray
    occupancy_grid: OccupancyGrid | None
    metrics: dict[str, float]
    runtime_s: float
    diagnostics: dict[str, float | int | str | None]
    world: World3D


@dataclass(frozen=True)
class RunStarted:
    """Identity and geometry available before the first simulation step."""

    experiment_id: str
    created_utc: str
    config: ExperimentConfig
    initial_pose: np.ndarray
    total_steps: int
    world: World3D


@dataclass(frozen=True)
class RunStep:
    """One completed step and only the map points added at that step."""

    index: int
    total_steps: int
    timestamp_s: float
    ground_truth: np.ndarray
    estimated: np.ndarray
    map_points_added: np.ndarray
    map_points_3d_added: np.ndarray
    occupancy_grid: OccupancyGrid | None


@dataclass(frozen=True)
class RunPreview:
    """Partial geometry used by the same canvases as a finished run."""

    world: World3D
    timestamps: np.ndarray
    ground_truth: np.ndarray
    estimated: np.ndarray
    map_points: np.ndarray
    map_points_3d: np.ndarray
    occupancy_grid: OccupancyGrid | None


def _experiment_id(config: ExperimentConfig, created_utc: str) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, ensure_ascii=False) + created_utc
    return "EXP-" + sha256(payload.encode("utf-8")).hexdigest()[:12].upper()


def run_simulation(
    config: ExperimentConfig,
    *,
    on_start: Callable[[RunStarted], None] | None = None,
    on_step: Callable[[RunStep], None] | None = None,
) -> RunResult:
    """Run one experiment; optional callbacks expose complete steps in order."""

    started = time.perf_counter()
    created = datetime.now(timezone.utc).isoformat()
    stream = ObservationStream(config)
    initial = stream.initial_pose
    experiment_id = _experiment_id(config, created)
    callback_duration = 0.0
    if on_start is not None:
        callback_started = time.perf_counter()
        on_start(
            RunStarted(
                experiment_id,
                created,
                config,
                initial.as_array().copy(),
                len(stream.timestamps) - 1,
                stream.world,
            )
        )
        callback_duration += time.perf_counter() - callback_started
    if config.slam.algorithm == "dead_reckoning":
        algorithm = DeadReckoningSLAM(
            config.robot.wheel_radius_m, config.robot.wheel_base_m, config.lidar.max_range_m
        )
    elif config.slam.algorithm == "wheel_imu_fusion":
        algorithm = WheelIMUFusionSLAM(
            config.robot.wheel_radius_m, config.robot.wheel_base_m, config.lidar.max_range_m
        )
    elif config.slam.algorithm == "adaptive_hybrid":
        algorithm = AdaptiveHybridSLAM(
            config.robot.wheel_radius_m,
            config.robot.wheel_base_m,
            config.lidar.max_range_m,
            config.slam.icp_max_iterations,
            config.slam.icp_max_correspondence_m,
            config.slam.icp_prior_weight,
            config.slam.hybrid_use_imu,
            config.slam.hybrid_adaptive_trust,
            config.slam.hybrid_gate_map,
        )
    else:
        algorithm = ICPOdometrySLAM(
            config.robot.wheel_radius_m,
            config.robot.wheel_base_m,
            config.lidar.max_range_m,
            config.slam.icp_max_iterations,
            config.slam.icp_max_correspondence_m,
            config.slam.icp_prior_weight,
        )
    algorithm.initialize(initial)
    timestamps = stream.timestamps
    ground_truth = [initial.as_array()]
    estimated = [initial.as_array()]
    lidar3d = MultiLayerLidar(config.lidar3d) if config.lidar3d.enabled else None
    occupancy_grid = (
        OccupancyGrid(config.occupancy_grid.resolution_m, config.occupancy_grid.beam_stride)
        if config.occupancy_grid.enabled
        else None
    )
    clouds_3d: list[np.ndarray] = []
    previous_map_count = 0
    blocked_steps = 0
    for index, frame in enumerate(stream.frames(), start=1):
        blocked_steps += int(frame.blocked_by_wall)
        estimated_pose = algorithm.update(
            frame.timestamp_s,
            frame.measured_left_rate_rad_s,
            frame.measured_right_rate_rad_s,
            config.simulation.timestep_s,
            frame.measured_lidar,
            frame.measured_imu,
        )
        ground_truth.append(frame.ground_truth.as_array())
        estimated.append(estimated_pose.as_array())
        if occupancy_grid is not None:
            occupancy_grid.observe(frame.measured_lidar, estimated_pose, config.lidar.max_range_m)
        new_cloud = np.empty((0, 3), dtype=float)
        if lidar3d is not None and frame.measured_lidar3d is not None:
            new_cloud = lidar3d.point_cloud(frame.measured_lidar3d, estimated_pose)
            clouds_3d.append(new_cloud)
        if on_step is not None:
            callback_started = time.perf_counter()
            current_map = algorithm.get_map()
            if len(current_map) < previous_map_count:
                raise RuntimeError("the incremental point map shrank during a run")
            on_step(
                RunStep(
                    index,
                    len(timestamps) - 1,
                    float(frame.timestamp_s),
                    frame.ground_truth.as_array().copy(),
                    estimated_pose.as_array().copy(),
                    current_map[previous_map_count:].copy(),
                    new_cloud.copy(),
                    deepcopy(occupancy_grid),
                )
            )
            previous_map_count = len(current_map)
            callback_duration += time.perf_counter() - callback_started
    truth_array = np.asarray(ground_truth)
    estimate_array = np.asarray(estimated)
    metrics = trajectory_metrics(truth_array, estimate_array)
    metrics["ground_truth_static_wall_contact_steps"] = float(
        static_wall_contact_steps(truth_array, stream.world)
    )
    metrics["wall_blocked_steps"] = float(blocked_steps)
    map_points = algorithm.get_map()
    map_points_3d = np.vstack(clouds_3d) if clouds_3d else np.empty((0, 3), dtype=float)
    metrics.update(point_map_metrics(map_points, stream.world))
    if config.lidar3d.enabled:
        metrics.update(wall_point_map_metrics_3d(map_points_3d, stream.world))
    if occupancy_grid is not None:
        metrics.update(occupancy_grid.metrics(stream.world))
    diagnostics = algorithm.get_diagnostics()
    if config.slam.algorithm in {"icp_odometry", "adaptive_hybrid"} and int(diagnostics["updates"]) > 1:
        attempted_registrations = int(diagnostics["updates"]) - 1
        metrics["icp_registration_availability"] = (
            attempted_registrations - int(diagnostics["fallbacks"])
        ) / attempted_registrations
    if config.robustness_score.enabled:
        metrics.update(robustness_score_metrics(metrics, config.robustness_score))
    metrics["runtime_s"] = float(max(0.0, time.perf_counter() - started - callback_duration))
    return RunResult(
        experiment_id=experiment_id,
        created_utc=created,
        software_version=__version__,
        source_sha256=source_sha256(),
        config=config,
        timestamps=timestamps,
        ground_truth=truth_array,
        estimated=estimate_array,
        map_points=map_points,
        map_points_3d=map_points_3d,
        occupancy_grid=occupancy_grid,
        metrics=metrics,
        runtime_s=metrics["runtime_s"],
        diagnostics=diagnostics,
        world=stream.world,
    )
