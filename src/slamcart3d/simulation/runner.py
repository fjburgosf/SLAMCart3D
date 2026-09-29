"""End-to-end vertical slice for deterministic SLAM robustness experiments."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256

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


def _experiment_id(config: ExperimentConfig, created_utc: str) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, ensure_ascii=False) + created_utc
    return "EXP-" + sha256(payload.encode("utf-8")).hexdigest()[:12].upper()


def run_simulation(config: ExperimentConfig) -> RunResult:
    """Run the complete headless vertical slice and calculate objective metrics."""

    started = time.perf_counter()
    created = datetime.now(timezone.utc).isoformat()
    stream = ObservationStream(config)
    initial = stream.initial_pose
    if config.slam.algorithm == "dead_reckoning":
        algorithm = DeadReckoningSLAM(
            config.robot.wheel_radius_m, config.robot.wheel_base_m, config.lidar.max_range_m
        )
    elif config.slam.algorithm == "wheel_imu_fusion":
        algorithm = WheelIMUFusionSLAM(
            config.robot.wheel_radius_m, config.robot.wheel_base_m, config.lidar.max_range_m
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
    for frame in stream.frames():
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
        if lidar3d is not None and frame.measured_lidar3d is not None:
            clouds_3d.append(lidar3d.point_cloud(frame.measured_lidar3d, estimated_pose))
    truth_array = np.asarray(ground_truth)
    estimate_array = np.asarray(estimated)
    metrics = trajectory_metrics(truth_array, estimate_array)
    metrics["ground_truth_static_wall_contact_steps"] = float(
        static_wall_contact_steps(truth_array, stream.world)
    )
    map_points = algorithm.get_map()
    map_points_3d = np.vstack(clouds_3d) if clouds_3d else np.empty((0, 3), dtype=float)
    metrics.update(point_map_metrics(map_points, stream.world))
    if config.lidar3d.enabled:
        metrics.update(wall_point_map_metrics_3d(map_points_3d, stream.world))
    if occupancy_grid is not None:
        metrics.update(occupancy_grid.metrics(stream.world))
    diagnostics = algorithm.get_diagnostics()
    if config.slam.algorithm == "icp_odometry" and int(diagnostics["updates"]) > 1:
        attempted_registrations = int(diagnostics["updates"]) - 1
        metrics["icp_registration_availability"] = (
            attempted_registrations - int(diagnostics["fallbacks"])
        ) / attempted_registrations
    if config.robustness_score.enabled:
        metrics.update(robustness_score_metrics(metrics, config.robustness_score))
    metrics["runtime_s"] = float(time.perf_counter() - started)
    return RunResult(
        experiment_id=_experiment_id(config, created),
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
