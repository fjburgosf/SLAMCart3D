"""End-to-end vertical slice for deterministic SLAM robustness experiments."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Callable

import numpy as np

from slamcart3d import __version__
from slamcart3d.core.config import ExperimentConfig
from slamcart3d.core.models import Pose2D
from slamcart3d.core.provenance import source_sha256
from slamcart3d.mapping.voxel import VoxelMap
from slamcart3d.metrics.map_quality import point_map_metrics
from slamcart3d.metrics.map_quality_3d import has_evaluable_wall_surface, wall_point_map_metrics_3d
from slamcart3d.metrics.occupancy_grid import OccupancyGrid
from slamcart3d.metrics.robustness_score import robustness_score_metrics
from slamcart3d.metrics.trajectory import trajectory_metrics
from slamcart3d.metrics.trajectory_geometry import static_wall_contact_steps
from slamcart3d.sensors.lidar3d import MultiLayerLidar
from slamcart3d.simulation.observations import ObservationStream
from slamcart3d.slam.factory import make_algorithm
from slamcart3d.slam.hybrid3d import Hybrid3DSLAM
from slamcart3d.slam.loop_closure import PoseGraphHybridSLAM, rotation
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
    online_estimated: np.ndarray | None = None
    estimated_3d: np.ndarray | None = None
    voxel_map: VoxelMap | None = None


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
    navigation: dict[str, float | int | str | None] = field(default_factory=dict)
    estimated_3d: np.ndarray | None = None
    voxel_map: VoxelMap | None = None


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
    navigation: dict[str, float | int | str | None] = field(default_factory=dict)
    estimated_3d: np.ndarray | None = None
    voxel_map: VoxelMap | None = None


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
    algorithm = make_algorithm(config)
    algorithm.initialize(initial)
    timestamps = stream.timestamps
    ground_truth = [initial.as_array()]
    estimated = [initial.as_array()]
    lidar3d = MultiLayerLidar(config.lidar3d) if config.lidar3d.enabled else None
    occupancy_grid = (
        OccupancyGrid(config.occupancy_grid.resolution_m, config.occupancy_grid.beam_stride)
        if config.occupancy_grid.enabled or config.simulation.motion_profile == "frontier"
        else None
    )
    clouds_3d: list[np.ndarray] = []
    states_3d = [np.array([initial.x, initial.y, 0.0, 0.0, 0.0, initial.yaw])]
    voxel_map = (
        VoxelMap(config.reconstruction.resolution_m, config.reconstruction.min_hits)
        if (config.lidar3d.enabled and config.reconstruction.enabled)
        else None
    )
    previous_map_count = 0
    blocked_steps = 0
    scans_for_rebuild = []
    for index, frame in enumerate(stream.frames(algorithm.get_pose), start=1):
        blocked_steps += int(frame.blocked_by_wall)
        update = algorithm.update_3d if isinstance(algorithm, Hybrid3DSLAM) else algorithm.update
        extra = (frame.measured_lidar3d,) if isinstance(algorithm, Hybrid3DSLAM) else ()
        estimated_pose = update(
            frame.timestamp_s,
            frame.measured_left_rate_rad_s,
            frame.measured_right_rate_rad_s,
            config.simulation.timestep_s,
            frame.measured_lidar,
            frame.measured_imu,
            *extra,
        )
        ground_truth.append(frame.ground_truth.as_array())
        estimated.append(estimated_pose.as_array())
        states_3d.append(
            algorithm.state.copy()
            if isinstance(algorithm, Hybrid3DSLAM)
            else np.array([estimated_pose.x, estimated_pose.y, 0.0, 0.0, 0.0, estimated_pose.yaw])
        )
        if occupancy_grid is not None:
            occupancy_grid.observe(frame.measured_lidar, estimated_pose, config.lidar.max_range_m)
        if isinstance(algorithm, PoseGraphHybridSLAM):
            scans_for_rebuild.append(frame.measured_lidar)
        new_cloud = np.empty((0, 3), dtype=float)
        if lidar3d is not None and frame.measured_lidar3d is not None:
            new_cloud = (
                algorithm._last_cloud.copy()
                if isinstance(algorithm, Hybrid3DSLAM)
                else (lidar3d.point_cloud(frame.measured_lidar3d, estimated_pose))
            )
            clouds_3d.append(new_cloud)
        origin = (
            algorithm.sensor_origin()
            if isinstance(algorithm, Hybrid3DSLAM)
            else np.array([estimated_pose.x, estimated_pose.y, config.lidar3d.sensor_height_m])
        )
        if voxel_map is not None:
            voxel_map.observe(origin, new_cloud)
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
                    dict(stream.navigation_diagnostics),
                    states_3d[-1].copy(),
                    deepcopy(voxel_map) if index % 10 == 0 or index == len(timestamps) - 1 else None,
                )
            )
            previous_map_count = len(current_map)
            callback_duration += time.perf_counter() - callback_started
    truth_array = np.asarray(ground_truth)
    estimate_array = np.asarray(estimated)
    online_estimated = None
    if isinstance(algorithm, PoseGraphHybridSLAM):
        online_estimated = estimate_array.copy()
        estimate_array = algorithm.finalize()
        states_3d = [np.array([pose[0], pose[1], 0.0, 0.0, 0.0, pose[2]]) for pose in estimate_array]
        for index, cloud in enumerate(clouds_3d):
            old, new = online_estimated[index + 1], estimate_array[index + 1]
            cloud[:, :2] = (cloud[:, :2] - old[:2]) @ rotation(old[2]) @ rotation(new[2]).T + new[:2]
        if voxel_map is not None:
            voxel_map = VoxelMap(config.reconstruction.resolution_m, config.reconstruction.min_hits)
            for cloud, pose in zip(clouds_3d, estimate_array[1:], strict=True):
                voxel_map.observe(np.array([pose[0], pose[1], config.lidar3d.sensor_height_m]), cloud)
        if occupancy_grid is not None:
            occupancy_grid = OccupancyGrid(
                config.occupancy_grid.resolution_m, config.occupancy_grid.beam_stride
            )
            for scan, pose in zip(scans_for_rebuild, estimate_array[1:], strict=True):
                occupancy_grid.observe(scan, Pose2D(*pose), config.lidar.max_range_m)
    metrics = trajectory_metrics(truth_array, estimate_array)
    metrics["ground_truth_static_wall_contact_steps"] = float(
        static_wall_contact_steps(truth_array, stream.world)
    )
    metrics["wall_blocked_steps"] = float(blocked_steps)
    map_points = algorithm.get_map()
    map_points_3d = np.vstack(clouds_3d) if clouds_3d else np.empty((0, 3), dtype=float)
    metrics.update(point_map_metrics(map_points, stream.world))
    if config.lidar3d.enabled and has_evaluable_wall_surface(stream.world):
        metrics.update(wall_point_map_metrics_3d(map_points_3d, stream.world))
    if occupancy_grid is not None:
        metrics.update(occupancy_grid.metrics(stream.world))
    diagnostics = algorithm.get_diagnostics()
    diagnostics.update(stream.navigation_diagnostics)
    if online_estimated is not None:
        diagnostics["online_ate_rmse_m"] = trajectory_metrics(truth_array, online_estimated)["ate_rmse_m"]
    if (
        config.slam.algorithm in {"icp_odometry", "adaptive_hybrid", "posegraph_hybrid", "hybrid_3d"}
        and int(diagnostics["updates"]) > 1
    ):
        attempted_registrations = int(diagnostics["updates"]) - 1
        metrics["icp_registration_availability"] = (
            attempted_registrations - int(diagnostics["fallbacks"])
        ) / attempted_registrations
    if isinstance(algorithm, Hybrid3DSLAM):
        states = np.asarray(states_3d)
        metrics["pose3d_z_rmse_m"] = float(np.sqrt(np.mean(states[:, 2] ** 2)))
        metrics["pose3d_roll_pitch_rmse_rad"] = float(np.sqrt(np.mean(states[:, 3:5] ** 2)))
    if voxel_map is not None:
        metrics["reconstruction_occupied_voxels"] = float(len(voxel_map.occupied))
        metrics["reconstruction_free_voxels"] = float(len(voxel_map.free))
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
        online_estimated=online_estimated,
        estimated_3d=np.asarray(states_3d),
        voxel_map=voxel_map,
    )
