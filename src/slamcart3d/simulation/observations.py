"""Deterministic sensor stream reusable by SLAM runs and dataset export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from slamcart3d.core.config import ExperimentConfig
from slamcart3d.core.models import CameraFrame, IMUSample, LidarScan, LidarScan3D, Pose2D
from slamcart3d.degradation.engine import DegradationEngine
from slamcart3d.navigation.frontier import FrontierExplorer
from slamcart3d.robots.differential_drive import DifferentialDrive
from slamcart3d.robots.geometry import cart_clearance_radius
from slamcart3d.sensors.camera import SyntheticCamera
from slamcart3d.sensors.imu import PlanarIMU
from slamcart3d.sensors.lidar import PlanarLidar
from slamcart3d.sensors.lidar3d import MultiLayerLidar
from slamcart3d.simulation.motion import motion_command
from slamcart3d.worlds.geometry import WallSegment, World3D, make_world


@dataclass(frozen=True)
class SimulationFrame:
    """One synchronized true state and set of ideal/measured observations."""

    timestamp_s: float
    ground_truth: Pose2D
    blocked_by_wall: bool
    ideal_lidar: LidarScan
    measured_lidar: LidarScan
    commanded_left_rate_rad_s: float
    commanded_right_rate_rad_s: float
    measured_left_rate_rad_s: float
    measured_right_rate_rad_s: float
    ideal_imu: IMUSample | None
    measured_imu: IMUSample | None
    ideal_lidar3d: LidarScan3D | None
    measured_lidar3d: LidarScan3D | None
    ideal_camera: CameraFrame | None
    measured_camera: CameraFrame | None


class ObservationStream:
    """Replayable stream whose random state is recreated on each iteration."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.initial_pose = Pose2D(
            config.robot.initial_x_m,
            config.robot.initial_y_m,
            config.robot.initial_yaw_rad,
        )
        if config.simulation.world == "custom":
            self.world = World3D(
                name="custom",
                segments=tuple(WallSegment(**wall.__dict__) for wall in config.simulation.world_segments),
            )
        else:
            self.world = make_world(config.simulation.world, config.seed)
        self.timesteps = int(round(config.simulation.duration_s / config.simulation.timestep_s))
        self.timestamps = np.arange(self.timesteps + 1, dtype=float) * config.simulation.timestep_s

    def frames(self, pose_feedback: Callable[[], Pose2D] | None = None) -> Iterator[SimulationFrame]:
        """Generate the same observations for every traversal of this stream."""

        config = self.config
        robot = DifferentialDrive(config.robot.wheel_radius_m, config.robot.wheel_base_m, self.initial_pose)
        lidar = PlanarLidar(config.lidar.beam_count, config.lidar.field_of_view_deg, config.lidar.max_range_m)
        lidar3d = MultiLayerLidar(config.lidar3d) if config.lidar3d.enabled else None
        camera = SyntheticCamera(config.camera) if config.camera.enabled else None
        imu = PlanarIMU()
        degradation = DegradationEngine(config.degradation, config.seed + 1)
        previous_velocity = 0.0
        navigator_robot = DifferentialDrive(
            config.robot.wheel_radius_m, config.robot.wheel_base_m, self.initial_pose
        )
        explorer = (
            FrontierExplorer(
                config.navigation,
                cart_clearance_radius(config.robot.wheel_radius_m, config.robot.wheel_base_m),
                config.simulation.linear_velocity_m_s,
                config.lidar.max_range_m,
            )
            if config.simulation.motion_profile == "frontier"
            else None
        )
        self.navigation_diagnostics = {}
        previous_scan = None
        for step_index, timestamp in enumerate(self.timestamps[1:]):
            progress = (step_index + 1) / self.timesteps
            if explorer is None:
                linear_velocity, yaw_rate = motion_command(config.simulation, step_index, self.timesteps)
            else:
                estimated_pose = pose_feedback() if pose_feedback is not None else navigator_robot.pose
                linear_velocity, yaw_rate = explorer.command(estimated_pose, previous_scan)
                self.navigation_diagnostics = explorer.diagnostics()
            left, right = robot.wheel_rates(linear_velocity, yaw_rate)
            previous_pose = robot.pose
            true_pose = robot.step(left, right, config.simulation.timestep_s)
            blocked_by_wall = config.simulation.stop_at_walls and self.world.path_clearance_m(
                (previous_pose.x, previous_pose.y),
                (true_pose.x, true_pose.y),
                float(timestamp),
            ) < cart_clearance_radius(config.robot.wheel_radius_m, config.robot.wheel_base_m)
            if blocked_by_wall:
                robot.pose = previous_pose
                true_pose = previous_pose
                left = right = 0.0
                linear_velocity = 0.0
                yaw_rate = 0.0
            ideal_scan = lidar.scan(float(timestamp), true_pose, self.world)
            measured_scan = degradation.degrade_lidar(
                ideal_scan, config.lidar.max_range_m, progress, true_pose
            )
            ideal_scan3d = lidar3d.scan(float(timestamp), true_pose, self.world) if lidar3d else None
            measured_scan3d = (
                degradation.degrade_lidar3d(ideal_scan3d, config.lidar3d.max_range_m, progress, true_pose)
                if ideal_scan3d is not None
                else None
            )
            ideal_camera = camera.capture(float(timestamp), true_pose, self.world) if camera else None
            measured_camera = (
                degradation.degrade_camera(ideal_camera, progress, true_pose)
                if ideal_camera is not None
                else None
            )
            measured_left, measured_right = degradation.degrade_wheel_rates(left, right, progress, true_pose)
            navigator_robot.step(measured_left, measured_right, config.simulation.timestep_s)
            previous_scan = measured_scan
            ideal_imu = (
                imu.sample(
                    float(timestamp),
                    linear_velocity,
                    previous_velocity,
                    yaw_rate,
                    config.simulation.timestep_s,
                )
                if config.imu.enabled
                else None
            )
            measured_imu = (
                degradation.degrade_imu(
                    ideal_imu, config.simulation.timestep_s, config.imu, progress, true_pose
                )
                if ideal_imu is not None
                else None
            )
            previous_velocity = linear_velocity
            yield SimulationFrame(
                timestamp_s=float(timestamp),
                ground_truth=true_pose,
                blocked_by_wall=blocked_by_wall,
                ideal_lidar=ideal_scan,
                measured_lidar=measured_scan,
                commanded_left_rate_rad_s=left,
                commanded_right_rate_rad_s=right,
                measured_left_rate_rad_s=measured_left,
                measured_right_rate_rad_s=measured_right,
                ideal_imu=ideal_imu,
                measured_imu=measured_imu,
                ideal_lidar3d=ideal_scan3d,
                measured_lidar3d=measured_scan3d,
                ideal_camera=ideal_camera,
                measured_camera=measured_camera,
            )
