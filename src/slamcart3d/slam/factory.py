"""Single algorithm construction path shared by simulation and recorded replay."""

from slamcart3d.core.config import ExperimentConfig
from slamcart3d.slam.adaptive_hybrid import AdaptiveHybridSLAM
from slamcart3d.slam.dead_reckoning import DeadReckoningSLAM
from slamcart3d.slam.hybrid3d import Hybrid3DSLAM
from slamcart3d.slam.icp_odometry import ICPOdometrySLAM
from slamcart3d.slam.loop_closure import PoseGraphHybridSLAM
from slamcart3d.slam.wheel_imu_fusion import WheelIMUFusionSLAM


def make_algorithm(config: ExperimentConfig):
    geometry = (config.robot.wheel_radius_m, config.robot.wheel_base_m, config.lidar.max_range_m)
    name = config.slam.algorithm
    if name == "dead_reckoning":
        return DeadReckoningSLAM(*geometry)
    if name == "wheel_imu_fusion":
        return WheelIMUFusionSLAM(*geometry)
    registration = (
        config.slam.icp_max_iterations,
        config.slam.icp_max_correspondence_m,
        config.slam.icp_prior_weight,
    )
    if name == "icp_odometry":
        return ICPOdometrySLAM(*geometry, *registration)
    if name in {"adaptive_hybrid", "posegraph_hybrid", "hybrid_3d"}:
        kind = {
            "posegraph_hybrid": PoseGraphHybridSLAM,
            "adaptive_hybrid": AdaptiveHybridSLAM,
            "hybrid_3d": Hybrid3DSLAM,
        }[name]
        extras = (
            {"sensor_height_m": config.lidar3d.sensor_height_m, "max_range_m": config.lidar3d.max_range_m}
            if name == "hybrid_3d"
            else {}
        )
        return kind(
            *geometry,
            *registration,
            config.slam.hybrid_use_imu,
            config.slam.hybrid_adaptive_trust,
            config.slam.hybrid_gate_map,
            **extras,
        )
    raise ValueError(f"unsupported SLAM algorithm: {name}")
