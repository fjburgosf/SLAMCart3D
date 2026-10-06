"""Export raw synthetic observations without executing a SLAM algorithm."""

from __future__ import annotations

import csv
import json
from contextlib import ExitStack
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from slamcart3d import __version__
from slamcart3d.core.config import ExperimentConfig
from slamcart3d.core.provenance import source_sha256
from slamcart3d.simulation.observations import ObservationStream


def generate_dataset(config: ExperimentConfig, output_root: str | Path = "datasets") -> Path:
    """Write reproducible ground truth, wheel, LiDAR, and optional IMU samples."""

    stream = ObservationStream(config)
    created = datetime.now(timezone.utc).isoformat()
    identity = json.dumps(config.to_dict(), sort_keys=True, ensure_ascii=False) + created
    dataset_id = "DATASET-" + sha256(identity.encode("utf-8")).hexdigest()[:12].upper()
    destination = Path(output_root) / dataset_id
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "config.yaml").write_text(
        yaml.safe_dump({"experiment": config.to_dict()}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    metadata = {
        "dataset_id": dataset_id,
        "data_origin": "synthetic_simulation",
        "created_utc": created,
        "software_version": __version__,
        "source_sha256": source_sha256(),
        "seed": config.seed,
        "world": stream.world.name,
        "moving_objects": len(stream.world.moving_boxes),
        "frames": stream.timesteps,
        "lidar_beams": config.lidar.beam_count,
        "lidar3d_enabled": config.lidar3d.enabled,
        "lidar3d_rays_per_frame": (
            config.lidar3d.azimuth_beams * config.lidar3d.elevation_beams if config.lidar3d.enabled else 0
        ),
        "lidar3d_world_coordinates": "measured ranges transformed by ground-truth robot pose",
        "lidar_missing_value": "empty CSV field",
        "lidar_ideal_no_hit_value_m": config.lidar.max_range_m,
        "imu_enabled": config.imu.enabled,
        "imu_missing_value": "empty CSV field",
        "camera_enabled": config.camera.enabled,
        "camera_resolution_px": [config.camera.width_px, config.camera.height_px]
        if config.camera.enabled
        else None,
        "camera_rgb_model": "deterministic schematic colors on ray-cast 2.5D geometry",
        "camera_depth_unit": "metres along each perspective ray",
        "camera_missing_frame": "blank paths in camera_index.csv",
        "algorithm_executed": False,
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    angles = np.linspace(
        -np.deg2rad(config.lidar.field_of_view_deg) / 2.0,
        np.deg2rad(config.lidar.field_of_view_deg) / 2.0,
        config.lidar.beam_count,
    )
    with (destination / "lidar_angles.csv").open("w", encoding="utf-8", newline="") as angles_file:
        writer = csv.writer(angles_file)
        writer.writerow(["beam_index", "angle_rad"])
        writer.writerows(enumerate(angles))
    with ExitStack() as stack:

        def writer(filename: str):
            return csv.writer(
                stack.enter_context((destination / filename).open("w", encoding="utf-8", newline=""))
            )

        truth = writer("ground_truth.csv")
        wheel = writer("wheel_odometry.csv")
        lidar = writer("lidar_ranges.csv")
        ideal = writer("lidar_ideal_ranges.csv")
        imu_writer = writer("imu.csv") if config.imu.enabled else None
        lidar3d_writer = writer("lidar3d.csv") if config.lidar3d.enabled else None
        camera_writer = writer("camera_index.csv") if config.camera.enabled else None
        camera_root = destination / "camera"
        if camera_writer is not None:
            for subfolder in ("ideal_rgb", "measured_rgb", "ideal_depth", "measured_depth"):
                (camera_root / subfolder).mkdir(parents=True)
            camera_writer.writerow(
                ["timestamp_s", "ideal_rgb_png", "measured_rgb_png", "ideal_depth_npy", "measured_depth_npy"]
            )
        truth.writerow(["timestamp_s", "x_m", "y_m", "yaw_rad"])
        wheel.writerow(
            [
                "timestamp_s",
                "left_command_rad_s",
                "right_command_rad_s",
                "left_measured_rad_s",
                "right_measured_rad_s",
            ]
        )
        range_header = ["timestamp_s", *(f"range_{index:03d}_m" for index in range(config.lidar.beam_count))]
        lidar.writerow(range_header)
        ideal.writerow(range_header)
        if imu_writer is not None:
            imu_writer.writerow(
                [
                    "timestamp_s",
                    "ideal_yaw_rate_rad_s",
                    "measured_yaw_rate_rad_s",
                    "ideal_longitudinal_acceleration_m_s2",
                    "measured_longitudinal_acceleration_m_s2",
                    "ideal_roll_rate_rad_s",
                    "measured_roll_rate_rad_s",
                    "ideal_pitch_rate_rad_s",
                    "measured_pitch_rate_rad_s",
                    "ideal_lateral_acceleration_m_s2",
                    "measured_lateral_acceleration_m_s2",
                    "ideal_vertical_specific_force_m_s2",
                    "measured_vertical_specific_force_m_s2",
                ]
            )
        if lidar3d_writer is not None:
            lidar3d_writer.writerow(
                [
                    "timestamp_s",
                    "ray_index",
                    "azimuth_robot_rad",
                    "elevation_rad",
                    "ideal_slant_range_m",
                    "measured_slant_range_m",
                    "x_sensor_m",
                    "y_sensor_m",
                    "z_sensor_m",
                    "x_world_m",
                    "y_world_m",
                    "z_world_m",
                ]
            )
        truth.writerow([0.0, *stream.initial_pose.as_array()])
        for frame_index, frame in enumerate(stream.frames(), start=1):
            truth.writerow([frame.timestamp_s, *frame.ground_truth.as_array()])
            wheel.writerow(
                [
                    frame.timestamp_s,
                    frame.commanded_left_rate_rad_s,
                    frame.commanded_right_rate_rad_s,
                    frame.measured_left_rate_rad_s,
                    frame.measured_right_rate_rad_s,
                ]
            )
            lidar.writerow(
                [
                    frame.timestamp_s,
                    *("" if not np.isfinite(value) else value for value in frame.measured_lidar.ranges),
                ]
            )
            ideal.writerow([frame.timestamp_s, *frame.ideal_lidar.ranges])
            if imu_writer is not None:
                assert frame.ideal_imu is not None and frame.measured_imu is not None
                imu_writer.writerow(
                    [
                        frame.timestamp_s,
                        frame.ideal_imu.yaw_rate_rad_s,
                        ""
                        if not np.isfinite(frame.measured_imu.yaw_rate_rad_s)
                        else frame.measured_imu.yaw_rate_rad_s,
                        frame.ideal_imu.longitudinal_acceleration_m_s2,
                        ""
                        if not np.isfinite(frame.measured_imu.longitudinal_acceleration_m_s2)
                        else frame.measured_imu.longitudinal_acceleration_m_s2,
                        frame.ideal_imu.roll_rate_rad_s,
                        ""
                        if not np.isfinite(frame.measured_imu.roll_rate_rad_s)
                        else frame.measured_imu.roll_rate_rad_s,
                        frame.ideal_imu.pitch_rate_rad_s,
                        ""
                        if not np.isfinite(frame.measured_imu.pitch_rate_rad_s)
                        else frame.measured_imu.pitch_rate_rad_s,
                        frame.ideal_imu.lateral_acceleration_m_s2,
                        ""
                        if not np.isfinite(frame.measured_imu.lateral_acceleration_m_s2)
                        else frame.measured_imu.lateral_acceleration_m_s2,
                        frame.ideal_imu.vertical_specific_force_m_s2,
                        ""
                        if not np.isfinite(frame.measured_imu.vertical_specific_force_m_s2)
                        else frame.measured_imu.vertical_specific_force_m_s2,
                    ]
                )
            if lidar3d_writer is not None:
                assert frame.ideal_lidar3d is not None and frame.measured_lidar3d is not None
                for index, (azimuth, elevation, ideal_range, measured_range) in enumerate(
                    zip(
                        frame.ideal_lidar3d.azimuths,
                        frame.ideal_lidar3d.elevations,
                        frame.ideal_lidar3d.ranges,
                        frame.measured_lidar3d.ranges,
                        strict=True,
                    )
                ):
                    point: list[float | str] = ["", "", "", "", "", ""]
                    if np.isfinite(measured_range) and measured_range < config.lidar3d.max_range_m - 1e-12:
                        horizontal = measured_range * np.cos(elevation)
                        x_sensor = horizontal * np.cos(azimuth)
                        y_sensor = horizontal * np.sin(azimuth)
                        z_sensor = measured_range * np.sin(elevation)
                        global_angle = frame.ground_truth.yaw + azimuth
                        point = [
                            x_sensor,
                            y_sensor,
                            z_sensor,
                            frame.ground_truth.x + horizontal * np.cos(global_angle),
                            frame.ground_truth.y + horizontal * np.sin(global_angle),
                            config.lidar3d.sensor_height_m + z_sensor,
                        ]
                    lidar3d_writer.writerow(
                        [
                            frame.timestamp_s,
                            index,
                            azimuth,
                            elevation,
                            ideal_range,
                            "" if not np.isfinite(measured_range) else measured_range,
                            *point,
                        ]
                    )
            if camera_writer is not None:
                assert frame.ideal_camera is not None
                filename = f"frame_{frame_index:06d}"
                ideal_rgb = Path("camera") / "ideal_rgb" / f"{filename}.png"
                ideal_depth = Path("camera") / "ideal_depth" / f"{filename}.npy"
                Image.fromarray(frame.ideal_camera.rgb).save(destination / ideal_rgb)
                np.save(destination / ideal_depth, frame.ideal_camera.depth_m)
                measured_rgb = ""
                measured_depth = ""
                if frame.measured_camera is not None:
                    measured_rgb_path = Path("camera") / "measured_rgb" / f"{filename}.png"
                    measured_depth_path = Path("camera") / "measured_depth" / f"{filename}.npy"
                    Image.fromarray(frame.measured_camera.rgb).save(destination / measured_rgb_path)
                    np.save(destination / measured_depth_path, frame.measured_camera.depth_m)
                    measured_rgb = measured_rgb_path.as_posix()
                    measured_depth = measured_depth_path.as_posix()
                camera_writer.writerow(
                    [
                        frame.timestamp_s,
                        ideal_rgb.as_posix(),
                        measured_rgb,
                        ideal_depth.as_posix(),
                        measured_depth,
                    ]
                )
    return destination
