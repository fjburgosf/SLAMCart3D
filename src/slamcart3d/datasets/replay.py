"""Replay measured CSV/CARMEN data without reading simulator geometry."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import numpy as np

from slamcart3d.core.config import ExperimentConfig
from slamcart3d.core.models import IMUSample, LidarScan, Pose2D, wrap_angle
from slamcart3d.core.provenance import source_sha256
from slamcart3d.metrics.trajectory import trajectory_metrics
from slamcart3d.slam.factory import make_algorithm
from slamcart3d.slam.loop_closure import PoseGraphHybridSLAM, relative_pose


@dataclass(frozen=True)
class RecordedFrame:
    timestamp_s: float
    left_rate: float
    right_rate: float
    scan: LidarScan
    imu: IMUSample | None = None


@dataclass(frozen=True)
class ReplayResult:
    timestamps: np.ndarray
    estimated: np.ndarray
    map_points: np.ndarray
    reference: np.ndarray | None
    metrics: dict[str, float]
    metadata: dict


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _validate_times(times: np.ndarray) -> None:
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("recorded timestamps must be finite and strictly increasing")


def load_carmen(path: Path, config: ExperimentConfig, max_frames: int, timestamp_policy: str = "strict"):
    """Read old FLASER only. Odometry is motion input, never ground truth.

    FLASER angles use a declared 180-degree forward fan. The supported legacy
    convention assumes a colocated laser; nonzero logged offsets are rejected.
    Corrected laser pose fields are intentionally ignored. Motion increments
    use the odometry fields; lateral increments are reported as a limitation.
    """

    records = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            parts = line.split()
            if not parts:
                continue
            if parts[:2] == ["PARAM", "robot_frontlaser_offset"] and float(parts[2]) != 0:
                raise ValueError("CARMEN nonzero laser offset requires an explicit calibration adapter")
            if parts[0] != "FLASER":
                continue
            try:
                beams = int(parts[1])
                if beams < 2 or beams > 10_000 or len(parts) < beams + 11:
                    raise ValueError("invalid FLASER field count")
                ranges = np.array([float(value) for value in parts[2 : 2 + beams]])
                odom = np.array([float(value) for value in parts[beams + 5 : beams + 8]])
                timestamp = float(parts[beams + 8])
                if not np.isfinite(odom).all() or np.any(ranges < 0) or np.isinf(ranges).any():
                    raise ValueError("invalid FLASER measurements")
                records.append((timestamp, ranges, odom))
            except (ValueError, IndexError) as exc:
                raise ValueError(f"CARMEN line {number}: {exc}") from exc
            if len(records) >= max_frames + 1:
                break
    if len(records) < 2:
        raise ValueError("recording requires at least two FLASER scans")
    times = np.array([record[0] for record in records])
    inversions = int(np.count_nonzero(np.diff(times) <= 0))
    duplicate_count = 0
    if timestamp_policy not in {"strict", "sort_unique"}:
        raise ValueError("timestamp policy must be strict or sort_unique")
    if timestamp_policy == "sort_unique":
        if not np.isfinite(times).all():
            raise ValueError("recorded timestamps must be finite and strictly increasing")
        ordered = sorted(records, key=lambda item: item[0])
        records = []
        for record in ordered:
            if records and record[0] == records[-1][0]:
                duplicate_count += 1
            else:
                records.append(record)
        times = np.array([record[0] for record in records])
        if len(records) < 2:
            raise ValueError("recording requires at least two FLASER scans")
    _validate_times(times)
    initial_time = times[0]
    frames = []
    lateral = 0.0
    for previous, current in zip(records[:-1], records[1:], strict=True):
        dt = current[0] - previous[0]
        delta = relative_pose(previous[2], current[2])
        lateral += abs(float(delta[1]))
        yaw_rate = float(delta[2]) / dt
        # Recover constant-twist forward speed from the finite displacement.
        sinc = math.sin(delta[2]) / delta[2] if abs(delta[2]) > 1e-9 else 1.0
        speed = float(delta[0]) / (dt * sinc) if abs(sinc) > 1e-6 else 0.0
        left = (speed - config.robot.wheel_base_m * yaw_rate / 2) / config.robot.wheel_radius_m
        right = (speed + config.robot.wheel_base_m * yaw_rate / 2) / config.robot.wheel_radius_m
        angles = np.linspace(-np.pi / 2, np.pi / 2, len(current[1]), endpoint=False)
        timestamp = current[0] - initial_time
        frames.append(RecordedFrame(timestamp, left, right, LidarScan(timestamp, angles, current[1])))
    return frames, {
        "format": "CARMEN_FLASER",
        "data_origin": "external_recording_unverified",
        "ground_truth_from_log": False,
        "timestamp_policy": timestamp_policy,
        "original_temporal_inversions": inversions,
        "duplicate_timestamps_discarded": duplicate_count,
        "subset_selection": "first max_frames+1 file-order scans before optional sorting",
        "first_ipc_timestamp_s": float(initial_time),
        "odometry_lateral_displacement_sum_m": lateral,
        "limitations": "declared 180-degree forward fan; zero laser offset; differential-motion approximation; no IMU",
    }


def load_measured_csv(path: Path, max_frames: int):
    ranges = _rows(path / "lidar_ranges.csv")[:max_frames]
    wheels = _rows(path / "wheel_odometry.csv")[:max_frames]
    angle_rows = _rows(path / "lidar_angles.csv")
    angles = np.array([float(row["angle_rad"]) for row in angle_rows])
    if len(angles) < 2 or not np.isfinite(angles).all() or len(ranges) != len(wheels) or not ranges:
        raise ValueError("recorded LiDAR/wheel rows must be nonempty and synchronized")
    imu_path = path / "imu.csv"
    imu_rows = _rows(imu_path)[:max_frames] if imu_path.is_file() else []
    if imu_rows and len(imu_rows) != len(ranges):
        raise ValueError("recorded IMU rows must be synchronized")
    frames = []
    for index, (scan, wheel) in enumerate(zip(ranges, wheels, strict=True)):
        timestamp = float(scan["timestamp_s"])
        if abs(timestamp - float(wheel["timestamp_s"])) > 1e-8:
            raise ValueError("recorded LiDAR/wheel timestamps do not match")
        values = [scan[f"range_{beam:03d}_m"] for beam in range(len(angles))]
        measured = np.array([float(value) if value.strip() else np.nan for value in values])
        rates = [float(wheel[f"{side}_measured_rad_s"]) for side in ("left", "right")]
        if not np.isfinite(rates).all() or np.isinf(measured).any() or np.any(measured < 0):
            raise ValueError("recorded measured values are invalid")
        imu = None
        if imu_rows:
            row = imu_rows[index]
            if abs(timestamp - float(row["timestamp_s"])) > 1e-8:
                raise ValueError("recorded IMU timestamps do not match")
            value = row["measured_yaw_rate_rad_s"]
            yaw = float(value) if value.strip() else float("nan")
            imu = IMUSample(timestamp, yaw, 0.0)
        frames.append(RecordedFrame(timestamp, *rates, LidarScan(timestamp, angles.copy(), measured), imu))
    _validate_times(np.array([frame.timestamp_s for frame in frames]))
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    return frames, {
        "format": "measured_CSV",
        "data_origin": metadata.get("data_origin", "undeclared"),
        "limitations": "measured channels only; no ideal sensor channels consumed",
    }


def replay_recording(
    path: str | Path,
    config: ExperimentConfig,
    *,
    reference_path: str | Path | None = None,
    max_frames: int = 1000,
    timestamp_policy: str = "strict",
) -> ReplayResult:
    if config.slam.algorithm == "hybrid_3d":
        raise ValueError(
            "hybrid_3d requires synchronized multilayer LiDAR measurements; this replay format contains planar LiDAR only"
        )
    if type(max_frames) is not int or not 2 <= max_frames <= 20_000:
        raise ValueError("max_frames must be an integer between 2 and 20000")
    source = Path(path)
    frames, metadata = (
        load_measured_csv(source, max_frames)
        if source.is_dir()
        else load_carmen(source, config, max_frames, timestamp_policy)
    )
    initial = Pose2D(config.robot.initial_x_m, config.robot.initial_y_m, config.robot.initial_yaw_rad)
    algorithm = make_algorithm(config)
    algorithm.initialize(initial)
    trajectory = [initial.as_array()]
    timestamps = [0.0]
    for frame in frames:
        dt = frame.timestamp_s - timestamps[-1]
        if dt <= 0:
            raise ValueError("first recorded timestamp must be greater than zero")
        trajectory.append(
            algorithm.update(
                frame.timestamp_s, frame.left_rate, frame.right_rate, dt, frame.scan, frame.imu
            ).as_array()
        )
        timestamps.append(frame.timestamp_s)
    estimate = algorithm.finalize() if isinstance(algorithm, PoseGraphHybridSLAM) else np.asarray(trajectory)
    times = np.asarray(timestamps)
    reference, metrics = None, {}
    if reference_path:
        rows = _rows(Path(reference_path))
        ref_times = np.array([float(row["timestamp_s"]) for row in rows])
        poses = np.array([[float(row[key]) for key in ("x_m", "y_m", "yaw_rad")] for row in rows])
        _validate_times(ref_times)
        if (
            len(poses) < 2
            or not np.isfinite(poses).all()
            or ref_times[0] > times[0]
            or ref_times[-1] < times[-1]
        ):
            raise ValueError(
                "reference poses must be finite and cover the entire recording; no extrapolation"
            )
        reference = np.column_stack(
            [
                np.interp(times, ref_times, poses[:, 0]),
                np.interp(times, ref_times, poses[:, 1]),
                wrap_angle(np.interp(times, ref_times, np.unwrap(poses[:, 2]))),
            ]
        )
        metrics = trajectory_metrics(reference, estimate)
    files = sorted(source.glob("*.csv")) if source.is_dir() else [source]
    if source.is_dir() and (source / "metadata.json").is_file():
        files.append(source / "metadata.json")
    metadata.update(
        frames=len(frames),
        algorithm=config.slam.algorithm,
        source_sha256=source_sha256(),
        input_sha256={file.name: sha256(file.read_bytes()).hexdigest() for file in files},
        reference_available=reference is not None,
        reference_sha256=sha256(Path(reference_path).read_bytes()).hexdigest() if reference_path else None,
        accuracy_claim="not evaluated without independent reference",
        diagnostics=algorithm.get_diagnostics(),
        config=config.to_dict(),
    )
    return ReplayResult(times, estimate, algorithm.get_map(), reference, metrics, metadata)


def export_replay(result: ReplayResult, output_root: str | Path) -> Path:
    target = Path(output_root) / f"REPLAY-{uuid4().hex[:12].upper()}"
    target.mkdir(parents=True, exist_ok=False)
    metadata = dict(result.metadata, created_utc=datetime.now(timezone.utc).isoformat())
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    np.savetxt(
        target / "trajectory.csv",
        np.column_stack((result.timestamps, result.estimated)),
        delimiter=",",
        header="timestamp_s,x_m,y_m,yaw_rad",
        comments="",
    )
    np.savetxt(target / "point_map.csv", result.map_points, delimiter=",", header="x_m,y_m", comments="")
    with (target / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        writer.writerows(result.metrics.items())
    return target
