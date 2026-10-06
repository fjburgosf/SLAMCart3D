"""Traceable CSV/JSON/YAML exports for one experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

from slamcart3d.core.provenance import source_sha256
from slamcart3d.metrics.map_quality import MAP_MATCH_TOLERANCE_M, WALL_SAMPLE_SPACING_M
from slamcart3d.metrics.map_quality_3d import (
    GROUND_EXCLUSION_HEIGHT_M,
    WALL_3D_MATCH_TOLERANCE_M,
    WALL_SURFACE_SAMPLE_SPACING_M,
)
from slamcart3d.simulation.runner import RunResult, RunStarted, RunStep


class IncrementalRunWriter:
    """Keep each observed step on disk, even if a later step fails."""

    def __init__(self, started: RunStarted, output_root: str | Path) -> None:
        self.target = Path(output_root) / started.experiment_id
        self.target.mkdir(parents=True, exist_ok=False)
        self.started = started
        self._source_hash = source_sha256()
        self.completed_steps = 0
        self._last_timestamp_s = 0.0
        self._state = "running"
        self._error: str | None = None
        (self.target / "config.yaml").write_text(
            yaml.safe_dump({"experiment": started.config.to_dict()}, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        self._trajectory_file = (self.target / "trajectory_live.csv").open("x", encoding="utf-8", newline="")
        self._state_file = (
            (self.target / "trajectory_3d_live.csv").open("x", encoding="utf-8", newline="")
            if started.config.lidar3d.enabled
            else None
        )
        self._state_writer = csv.writer(self._state_file) if self._state_file else None
        if self._state_writer:
            self._state_writer.writerow(
                ["step", "timestamp_s", "x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad"]
            )
        self._map_file = (self.target / "point_cloud_live.csv").open("x", encoding="utf-8", newline="")
        self._cloud_file = (
            (self.target / "point_cloud_3d_live.csv").open("x", encoding="utf-8", newline="")
            if started.config.lidar3d.enabled
            else None
        )
        self._trajectory = csv.writer(self._trajectory_file)
        self._map = csv.writer(self._map_file)
        self._cloud = csv.writer(self._cloud_file) if self._cloud_file is not None else None
        self._trajectory.writerow(
            ["step", "timestamp_s", "gt_x_m", "gt_y_m", "gt_yaw_rad", "est_x_m", "est_y_m", "est_yaw_rad"]
        )
        self._map.writerow(["step", "x_m", "y_m"])
        if self._cloud is not None:
            self._cloud.writerow(["step", "x_m", "y_m", "z_m"])
        self._flush()
        self._write_status()

    def _flush(self) -> None:
        self._trajectory_file.flush()
        if self._state_file:
            self._state_file.flush()
        self._map_file.flush()
        if self._cloud_file is not None:
            self._cloud_file.flush()

    def _write_status(self) -> None:
        content = {
            "experiment_id": self.started.experiment_id,
            "created_utc": self.started.created_utc,
            "source_sha256": self._source_hash,
            "state": self._state,
            "completed_steps": self.completed_steps,
            "total_steps": self.started.total_steps,
            "last_timestamp_s": self._last_timestamp_s,
        }
        if self._error is not None:
            content["error"] = self._error
        temporary = self.target / "status.json.tmp"
        temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.target / "status.json")

    def append(self, step: RunStep) -> None:
        if self._state != "running" or step.index != self.completed_steps + 1:
            raise ValueError("incremental steps must be written exactly once and in order")
        self._trajectory.writerow([step.index, step.timestamp_s, *step.ground_truth, *step.estimated])
        if self._state_writer and step.estimated_3d is not None:
            self._state_writer.writerow([step.index, step.timestamp_s, *step.estimated_3d])
        self._map.writerows([step.index, *point] for point in step.map_points_added)
        if self._cloud is not None:
            self._cloud.writerows([step.index, *point] for point in step.map_points_3d_added)
        self._flush()
        self.completed_steps = step.index
        self._last_timestamp_s = step.timestamp_s
        self._write_status()

    def close(self) -> None:
        self._trajectory_file.close()
        if self._state_file:
            self._state_file.close()
        self._map_file.close()
        if self._cloud_file is not None:
            self._cloud_file.close()

    def complete(self) -> None:
        self._state = "complete"
        self._write_status()

    def fail(self, error: str) -> None:
        self._state = "failed"
        self._error = error
        self._write_status()


def export_result(
    result: RunResult, output_root: str | Path = "results", *, reuse_existing: bool = False
) -> Path:
    """Write one self-contained result directory and return its path."""

    target = Path(output_root) / result.experiment_id
    if reuse_existing:
        if not target.is_dir() or (target / "metadata.json").exists():
            raise FileExistsError("expected an unfinished incremental result directory")
    else:
        target.mkdir(parents=True, exist_ok=False)
    (target / "figures").mkdir()
    (target / "map").mkdir()
    with (target / "config.yaml").open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump({"experiment": result.config.to_dict()}, stream, sort_keys=False, allow_unicode=True)
    metadata = {
        "experiment_id": result.experiment_id,
        "created_utc": result.created_utc,
        "software_version": result.software_version,
        "source_sha256": result.source_sha256,
        "seed": result.config.seed,
        "algorithm": result.diagnostics["algorithm"],
        "runtime_s": result.runtime_s,
        "diagnostics": result.diagnostics,
        "trajectory_evaluation": {
            "ate_and_legacy_final_error": "rigid SE(2) alignment before evaluation",
            "position_unaligned": "direct distance in initial world frame; no alignment",
            "final_trajectory": "pose graph optimized"
            if result.online_estimated is not None
            else "online estimate",
            "incremental_trajectory": "online estimate, never retrospectively overwritten",
        },
        "ground_truth_wall_contact_interpretation": (
            "count of sampled centerline chords touching any static wall; no collision response, "
            "robot radius, dynamic obstacle check, or continuous-arc guarantee"
        ),
        "lidar3d_point_count": int(len(result.map_points_3d)),
        "lidar3d_map_pose_source": (
            "measured XYZ registration; x,y,z,roll,pitch,yaw with soft level-ground prior"
            if result.config.slam.algorithm == "hybrid_3d"
            else "estimated planar pose and fixed sensor height; not 6DoF SLAM"
            if result.config.lidar3d.enabled
            else None
        ),
        "reconstruction": (
            {
                "method": "external boundary triangles of measured occupied voxels",
                "resolution_m": result.voxel_map.resolution_m,
                "min_hits": result.voxel_map.min_hits,
                "evidence": "one endpoint hit per cell per scan; finite ray interiors free",
                "unknown": "unobserved space is not filled; surface is not necessarily watertight",
                "probabilities": "categorical evidence, not calibrated probabilities",
                "world_reference_used": False,
            }
            if result.voxel_map is not None
            else None
        ),
        "map_evaluation": {
            "reference": "static wall ground-plane segments",
            "wall_sample_spacing_m": WALL_SAMPLE_SPACING_M,
            "match_tolerance_m": MAP_MATCH_TOLERANCE_M,
            "alignment": "none; initial world frame",
        },
        "map3d_evaluation": (
            {
                "reference": "static vertical wall surfaces only",
                "available": "map3d_wall_f1_0p25" in result.metrics,
                "unavailable_reason": (
                    None
                    if "map3d_wall_f1_0p25" in result.metrics
                    else "no static wall surface above the ground exclusion height"
                ),
                "ground_exclusion_height_m": GROUND_EXCLUSION_HEIGHT_M,
                "wall_surface_sample_spacing_m": WALL_SURFACE_SAMPLE_SPACING_M,
                "match_tolerance_m": WALL_3D_MATCH_TOLERANCE_M,
                "alignment": "none; estimated pose in initial world frame",
            }
            if result.config.lidar3d.enabled
            else None
        ),
        "occupancy_evaluation": (
            {
                "reference": "rasterized static ground-plane wall segments",
                "resolution_m": result.config.occupancy_grid.resolution_m,
                "beam_stride": result.config.occupancy_grid.beam_stride,
                "frame": "estimated planar pose in initial world frame",
                "missing_beams": "ignored; not treated as free",
                "unknown_cells": "included as missing occupied coverage, excluded from observed-free accuracy",
                "map_model": "categorical occupied-over-free precedence; no probability calibration",
            }
            if result.occupancy_grid is not None
            else None
        ),
        "srs_definition": (
            {
                "components": [
                    "ate_rmse_m",
                    "rpe_translation_rmse_m",
                    "map_f1_0p25",
                    "tracking_availability",
                ],
                "error_quality": "scale / (scale + error)",
                "bounded_quality": "clip((value - floor) / (1 - floor), 0, 1)",
                "aggregation": "sum(weight * quality) / sum(weight)",
                "weight_sensitivity": "(quality - srs) / sum(weight)",
                "runtime_included": False,
            }
            if result.config.robustness_score.enabled
            else None
        ),
    }
    (target / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    with (target / "trajectory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["timestamp_s", "gt_x_m", "gt_y_m", "gt_yaw_rad", "est_x_m", "est_y_m", "est_yaw_rad"]
        )
        for timestamp, truth, estimate in zip(
            result.timestamps, result.ground_truth, result.estimated, strict=True
        ):
            writer.writerow([timestamp, *truth, *estimate])
    with (target / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        writer.writerows(result.metrics.items())
    if result.online_estimated is not None:
        np.savetxt(
            target / "trajectory_online.csv",
            np.column_stack((result.timestamps, result.online_estimated)),
            delimiter=",",
            header="timestamp_s,x_m,y_m,yaw_rad",
            comments="",
        )
    np.savetxt(
        target / "map" / "point_cloud.csv", result.map_points, delimiter=",", header="x_m,y_m", comments=""
    )
    if result.config.lidar3d.enabled:
        np.savetxt(
            target / "map" / "point_cloud_3d.csv",
            result.map_points_3d,
            delimiter=",",
            header="x_m,y_m,z_m",
            comments="",
        )
        if result.estimated_3d is not None:
            np.savetxt(
                target / "trajectory_3d.csv",
                np.column_stack((result.timestamps, result.estimated_3d)),
                delimiter=",",
                header="timestamp_s,x_m,y_m,z_m,roll_rad,pitch_rad,yaw_rad",
                comments="",
            )
    if result.voxel_map is not None:
        with (target / "map" / "voxel_grid_3d.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["cell_x", "cell_y", "cell_z", "center_x_m", "center_y_m", "center_z_m", "state", "hits"]
            )
            writer.writerows(result.voxel_map.rows())
        mesh = result.voxel_map.surface()
        with (target / "map" / "reconstructed_surface.ply").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write("ply\nformat ascii 1.0\ncomment measured voxel boundary; units meters\n")
            stream.write(
                f"element vertex {len(mesh.vertices)}\nproperty float x\nproperty float y\nproperty float z\n"
            )
            stream.write(
                f"element face {len(mesh.triangles)}\nproperty list uchar int vertex_indices\nend_header\n"
            )
            for vertex in mesh.vertices:
                stream.write(" ".join(f"{value:.9g}" for value in vertex) + "\n")
            for triangle in mesh.triangles:
                stream.write("3 " + " ".join(str(value) for value in triangle) + "\n")
    if result.occupancy_grid is not None:
        with (target / "map" / "occupancy_grid.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["cell_x", "cell_y", "center_x_m", "center_y_m", "estimated", "reference"])
            writer.writerows(result.occupancy_grid.rows(result.world))
    return target
