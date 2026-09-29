"""Traceable CSV/JSON/YAML exports for one experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

from slamcart3d.metrics.map_quality import MAP_MATCH_TOLERANCE_M, WALL_SAMPLE_SPACING_M
from slamcart3d.metrics.map_quality_3d import (
    GROUND_EXCLUSION_HEIGHT_M,
    WALL_3D_MATCH_TOLERANCE_M,
    WALL_SURFACE_SAMPLE_SPACING_M,
)
from slamcart3d.simulation.runner import RunResult


def export_result(result: RunResult, output_root: str | Path = "results") -> Path:
    """Write one self-contained result directory and return its path."""

    target = Path(output_root) / result.experiment_id
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
        "ground_truth_wall_contact_interpretation": (
            "count of sampled centerline chords touching any static wall; no collision response, "
            "robot radius, dynamic obstacle check, or continuous-arc guarantee"
        ),
        "lidar3d_point_count": int(len(result.map_points_3d)),
        "lidar3d_map_pose_source": (
            "estimated planar pose and fixed sensor height; not 6DoF SLAM"
            if result.config.lidar3d.enabled
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
                "ground_exclusion_height_m": GROUND_EXCLUSION_HEIGHT_M,
                "wall_surface_sample_spacing_m": WALL_SURFACE_SAMPLE_SPACING_M,
                "match_tolerance_m": WALL_3D_MATCH_TOLERANCE_M,
                "alignment": "none; estimated planar pose in initial world frame",
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
    if result.occupancy_grid is not None:
        with (target / "map" / "occupancy_grid.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["cell_x", "cell_y", "center_x_m", "center_y_m", "estimated", "reference"])
            writer.writerows(result.occupancy_grid.rows(result.world))
    return target
