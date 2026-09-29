"""Publication-oriented plots that do not affect the simulation clock."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from slamcart3d.metrics.direction import success_relation
from slamcart3d.simulation.runner import RunResult

PARAMETER_LABELS = {
    "lidar_noise_std_m": "LiDAR range noise σ_L [m]",
    "lidar_bias_m": "LiDAR range bias b_L [m]",
    "lidar_dropout_probability": "LiDAR dropout probability p_L [1]",
    "lidar_outlier_probability": "LiDAR outlier probability p_O [1]",
    "lidar_range_reduction_fraction": "LiDAR range reduction fraction q_R [1]",
    "lidar_sector_dropout_center_deg": "LiDAR missing-sector center [deg]",
    "lidar_sector_dropout_width_deg": "LiDAR missing-sector width [deg]",
    "wheel_rate_noise_std_rad_s": "Wheel rate noise σ_w [rad/s]",
    "wheel_rate_bias_rad_s": "Wheel rate bias b_w [rad/s]",
    "odometry_dropout_probability": "Odometry dropout probability p_D [1]",
    "imu_gyro_bias_rad_s": "IMU gyroscope bias b_g [rad/s]",
    "imu_gyro_noise_std_rad_s": "IMU gyroscope noise σ_g [rad/s]",
    "imu_gyro_random_walk_std_rad_s_sqrt_s": "IMU gyro bias walk σ_bg [rad/s/√s]",
    "imu_accel_bias_m_s2": "IMU forward-acceleration bias b_a [m/s²]",
    "imu_accel_noise_std_m_s2": "IMU forward-acceleration noise σ_a [m/s²]",
    "imu_accel_random_walk_std_m_s2_sqrt_s": "IMU accel bias walk σ_ba [m/s²/√s]",
    "imu_dropout_probability": "IMU packet dropout probability p_I [1]",
    "simulation.linear_velocity_m_s": "Commanded forward speed v [m/s]",
    "simulation.yaw_rate_rad_s": "Commanded yaw rate ω [rad/s]",
    "simulation.motion_frequency_hz": "Yaw modulation frequency f [Hz]",
}
METRIC_LABELS = {
    "ate_rmse_m": "ATE RMSE [m]",
    "rpe_translation_rmse_m": "Translational RPE RMSE [m]",
    "rpe_rotation_rmse_rad": "Rotational RPE RMSE [rad]",
    "final_position_error_m": "Final position error [m]",
    "tracking_availability": "Tracking availability [1]",
    "icp_registration_availability": "ICP registration availability [1]",
    "map_accuracy_mean_m": "Map point-to-wall distance [m]",
    "map_completeness_mean_m": "Wall-to-map distance [m]",
    "map_chamfer_mean_m": "Map Chamfer distance [m]",
    "map_precision_0p25": "Map precision at 0.25 m [1]",
    "map_recall_0p25": "Map recall at 0.25 m [1]",
    "map_f1_0p25": "Map F1 at 0.25 m [1]",
    "map3d_wall_accuracy_mean_m": "3D cloud-to-wall distance [m]",
    "map3d_wall_completeness_mean_m": "3D wall-to-cloud distance [m]",
    "map3d_wall_chamfer_mean_m": "3D wall Chamfer distance [m]",
    "map3d_wall_precision_0p25": "3D wall precision at 0.25 m [1]",
    "map3d_wall_recall_0p25": "3D wall recall at 0.25 m [1]",
    "map3d_wall_f1_0p25": "3D wall F1 at 0.25 m [1]",
    "occupancy_occupied_iou": "Occupied-cell IoU [1]",
    "srs": "SRS robustness index [0, 1]",
}


def _threshold_label(metric_name: str, threshold: float) -> str:
    label = METRIC_LABELS.get(metric_name, metric_name)
    relation = success_relation(metric_name)
    unit_start = label.rfind(" [")
    if unit_start < 0:
        return f"{label} {relation} {threshold:g}"
    unit = label[unit_start + 2 : -1]
    suffix = "" if unit == "1" else f" {unit}"
    return f"{label[:unit_start]} {relation} {threshold:g}{suffix}"


def _prepare_plot_cache(destination: Path) -> None:
    """Keep the plotting library's writable cache beside exported figures."""

    cache = destination.parent / ".plot-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache.resolve())


def save_run_figure(result: RunResult, path: str | Path) -> Path:
    """Save world, map, and trajectories with SI-labelled axes."""

    destination = Path(path)
    _prepare_plot_cache(destination)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.0, 7.0), constrained_layout=True)
    for wall in result.world.segments:
        axis.plot([wall.x1, wall.x2], [wall.y1, wall.y2], color="black", linewidth=2.0)
    for index, wall in enumerate(
        result.world.segments_at(float(result.timestamps[-1]))[len(result.world.segments) :]
    ):
        axis.plot(
            [wall.x1, wall.x2],
            [wall.y1, wall.y2],
            color="#a855f7",
            linewidth=2.0,
            label="Moving object (final position)" if index == 0 else None,
        )
    if len(result.map_points):
        axis.scatter(result.map_points[:, 0], result.map_points[:, 1], s=1, alpha=0.18, label="Estimated map")
    axis.plot(result.ground_truth[:, 0], result.ground_truth[:, 1], linewidth=2.0, label="Ground truth")
    algorithm_labels = {
        "dead_reckoning": "Dead reckoning",
        "wheel_imu_fusion": "Wheel/IMU odometry",
        "icp_odometry": "ICP odometry",
        "adaptive_hybrid": "Adaptive hybrid scan-to-map",
    }
    algorithm_label = algorithm_labels.get(
        str(result.diagnostics["algorithm"]), str(result.diagnostics["algorithm"])
    )
    axis.plot(result.estimated[:, 0], result.estimated[:, 1], linestyle="--", label=algorithm_label)
    axis.set(xlabel="x [m]", ylabel="y [m]", title="SLAMCart3D trajectory and LiDAR map")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def save_occupancy_figure(result: RunResult, path: str | Path) -> Path:
    """Save observed occupancy while showing reference-wall cells left unknown."""

    grid = result.occupancy_grid
    if grid is None:
        raise ValueError("occupancy figure requires an enabled occupancy grid")
    destination = Path(path)
    _prepare_plot_cache(destination)
    import matplotlib.pyplot as plt

    rows = grid.rows(result.world)
    figure, axis = plt.subplots(figsize=(8.0, 6.5), constrained_layout=True)
    for state, color, label in (
        ("free", "#cfe8ff", "Observed free"),
        ("unknown", "#94a3b8", "Unobserved reference wall"),
        ("occupied", "#d95f0e", "Estimated occupied"),
    ):
        centers = np.asarray([(row[2], row[3]) for row in rows if row[4] == state], dtype=float)
        if len(centers):
            axis.scatter(centers[:, 0], centers[:, 1], marker="s", s=13, c=color, label=label)
    for wall in result.world.segments:
        axis.plot([wall.x1, wall.x2], [wall.y1, wall.y2], color="#334155", linewidth=0.6)
    axis.plot(result.ground_truth[:, 0], result.ground_truth[:, 1], color="#2563eb", label="Ground truth")
    axis.plot(
        result.estimated[:, 0],
        result.estimated[:, 1],
        color="#16a34a",
        linestyle="--",
        label="Estimated trajectory",
    )
    axis.set(
        xlabel="x [m]",
        ylabel="y [m]",
        title=f"SLAMCart3D observed occupancy ({grid.resolution_m:g} m cells)",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2)
    axis.legend(loc="best", fontsize=8)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def save_robustness_map(study: object, path: str | Path) -> Path:
    """Save a two-variable failure-probability surface and ASRF samples."""

    destination = Path(path)
    _prepare_plot_cache(destination)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.5, 6.0), constrained_layout=True)
    image = axis.pcolormesh(
        study.x_axis,
        study.y_axis,
        study.success_probability,
        shading="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="RdYlGn",
    )
    if np.min(study.success_probability) <= 0.5 <= np.max(study.success_probability):
        axis.contour(study.x_axis, study.y_axis, study.success_probability, levels=[0.5], colors="black")
    colors = np.where(study.asrf.failed, "#b2182b", "#2166ac")
    axis.scatter(study.asrf.samples[:, 0], study.asrf.samples[:, 1], c=colors, edgecolor="white", s=30)
    axis.set(
        xlabel=PARAMETER_LABELS.get(study.parameter_names[0], study.parameter_names[0]),
        ylabel=PARAMETER_LABELS.get(study.parameter_names[1], study.parameter_names[1]),
        title=f"SLAM robustness map: P({_threshold_label(study.metric_name, study.failure_threshold)})",
    )
    figure.colorbar(image, ax=axis, label="Estimated success probability [1] (uncalibrated)")
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def save_uncertainty_map(study: object, path: str | Path) -> Path:
    """Save the approximate kernel-posterior uncertainty surface."""

    destination = Path(path)
    _prepare_plot_cache(destination)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.5, 6.0), constrained_layout=True)
    image = axis.pcolormesh(
        study.x_axis,
        study.y_axis,
        study.failure_uncertainty,
        shading="auto",
        cmap="magma",
    )
    axis.scatter(study.asrf.samples[:, 0], study.asrf.samples[:, 1], color="cyan", edgecolor="black", s=25)
    axis.set(
        xlabel=PARAMETER_LABELS.get(study.parameter_names[0], study.parameter_names[0]),
        ylabel=PARAMETER_LABELS.get(study.parameter_names[1], study.parameter_names[1]),
        title="Approximate failure-boundary uncertainty",
    )
    figure.colorbar(image, ax=axis, label="Approximate posterior standard deviation [1]")
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def save_scene_3d(result: RunResult, path: str | Path) -> Path:
    """Render vertical world surfaces and planar trajectories in 3D."""

    destination = Path(path)
    _prepare_plot_cache(destination)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    figure = plt.figure(figsize=(8.0, 6.5), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    final_segments = result.world.segments_at(float(result.timestamps[-1]))
    for index, wall in enumerate(final_segments):
        quad = [
            (wall.x1, wall.y1, 0.0),
            (wall.x2, wall.y2, 0.0),
            (wall.x2, wall.y2, wall.height_m),
            (wall.x1, wall.y1, wall.height_m),
        ]
        axis.add_collection3d(
            Poly3DCollection(
                [quad],
                facecolors="#a855f7" if index >= len(result.world.segments) else "#8a969f",
                edgecolors="#3f4a53",
                alpha=0.55 if index >= len(result.world.segments) else 0.45,
            )
        )
    axis.plot(result.ground_truth[:, 0], result.ground_truth[:, 1], 0.12, label="Ground truth", linewidth=2.2)
    axis.plot(
        result.estimated[:, 0],
        result.estimated[:, 1],
        0.17,
        label=str(result.diagnostics["algorithm"]).replace("_", " "),
        linestyle="--",
    )
    axis.scatter(*result.ground_truth[0, :2], 0.12, color="green", s=45, label="Start")
    axis.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title="SLAMCart3D extruded indoor world")
    x_values = [coordinate for wall in final_segments for coordinate in (wall.x1, wall.x2)]
    y_values = [coordinate for wall in final_segments for coordinate in (wall.y1, wall.y2)]
    x_values.extend(result.ground_truth[:, 0])
    y_values.extend(result.ground_truth[:, 1])
    x_values.extend(result.estimated[:, 0])
    y_values.extend(result.estimated[:, 1])
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_span, y_span = max(x_max - x_min, 1.0), max(y_max - y_min, 1.0)
    z_max = max(wall.height_m for wall in final_segments) + 0.5
    axis.set_xlim(x_min - 0.05 * x_span, x_max + 0.05 * x_span)
    axis.set_ylim(y_min - 0.05 * y_span, y_max + 0.05 * y_span)
    axis.set_zlim(0.0, z_max)
    axis.set_box_aspect((x_span, y_span, z_max), zoom=0.9)
    axis.view_init(elev=30, azim=-55)
    axis.legend(loc="upper left")
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def save_point_cloud_3d(result: RunResult, path: str | Path) -> Path:
    """Plot a measured 3D run map located with estimated planar robot poses."""

    if not result.config.lidar3d.enabled:
        raise ValueError("a 3D point-map figure requires an enabled multilayer LiDAR")
    destination = Path(path)
    _prepare_plot_cache(destination)
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(8.4, 6.8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    cloud = result.map_points_3d
    if len(cloud):
        sample = cloud[:: max(1, (len(cloud) + 7_999) // 8_000)]
        points = axis.scatter(
            sample[:, 0],
            sample[:, 1],
            sample[:, 2],
            c=sample[:, 2],
            cmap="viridis",
            s=2.0,
            alpha=0.65,
            rasterized=True,
        )
        figure.colorbar(points, ax=axis, shrink=0.58, pad=0.08, label="Measured point height z [m]")
        xy = np.vstack((cloud[:, :2], result.ground_truth[:, :2], result.estimated[:, :2]))
        z_min = min(0.0, float(np.min(cloud[:, 2])))
        z_max = max(0.0, float(np.max(cloud[:, 2])))
    else:
        axis.text2D(0.05, 0.95, "No valid 3D returns", transform=axis.transAxes)
        xy = np.vstack((result.ground_truth[:, :2], result.estimated[:, :2]))
        z_min, z_max = 0.0, 1.0
    axis.plot(
        result.ground_truth[:, 0],
        result.ground_truth[:, 1],
        0.0,
        color="#16a34a",
        linewidth=2.2,
        label="Ground-truth path (z = 0 m)",
    )
    axis.plot(
        result.estimated[:, 0],
        result.estimated[:, 1],
        0.0,
        color="#ea580c",
        linestyle="--",
        linewidth=2.2,
        label="Estimated SE(2) path (z = 0 m)",
    )
    xy_min = np.min(xy, axis=0)
    xy_max = np.max(xy, axis=0)
    xy_span = np.maximum(xy_max - xy_min, 1.0)
    z_span = max(z_max - z_min, 1.0)
    axis.set_xlim(xy_min[0] - 0.05 * xy_span[0], xy_max[0] + 0.05 * xy_span[0])
    axis.set_ylim(xy_min[1] - 0.05 * xy_span[1], xy_max[1] + 0.05 * xy_span[1])
    axis.set_zlim(z_min - 0.05 * z_span, z_max + 0.05 * z_span)
    axis.set_box_aspect((xy_span[0], xy_span[1], z_span), zoom=0.9)
    axis.set(
        xlabel="x [m]",
        ylabel="y [m]",
        zlabel="z [m]",
        title="SLAMCart3D measured 3D LiDAR map (estimated planar pose)",
    )
    axis.view_init(elev=28, azim=-60)
    axis.legend(loc="upper left")
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def save_validation_curve(summary: list[dict[str, float | int | str]], path: str | Path) -> Path:
    """Plot mean boundary error and approximate 95% intervals by budget."""

    destination = Path(path)
    _prepare_plot_cache(destination)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    methods = ("grid", "random", "lhs", "asrf")
    for method in methods:
        rows = sorted(
            (row for row in summary if row["method"] == method), key=lambda row: int(row["requested_budget"])
        )
        budgets = np.array([int(row["requested_budget"]) for row in rows])
        means = np.array([float(row["boundary_classification_error_mean"]) for row in rows])
        lows = np.array([float(row["boundary_classification_error_ci95_low"]) for row in rows])
        highs = np.array([float(row["boundary_classification_error_ci95_high"]) for row in rows])
        label = method.upper() if method in {"lhs", "asrf"} else method.title()
        axis.plot(budgets, means, marker="o", label=label)
        axis.fill_between(budgets, np.maximum(lows, 0.0), np.minimum(highs, 1.0), alpha=0.15)
    axis.set(
        xlabel="Requested simulation budget [evaluations]",
        ylabel="Boundary classification error [fraction]",
        title="Known-frontier reconstruction across design seeds",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination
