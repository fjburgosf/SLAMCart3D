# SLAMCart3D

SLAMCart3D is a reproducible scientific framework for studying how controlled sensor, motion, and environment degradation changes localization and mapping performance. Its central research question is where the transition between robust and failed operation lies, and how that frontier can be estimated with fewer simulations than uniform exploration.

Authors: Francisco Javier Burgos Flórez, Juan Pablo Hoyos Sánchez, and Juan Guillermo Popayán Hernández.

Repository: https://github.com/fjburgosf/SLAMCart3D

## Scientific specification

For experimental conditions `theta`, SLAMCart3D measures a performance margin `g(theta) = J(theta) - J_crit`. Negative values denote acceptable operation and non-negative values denote critical degradation. The proposed Adaptive SLAM Robustness Frontier (ASRF) uses a probabilistic radial-basis surrogate and sequential acquisition:

`a(theta) = 4 p_fail(theta) (1 - p_fail(theta)) + lambda d_nearest(theta)`

The first term concentrates evaluations near the estimated success/failure boundary. The second preserves domain coverage. The synthetic validator uses a mathematically known nonlinear frontier and reports classification error, false-negative rate, and false-positive rate. This initial formulation is a testable method, not evidence that the research hypothesis is true.

## Architecture

- `core`: typed configuration and shared models.
- `worlds`, `robots`, `sensors`: deterministic simulation components.
- `degradation`: noise, bias, dropout, and outlier injection that never alters ground truth.
- `slam`: replaceable algorithm interface, dead-reckoning, wheel/IMU-fusion odometry, and scan-to-scan ICP LiDAR odometry baselines.
- `metrics`: SE(2) alignment, ATE, RPE, final drift, and availability.
- `metrics/map_quality`: directed point-to-wall distances and 2D map precision, recall, and F1 against sampled ground-truth walls.
- `simulation`, `experiments`: headless orchestration and Python API.
- `robustness`: ASRF, Latin hypercube initialization, and known-frontier validation.
- `io`, `visualization`: traceable exports and optional figures, isolated from the simulation clock.

The current 0.1.0 development version simulates a differential-drive robot in six seeded, extruded 2.5D worlds (`indoor_simple`, `indoor_corridor`, `indoor_complex`, `warehouse_shelves`, `indoor_dynamic`, `outdoor_blocks`); ray-casts a planar LiDAR; samples a six-axis IMU constrained to planar motion; injects controlled degradation; builds a point map using dead reckoning, wheel/IMU fusion, or point-to-line ICP LiDAR odometry; evaluates trajectories; and exports results. The complex indoor world has connected rooms and a central corridor. The dynamic world includes one scripted oscillating object with no collision response. The outdoor world is a flat open plane with sparse building footprints, not a terrain or weather simulation. These algorithms are baselines without loop closure. The same deterministic observation stream also generates datasets without executing an algorithm.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[test,visualization,gui]"
```

An isolated Windows development build can be made with `.[build]` installed and
`.venv\Scripts\python scripts\build_windows.py`. Each build gets a new folder
under `dist/`; the script keeps all build and temporary files in this checkout.
For a diagnostic console build, add `--console`. The frozen application bundles
the example configurations and stores results beside the executable by default.
If that directory is protected, export fails with a permission error and does
not silently create results elsewhere; choose another output folder explicitly.
Run its non-writing smoke test with `SLAMCart3D.exe --self-test` (exit code 0
means success). This is a development build, not yet the final distribution ZIP.

## Quick start

```powershell
slamcart3d run configs/example_01_clean_environment.yaml --plot
slamcart3d-gui
slamcart3d run configs/example_02_lidar_noise.yaml --plot
slamcart3d run configs/example_05_algorithm_comparison_icp.yaml --plot
slamcart3d run configs/example_03_imu_drift.yaml --plot
slamcart3d run configs/example_04_dynamic_environment.yaml --plot
slamcart3d run configs/example_outdoor_blocks.yaml --plot
slamcart3d run configs/example_indoor_complex.yaml --plot
slamcart3d run configs/example_occupancy_grid.yaml --plot
slamcart3d run configs/example_lidar_occlusion.yaml --plot
slamcart3d run configs/example_motion_stop_turn.yaml --plot
slamcart3d run configs/example_scripted_trajectory.yaml --plot
slamcart3d run configs/example_temporal_dropout.yaml --plot
slamcart3d run configs/example_spatial_dropout.yaml --plot
slamcart3d run configs/example_srs.yaml
slamcart3d generate-dataset configs/example_multilayer_lidar.yaml
slamcart3d generate-dataset configs/example_camera_rgbd.yaml
slamcart3d run configs/example_multilayer_lidar.yaml
slamcart3d run configs/example_multilayer_lidar.yaml --plot
slamcart3d run configs/example_multilayer_lidar.yaml --plot --figure-format pdf
slamcart3d batch configs/example_motion_stop_turn.yaml --seeds 0,1,2,3,4 --algorithms dead_reckoning,wheel_imu_fusion --worlds indoor_simple,indoor_corridor --workers 2
slamcart3d generate-dataset configs/example_01_clean_environment.yaml
slamcart3d validate-asrf --budget 50 --seed 42
slamcart3d compare-asrf --budget 50 --seed 42
slamcart3d benchmark-asrf --budgets 25,50,100 --seeds 0,1,2,3,4 --plot
slamcart3d robustness-study configs/example_07_asrf.yaml --budget 20 --seeds 0,1,2,3,4 --plot
slamcart3d compare-real configs/example_07_asrf.yaml --budget 25 --training-seeds 0,1,2,3,4 --validation-seeds 5,6,7,8,9 --grid-resolution 11
.venv\Scripts\python.exe scripts\run_workspace_checks.py
```

Python API:

```python
from slamcart3d import Experiment

result = Experiment.from_file("configs/example_01_clean_environment.yaml").run()
print(result.metrics)
```

Each exported run records its configuration, seed, software version, SHA-256 fingerprint of the package source, timestamp, trajectory, metrics, point map, runtime, and unique experiment identifier under `results/EXP-.../`. Dataset, batch, and robustness-study metadata carry the same source fingerprint.

`generate-dataset` writes ground truth, measured wheel rates, ideal and degraded LiDAR ranges, beam angles, optional ideal/measured six-axis IMU channels, configuration, and provenance to `datasets/DATASET-.../`. Dropped measurements are empty CSV fields. For the level-ground robot, ideal angular rates are `(0, 0, yaw_rate)` and body-frame specific force is `(forward acceleration, speed × yaw_rate, 9.80665 m/s²)`. The latter is a fixed-gravity, planar-kinematic approximation, not a full rigid-body inertial model or 6DoF pose ground truth. Gyroscope and accelerometer white noise, bias random walk, saturation, and packet dropout affect the measured channels; the configured fixed gyro bias applies to yaw and fixed accelerometer bias to forward acceleration, while auxiliary axes start with zero fixed bias. Auxiliary axes use their own random sequence so the legacy yaw/forward stream remains reproducible. The wheel/IMU baseline uses only gyroscope yaw rate and falls back to wheel yaw on IMU dropout. Run and robustness figures are requested with `--plot`; the core installation runs headless without Matplotlib.

When `lidar3d.enabled` is true, the same observation stream also casts a multielevation LiDAR against finite-height walls and flat ground. Dataset `lidar3d.csv` contains each ideal/degraded slant range plus local-sensor and world XYZ coordinates; those world coordinates use the known ground-truth robot pose. Missing and no-hit points have blank coordinates. Sensor noise can place a measured point slightly beyond an ideal surface, including just below the nominal ground plane. A simulation run separately accumulates the measured 3D returns using the **estimated planar pose** and fixed sensor height in `map/point_cloud_3d.csv`; the Qt 3D tab can rotate and zoom this cloud alongside wall outlines and both trajectories. The optional sensor leaves planar LiDAR/odometry/IMU random streams unchanged. The current SLAM baselines still consume the planar scan; the optional wall-surface metrics assess the 3D cloud against known static walls while excluding ground-like returns. This is not a 6DoF 3D SLAM algorithm or a fully dynamic 3D physics engine.

For a multilayer run, `--plot` also writes `figures/point_cloud_3d.png`: measured 3D returns colored by height, with ground-truth and estimated planar trajectories shown at ground level. The image is computed from that run's data, not from a GUI screenshot. `figures/scene_3d.png` separately shows the extruded world geometry and trajectories. Use `--figure-format svg` or `--figure-format pdf` for vector figures instead of the default PNG.

With multilayer LiDAR enabled, additional `map3d_wall_*` metrics assess the estimated-pose cloud against **static vertical wall surfaces**. Returns at or below 0.1 m height are excluded as ground-like; reference wall sampling also begins above that height. Accuracy is the mean exact point-to-wall-rectangle distance; completeness is the mean sampled-wall-to-cloud distance (wall surface spacing 0.2 m); precision, recall, and F1 use a 0.25 m threshold. Unseen walls lower recall, and moving-object returns can lower precision. These metrics do not evaluate the ground surface, camera reconstruction, or 6DoF pose. Their reference and thresholds are recorded in run metadata.

The optional RGB-D camera casts perspective rays into the same 2.5D geometry and assigns deterministic schematic colors to walls, ground, and sky. It is not photorealistic and the current SLAM baselines do not use its images. Camera conditions are therefore not ASRF axes yet. `generate-dataset` saves ideal/measured RGB PNGs, metric ray-depth NumPy arrays, and `camera_index.csv`; a dropped frame has blank measured paths. Camera degradations include RGB/depth noise, box blur, lighting multiplier, global contrast reduction, rectangular occlusion, and frame dropout. Frame dropout follows the configured temporal and spatial profiles. The camera uses an independent seeded random stream, so enabling it does not change LiDAR, IMU, or wheel samples. The CPU ray caster limits images to 160 × 120 pixels. `example_camera_rgbd.yaml` is a reproducible camera dataset configuration.

The optional SLAM Robustness Score (SRS) is **disabled by default**. When enabled, it retains ATE, translational RPE, 2D map F1, and tracking availability as separate metrics, while adding an index in [0, 1]. Error quality is `scale / (scale + error)`, so a configured scale represents half quality. F1 and availability quality are `clip((value - floor)/(1 - floor), 0, 1)`. SRS is the weighted mean of the four qualities; the default weights are equal. The run also exports each quality and the exact local derivative `∂SRS/∂wᵢ = (qualityᵢ - SRS)/Σw` for weight sensitivity. Scales, floors, and weights are part of the saved configuration, not universal safety thresholds. Runtime is deliberately excluded to keep this score reproducible across machines. ASRF can use SRS with a user-chosen failure threshold only when the score is enabled. `example_srs.yaml` demonstrates the option.

LiDAR degradation parameters include per-beam noise, bias, random dropout, outliers, effective-range reduction, and a missing angular sector in the robot frame. Out-of-range and sector-masked beams are missing observations, not obstacle points.

An ideal range equal to the configured LiDAR maximum means no wall was hit. The measured stream exports it as a missing field even when range noise is enabled; this avoids false distant map points in open worlds.

Motion profiles are `constant` (the default), `sinusoidal_yaw` (cosine yaw-rate modulation at `motion_frequency_hz`), `stop_turn` (alternating translation and in-place turns), and `scripted`. Scripted motion uses an ordered YAML `motion_segments` list; each segment specifies `duration_s`, `linear_velocity_m_s`, and `yaw_rate_rad_s`. Every segment duration must be a positive multiple of `timestep_s`, and the segments must exactly cover `duration_s`. The GUI can load, preserve, and run a scripted configuration, but segment editing is currently done in YAML. `example_scripted_trajectory.yaml` executes four translation/turn segments and, with zero wheel noise, yields final ground truth `(5.4 m, 4.4 m, 0 rad)` and zero ATE in the wheel-odometry baseline. These are deterministic kinematic commands, not collision-aware navigation or rigid-body dynamics.

Every run now reports `ground_truth_static_wall_contact_steps`: the number of consecutive sampled robot-center chords touching at least one static wall. The value is zero in the straight-corridor `example_indoor_complex.yaml` and `example_occupancy_grid.yaml` examples. A positive value warns that the prescribed path may be physically invalid, because the simulator does not stop or redirect the robot at walls. This sampled-centerline diagnostic does not account for chassis radius, moving objects, or collisions between observation instants.
All 18 shipped example configurations are exercised by integration tests and currently report zero sampled-centerline/static-wall contacts; that check does not establish collision-free chassis motion.

Dropout can also change over simulated time. `degradation.dropout_time_profile` is `constant` (default), `linear_ramp` (zero to the configured LiDAR/odometry/IMU dropout probabilities), or `abrupt` (zero before `dropout_transition_fraction`, configured probabilities thereafter). Time is normalized from the first to last simulation step; sensor noise and bias remain constant under this profile. The schedule affects only measurements, never ground truth. `example_temporal_dropout.yaml` demonstrates a reproducible ramp.

Dropout may additionally be confined to a rectangular world-coordinate region using `dropout_region_enabled` and `dropout_region_{x,y}_{min,max}_m`. Its boundaries are inclusive. Outside this region the configured LiDAR (planar and multilayer), odometry, and IMU packet-dropout probabilities are zero; inside, the temporal profile applies normally. The region is evaluated against ground-truth robot position so it does not move with localization error. Noise and bias are unaffected. The GUI exposes the zone switch and bounds; `example_spatial_dropout.yaml` illustrates the setting.

Map precision, recall, and F1 use a 0.25 m wall-matching tolerance. Accuracy and completeness are directed mean nearest-neighbor distances; their average is the reported Chamfer distance. Unobserved walls reduce recall, so these metrics combine mapping quality with trajectory coverage. They assess a 2D point map, not 3D reconstruction. ASRF also accepts `simulation.linear_velocity_m_s`, `simulation.yaw_rate_rad_s`, and `simulation.motion_frequency_hz` as study axes. The first two do not apply to scripted motion because segment commands override them, and the frequency axis requires sinusoidal motion. Failure thresholds respect metric direction: low error is good, while high availability or map F1 is good.

An optional sparse 2D occupancy map uses valid planar LiDAR hits and estimated poses. It marks hit cells occupied, ray interiors free, and all other cells unknown; missing and no-hit beams never imply free space. `occupancy_grid.enabled` is off by default, with configurable resolution (0.05–1.0 m) and beam stride. The static wall reference marks every cell touched by a wall, including both sides of exact grid boundaries. Occupied precision, recall, F1, and IoU compare the estimated occupied cells against **all** reference-wall cells, so unseen rooms lower recall. Observed-free accuracy checks only cells marked free by measured rays, not free-space recall. Finite resolution can make a physically free ray share a wall-touching cell, so even ideal poses need not score 1.0 on this metric. This categorical grid is not a calibrated occupancy probability model. Exports include `map/occupancy_grid.csv` with observed/free/unknown states; `--plot` adds `figures/occupancy_grid.png`. `example_occupancy_grid.yaml` keeps the robot in the central corridor because collision response is not modeled. ASRF can use occupied IoU only when the grid is enabled.

Map metrics use the static walls as reference. In `indoor_dynamic`, returns from the moving object therefore count as non-static map points; the viewer and saved figures show its final position only, while LiDAR ranges were computed from its position at each observation time.

`tracking_availability` means the fraction of finite pose estimates; a baseline can still report finite poses while localization is poor. For ICP, `icp_registration_availability` separately reports the fraction of post-initial scans registered instead of falling back to wheel prediction.

`batch` evaluates the Cartesian product of selected worlds, algorithms, and seeds. `--workers 1` runs serially; `--workers auto` or a positive integer uses independent processes. The CSV row order and scientific outputs are independent of worker count; runtime is not. Each batch keeps full individual run exports under `results/BATCH-.../runs/`, plus `runs.csv`, `summary.csv`, `paired_effects.csv`, and provenance metadata. The summary includes mean, median, sample standard deviation, IQR, and a normal-approximation 95% interval; interpret the latter cautiously with few seeds. `paired_effects.csv` compares algorithms within each world at the same seeds, reporting algorithm A minus B differences and paired Cohen's dz (mean difference divided by sample SD). An undefined dz is blank. With few seeds, these are descriptive effect sizes, not significance claims.

The optional Qt desktop interface loads/saves YAML, edits principal robot/sensor conditions, switches between Spanish and English without resetting values, and runs single experiments, serial multi-seed/algorithm/world batches, two-variable ASRF studies, or synthetic dataset generation in background threads. It displays trajectories, a point map, an optional observed occupancy grid, metrics, batch means, an exploratory success-probability map with sampled conditions, and a rotatable/zoomable estimated-pose 3D point cloud when the optional multilayer sensor is enabled. The occupancy grid, multilayer sensor, and schematic RGB-D camera can be switched on in the experiment form; their detailed geometry and degradation remain configurable in YAML. The dataset tab previews a measured RGB camera frame when available. Each batch or ASRF study exports its underlying runs/statistics; the GUI also saves the shown ASRF map as PNG. Advanced parameters remain in the YAML and are preserved when a configuration is loaded. Parallel batches remain available in the CLI. The GUI is an initial connected interface, not yet the complete multi-module workbench.

ASRF maps include the kernel-smoothed success/failure estimate and an approximate posterior standard deviation. These are exploratory surfaces; neither has yet been calibrated as a predictive probability or confidence interval. Each study exports its full configuration and individual seed results in `seed_runs.csv` alongside summaries and maps.

Each ASRF study also writes `sensitivity.csv` with tie-aware Spearman rank correlations between each sampled condition and the median selected metric. These associations describe only the adaptively sampled conditions: they are not causal effects, population-wide feature importance, or significance tests. An undefined correlation (constant axis or fewer than three finite pairs) is left blank.

`benchmark-asrf` repeats the mathematically known-frontier test across budgets and independent design seeds. It exports per-run Grid/Random/LHS/ASRF errors, summaries, ablations, and an optional convergence figure. Synthetic objective evaluations are cheap, so runtime differences there chiefly reflect sampling overhead.

`compare-real` compares the four sampling designs on actual SLAMCart3D runs with the same surrogate and threshold. Grid uses exactly the requested budget even when it is not a square number. Validation uses an independent set of seeds on a finite grid; this is an empirical approximation to a robustness frontier, not an exact ground truth. The command refuses a validation grid containing only success or only failure, where one of FNR/FPR would be undefined. It exports method classification scores, Brier score against all held-out binary seed outcomes, 10-bin expected calibration error (ECE10), a `reliability.csv` table, every training/validation condition, per-seed metric values, and full provenance. Brier and ECE10 are **diagnostics**, not evidence that the probability map is calibrated. Classification accuracy and probability quality can rank methods differently. Budget 8 exercises ASRF's initial LHS design only; use a larger budget (default 25) to test adaptive refinement. The computation can be expensive: validation alone runs `grid_resolution² × number of validation seeds` simulations.

To explore LiDAR robustness with ICP, select the ICP configuration and set two degradation axes explicitly:

```powershell
slamcart3d robustness-study configs/example_08_icp_lidar_frontier.yaml --x-param lidar_noise_std_m --x-max 0.25 --y-param lidar_dropout_probability --y-max 0.6 --threshold 0.05 --budget 20 --seeds 0,1,2,3,4 --plot
```

## Scope and next validation steps

This development version establishes an executable scientific path, repeated-seed statistics, synthetic comparison against Grid/Random/LHS, ASRF ablations, an ASRF robustness map driven by real simulations, a schematic RGB-D dataset camera, a planar-constrained six-axis IMU, an optional categorical occupancy grid, and an initial desktop GUI. It does not claim comparative superiority or implement loop closure, visual odometry, full 6DoF robot motion, collision-aware navigation, or a complete multi-module GUI. Those claims require further implementation and measured experiments.

Observed comparisons do not establish ASRF superiority. In one 25-condition-per-method dead-reckoning study with 5 held-out seeds on an 11 × 11 validation grid, classification error was 0.000 for Grid and 0.066 for ASRF; held-out Brier scores were 0.077 and 0.100 respectively (`REAL-COMPARISON-2C58789F42B4`). In a separate ICP/LiDAR study with a 9 × 9 grid, error was 0.086 for Grid and 0.099 for ASRF (`REAL-COMPARISON-2CD81507EC8C`). A third study used occupancy IoU, 12 conditions per method, 3 training and 3 held-out seeds, and a 7 × 7 grid: classification error was 0.245 for Random, 0.265 for Grid/LHS, and 0.388 for ASRF (`REAL-COMPARISON-83BC69CDF8D0`). At budget 12, ASRF has only four adaptive selections after its initial eight. These are finite-grid, scenario-specific estimates, not exact frontiers or safety guarantees. The known synthetic-frontier suite did favor ASRF at budgets 25, 50, and 100 across five design seeds, showing why both synthetic and real-simulation validation are retained.
