# SLAMCart3D

SLAMCart3D is a reproducible scientific framework for studying how controlled sensor, motion, and environment degradation changes localization and mapping performance. Its central research question is where the transition between robust and failed operation lies, and how that frontier can be estimated with fewer simulations than uniform exploration.

Authors: Francisco Javier Burgos Flórez and Juan Guillermo Popayán Hernández.

Contact: fjburgosf@gmail.com.

Repository: https://github.com/fjburgosf/SLAMCart3D

## Scientific specification

For experimental conditions `theta`, SLAMCart3D measures a performance margin `g(theta) = J(theta) - J_crit`. Negative values denote acceptable operation and non-negative values denote critical degradation. The proposed Adaptive SLAM Robustness Frontier (ASRF) uses a probabilistic radial-basis surrogate and sequential acquisition:

`a(theta) = 4 p_fail(theta) (1 - p_fail(theta)) + lambda d_nearest(theta)`

The first term concentrates evaluations near the estimated success/failure boundary. The second preserves domain coverage. The synthetic validator uses a mathematically known nonlinear frontier and reports classification error, false-negative rate, and false-positive rate. This initial formulation is a testable method, not evidence that the research hypothesis is true.

## Architecture

- `core`: typed configuration and shared models.
- `worlds`, `robots`, `sensors`: deterministic simulation components.
- `degradation`: noise, bias, dropout, and outlier injection that never alters ground truth.
- `slam`: three localization/mapping baselines, a local adaptive hybrid, its scan-verified pose-graph extension, and an XYZ scan-to-map hybrid.
- `mapping`: measured ray evidence in sparse 3D voxels and exportable boundary triangles.
- `navigation`: measured-grid frontier exploration with conservative obstacle inflation.
- `datasets/replay`: measured CSV and legacy CARMEN FLASER playback, with optional independent reference evaluation.
- `metrics`: SE(2) alignment, ATE, RPE, final drift, and availability.
- `metrics/map_quality`: directed point-to-wall distances and 2D map precision, recall, and F1 against sampled ground-truth walls.
- `simulation`, `experiments`: headless orchestration and Python API.
- `robustness`: ASRF, Latin hypercube initialization, and known-frontier validation.
- `io`, `visualization`: traceable exports and optional figures, isolated from the simulation clock.

Version 1.0.0 simulates a differential-drive robot in seven seeded, extruded 2.5D worlds (`indoor_simple`, `indoor_corridor`, `indoor_complex`, `warehouse_shelves`, `indoor_maze`, `indoor_dynamic`, `outdoor_blocks`) or a custom world made from finite-height wall segments. It models measured wheels, planar and multilayer LiDAR and a planar-constrained six-axis IMU, runs six estimators, and exports trajectories and maps. The planar pose-graph hybrid adds scan-verified loop constraints and final optimization. The dynamic world has one scripted oscillating object; the outdoor world has flat ground and sparse building footprints. Static walls stop the cart by default using a simplified contact guard, not a physics engine. The same observation stream also generates datasets without executing SLAM.

The desktop 3D view separates true walls and cart from estimated returns and measured voxel surfaces. **Reconstruction only** hides ground-truth walls and cart; the gridded plane is a coordinate guide, not an inferred floor. Unknown areas are not filled. Enable reconstruction explicitly, select its voxel size and supporting-scan threshold, or load `example_hybrid_3d.yaml`. The timeline includes only evidence acquired by the selected step. `trajectory_live.csv`, `point_cloud_live.csv`, optional `point_cloud_3d_live.csv`, `trajectory_3d_live.csv`, and `status.json` are flushed while the experiment runs. Final exports add `trajectory_3d.csv`, `map/voxel_grid_3d.csv` and `map/reconstructed_surface.ply` when applicable. The PLY contains boundary triangles of occupied voxels in metres; it is a partial, block-shaped measured map, not a guaranteed watertight architectural model. A failed run retains partial files and records its state.

`hybrid_3d` registers measured multilayer XYZ returns to its own map using Huber-weighted point-to-plane ICP, wheel/IMU prediction, adaptive trust and map gating. Plane normals come from local PCA of measured XYZ; unsupported neighborhoods are rejected. Its pose is `[x,y,z,roll,pitch,yaw]`, with rotation `Rz(yaw) Ry(pitch) Rx(roll)`. A stronger soft prior keeps z/roll/pitch near level ground; measured XYZ can correct all six components. The simulated vehicle still has planar physical motion. This is not a validated unrestricted 6DoF system for slopes, drones or arbitrary terrain. This estimator has no 3D loop closure and no camera input. Existing five methods remain planar. Recorded CSV/FLASER replay currently has only planar LiDAR, so it rejects `hybrid_3d` explicitly. The XYZ extension is an implemented research candidate built on established registration methods, not a demonstrated novel or universally superior algorithm.

To modify the test scene in the GUI, select **Custom environment** and enter one wall per line in the 3D walls field as `x1, y1, x2, y2, height`, in metres. The same geometry can be supplied in YAML through `simulation.world: custom` and `simulation.world_segments`; see `configs/example_custom_room.yaml`. These wall surfaces are the known simulated environment. They are distinct from the estimated point cloud, which depends on sensor observations and pose error.

The proposed `adaptive_hybrid` method predicts planar motion from wheel rates and optional IMU yaw rate. It then registers a subsampled planar LiDAR scan to its accumulated point map with a robust residual and pose prior. Its online trust state is an exponential moving average of inlier fraction and registration residual; that state changes how strongly the next scan correction is trusted. Unreliable registrations fall back to motion prediction, and a map gate can exclude such scans. This adapts from observations, not labelled training data or ground truth, and is **not** a deep-learning model. The method is local and has no loop closure. `benchmark-hybrid` compares it with the three baselines and ablations that disable IMU input, online trust, or map gating under predefined scenarios and paired seeds. Report all ATE, RPE, map F1, runtime, and paired differences; no single metric or scenario establishes general superiority.

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
The desktop launcher keeps temporary files and caches in `.runtime/` beside the
executable (or in the current folder when launched from source). It does not
fall back to AppData or another output folder when this location is protected.
Run its smoke test with `SLAMCart3D.exe --self-test` (exit code 0 means success).
The test creates only local runtime directories and does not export experiments.
This is a development build, pending validation on a clean Windows installation.

## Quick start

Use **Tutorial (F1)** for an optional eight-step guide linked to the real interface.
It explains the 3D world, time slider, map quality and scientific comparisons in
Spanish or English. **Prepare 30 s practice** asks before replacing the parameters
with an autonomous exploration example and preserves the output folder. Reading
or navigating the guide never starts a calculation; press **Run experiment** yourself.
The guide can be closed and reopened at the same step during the current session.

In the desktop interface, use **Example configuration** to choose one of the 26 bundled
examples by its descriptive name, then press **Load example**. This replaces the form
parameters but does not start a calculation. Use **Workflow** to open single simulation,
batch comparison, ASRF robustness, a one-variable controlled study, dataset generation,
or recorded measurement replay. Choosing a workflow only opens its tab; run it with
that tab's action button. You can still browse your own YAML/JSON configuration files.
Example titles follow the selected interface language. Loading is disabled while a
background task is active, and bundled examples are located independently of the
directory from which the application was launched.

Choose scenes directly with **Experiment → Environment**. **Indoor maze** is a closed
15 × 12 m scene with 3 m wide passages and 2.5 m high walls. A seeded depth-first
generator connects all 20 cells without enclosed islands; junctions and dead ends
make it a mapping test, not an entrance-to-exit puzzle. The same seed reproduces
the layout; another seed produces another layout. Selecting a scene does not change
the motion profile, start a run, or replace previously displayed results.
For ready-to-run settings, select **Maze: autonomous exploration and 3D map** and
press **Load example**, then **Run experiment**. That example enables measured-frontier
exploration, the adaptive hybrid, multilayer LiDAR and occupancy mapping for 60 s
of simulated time. Exploration receives measurements and estimated poses, not the
maze's true walls or a precomputed solution. Completing 60 s does not guarantee
complete coverage. The floor grid is a scale reference, not a navigation constraint.

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
slamcart3d run configs/example_custom_room.yaml --plot
slamcart3d run configs/example_adaptive_hybrid.yaml --plot
slamcart3d run configs/example_autonomous_exploration.yaml --plot
slamcart3d run configs/example_loop_closure.yaml --plot
slamcart3d replay recording.clf configs/example_recorded_lidar.yaml --frames 1000 --timestamp-policy sort_unique --output results
slamcart3d sweep configs/example_adaptive_hybrid.yaml --parameter wheel_rate_noise_std_rad_s --min 0 --max 0.5 --levels 5 --seeds 0,1,2,3,4 --metric ate_rmse_m --threshold 0.05
slamcart3d benchmark-hybrid configs/example_adaptive_hybrid.yaml --seeds 5,6,7,8,9 --plot
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

When `lidar3d.enabled` is true, the observation stream also casts a multielevation LiDAR against finite-height walls and flat ground. Dataset `lidar3d.csv` contains ideal/degraded slant ranges and local-sensor and world XYZ coordinates; its world coordinates use the known ground-truth robot pose for sensor analysis. Missing and no-hit points have blank coordinates. Noise can place a measured point slightly beyond an ideal surface, including below the nominal ground plane. A simulation run instead places measured returns in `map/point_cloud_3d.csv` using its estimated transformation: planar pose and mounting height for the five planar methods, or the six-component pose for `hybrid_3d`. The planar methods consume the planar scan; `hybrid_3d` also registers measured multilayer XYZ. The optional sensor leaves existing planar LiDAR, odometry and IMU random streams unchanged. Wall-surface metrics compare returns with known static walls while excluding ground-like returns; they do not evaluate the exported voxel triangles. Physical motion remains on level ground, without unrestricted 6DoF or a full dynamic physics engine.

For a multilayer run, `--plot` also writes `figures/point_cloud_3d.png`: measured 3D returns colored by height, with ground-truth and estimated planar trajectories shown at ground level. The image is computed from that run's data, not from a GUI screenshot. `figures/scene_3d.png` separately shows the extruded world geometry and trajectories. Use `--figure-format svg` or `--figure-format pdf` for vector figures instead of the default PNG.

With multilayer LiDAR enabled, additional `map3d_wall_*` metrics assess the estimated-pose cloud against **static vertical wall surfaces**. Returns at or below 0.1 m height are excluded as ground-like; reference wall sampling also begins above that height. Accuracy is the mean exact point-to-wall-rectangle distance; completeness is the mean sampled-wall-to-cloud distance (wall surface spacing 0.2 m); precision, recall, and F1 use a 0.25 m threshold. Unseen walls lower recall, and moving-object returns can lower precision. These metrics do not evaluate the ground surface, camera reconstruction, or 6DoF pose. Their reference and thresholds are recorded in run metadata.

The optional RGB-D camera casts perspective rays into the same 2.5D geometry and assigns deterministic schematic colors to walls, ground, and sky. It is not photorealistic and the current SLAM baselines do not use its images. Camera conditions are therefore not ASRF axes yet. `generate-dataset` saves ideal/measured RGB PNGs, metric ray-depth NumPy arrays, and `camera_index.csv`; a dropped frame has blank measured paths. Camera degradations include RGB/depth noise, box blur, lighting multiplier, global contrast reduction, rectangular occlusion, and frame dropout. Frame dropout follows the configured temporal and spatial profiles. The camera uses an independent seeded random stream, so enabling it does not change LiDAR, IMU, or wheel samples. The CPU ray caster limits images to 160 × 120 pixels. `example_camera_rgbd.yaml` is a reproducible camera dataset configuration.

The optional SLAM Robustness Score (SRS) is **disabled by default**. When enabled, it retains ATE, translational RPE, 2D map F1, and tracking availability as separate metrics, while adding an index in [0, 1]. Error quality is `scale / (scale + error)`, so a configured scale represents half quality. F1 and availability quality are `clip((value - floor)/(1 - floor), 0, 1)`. SRS is the weighted mean of the four qualities; the default weights are equal. The run also exports each quality and the exact local derivative `∂SRS/∂wᵢ = (qualityᵢ - SRS)/Σw` for weight sensitivity. Scales, floors, and weights are part of the saved configuration, not universal safety thresholds. Runtime is deliberately excluded to keep this score reproducible across machines. ASRF can use SRS with a user-chosen failure threshold only when the score is enabled. `example_srs.yaml` demonstrates the option.

LiDAR degradation parameters include per-beam noise, bias, random dropout, outliers, effective-range reduction, and a missing angular sector in the robot frame. Out-of-range and sector-masked beams are missing observations, not obstacle points.

An ideal range equal to the configured LiDAR maximum means no wall was hit. The measured stream exports it as a missing field even when range noise is enabled; this avoids false distant map points in open worlds.

Prescribed motion profiles are `constant` (the default), `sinusoidal_yaw` (cosine yaw-rate modulation at `motion_frequency_hz`), `stop_turn` (alternating translation and in-place turns), and `scripted`. Scripted motion uses an ordered YAML `motion_segments` list with duration, forward speed and yaw rate. Durations must be positive multiples of `timestep_s` and sum to `duration_s`. The GUI preserves these segments; editing is done in YAML. `example_scripted_trajectory.yaml`, with zero wheel noise, finishes at ground truth `(5.4 m, 4.4 m, 0 rad)` with zero wheel-odometry ATE. Prescribed commands alone do not explore an environment.

## Exploration and loop closure

`motion_profile: frontier` instead chooses destinations from the boundary between observed free cells and unknown space. Its four-connected breadth-first search uses measured planar scans and the current **estimated** pose, never true poses or world geometry. Occupied cells are inflated by the cart clearance plus `navigation.safety_margin_m`. A measured-range guard stops forward motion near an obstacle; the separate simulator guard remains enabled. In the GUI, **Prepare autonomous exploration** selects the frontier profile, pose-graph hybrid, occupancy grid and multilayer sensor; then press **Run experiment**. This is finite-duration exploration, not a guarantee of complete mapping. Sensor loss stops motion; no reachable frontier causes a scanning turn, which can persist with sparse data or an inconsistent accumulated grid. State and selected destination appear below the 3D view.

`posegraph_hybrid` keeps the online adaptive estimate and verifies revisits by temporal separation, estimated spatial proximity, non-collinear scan geometry, point-to-line residual and bidirectional overlap. It anchors the initial pose and uses robust Gauss-Newton optimization **at completion**, then rebuilds approved 2D map scans, 3D returns and occupancy with corrected poses. `trajectory_online.csv` retains the online trajectory; `trajectory.csv` contains the final optimized one. Incremental files remain online and are not retroactively overwritten. This development implementation limits the graph to 120 keyframes; conservative gates can miss loops and repeated geometry can still create false matches. Lower graph cost does not imply lower true position error. Compare both online and final errors rather than assuming improvement.

The quality panel reports current **unaligned** position/heading error, wall distance and wall coverage, with explicit configured thresholds (`evaluation`). ATE and the legacy final-error metric use rigid SE(2) alignment and are labelled accordingly. Intermediate map assessment is sampled and refreshed every ten live steps; its assessment time is shown. Completed-run metrics use full data. Wall coverage uses a fixed 0.25 m matching tolerance and is distinct from accuracy. **Observed areas** colors only measured occupancy cells; gray is unknown, green observed free, red estimated occupied. **Color wall error** uses the known reference only for evaluation: green within the selected wall-distance limit, red outside, ground cyan. Hiding the reference disables that reference-dependent coloring; it does not change SLAM.

## Recorded measurements

The **Recorded data** tab and `replay` command process measured wheel/LiDAR CSV channels or legacy CARMEN FLASER logs through the selected estimator. They do not use ideal sensor channels, corrected laser-pose fields or simulated walls. CARMEN odometry is a motion input, **not ground truth**. The adapter declares a 180-degree forward fan and colocated laser; nonzero logged laser offsets require another calibration adapter. It approximates differential motion and reports lateral displacement not represented by that model. `example_recorded_lidar.yaml` sets the initial coordinate frame; its synthetic room is irrelevant to replay.

Times are strict by default. Explicit `--timestamp-policy sort_unique` sorts the selected CARMEN subset and keeps the first scan at duplicate times, recording inversions and discarded duplicates. It never changes the original file. Subset selection is the first `frames+1` file-order scans before sorting; the first scan initializes timing/odometry. Measured CSV timestamps remain strict. An optional `--reference reference.csv` accepts independent `timestamp_s,x_m,y_m,yaw_rad` in the same relative time and coordinate frame, with interpolation but no extrapolation. Without such a reference, accuracy metrics remain absent. Drawing a plausible map or successfully reading a real log is interoperability evidence, not validated localization accuracy.

For fair estimator comparisons with autonomous motion, save and replay the **same measurements**: feedback-dependent exploration can yield different real trajectories for different algorithms even with the same seed. The fixed-trajectory `benchmark-hybrid` refuses the frontier profile and compares eight methods/ablations across four conditions and five paired seeds (160 runs).

With `simulation.stop_at_walls: true` (default), a step is stopped if its centerline chord comes closer to a wall than a conservative circle enclosing the displayed chassis and wheels; `wall_blocked_steps` counts these events. The schematic chassis has length `7r`, width `0.9b`, and heights `0.8r` to `4.3r`, where `r` is wheel radius and `b` is wheelbase. Wheels touch `z=0`; their axle separation is `b`. The clearance radius is `max(hypot(3.5r, 0.45b), hypot(r, b/2 + 0.25r))`. This intentionally conservative model is not detailed contact physics, sliding or path planning. Walls are evaluated at the step timestamp, including scripted moving walls; their continuous swept motion is not resolved. An initial overlapping pose is not automatically repaired. Set the option to `false` only for intentional wall-penetration diagnostics. `ground_truth_static_wall_contact_steps` remains a sampled-centerline warning.

The 3D tab displays an orbitable perspective world with a metric gridded floor, shaded finite-height walls, an oriented chassis, ground-contact wheels, and an XYZ orientation triad. Reset and top-view controls recover the camera; perspective can be disabled for an orthographic measurement view. Wall cutaway lowers only near perimeter walls on screen to expose the interior: sensor rays and exported geometry still use their full configured heights. Hide the measured cloud to inspect the environment, or hide the true reference to inspect only the estimate. An estimated cart may drift outside the world; it is never clamped to ground truth. The map is a measured point cloud positioned using estimated planar poses, not a filled ground-truth mesh or full 6DoF reconstruction.

Dropout can also change over simulated time. `degradation.dropout_time_profile` is `constant` (default), `linear_ramp` (zero to the configured LiDAR/odometry/IMU dropout probabilities), or `abrupt` (zero before `dropout_transition_fraction`, configured probabilities thereafter). Time is normalized from the first to last simulation step; sensor noise and bias remain constant under this profile. The schedule affects only measurements, never ground truth. `example_temporal_dropout.yaml` demonstrates a reproducible ramp.

Dropout may additionally be confined to a rectangular world-coordinate region using `dropout_region_enabled` and `dropout_region_{x,y}_{min,max}_m`. Its boundaries are inclusive. Outside this region the configured LiDAR (planar and multilayer), odometry, and IMU packet-dropout probabilities are zero; inside, the temporal profile applies normally. The region is evaluated against ground-truth robot position so it does not move with localization error. Noise and bias are unaffected. The GUI exposes the zone switch and bounds; `example_spatial_dropout.yaml` illustrates the setting.

Map precision, recall, and F1 use a 0.25 m wall-matching tolerance. Accuracy and completeness are directed mean nearest-neighbor distances; their average is the reported Chamfer distance. Unobserved walls reduce recall, so these metrics combine mapping quality with trajectory coverage. They assess a 2D point map, not 3D reconstruction. ASRF also accepts `simulation.linear_velocity_m_s`, `simulation.yaw_rate_rad_s`, and `simulation.motion_frequency_hz` as study axes. The first two do not apply to scripted motion because segment commands override them, and the frequency axis requires sinusoidal motion. Failure thresholds respect metric direction: low error is good, while high availability or map F1 is good.

An optional sparse 2D occupancy map uses valid planar LiDAR hits and estimated poses. It marks hit cells occupied, ray interiors free, and all other cells unknown; missing and no-hit beams never imply free space. `occupancy_grid.enabled` is off by default, with configurable resolution (0.05–1.0 m) and beam stride. The static wall reference marks every cell touched by a wall, including both sides of exact grid boundaries. Occupied precision, recall, F1, and IoU compare the estimated occupied cells against **all** reference-wall cells, so unseen rooms lower recall. Observed-free accuracy checks only cells marked free by measured rays, not free-space recall. Finite resolution can make a physically free ray share a wall-touching cell, so even ideal poses need not score 1.0 on this metric. This categorical grid is not a calibrated occupancy probability model. Exports include `map/occupancy_grid.csv` with observed/free/unknown states; `--plot` adds `figures/occupancy_grid.png`. `example_occupancy_grid.yaml` keeps the robot in the central corridor. ASRF can use occupied IoU only when the grid is enabled.

Map metrics use the static walls as reference. In `indoor_dynamic`, returns from the moving object therefore count as non-static map points; the viewer and saved figures show its final position only, while LiDAR ranges were computed from its position at each observation time.

`tracking_availability` means the fraction of finite pose estimates; a baseline can still report finite poses while localization is poor. For ICP, `icp_registration_availability` separately reports the fraction of post-initial scans registered instead of falling back to wheel prediction.

`batch` evaluates the Cartesian product of selected worlds, algorithms, and seeds. `--workers 1` runs serially; `--workers auto` or a positive integer uses independent processes. The CSV row order and scientific outputs are independent of worker count; runtime is not. Each batch keeps full individual run exports under `results/BATCH-.../runs/`, plus `runs.csv`, `summary.csv`, `paired_effects.csv`, and provenance metadata. The summary includes mean, median, sample standard deviation, IQR, and a normal-approximation 95% interval; interpret the latter cautiously with few seeds. `paired_effects.csv` compares algorithms within each world at the same seeds, reporting algorithm A minus B differences and paired Cohen's dz (mean difference divided by sample SD). An undefined dz is blank. With few seeds, these are descriptive effect sizes, not significance claims.

The optional Qt desktop interface loads/saves YAML, edits principal robot/sensor conditions, switches between Spanish and English without resetting values, and runs single experiments, serial multi-seed/algorithm/world batches, one-variable controlled sweeps, two-variable ASRF studies, or synthetic dataset generation in background threads. It displays trajectories, a point map, an optional observed occupancy grid, metrics, batch means, an exploratory success-probability map with sampled conditions, and a rotatable/zoomable 3D scene. The 3D view now distinguishes the gridded floor, solid-height walls, oriented true/estimated carts, and estimated point cloud; its reference toggle hides the simulated truth without changing the estimator. The occupancy grid, multilayer sensor, and schematic RGB-D camera can be switched on in the experiment form; their detailed geometry and degradation remain configurable in YAML. The dataset tab previews a measured RGB camera frame when available. Each batch, sweep, or ASRF study exports its underlying runs/statistics and displayed graph. Advanced parameters remain in the YAML and are preserved when a configuration is loaded. Parallel batches and the full hybrid/ablation benchmark remain in the CLI. The GUI is an initial connected interface, not yet the complete multi-module workbench.

ASRF maps include the kernel-smoothed success/failure estimate and an approximate posterior standard deviation. These are exploratory surfaces; neither has yet been calibrated as a predictive probability or confidence interval. Each study exports its full configuration and individual seed results in `seed_runs.csv` alongside summaries and maps.

Each ASRF study also writes `sensitivity.csv` with tie-aware Spearman rank correlations between each sampled condition and the median selected metric. These associations describe only the adaptively sampled conditions: they are not causal effects, population-wide feature importance, or significance tests. An undefined correlation (constant axis or fewer than three finite pairs) is left blank.

`benchmark-asrf` repeats the mathematically known-frontier test across budgets and independent design seeds. It exports per-run Grid/Random/LHS/ASRF errors, summaries, ablations, and an optional convergence figure. Synthetic objective evaluations are cheap, so runtime differences there chiefly reflect sampling overhead.

`compare-real` compares the four sampling designs on actual SLAMCart3D runs with the same surrogate and threshold. Grid uses exactly the requested budget even when it is not a square number. Validation uses an independent set of seeds on a finite grid; this is an empirical approximation to a robustness frontier, not an exact ground truth. The command refuses a validation grid containing only success or only failure, where one of FNR/FPR would be undefined. It exports method classification scores, Brier score against all held-out binary seed outcomes, 10-bin expected calibration error (ECE10), a `reliability.csv` table, every training/validation condition, per-seed metric values, and full provenance. Brier and ECE10 are **diagnostics**, not evidence that the probability map is calibrated. Classification accuracy and probability quality can rank methods differently. Budget 8 exercises ASRF's initial LHS design only; use a larger budget (default 25) to test adaptive refinement. The computation can be expensive: validation alone runs `grid_resolution² × number of validation seeds` simulations.

To explore LiDAR robustness with ICP, select the ICP configuration and set two degradation axes explicitly:

```powershell
slamcart3d robustness-study configs/example_08_icp_lidar_frontier.yaml --x-param lidar_noise_std_m --x-max 0.25 --y-param lidar_dropout_probability --y-max 0.6 --threshold 0.05 --budget 20 --seeds 0,1,2,3,4 --plot
```

## Scope and next validation steps

The desktop 3D view uses shaded walls, a metric floor and ground-contact wheels. Its cutaway changes display only. Configuration loading rejects ambiguous types; failed exports retain their details and release the action buttons for retry. The packaged `--self-test` exercises configuration dialogs, all six background actions and their exports, exploration preparation, timeline navigation and 3D controls.

This development version includes repeatable simulation studies, hybrid ablations, ASRF, measured-grid exploration, final scan-verified loop optimization and recorded-data replay. It does not claim comparative superiority, exhaustive exploration or novel loop-closure theory. It does not implement visual odometry, full 6DoF robot motion, full contact physics or calibrated probabilistic occupancy. Clean-machine Windows validation and independent real-world accuracy evaluation remain pending.

Observed comparisons do not establish ASRF superiority. In one 25-condition-per-method dead-reckoning study with 5 held-out seeds on an 11 × 11 validation grid, classification error was 0.000 for Grid and 0.066 for ASRF; held-out Brier scores were 0.077 and 0.100 respectively (`REAL-COMPARISON-2C58789F42B4`). In a separate ICP/LiDAR study with a 9 × 9 grid, error was 0.086 for Grid and 0.099 for ASRF (`REAL-COMPARISON-2CD81507EC8C`). A third study used occupancy IoU, 12 conditions per method, 3 training and 3 held-out seeds, and a 7 × 7 grid: classification error was 0.245 for Random, 0.265 for Grid/LHS, and 0.388 for ASRF (`REAL-COMPARISON-83BC69CDF8D0`). At budget 12, ASRF has only four adaptive selections after its initial eight. These are finite-grid, scenario-specific estimates, not exact frontiers or safety guarantees. The known synthetic-frontier suite did favor ASRF at budgets 25, 50, and 100 across five design seeds, showing why both synthetic and real-simulation validation are retained.
