"""Command-line interface for headless reproducible operation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from slamcart3d import __version__
from slamcart3d.core.config import ExperimentConfig
from slamcart3d.datasets.generator import generate_dataset
from slamcart3d.datasets.replay import export_replay, replay_recording
from slamcart3d.experiments.batch import export_batch, run_batch
from slamcart3d.experiments.experiment import Experiment
from slamcart3d.experiments.hybrid_validation import (
    DEFAULT_SCENARIOS,
    METHODS,
    export_hybrid_benchmark,
    run_hybrid_benchmark,
)
from slamcart3d.experiments.sweep import SWEEP_PARAMETERS, export_parameter_sweep, run_parameter_sweep
from slamcart3d.metrics.direction import HIGHER_IS_BETTER, LOWER_IS_BETTER
from slamcart3d.robustness.benchmark import compare_sampling_methods, run_asrf_ablation, validate_asrf
from slamcart3d.robustness.real_comparison import compare_real_sampling, export_real_comparison
from slamcart3d.robustness.study import (
    SIMULATION_PARAMETER_NAMES,
    STUDY_DEGRADATION_PARAMETER_NAMES,
    export_frontier_study,
    run_frontier_study,
)
from slamcart3d.robustness.validation import export_validation_suite, run_validation_suite


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slamcart3d", description="Scientific SLAM robustness framework")
    parser.add_argument("--version", action="version", version=f"SLAMCart3D {__version__}")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one YAML/JSON experiment")
    run.add_argument("config", type=Path)
    run.add_argument("--output", type=Path, default=Path("results"))
    run.add_argument(
        "--plot", action="store_true", help="save 2D/3D run figures (requires visualization extra)"
    )
    run.add_argument(
        "--figure-format",
        choices=("png", "svg", "pdf"),
        default="png",
        help="format for figures saved by --plot (default: png)",
    )
    dataset = commands.add_parser("generate-dataset", help="save synthetic observations without running SLAM")
    dataset.add_argument("config", type=Path)
    dataset.add_argument("--output", type=Path, default=Path("datasets"))
    replay = commands.add_parser("replay", help="run measured CSV or CARMEN FLASER; reference is optional")
    replay.add_argument("recording", type=Path)
    replay.add_argument("config", type=Path)
    replay.add_argument("--reference", type=Path)
    replay.add_argument("--frames", type=int, default=1000)
    replay.add_argument("--timestamp-policy", choices=("strict", "sort_unique"), default="strict")
    replay.add_argument("--output", type=Path, default=Path("results"))
    batch = commands.add_parser("batch", help="compare algorithms and worlds over independent seeds")
    batch.add_argument("config", type=Path)
    batch.add_argument("--seeds", default="0,1,2,3,4")
    batch.add_argument("--algorithms", default="dead_reckoning,icp_odometry,wheel_imu_fusion,adaptive_hybrid")
    batch.add_argument("--worlds", default="indoor_simple")
    batch.add_argument("--workers", default="1", help="positive integer or auto")
    batch.add_argument("--output", type=Path, default=Path("results"))
    sweep = commands.add_parser("sweep", help="vary one condition over paired seeds and report a metric")
    sweep.add_argument("config", type=Path)
    sweep.add_argument("--parameter", choices=SWEEP_PARAMETERS, default="lidar_noise_std_m")
    sweep.add_argument("--min", type=float, default=0.0)
    sweep.add_argument("--max", type=float, default=0.4)
    sweep.add_argument("--levels", type=int, default=5)
    sweep.add_argument("--seeds", default="0,1,2,3,4")
    sweep.add_argument(
        "--metric", choices=("ate_rmse_m", "rpe_translation_rmse_m", "map_f1_0p25"), default="ate_rmse_m"
    )
    sweep.add_argument("--threshold", type=float, default=0.5)
    sweep.add_argument("--output", type=Path, default=Path("results"))
    hybrid = commands.add_parser(
        "benchmark-hybrid", help="compare local SLAM methods and hybrid ablations under paired conditions"
    )
    hybrid.add_argument("config", type=Path)
    hybrid.add_argument("--seeds", default="5,6,7,8,9")
    hybrid.add_argument("--scenarios", default=",".join(s.name for s in DEFAULT_SCENARIOS))
    hybrid.add_argument("--methods", default=",".join(METHODS))
    hybrid.add_argument("--output", type=Path, default=Path("results"))
    hybrid.add_argument("--plot", action="store_true")
    benchmark = commands.add_parser("validate-asrf", help="validate ASRF against a known synthetic frontier")
    benchmark.add_argument("--budget", type=int, default=50)
    benchmark.add_argument("--seed", type=int, default=42)
    comparison = commands.add_parser("compare-asrf", help="compare samplers and run ASRF ablations")
    comparison.add_argument("--budget", type=int, default=50)
    comparison.add_argument("--seed", type=int, default=42)
    validation = commands.add_parser(
        "benchmark-asrf", help="repeat sampler comparisons over budgets and seeds"
    )
    validation.add_argument("--budgets", default="25,50,100")
    validation.add_argument("--seeds", default="0,1,2,3,4")
    validation.add_argument("--output", type=Path, default=Path("results"))
    validation.add_argument("--plot", action="store_true")
    study = commands.add_parser("robustness-study", help="run a real-simulation two-variable ASRF study")
    study.add_argument("config", type=Path)
    study.add_argument("--budget", type=int, default=20)
    study.add_argument("--seeds", default="0,1,2,3,4")
    parameters = list(STUDY_DEGRADATION_PARAMETER_NAMES) + list(SIMULATION_PARAMETER_NAMES)
    study.add_argument("--x-param", choices=parameters, default="wheel_rate_noise_std_rad_s")
    study.add_argument("--y-param", choices=parameters, default="odometry_dropout_probability")
    study.add_argument("--x-min", type=float, default=0.0)
    study.add_argument("--x-max", type=float, default=0.8)
    study.add_argument("--y-min", type=float, default=0.0)
    study.add_argument("--y-max", type=float, default=0.5)
    study.add_argument("--metric", choices=sorted(HIGHER_IS_BETTER | LOWER_IS_BETTER), default="ate_rmse_m")
    study.add_argument("--threshold", type=float, default=0.05)
    study.add_argument("--output", type=Path, default=Path("results"))
    study.add_argument("--plot", action="store_true", help="save the robustness map figure")
    real = commands.add_parser("compare-real", help="compare sampling methods using real SLAM simulations")
    real.add_argument("config", type=Path)
    real.add_argument("--budget", type=int, default=25)
    real.add_argument("--training-seeds", default="0,1,2,3,4")
    real.add_argument("--validation-seeds", default="5,6,7,8,9")
    real.add_argument("--grid-resolution", type=int, default=11)
    real.add_argument("--x-param", choices=parameters, default="wheel_rate_noise_std_rad_s")
    real.add_argument("--y-param", choices=parameters, default="odometry_dropout_probability")
    real.add_argument("--x-min", type=float, default=0.0)
    real.add_argument("--x-max", type=float, default=0.8)
    real.add_argument("--y-min", type=float, default=0.0)
    real.add_argument("--y-max", type=float, default=0.5)
    real.add_argument("--metric", choices=sorted(HIGHER_IS_BETTER | LOWER_IS_BETTER), default="ate_rmse_m")
    real.add_argument("--threshold", type=float, default=0.05)
    real.add_argument("--output", type=Path, default=Path("results"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")
    if args.command == "run":
        target = Experiment.from_file(args.config).run_and_export(
            args.output, plot=args.plot, figure_format=args.figure_format
        )
        logging.getLogger(__name__).info("results exported to %s", target)
        return 0
    if args.command == "generate-dataset":
        target = generate_dataset(Experiment.from_file(args.config).config, args.output)
        logging.getLogger(__name__).info("dataset exported to %s", target)
        return 0
    if args.command == "replay":
        result = replay_recording(
            args.recording,
            ExperimentConfig.load(args.config),
            reference_path=args.reference,
            max_frames=args.frames,
            timestamp_policy=args.timestamp_policy,
        )
        target = export_replay(result, args.output)
        logging.getLogger(__name__).info(
            "recorded-data result exported to %s; reference available: %s",
            target,
            result.reference is not None,
        )
        return 0
    if args.command == "batch":
        seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
        algorithms = tuple(value.strip() for value in args.algorithms.split(",") if value.strip())
        worlds = tuple(value.strip() for value in args.worlds.split(",") if value.strip())
        workers = "auto" if args.workers == "auto" else int(args.workers)
        result = run_batch(Experiment.from_file(args.config).config, seeds, algorithms, worlds, workers)
        target = export_batch(result, args.output)
        logging.getLogger(__name__).info("batch exported to %s", target)
        return 0
    if args.command == "sweep":
        seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
        levels = tuple(float(value) for value in np.linspace(args.min, args.max, args.levels))
        result = run_parameter_sweep(
            Experiment.from_file(args.config).config,
            args.parameter,
            levels,
            seeds,
            args.metric,
            args.threshold,
        )
        target = export_parameter_sweep(result, args.output)
        logging.getLogger(__name__).info("controlled study exported to %s", target)
        return 0
    if args.command == "benchmark-hybrid":
        seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
        names = tuple(value.strip() for value in args.scenarios.split(",") if value.strip())
        scenarios = tuple(s for s in DEFAULT_SCENARIOS if s.name in names)
        if len(scenarios) != len(names) or len(set(names)) != len(names):
            raise ValueError("scenarios must be unique names from the predefined benchmark")
        methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
        result = run_hybrid_benchmark(Experiment.from_file(args.config).config, scenarios, seeds, methods)
        target = export_hybrid_benchmark(result, args.output, plot=args.plot)
        logging.getLogger(__name__).info("hybrid benchmark exported to %s", target)
        return 0
    if args.command == "validate-asrf":
        metrics = validate_asrf(args.budget, args.seed)
        logging.getLogger(__name__).info("ASRF validation %s", json.dumps(metrics, sort_keys=True))
        return 0
    if args.command == "compare-asrf":
        report = {
            "comparison": compare_sampling_methods(args.budget, args.seed),
            "ablation": run_asrf_ablation(args.budget, args.seed),
        }
        logging.getLogger(__name__).info("ASRF comparison %s", json.dumps(report, sort_keys=True))
        return 0
    if args.command == "benchmark-asrf":
        budgets = tuple(int(value.strip()) for value in args.budgets.split(",") if value.strip())
        seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
        target = export_validation_suite(run_validation_suite(budgets, seeds), args.output, plot=args.plot)
        logging.getLogger(__name__).info("ASRF validation suite exported to %s", target)
        return 0
    if args.command == "compare-real":
        training = tuple(int(value.strip()) for value in args.training_seeds.split(",") if value.strip())
        validation = tuple(int(value.strip()) for value in args.validation_seeds.split(",") if value.strip())
        comparison = compare_real_sampling(
            Experiment.from_file(args.config).config,
            (args.x_param, args.y_param),
            np.array([args.x_min, args.y_min]),
            np.array([args.x_max, args.y_max]),
            args.metric,
            args.threshold,
            training,
            validation,
            args.budget,
            args.grid_resolution,
        )
        target = export_real_comparison(comparison, args.output)
        logging.getLogger(__name__).info("real-simulation comparison exported to %s", target)
        return 0
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    base = Experiment.from_file(args.config).config
    result = run_frontier_study(
        base,
        (args.x_param, args.y_param),
        np.array([args.x_min, args.y_min]),
        np.array([args.x_max, args.y_max]),
        args.metric,
        args.threshold,
        seeds,
        args.budget,
    )
    target = export_frontier_study(result, args.output, plot=args.plot)
    logging.getLogger(__name__).info("robustness study exported to %s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
