"""Python API entry point."""

from __future__ import annotations

from pathlib import Path

from slamcart3d.core.config import ExperimentConfig
from slamcart3d.io.export import export_result
from slamcart3d.simulation.runner import RunResult, run_simulation
from slamcart3d.visualization.plots import (
    save_occupancy_figure,
    save_point_cloud_3d,
    save_run_figure,
    save_scene_3d,
)


class Experiment:
    """A reproducible experiment executable from Python, CLI, or a future GUI."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    @classmethod
    def from_file(cls, path: str | Path) -> "Experiment":
        return cls(ExperimentConfig.load(path))

    def run(self) -> RunResult:
        return run_simulation(self.config)

    def run_and_export(
        self, output_root: str | Path = "results", plot: bool = False, figure_format: str = "png"
    ) -> Path:
        if figure_format not in {"png", "svg", "pdf"}:
            raise ValueError("figure_format must be png, svg, or pdf")
        result = self.run()
        target = export_result(result, output_root)
        if plot:
            save_run_figure(result, target / "figures" / f"trajectory_map.{figure_format}")
            save_scene_3d(result, target / "figures" / f"scene_3d.{figure_format}")
            if result.occupancy_grid is not None:
                save_occupancy_figure(result, target / "figures" / f"occupancy_grid.{figure_format}")
            if result.config.lidar3d.enabled:
                save_point_cloud_3d(result, target / "figures" / f"point_cloud_3d.{figure_format}")
        return target
