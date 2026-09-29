import json

import numpy as np
import pytest

from slamcart3d.core.config import (
    DegradationConfig,
    ExperimentConfig,
    Lidar3DConfig,
    LidarConfig,
    SimulationConfig,
)
from slamcart3d.experiments.experiment import Experiment
from slamcart3d.io.export import export_result


def test_vertical_slice_is_reproducible_and_exportable(tmp_path) -> None:
    config = ExperimentConfig(simulation=SimulationConfig(duration_s=1.0, timestep_s=0.1))
    first = Experiment(config).run()
    second = Experiment(config).run()
    np.testing.assert_allclose(first.ground_truth, second.ground_truth)
    np.testing.assert_allclose(first.estimated, second.estimated)
    assert first.metrics["ate_rmse_m"] >= 0.0
    target = export_result(first, tmp_path)
    assert (target / "trajectory.csv").exists()
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["seed"] == config.seed
    assert metadata["software_version"] == "0.1.0"
    assert len(metadata["source_sha256"]) == 64
    assert metadata["source_sha256"] == first.source_sha256
    assert metadata["map_evaluation"]["match_tolerance_m"] == 0.25
    assert first.metrics["map_chamfer_mean_m"] >= 0.0


def test_multilayer_run_exports_real_3d_figure(tmp_path) -> None:
    config = ExperimentConfig(
        lidar=LidarConfig(beam_count=8),
        lidar3d=Lidar3DConfig(enabled=True, azimuth_beams=3, elevation_beams=3),
        simulation=SimulationConfig(duration_s=0.2, timestep_s=0.1),
    )
    target = Experiment(config).run_and_export(tmp_path, plot=True)
    figure = target / "figures" / "point_cloud_3d.png"
    assert figure.is_file()
    assert figure.stat().st_size > 10_000
    assert (target / "figures" / "scene_3d.png").is_file()
    assert (target / "map" / "point_cloud_3d.csv").is_file()


def test_empty_multilayer_cloud_still_exports_explanatory_figure(tmp_path) -> None:
    config = ExperimentConfig(
        lidar=LidarConfig(beam_count=8),
        lidar3d=Lidar3DConfig(enabled=True, azimuth_beams=3, elevation_beams=3),
        degradation=DegradationConfig(lidar_dropout_probability=1.0),
        simulation=SimulationConfig(duration_s=0.2, timestep_s=0.1),
    )
    target = Experiment(config).run_and_export(tmp_path, plot=True)
    assert (target / "figures" / "point_cloud_3d.png").stat().st_size > 10_000
    assert (target / "map" / "point_cloud_3d.csv").read_text(encoding="utf-8").strip() == "x_m,y_m,z_m"


@pytest.mark.parametrize("figure_format", ("svg", "pdf"))
def test_multilayer_run_supports_vector_figure_formats(tmp_path, figure_format) -> None:
    config = ExperimentConfig(
        lidar=LidarConfig(beam_count=8),
        lidar3d=Lidar3DConfig(enabled=True, azimuth_beams=3, elevation_beams=3),
        simulation=SimulationConfig(duration_s=0.2, timestep_s=0.1),
    )
    target = Experiment(config).run_and_export(tmp_path, plot=True, figure_format=figure_format)
    for name in ("trajectory_map", "scene_3d", "point_cloud_3d"):
        figure = target / "figures" / f"{name}.{figure_format}"
        assert figure.stat().st_size > 1_000
        if figure_format == "pdf":
            assert figure.read_bytes().startswith(b"%PDF")
        else:
            assert b"<svg" in figure.read_bytes()[:1_000]


def test_unknown_figure_format_is_rejected_before_creating_result(tmp_path) -> None:
    with pytest.raises(ValueError, match="figure_format"):
        Experiment(ExperimentConfig()).run_and_export(tmp_path, plot=True, figure_format="bmp")
    assert not list(tmp_path.iterdir())
