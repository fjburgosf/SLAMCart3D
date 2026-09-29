import json

from slamcart3d.core.config import ExperimentConfig, SimulationConfig, SLAMConfig
from slamcart3d.experiments.experiment import Experiment
from slamcart3d.io.export import export_result


def test_one_step_icp_export_has_valid_json_without_nan(tmp_path) -> None:
    config = ExperimentConfig(
        simulation=SimulationConfig(duration_s=0.1, timestep_s=0.1),
        slam=SLAMConfig(algorithm="icp_odometry"),
    )
    result = Experiment(config).run()
    destination = export_result(result, tmp_path)
    metadata = json.loads(
        (destination / "metadata.json").read_text(encoding="utf-8"), parse_constant=_reject_nonfinite
    )
    assert metadata["diagnostics"]["last_icp_rmse_m"] is None


def _reject_nonfinite(value: str) -> None:
    raise AssertionError(f"non-finite JSON value: {value}")
