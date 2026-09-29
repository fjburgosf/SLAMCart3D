import numpy as np

from slamcart3d.core.config import ExperimentConfig, LidarConfig, SimulationConfig
from slamcart3d.metrics.trajectory_geometry import static_wall_contact_steps
from slamcart3d.simulation.runner import run_simulation
from slamcart3d.worlds.geometry import WallSegment, World3D


def test_static_wall_contact_counts_crossing_steps_not_walls() -> None:
    world = World3D(
        "cross",
        (
            WallSegment(2.0, 0.0, 2.0, 2.0),
            WallSegment(2.0, 1.0, 3.0, 1.0),
        ),
    )
    path = np.array([[1.0, 1.0], [3.5, 1.0], [3.5, 2.0]])
    assert static_wall_contact_steps(path, world) == 1
    assert static_wall_contact_steps(np.array([[0.0, 0.0], [1.0, 0.0]]), world) == 0


def test_stationary_point_inside_diagonal_bounding_box_is_not_a_contact() -> None:
    world = World3D("diagonal", (WallSegment(0.0, 0.0, 2.0, 2.0),))
    assert static_wall_contact_steps(np.array([[1.0, 0.5], [1.0, 0.5]]), world) == 0
    assert static_wall_contact_steps(np.array([[1.0, 1.0], [1.0, 1.0]]), world) == 1


def test_complex_world_flags_curved_wall_crossing_but_not_corridor_path() -> None:
    base = ExperimentConfig(
        lidar=LidarConfig(beam_count=8),
        simulation=SimulationConfig(
            duration_s=8.0,
            timestep_s=0.1,
            world="indoor_complex",
            linear_velocity_m_s=0.35,
            yaw_rate_rad_s=0.0,
        ),
    )
    assert run_simulation(base).metrics["ground_truth_static_wall_contact_steps"] == 0.0
    curved = ExperimentConfig(
        lidar=base.lidar,
        simulation=SimulationConfig(
            duration_s=8.0,
            timestep_s=0.1,
            world="indoor_complex",
            linear_velocity_m_s=0.35,
            yaw_rate_rad_s=0.18,
            stop_at_walls=False,
        ),
    )
    assert run_simulation(curved).metrics["ground_truth_static_wall_contact_steps"] > 0.0
