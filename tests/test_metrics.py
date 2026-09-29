import numpy as np

from slamcart3d.metrics.trajectory import absolute_trajectory_error, align_se2, relative_pose_error


def test_alignment_removes_rigid_transform() -> None:
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.2]])
    angle = 0.4
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    estimate = reference.copy()
    estimate[:, :2] = reference[:, :2] @ rotation + [3.0, -2.0]
    estimate[:, 2] += angle
    aligned = align_se2(reference, estimate)
    np.testing.assert_allclose(aligned, reference, atol=1e-10)
    assert absolute_trajectory_error(reference, estimate) < 1e-10


def test_rpe_is_zero_for_identical_trajectories() -> None:
    trajectory = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert relative_pose_error(trajectory, trajectory) == (0.0, 0.0)
