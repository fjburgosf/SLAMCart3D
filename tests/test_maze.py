"""Maze topology and sensing, checked against the actual finite-height walls."""

from collections import deque

import numpy as np
import pytest

from slamcart3d.core.config import ExperimentConfig
from slamcart3d.robots.geometry import cart_clearance_radius
from slamcart3d.worlds.geometry import make_world


def maze_graph(world):
    centers = {(x, y): (3.0 * x + 1.5, 3.0 * y + 1.5) for y in range(4) for x in range(5)}
    graph = {cell: set() for cell in centers}
    for (x, y), start in centers.items():
        for neighbor in ((x + 1, y), (x, y + 1)):
            if neighbor in centers:
                clearance = world.path_clearance_m(start, centers[neighbor], 0.0)
                # A carved 3 m passage must admit the full cart, not just a ray.
                assert clearance == pytest.approx(0.0) or clearance == pytest.approx(1.5)
                if clearance > 1.0:
                    graph[(x, y)].add(neighbor)
                    graph[neighbor].add((x, y))
    return graph


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 8, 42, 43, 123, 2026])
def test_maze_is_a_reproducible_connected_tree_with_safe_passages(seed):
    world = make_world("indoor_maze", seed)
    assert world == make_world("indoor_maze", seed)
    assert len(world.segments) == len(set(world.segments)) == 16
    assert all(wall.height_m == 2.5 for wall in world.segments)
    assert not world.moving_boxes
    graph = maze_graph(world)
    visited = {(1, 1)}
    pending = deque(visited)
    while pending:
        for neighbor in graph[pending.popleft()] - visited:
            visited.add(neighbor)
            pending.append(neighbor)
    assert len(visited) == 20
    assert sum(map(len, graph.values())) // 2 == 19  # no isolated rooms or cycles
    assert sum(len(neighbors) == 1 for neighbors in graph.values()) >= 2
    assert world.path_clearance_m((5.0, 4.0), (5.0, 4.0), 0.0) >= 1.0
    robot = ExperimentConfig().robot
    assert cart_clearance_radius(robot.wheel_radius_m, robot.wheel_base_m) < 1.0
    for x, y, angle, expected in ((1.5, 1.5, np.pi, 1.5), (13.5, 10.5, 0.0, 1.5)):
        assert world.ray_distance(x, y, angle, 12.0) == pytest.approx(expected)


def test_seed_changes_maze_layout_and_example_has_junctions():
    first = make_world("indoor_maze", 42)
    assert first != make_world("indoor_maze", 43)
    graph = maze_graph(first)
    assert any(len(neighbors) >= 3 for neighbors in graph.values())


def test_maze_multilayer_rays_respect_floor_and_finite_wall_height():
    world = make_world("indoor_maze", 42)
    assert world.ray_distance_3d(1.5, 1.5, 0.5, 0.0, -np.pi / 4, 12.0) == pytest.approx(np.sqrt(0.5))
    assert world.ray_distance_3d(1.5, 1.5, 1.0, np.pi, 0.0, 12.0) == pytest.approx(1.5)
    assert world.ray_distance_3d(1.5, 1.5, 3.0, np.pi, 0.0, 12.0) == 12.0
