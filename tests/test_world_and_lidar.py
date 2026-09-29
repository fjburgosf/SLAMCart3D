import numpy as np

from slamcart3d.core.models import Pose2D
from slamcart3d.sensors.lidar import PlanarLidar
from slamcart3d.worlds.geometry import indoor_simple, make_world


def test_world_generation_is_seed_deterministic() -> None:
    assert indoor_simple(7) == indoor_simple(7)
    assert indoor_simple(7) != indoor_simple(8)


def test_forward_ray_hits_outer_wall() -> None:
    world = indoor_simple(42)
    lidar = PlanarLidar(3, 180.0, 20.0)
    scan = lidar.scan(0.0, Pose2D(5.0, 5.0, 0.0), world)
    assert scan.ranges[1] == 5.0


def test_named_worlds_are_distinct_and_reproducible() -> None:
    names = (
        "indoor_simple",
        "indoor_corridor",
        "indoor_complex",
        "warehouse_shelves",
        "indoor_dynamic",
        "outdoor_blocks",
    )
    worlds = [make_world(name, 7) for name in names]
    assert [world.name for world in worlds] == list(names)
    assert all(world == make_world(world.name, 7) for world in worlds)
    assert all(world != make_world(world.name, 8) for world in worlds)
    assert len({world.segments_at(0.0) for world in worlds}) == len(names)


def test_new_worlds_provide_finite_lidar_observations() -> None:
    lidar = PlanarLidar(9, 180.0, 20.0)
    pose = Pose2D(5.0, 4.0, 0.0)
    for name in ("indoor_corridor", "indoor_complex", "warehouse_shelves"):
        scan = lidar.scan(0.0, pose, make_world(name, 42))
        assert np.isfinite(scan.ranges).all()
        assert np.all((scan.ranges > 0.0) & (scan.ranges <= 20.0))


def test_complex_world_has_reproducible_open_doorway_and_corridor() -> None:
    world = make_world("indoor_complex", 42)
    assert world == make_world("indoor_complex", 42)
    assert world != make_world("indoor_complex", 43)
    assert world.ray_distance(5.5, 4.0, np.pi / 2.0, 20.0) == 8.0
    assert world.ray_distance(5.5, 4.0, -np.pi / 2.0, 20.0) == 1.0
    assert world.ray_distance(5.0, 4.0, 0.0, 20.0) == 11.0


def test_dynamic_object_changes_rays_but_not_static_wall_reference() -> None:
    world = make_world("indoor_dynamic", 42)
    assert world == make_world("indoor_dynamic", 42)
    assert len(world.moving_boxes) == 1
    before = world.ray_distance(0.5, 5.3, 0.0, 12.0, 0.0)
    after = world.ray_distance(0.5, 5.3, 0.0, 12.0, 1.25)
    assert before != after
    assert world.segments == make_world("indoor_simple", 42).segments
    scan_before = PlanarLidar(3, 180.0, 12.0).scan(0.0, Pose2D(0.5, 5.3, 0.0), world)
    scan_after = PlanarLidar(3, 180.0, 12.0).scan(1.25, Pose2D(0.5, 5.3, 0.0), world)
    assert scan_before.ranges[1] != scan_after.ranges[1]


def test_open_outdoor_world_has_no_hit_directions() -> None:
    world = make_world("outdoor_blocks", 42)
    scan = PlanarLidar(9, 180.0, 12.0).scan(0.0, Pose2D(5.0, 5.0, 0.0), world)
    assert np.any(scan.ranges == 12.0)
    assert np.all(scan.ranges > 0.0)
