"""Metric display geometry and conservative swept footprint of the cart."""

from __future__ import annotations

import numpy as np


def transform_vertices(vertices: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Place local XYZ geometry with a planar pose, without changing its height."""

    x, y, yaw = map(float, pose)
    world = vertices.copy()
    c, s = np.cos(yaw), np.sin(yaw)
    world[:, 0] = x + c * vertices[:, 0] - s * vertices[:, 1]
    world[:, 1] = y + s * vertices[:, 0] + c * vertices[:, 1]
    return world


def cart_vertices(pose: np.ndarray, wheel_radius_m: float = 0.1, wheel_base_m: float = 0.5) -> np.ndarray:
    """Approximate chassis: length 7r, width 0.9b, bottom 0.8r, top 4.3r."""

    r, b = wheel_radius_m, wheel_base_m
    local = np.array(
        [
            (length, width, height)
            for height in (0.8 * r, 4.3 * r)
            for length, width in (
                (-3.5 * r, -0.45 * b),
                (3.5 * r, -0.45 * b),
                (3.5 * r, 0.45 * b),
                (-3.5 * r, 0.45 * b),
            )
        ]
    )
    return transform_vertices(local, pose)


def wheel_vertices(
    pose: np.ndarray, side: int, wheel_radius_m: float = 0.1, wheel_base_m: float = 0.5
) -> np.ndarray:
    """Two 16-vertex cylinder rings, axle along local Y, touching z=0."""

    r = wheel_radius_m
    angles = np.arange(16) * (2 * np.pi / 16)
    local = np.array(
        [
            (r * np.cos(angle), side * wheel_base_m / 2 + offset, r + r * np.sin(angle))
            for offset in (-0.25 * r, 0.25 * r)
            for angle in angles
        ]
    )
    return transform_vertices(local, pose)


def cart_clearance_radius(wheel_radius_m: float, wheel_base_m: float) -> float:
    """Circle enclosing chassis and wheels for any heading, in meters."""

    chassis = np.hypot(3.5 * wheel_radius_m, 0.45 * wheel_base_m)
    wheel = np.hypot(wheel_radius_m, wheel_base_m / 2 + 0.25 * wheel_radius_m)
    return float(max(chassis, wheel))
