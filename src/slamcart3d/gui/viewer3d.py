"""Metric, lit 3D world and measured point map with an orbit camera."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from slamcart3d.core.config import ExperimentConfig
from slamcart3d.mapping.voxel import SurfaceMesh, transform_points
from slamcart3d.robots.geometry import cart_vertices, transform_vertices, wheel_vertices

if TYPE_CHECKING:
    from slamcart3d.simulation.runner import RunPreview, RunResult


class PointCloudCanvas(QWidget):
    """CPU-rendered surfaces; the world reference never feeds the estimator."""

    def __init__(self) -> None:
        super().__init__()
        self.result: RunResult | RunPreview | None = None
        self.config = ExperimentConfig()
        self.empty_label = ""
        self.count_label = "puntos"
        self.legend_labels = ("Pared", "Nube", "Real", "Estimada")
        self.floor_label = "Piso z = 0 m"
        self.no_points_label = "Sin nube 3D: active LiDAR multicapa"
        self.cutaway_label = "Vista en corte: paredes exteriores cercanas rebajadas solo en pantalla"
        self.show_reference = True
        self.show_cloud = True
        self.show_surface = True
        self.surface_label = "Superficie medida"
        self._surface_owner = None
        self._surface_mesh = SurfaceMesh(np.empty((0, 3)), np.empty((0, 3), dtype=int))
        self.show_observation = False
        self.color_error = False
        self.observation_label = "Gris: desconocido · Verde: libre observado · Rojo: ocupado estimado"
        self.error_label = "Error de pared: verde ≤ {limit:g} m · rojo > {limit:g} m · piso cian"
        self.cutaway = True
        self.perspective = True
        self.azimuth_deg = 35.0
        self.elevation_deg = 38.0
        self.zoom = 1.0
        self._camera_distance = 40.0
        self._drag_position: QPointF | None = None
        self.setMinimumSize(560, 420)

    def reset_view(self) -> None:
        self.azimuth_deg, self.elevation_deg, self.zoom = 35.0, 38.0, 1.0
        self.update()

    def top_view(self) -> None:
        self.azimuth_deg, self.elevation_deg, self.zoom = 0.0, 90.0, 1.0
        self.update()

    def _project(self, points: np.ndarray, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Camera Y is up: positive world Z rises before screen-Y inversion."""

        relative = points - center
        yaw, pitch = np.deg2rad([self.azimuth_deg, self.elevation_deg])
        x = relative[:, 0] * np.cos(yaw) - relative[:, 1] * np.sin(yaw)
        y = relative[:, 0] * np.sin(yaw) + relative[:, 1] * np.cos(yaw)
        projected = np.column_stack((x, -y * np.sin(pitch) + relative[:, 2] * np.cos(pitch)))
        depth = y * np.cos(pitch) + relative[:, 2] * np.sin(pitch)
        if self.perspective:
            projected *= (self._camera_distance / np.maximum(self._camera_distance - depth, 0.1))[:, None]
        return projected, depth

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#101b2b"))
        if self.result is None:
            painter.setPen(QColor("#d1d5db"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.empty_label)
            return

        result = self.result
        config = getattr(result, "config", self.config)
        r, b = config.robot.wheel_radius_m, config.robot.wheel_base_m
        cloud = result.map_points_3d
        sample = cloud[:: max(1, (len(cloud) + 3_999) // 4_000)] if self.show_cloud else np.empty((0, 3))
        truth = np.column_stack((result.ground_truth[:, :2], np.full(len(result.ground_truth), 0.015)))
        estimate = np.column_stack((result.estimated[:, :2], np.full(len(result.estimated), 0.02)))
        state_3d = result.estimated_3d
        if state_3d is not None:
            estimate = state_3d[:, :3].copy()
            estimate[:, 2] += 0.02
        voxel_map = result.voxel_map
        if self.show_surface and voxel_map is not self._surface_owner:
            self._surface_owner = voxel_map
            self._surface_mesh = (
                voxel_map.surface()
                if voxel_map is not None
                else SurfaceMesh(np.empty((0, 3)), np.empty((0, 3), dtype=int))
            )
        mesh = self._surface_mesh
        surface_vertices = mesh.vertices if self.show_surface else np.empty((0, 3))
        walls = result.world.segments_at(float(result.timestamps[-1]))
        wall_geometry = [
            np.array([(w.x1, w.y1, 0), (w.x2, w.y2, 0), (w.x2, w.y2, w.height_m), (w.x1, w.y1, w.height_m)])
            for w in walls
        ]
        true_cart = cart_vertices(result.ground_truth[-1], r, b)
        estimated_cart = cart_vertices(result.estimated[-1], r, b)

        def placed(points, pose, solid):
            if solid or state_3d is None:
                return points
            local = points.copy()
            yaw = pose[2]
            local[:, :2] = (points[:, :2] - pose[:2]) @ np.array(
                [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
            )
            return transform_points(local, state_3d[-1])

        estimated_cart = placed(estimated_cart, result.estimated[-1], False)
        geometry = [estimate, estimated_cart]
        if self.show_reference:
            geometry.extend((truth, true_cart, *wall_geometry))
        if len(sample):
            geometry.append(sample)
        if len(surface_vertices):
            geometry.append(surface_vertices)
        scene = np.vstack(geometry)
        center = (scene.min(axis=0) + scene.max(axis=0)) / 2
        self._camera_distance = max(6.0, 2.8 * float(np.linalg.norm(np.ptp(scene, axis=0))))
        projected, _ = self._project(scene, center)
        minimum, maximum = projected.min(axis=0), projected.max(axis=0)
        bounds = maximum - minimum
        scale = self.zoom * min(
            (self.width() - 90) / max(float(bounds[0]), 1.0),
            (self.height() - 160) / max(float(bounds[1]), 1.0),
        )
        origin = (minimum + maximum) / 2

        def screen_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            xy, depth = self._project(points, center)
            xy = (xy - origin) * scale
            xy[:, 0] += self.width() / 2
            xy[:, 1] = self.height() / 2 + 20 - xy[:, 1]
            return xy, depth

        def polygon(points: np.ndarray) -> QPolygonF:
            return QPolygonF([QPointF(float(x), float(y)) for x, y in points])

        floor_geometry = (
            np.vstack([vertices[:2] for vertices in wall_geometry])
            if self.show_reference and wall_geometry
            else np.vstack((estimate, estimated_cart, sample, surface_vertices))
        )
        floor_min = floor_geometry[:, :2].min(axis=0) - 0.5
        floor_max = np.maximum(floor_geometry[:, :2].max(axis=0) + 0.5, floor_min + 2.0)
        floor_corners = np.array(
            [
                (floor_min[0], floor_min[1], 0),
                (floor_max[0], floor_min[1], 0),
                (floor_max[0], floor_max[1], 0),
                (floor_min[0], floor_max[1], 0),
            ]
        )
        floor_screen, _ = screen_points(floor_corners)
        painter.setPen(QPen(QColor("#7793a8"), 1.4))
        painter.setBrush(QColor("#526575"))
        painter.drawPolygon(polygon(floor_screen))
        spacing = max(1, int(np.ceil(float(np.max(floor_max - floor_min)) / 20)))
        for x in np.arange(np.floor(floor_min[0] / spacing) * spacing, floor_max[0], spacing):
            for y in np.arange(np.floor(floor_min[1] / spacing) * spacing, floor_max[1], spacing):
                lo, hi = np.maximum((x, y), floor_min), np.minimum((x + spacing, y + spacing), floor_max)
                tile = np.array([(lo[0], lo[1], 0), (hi[0], lo[1], 0), (hi[0], hi[1], 0), (lo[0], hi[1], 0)])
                tile_screen, _ = screen_points(tile)
                parity = int(round(x / spacing) + round(y / spacing)) % 2
                painter.setBrush(QColor("#566d7c" if parity else "#617987"))
                painter.setPen(QPen(QColor("#8296a1"), 0.5))
                painter.drawPolygon(polygon(tile_screen))
        grid = result.occupancy_grid
        if self.show_observation and grid is not None:
            resolution = grid.resolution_m
            # Unknown is the existing gray floor. Only measured cells are colored;
            # reference wall cells are never painted as observed free space.
            observed = sorted(grid.free | grid.occupied)
            for x, y in observed:
                if not (
                    floor_min[0] <= x * resolution <= floor_max[0]
                    and floor_min[1] <= y * resolution <= floor_max[1]
                ):
                    continue
                corners = np.array(
                    [
                        (x * resolution, y * resolution, 0.003),
                        ((x + 1) * resolution, y * resolution, 0.003),
                        ((x + 1) * resolution, (y + 1) * resolution, 0.003),
                        (x * resolution, (y + 1) * resolution, 0.003),
                    ]
                )
                projected_cell, _ = screen_points(corners)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor("#c9564e" if (x, y) in grid.occupied else "#477f60"))
                painter.drawPolygon(polygon(projected_cell))

        # Vertical surfaces, trajectories, wheel faces and cloud points share
        # one far-to-near order. No cart is forced in front of a wall.
        primitives: list[tuple[float, str, np.ndarray, QColor, float]] = []

        def add(kind: str, points: np.ndarray, color: QColor, width: float = 1.0) -> None:
            screen, depth = screen_points(points)
            primitives.append((float(np.mean(depth)), kind, screen, color, width))

        if len(surface_vertices):
            mesh_screen, mesh_depth = screen_points(surface_vertices)
            vertices = surface_vertices[mesh.triangles]
            normal = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
            normal /= np.maximum(np.linalg.norm(normal, axis=1), 1e-12)[:, None]
            shades = 0.7 + 0.3 * np.abs(normal @ np.array([0.4, 0.5, 0.75]))
            depths = mesh_depth[mesh.triangles].mean(axis=1)
            screens = mesh_screen[mesh.triangles]
            colors = (shades[:, None] * np.array([46, 194, 174])).astype(int)
            for depth, screen, color in zip(depths, screens, colors, strict=True):
                primitives.append((float(depth), "surface", screen, QColor(*map(int, color)), 0.5))

        cut_count = 0
        for index, vertices in enumerate(wall_geometry if self.show_reference else []):
            displayed = vertices.copy()
            endpoints = vertices[:2, :2]
            perimeter = any(
                np.allclose(endpoints[:, axis], bound, atol=1e-6)
                for axis in (0, 1)
                for bound in (floor_min[axis] + 0.5, floor_max[axis] - 0.5)
            )
            _, wall_depth = self._project(vertices, center)
            if self.cutaway and perimeter and np.mean(wall_depth[:2]) > 0:
                displayed[2:, 2] = min(0.22, float(vertices[2, 2]))
                cut_count += 1
            tangent = vertices[1, :2] - vertices[0, :2]
            normal = np.array([-tangent[1], tangent[0]]) / max(float(np.linalg.norm(tangent)), 1e-9)
            shade = 0.72 + 0.22 * abs(float(np.dot(normal, [0.6, 0.8])))
            base = np.array([166, 196, 216] if index < len(result.world.segments) else [183, 132, 221])
            color = QColor(*(int(v) for v in base * shade))
            # Subdivide large planes for local depth ordering: a whole wall's
            # average depth must not hide a cart that is in front of its base.
            columns = max(1, int(np.ceil(float(np.linalg.norm(tangent)) / 0.6)))
            rows = max(1, int(np.ceil(float(displayed[2, 2]) / 0.4)))
            for column in range(columns):
                left = displayed[0] + (displayed[1] - displayed[0]) * column / columns
                right = displayed[0] + (displayed[1] - displayed[0]) * (column + 1) / columns
                for row in range(rows):
                    bottom = float(displayed[2, 2]) * row / rows
                    top = float(displayed[2, 2]) * (row + 1) / rows
                    tile = np.array(
                        [
                            (left[0], left[1], bottom),
                            (right[0], right[1], bottom),
                            (right[0], right[1], top),
                            (left[0], left[1], top),
                        ]
                    )
                    add("surface", tile, color)
            add("line", displayed[[2, 3]], QColor("#e1edf4"), 2.0)

        trajectories = (
            ((truth, "#22c55e"), (estimate, "#ff8b36")) if self.show_reference else ((estimate, "#ff8b36"),)
        )
        for trajectory, color in trajectories:
            for start, end in zip(trajectory[:-1], trajectory[1:], strict=True):
                add("line", np.vstack((start, end)), QColor(color), 2.3)

        if len(sample):
            distances = None
            if self.color_error and self.show_reference:
                from slamcart3d.metrics.map_quality_3d import _point_to_wall_surface_distances

                distances = _point_to_wall_surface_distances(sample, result.world)
            z_min, z_span = float(sample[:, 2].min()), max(float(np.ptp(sample[:, 2])), 1e-9)
            for point_index, point in enumerate(sample):
                height = (float(point[2]) - z_min) / z_span
                color = QColor.fromRgbF(0.1 + 0.35 * height, 0.88 - 0.18 * height, 1.0)
                if distances is not None and point[2] > 0.1:
                    color = QColor(
                        "#45e691"
                        if distances[point_index] <= config.evaluation.wall_distance_limit_m
                        else "#ff635a"
                    )
                add("point", point[None, :], color, 2.5)

        carts = [(result.estimated[-1], estimated_cart, "#ff8b36", False)]
        if self.show_reference:
            carts.append((result.ground_truth[-1], true_cart, "#25d36c", True))
        for pose, chassis, color, solid in carts:
            shadow = chassis[:4].copy()
            shadow[:, 2] = 0.005
            shadow_screen, _ = screen_points(shadow)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(15, 27, 35, 85))
            painter.drawPolygon(polygon(shadow_screen))
            faces = ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7))
            for indices in faces:
                add("face" if solid else "outline", chassis[list(indices)], QColor(color), 1.5)
            for side in (-1, 1):
                wheels = placed(wheel_vertices(pose, side, r, b), pose, solid)
                for ring in (wheels[:16], wheels[16:]):
                    add("face" if solid else "outline", ring, QColor("#1d2936" if solid else color), 1.3)
                for i in range(16):
                    j = (i + 1) % 16
                    add(
                        "face" if solid else "outline",
                        wheels[[i, j, j + 16, i + 16]],
                        QColor("#334454" if solid else color),
                        0.7,
                    )
            sensor_z = config.lidar3d.sensor_height_m
            mast = placed(
                transform_vertices(np.array([(0, 0, 4.3 * r), (0, 0, sensor_z)]), pose), pose, solid
            )
            add("line", mast, QColor("#e9eef3" if solid else color), 2.5)
            angles = np.arange(16) * (2 * np.pi / 16)
            cap = transform_vertices(
                np.column_stack((0.8 * r * np.cos(angles), 0.8 * r * np.sin(angles), np.full(16, sensor_z))),
                pose,
            )
            cap = placed(cap, pose, solid)
            add("face" if solid else "outline", cap, QColor("#f0df72" if solid else color), 1.0)
            heading = transform_vertices(np.array([(0, 0, 4.35 * r), (3.1 * r, 0, 4.35 * r)]), pose)
            heading = placed(heading, pose, solid)
            add("line", heading, QColor("#132d22" if solid else color), 3.0)

        for _, kind, screen, color, width in sorted(primitives, key=lambda item: item[0]):
            painter.setPen(QPen(color.darker(125) if kind == "face" else color, width))
            painter.setBrush(color if kind in ("face", "surface") else Qt.BrushStyle.NoBrush)
            if kind == "surface":
                painter.setPen(QPen(color, 0.7))
            if kind in ("face", "surface", "outline"):
                painter.drawPolygon(polygon(screen))
            elif kind == "point":
                painter.drawPoint(QPointF(float(screen[0, 0]), float(screen[0, 1])))
            else:
                painter.drawPolyline(polygon(screen))

        axis_origin = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        saved = self.perspective
        self.perspective = False
        axes, _ = self._project(axis_origin, np.zeros(3))
        self.perspective = saved
        axes *= np.array([32, -32])
        axes += np.array([self.width() - 65, self.height() - 75])
        for index, (label, color) in enumerate(
            zip("XYZ", ("#ff7469", "#7be89c", "#71c6ff"), strict=True), start=1
        ):
            painter.setPen(QPen(QColor(color), 2.0))
            painter.drawLine(QPointF(*axes[0]), QPointF(*axes[index]))
            painter.drawText(QPointF(*(axes[index] + [3, -3])), label)
        painter.setPen(QColor("#e8f0f5"))
        painter.drawText(
            18,
            25,
            f"XYZ: {len(cloud):,} {self.count_label}"
            + (f"   z [{cloud[:, 2].min():.2f}, {cloud[:, 2].max():.2f}] m" if len(cloud) else ""),
        )
        painter.drawText(18, 46, f"{self.floor_label}   |   {spacing} m / □")
        if self.show_observation:
            painter.drawText(
                18,
                86,
                self.observation_label
                if grid is not None
                else (
                    "Sin ocupación: active el mapa de ocupación"
                    if self.floor_label.startswith(("Piso", "Guía"))
                    else "No occupancy: enable the occupancy map"
                ),
            )
        if cut_count:
            painter.setPen(QColor("#ffda87"))
            painter.drawText(18, 67, self.cutaway_label)
        if not len(cloud):
            painter.setPen(QColor("#facc15"))
            painter.drawText(18, 107, self.no_points_label)
        elif self.color_error and self.show_reference:
            painter.setPen(QColor("#e8f0f5"))
            painter.drawText(18, 107, self.error_label.format(limit=config.evaluation.wall_distance_limit_m))
        if self.show_surface:
            painter.setPen(QColor("#65dfc9"))
            painter.drawText(
                18,
                128,
                f"{self.surface_label}: {len(mesh.triangles):,} △"
                + (
                    f" · {voxel_map.resolution_m:g} m · ≥ {voxel_map.min_hits} obs."
                    if voxel_map is not None
                    else " · —"
                ),
            )
        legend = list(zip(("#a8c5d9", "#38d9ff", "#25d36c", "#ff8b36"), self.legend_labels, strict=True))
        if not self.show_reference:
            legend = [legend[1], legend[3]]
        if not self.show_cloud:
            legend = [(color, label) for color, label in legend if color != "#38d9ff"]
        if self.show_surface:
            legend.insert(0, ("#2ec2ae", self.surface_label))
        for index, (color, label) in enumerate(legend):
            painter.setPen(QColor(color))
            painter.drawText(18 + index * 150, self.height() - 16, label)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_position = event.position()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_position is None:
            super().mouseMoveEvent(event)
            return
        delta = event.position() - self._drag_position
        self.azimuth_deg = (self.azimuth_deg + delta.x() * 0.6) % 360
        self.elevation_deg = float(np.clip(self.elevation_deg + delta.y() * 0.6, 8.0, 90.0))
        self._drag_position = event.position()
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_position = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        self.zoom = float(np.clip(self.zoom * 1.12 ** (event.angleDelta().y() / 120), 0.2, 8.0))
        self.update()
        event.accept()
