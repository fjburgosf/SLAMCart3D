"""Interactive orthographic view of the optional estimated-pose 3D point map."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    from slamcart3d.simulation.runner import RunPreview, RunResult


def cart_vertices(pose: np.ndarray) -> np.ndarray:
    """Return an oriented, 0.7 x 0.45 x 0.35 m cart-shaped prism."""

    x, y, yaw = map(float, pose)
    local = np.array(
        [
            (length, width, height)
            for height in (0.08, 0.43)
            for length, width in ((-0.35, -0.225), (0.35, -0.225), (0.35, 0.225), (-0.35, 0.225))
        ]
    )
    c, s = np.cos(yaw), np.sin(yaw)
    world = local.copy()
    world[:, 0] = x + c * local[:, 0] - s * local[:, 1]
    world[:, 1] = y + s * local[:, 0] + c * local[:, 1]
    return world


class PointCloudCanvas(QWidget):
    """Rotate and zoom a sampled cloud; no 6DoF pose is inferred."""

    def __init__(self) -> None:
        super().__init__()
        self.result: RunResult | RunPreview | None = None
        self.empty_label = ""
        self.count_label = "puntos"
        self.legend_labels = ("Pared", "Nube", "Real", "Estimada")
        self.floor_label = "Piso z = 0 m"
        self.no_points_label = "Sin nube 3D: active LiDAR multicapa"
        self.show_reference = True
        self.azimuth_deg = 35.0
        self.elevation_deg = 30.0
        self.zoom = 1.0
        self._drag_position: QPointF | None = None
        self.setMinimumSize(560, 420)

    def _project(self, points: np.ndarray, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        relative = points - center
        yaw = np.deg2rad(self.azimuth_deg)
        pitch = np.deg2rad(self.elevation_deg)
        x = relative[:, 0] * np.cos(yaw) - relative[:, 1] * np.sin(yaw)
        y = relative[:, 0] * np.sin(yaw) + relative[:, 1] * np.cos(yaw)
        return np.column_stack((x, y * np.sin(pitch) - relative[:, 2] * np.cos(pitch))), (
            y * np.cos(pitch) + relative[:, 2] * np.sin(pitch)
        )

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#111827"))
        if self.result is None:
            painter.setPen(QColor("#d1d5db"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.empty_label)
            return

        result = self.result
        cloud = result.map_points_3d
        sample = cloud[:: max(1, (len(cloud) + 4_999) // 5_000)]
        truth = np.column_stack((result.ground_truth[:, :2], np.zeros(len(result.ground_truth))))
        estimate = np.column_stack((result.estimated[:, :2], np.zeros(len(result.estimated))))
        walls = result.world.segments_at(float(result.timestamps[-1]))
        wall_vertices = [
            np.array(
                [
                    (wall.x1, wall.y1, 0.0),
                    (wall.x2, wall.y2, 0.0),
                    (wall.x1, wall.y1, wall.height_m),
                    (wall.x2, wall.y2, wall.height_m),
                ]
            )
            for wall in walls
        ]
        true_cart = cart_vertices(result.ground_truth[-1])
        estimated_cart = cart_vertices(result.estimated[-1])
        visible_geometry = [estimate, estimated_cart]
        if self.show_reference:
            visible_geometry.extend((truth, true_cart, *wall_vertices))
        if len(sample):
            visible_geometry.append(sample)
        scene = np.vstack(visible_geometry)
        center = (np.min(scene, axis=0) + np.max(scene, axis=0)) / 2.0
        scene_projected, _ = self._project(scene, center)
        bounds = np.ptp(scene_projected, axis=0)
        scale = self.zoom * min(
            (self.width() - 70) / max(float(bounds[0]), 1.0),
            (self.height() - 100) / max(float(bounds[1]), 1.0),
        )

        def screen_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            projected, depth = self._project(points, center)
            screen = projected * scale + np.array([self.width() / 2, self.height() / 2])
            screen[:, 1] = self.height() - screen[:, 1]
            return screen, depth

        floor_geometry = (
            np.vstack([vertices[:2] for vertices in wall_vertices])
            if self.show_reference
            else np.vstack((estimate, sample if len(sample) else estimated_cart))
        )
        floor_min = np.min(floor_geometry[:, :2], axis=0)
        floor_max = np.max(floor_geometry[:, :2], axis=0)
        for axis in (0, 1):
            if floor_max[axis] - floor_min[axis] < 1.0:
                floor_min[axis] -= 1.0
                floor_max[axis] += 1.0
        floor_corners = np.array(
            [
                (floor_min[0], floor_min[1], 0.0),
                (floor_max[0], floor_min[1], 0.0),
                (floor_max[0], floor_max[1], 0.0),
                (floor_min[0], floor_max[1], 0.0),
            ]
        )
        floor_screen, _ = screen_points(floor_corners)
        painter.setPen(QPen(QColor("#7b94ac"), 2.0))
        painter.setBrush(QColor("#203145"))
        painter.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in floor_screen]))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#456078"), 0.8))
        grid_spacing = max(
            1, int(np.ceil(max(np.ptp(floor_geometry[:, 0]), np.ptp(floor_geometry[:, 1])) / 20))
        )
        for x in range(int(np.ceil(floor_min[0])), int(np.floor(floor_max[0])) + 1, grid_spacing):
            line, _ = screen_points(np.array([(x, floor_min[1], 0.0), (x, floor_max[1], 0.0)]))
            painter.drawLine(
                QPointF(float(line[0, 0]), float(line[0, 1])),
                QPointF(float(line[1, 0]), float(line[1, 1])),
            )
        for y in range(int(np.ceil(floor_min[1])), int(np.floor(floor_max[1])) + 1, grid_spacing):
            line, _ = screen_points(np.array([(floor_min[0], y, 0.0), (floor_max[0], y, 0.0)]))
            painter.drawLine(
                QPointF(float(line[0, 0]), float(line[0, 1])),
                QPointF(float(line[1, 0]), float(line[1, 1])),
            )
        for index, vertices in enumerate(wall_vertices if self.show_reference else []):
            screen, _ = screen_points(vertices)
            color = "#bb87dd" if index >= len(result.world.segments) else "#a8c5de"
            painter.setBrush(
                QColor(104, 148, 181, 140)
                if index < len(result.world.segments)
                else QColor(168, 85, 247, 140)
            )
            painter.drawPolygon(
                QPolygonF(
                    [QPointF(float(screen[point, 0]), float(screen[point, 1])) for point in (0, 1, 3, 2)]
                )
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(color), 1.0))
            for start, end in ((0, 1), (2, 3), (0, 2), (1, 3)):
                painter.drawLine(
                    QPointF(float(screen[start, 0]), float(screen[start, 1])),
                    QPointF(float(screen[end, 0]), float(screen[end, 1])),
                )

        if len(sample):
            screen, depth = screen_points(sample)
            z_min = float(np.min(sample[:, 2]))
            z_span = max(float(np.ptp(sample[:, 2])), 1e-9)
            for index in np.argsort(depth):
                height = (float(sample[index, 2]) - z_min) / z_span
                color = QColor.fromRgbF(0.16 + 0.66 * height, 0.75 - 0.26 * height, 0.95 - 0.30 * height)
                painter.setPen(QPen(color, 2.0))
                painter.drawPoint(QPointF(float(screen[index, 0]), float(screen[index, 1])))

        trajectories = (
            ((truth, "#22c55e"), (estimate, "#f97316")) if self.show_reference else ((estimate, "#f97316"),)
        )
        for trajectory, color in trajectories:
            screen, _ = screen_points(trajectory)
            painter.setPen(QPen(QColor(color), 2.3))
            painter.drawPolyline(QPolygonF([QPointF(float(x), float(y)) for x, y in screen]))
        carts = ((estimated_cart, "#f97316", False), (true_cart, "#22c55e", True))
        for vertices, color, is_true in carts if self.show_reference else carts[:1]:
            screen, _ = screen_points(vertices)
            painter.setPen(QPen(QColor(color), 2.3))
            painter.setBrush(QColor(color) if is_true else Qt.BrushStyle.NoBrush)
            painter.drawPolygon(
                QPolygonF(
                    [QPointF(float(screen[point, 0]), float(screen[point, 1])) for point in (4, 5, 6, 7)]
                )
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for start, end in ((0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7)):
                painter.drawLine(
                    QPointF(float(screen[start, 0]), float(screen[start, 1])),
                    QPointF(float(screen[end, 0]), float(screen[end, 1])),
                )
            painter.drawLine(
                QPointF(float(screen[4, 0]), float(screen[4, 1])),
                QPointF(float(screen[5, 0]), float(screen[5, 1])),
            )
            front_center = (screen[5] + screen[6]) / 2.0
            top_center = np.mean(screen[4:8], axis=0)
            painter.drawLine(
                QPointF(float(top_center[0]), float(top_center[1])),
                QPointF(float(front_center[0]), float(front_center[1])),
            )
            for corner in screen[:4]:
                painter.setBrush(QColor("#111827"))
                painter.drawEllipse(QPointF(float(corner[0]), float(corner[1])), 4.0, 4.0)
            painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QColor("#e5e7eb"))
        painter.drawText(
            18,
            26,
            f"XYZ: {len(cloud):,} {self.count_label}"
            + (f"   z [{np.min(cloud[:, 2]):.2f}, {np.max(cloud[:, 2]):.2f}] m" if len(cloud) else ""),
        )
        painter.setPen(QColor("#d5e7f6"))
        painter.drawText(18, 47, self.floor_label)
        if not len(cloud):
            painter.setPen(QColor("#facc15"))
            painter.drawText(18, 68, self.no_points_label)
        legend = list(zip(("#94a3b8", "#38bdf8", "#22c55e", "#f97316"), self.legend_labels, strict=True))
        if not self.show_reference:
            legend = [legend[1], legend[3]]
        for index, (color, label) in enumerate(legend):
            painter.setPen(QPen(QColor(color), 2.0))
            painter.drawText(18 + index * 145, self.height() - 16, label)

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
        self.azimuth_deg = (self.azimuth_deg + delta.x() * 0.6) % 360.0
        self.elevation_deg = float(np.clip(self.elevation_deg + delta.y() * 0.6, -80.0, 80.0))
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
        self.zoom = float(np.clip(self.zoom * 1.12 ** (event.angleDelta().y() / 120.0), 0.2, 8.0))
        self.update()
        event.accept()
