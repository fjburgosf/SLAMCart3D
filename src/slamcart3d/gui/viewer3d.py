"""Interactive orthographic view of the optional estimated-pose 3D point map."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    from slamcart3d.simulation.runner import RunResult


class PointCloudCanvas(QWidget):
    """Rotate and zoom a sampled cloud; no 6DoF pose is inferred."""

    def __init__(self) -> None:
        super().__init__()
        self.result: RunResult | None = None
        self.empty_label = ""
        self.count_label = "puntos"
        self.legend_labels = ("Pared", "Nube", "Real", "Estimada")
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
        if self.result is None or len(self.result.map_points_3d) == 0:
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
        scene = np.vstack((cloud, truth, estimate, *wall_vertices))
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

        for index, vertices in enumerate(wall_vertices):
            screen, _ = screen_points(vertices)
            color = "#a855f7" if index >= len(result.world.segments) else "#64748b"
            painter.setPen(QPen(QColor(color), 1.0))
            for start, end in ((0, 1), (2, 3), (0, 2), (1, 3)):
                painter.drawLine(
                    QPointF(float(screen[start, 0]), float(screen[start, 1])),
                    QPointF(float(screen[end, 0]), float(screen[end, 1])),
                )

        screen, depth = screen_points(sample)
        z_min = float(np.min(sample[:, 2]))
        z_span = max(float(np.ptp(sample[:, 2])), 1e-9)
        for index in np.argsort(depth):
            height = (float(sample[index, 2]) - z_min) / z_span
            color = QColor.fromRgbF(0.16 + 0.66 * height, 0.75 - 0.26 * height, 0.95 - 0.30 * height)
            painter.setPen(QPen(color, 2.0))
            painter.drawPoint(QPointF(float(screen[index, 0]), float(screen[index, 1])))

        for trajectory, color in ((truth, "#22c55e"), (estimate, "#f97316")):
            screen, _ = screen_points(trajectory)
            painter.setPen(QPen(QColor(color), 2.3))
            painter.drawPolyline(QPolygonF([QPointF(float(x), float(y)) for x, y in screen]))
        painter.setPen(QColor("#e5e7eb"))
        painter.drawText(
            18,
            26,
            f"XYZ: {len(cloud):,} {self.count_label}   z [{np.min(cloud[:, 2]):.2f}, {np.max(cloud[:, 2]):.2f}] m",
        )
        for index, (color, label) in enumerate(
            zip(("#94a3b8", "#38bdf8", "#22c55e", "#f97316"), self.legend_labels, strict=True)
        ):
            painter.setPen(QPen(QColor(color), 2.0))
            painter.drawText(18 + index * 112, self.height() - 16, label)

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
