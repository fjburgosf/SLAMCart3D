"""Qt worker and probability-map canvas for two-variable ASRF studies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from slamcart3d.core.config import ExperimentConfig
from slamcart3d.gui.paths import export_to_requested_path
from slamcart3d.robustness.study import FrontierStudyResult, export_frontier_study, run_frontier_study


class FrontierWorker(QThread):
    """Run a real-simulation study off the GUI thread and export its evidence."""

    completed = Signal(object, str)
    failed = Signal(str)

    def __init__(
        self,
        config: ExperimentConfig,
        parameter_names: tuple[str, str],
        lower: tuple[float, float],
        upper: tuple[float, float],
        metric_name: str,
        failure_threshold: float,
        seeds: tuple[int, ...],
        budget: int,
        output_root: Path,
    ) -> None:
        super().__init__()
        self.config = config
        self.parameter_names = parameter_names
        self.lower = lower
        self.upper = upper
        self.metric_name = metric_name
        self.failure_threshold = failure_threshold
        self.seeds = seeds
        self.budget = budget
        self.output_root = output_root

    def run(self) -> None:
        try:
            study = run_frontier_study(
                self.config,
                self.parameter_names,
                np.asarray(self.lower),
                np.asarray(self.upper),
                self.metric_name,
                self.failure_threshold,
                self.seeds,
                self.budget,
            )
            target = export_to_requested_path(
                self.output_root, lambda root: export_frontier_study(study, root)
            )
            self.completed.emit(study, str(target))
        except Exception as exc:
            self.failed.emit(str(exc))


class RobustnessCanvas(QWidget):
    """Display success probability and evaluated samples without Matplotlib."""

    def __init__(self) -> None:
        super().__init__()
        self.study: FrontierStudyResult | None = None
        self.x_label = ""
        self.y_label = ""
        self.empty_label = ""
        self.success_label = ""
        self.failure_label = ""
        self.setMinimumSize(500, 340)

    @staticmethod
    def _color(probability: float) -> QColor:
        if probability < 0.5:
            fraction = max(0.0, min(1.0, probability * 2.0))
            start, end = (239, 68, 68), (245, 158, 11)
        else:
            fraction = max(0.0, min(1.0, (probability - 0.5) * 2.0))
            start, end = (245, 158, 11), (34, 197, 94)
        return QColor(*(round(a + fraction * (b - a)) for a, b in zip(start, end, strict=True)))

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f8fafc"))
        if self.study is None:
            painter.setPen(QColor("#475569"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.empty_label)
            return
        study = self.study
        plot = QRectF(72, 24, max(10, self.width() - 100), max(10, self.height() - 105))
        probabilities = study.success_probability
        height, width = probabilities.shape
        image = QImage(width, height, QImage.Format.Format_RGB32)
        for row in range(height):
            for column in range(width):
                image.setPixelColor(column, height - row - 1, self._color(float(probabilities[row, column])))
        painter.drawImage(plot, image)
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.drawRect(plot)
        low_x, high_x = float(study.x_axis[0]), float(study.x_axis[-1])
        low_y, high_y = float(study.y_axis[0]), float(study.y_axis[-1])
        for sample, failed in zip(study.asrf.samples, study.asrf.failed, strict=True):
            x = plot.left() + (float(sample[0]) - low_x) / (high_x - low_x) * plot.width()
            y = plot.bottom() - (float(sample[1]) - low_y) / (high_y - low_y) * plot.height()
            painter.setPen(QPen(QColor("#0f172a"), 1.5))
            painter.setBrush(QColor("#f8fafc") if not failed else QColor("#0f172a"))
            painter.drawEllipse(QPointF(x, y), 4.0, 4.0)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QColor("#0f172a"))
        painter.drawText(int(plot.left()), int(plot.bottom() + 19), f"{low_x:.3g}")
        painter.drawText(int(plot.right() - 42), int(plot.bottom() + 19), f"{high_x:.3g}")
        painter.drawText(int(plot.left() - 58), int(plot.top() + 5), f"{high_y:.3g}")
        painter.drawText(int(plot.left() - 58), int(plot.bottom()), f"{low_y:.3g}")
        painter.drawText(int(plot.center().x() - 100), int(plot.bottom() + 41), self.x_label)
        painter.drawText(int(plot.left()), int(plot.bottom() + 63), self.y_label)
        painter.setPen(QColor("#334155"))
        painter.drawText(
            int(plot.left()),
            int(plot.bottom() + 82),
            f"{self.failure_label}  ●   0 — 0.5 — 1   ○  {self.success_label}",
        )
