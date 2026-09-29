"""Desktop controls for repeated one-variable scientific studies."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from slamcart3d.core.config import ExperimentConfig
from slamcart3d.experiments.sweep import SweepResult, export_parameter_sweep, run_parameter_sweep


class SweepWorker(QThread):
    """Execute paired-seed runs without blocking the interface."""

    completed = Signal(object, str)
    failed = Signal(str)

    def __init__(
        self,
        config: ExperimentConfig,
        parameter_name: str,
        levels: tuple[float, ...],
        seeds: tuple[int, ...],
        metric_name: str,
        failure_threshold: float,
        output_root: Path,
    ) -> None:
        super().__init__()
        self.config = config
        self.parameter_name = parameter_name
        self.levels = levels
        self.seeds = seeds
        self.metric_name = metric_name
        self.failure_threshold = failure_threshold
        self.output_root = output_root

    def run(self) -> None:
        try:
            study = run_parameter_sweep(
                self.config,
                self.parameter_name,
                self.levels,
                self.seeds,
                self.metric_name,
                self.failure_threshold,
            )
            target = export_parameter_sweep(study, self.output_root)
            self.completed.emit(study, str(target))
        except Exception as exc:
            self.failed.emit(str(exc))


class SweepCanvas(QWidget):
    """Draw means, descriptive confidence intervals and a failure threshold."""

    def __init__(self) -> None:
        super().__init__()
        self.study: SweepResult | None = None
        self.x_label = ""
        self.y_label = ""
        self.empty_label = ""
        self.setMinimumHeight(260)

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        if self.study is None:
            painter.setPen(QColor("#475569"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.empty_label)
            return
        study = self.study
        plot = QRectF(70, 28, max(50, self.width() - 110), max(50, self.height() - 90))
        rows = study.summary
        low_y = min(float(row["ci95_low"]) for row in rows)
        high_y = max(float(row["ci95_high"]) for row in rows)
        low_y = min(low_y, study.failure_threshold)
        high_y = max(high_y, study.failure_threshold)
        padding = max((high_y - low_y) * 0.1, 0.01)
        low_y = max(0.0, low_y - padding)
        high_y += padding
        low_x, high_x = study.levels[0], study.levels[-1]

        def position(level: float, value: float) -> QPointF:
            return QPointF(
                plot.left() + (level - low_x) / (high_x - low_x) * plot.width(),
                plot.bottom() - (value - low_y) / (high_y - low_y) * plot.height(),
            )

        painter.setPen(QPen(QColor("#cbd5e1"), 1.0))
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = plot.top() + fraction * plot.height()
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            value = high_y - fraction * (high_y - low_y)
            painter.setPen(QColor("#334155"))
            painter.drawText(5, int(y + 4), f"{value:.3g}")
            painter.setPen(QPen(QColor("#cbd5e1"), 1.0))
        painter.setPen(QPen(QColor("#ef4444"), 1.5, Qt.PenStyle.DashLine))
        threshold_y = position(low_x, study.failure_threshold).y()
        painter.drawLine(QPointF(plot.left(), threshold_y), QPointF(plot.right(), threshold_y))
        painter.setPen(QPen(QColor("#1d4ed8"), 2.5))
        for first, second in zip(rows, rows[1:]):
            painter.drawLine(
                position(float(first["level"]), float(first["mean"])),
                position(float(second["level"]), float(second["mean"])),
            )
        for row in rows:
            x = float(row["level"])
            lower = position(x, float(row["ci95_low"]))
            upper = position(x, float(row["ci95_high"]))
            center = position(x, float(row["mean"]))
            painter.drawLine(lower, upper)
            painter.drawLine(QPointF(lower.x() - 5, lower.y()), QPointF(lower.x() + 5, lower.y()))
            painter.drawLine(QPointF(upper.x() - 5, upper.y()), QPointF(upper.x() + 5, upper.y()))
            painter.setBrush(QColor("#1d4ed8"))
            painter.drawEllipse(center, 4.0, 4.0)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QColor("#334155"))
            painter.drawText(int(center.x() - 20), int(plot.bottom() + 18), f"{x:.3g}")
            painter.setPen(QPen(QColor("#1d4ed8"), 2.5))
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.drawRect(plot)
        painter.drawText(int(plot.center().x() - 70), self.height() - 10, self.x_label)
        painter.drawText(72, 17, self.y_label)
