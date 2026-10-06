"""Recorded-data replay page with real measured channels and honest evaluation."""

import numpy as np
from PySide6.QtCore import QPointF, QThread, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from slamcart3d.datasets.replay import export_replay, replay_recording


class ReplayWorker(QThread):
    completed = Signal(object, str)
    failed = Signal(str)

    def __init__(self, source, config, reference, frames, output, timestamp_policy="strict"):
        super().__init__()
        self.timestamp_policy = timestamp_policy
        self.source, self.config, self.reference, self.frames, self.output = (
            source,
            config,
            reference,
            frames,
            output,
        )

    def run(self):
        try:
            result = replay_recording(
                self.source,
                self.config,
                reference_path=self.reference or None,
                max_frames=self.frames,
                timestamp_policy=self.timestamp_policy,
            )
            target = export_replay(result, self.output)
            self.completed.emit(result, str(target))
        except Exception as exc:
            self.failed.emit(str(exc))


class RecordedCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.result = None
        self.setMinimumHeight(240)

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101b2b"))
        if self.result is None:
            return
        result = self.result
        sample = result.map_points[:: max(1, int(np.ceil(len(result.map_points) / 4000)))]
        scene = np.vstack((sample, result.estimated[:, :2]))
        lo, hi = scene.min(axis=0), scene.max(axis=0)
        scale = min((self.width() - 40) / max(hi[0] - lo[0], 1), (self.height() - 40) / max(hi[1] - lo[1], 1))

        def screen(points):
            xy = (points - (lo + hi) / 2) * scale
            return [QPointF(float(x + self.width() / 2), float(self.height() / 2 - y)) for x, y in xy]

        painter.setPen(QPen(QColor("#20ddff"), 2))
        for point in screen(sample):
            painter.drawPoint(point)
        painter.setPen(QPen(QColor("#ff963a"), 2))
        painter.drawPolyline(QPolygonF(screen(result.estimated[:, :2])))
        if result.reference is not None:
            painter.setPen(QPen(QColor("#35df85"), 2))
            painter.drawPolyline(QPolygonF(screen(result.reference[:, :2])))


class RecordedPage(QWidget):
    busy = Signal(bool)

    def __init__(self, get_config, get_output, can_start, show_error):
        super().__init__()
        self.get_config, self.get_output = get_config, get_output
        self.can_start, self.show_error = can_start, show_error
        self.worker = None
        self.last_result, self.last_path = None, None
        self.language = "es"
        self.failed = False
        layout = QVBoxLayout(self)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        form = QFormLayout()
        self.source = QLineEdit()
        self.reference = QLineEdit()
        self.frames = QSpinBox()
        self.frames.setRange(2, 20_000)
        self.frames.setValue(1000)
        self.file_button, self.folder_button, self.reference_button = (
            QPushButton(),
            QPushButton(),
            QPushButton(),
        )
        for field, controls in (
            (self.source, (self.file_button, self.folder_button)),
            (self.reference, (self.reference_button,)),
        ):
            row = QHBoxLayout()
            row.addWidget(field, 1)
            for control in controls:
                row.addWidget(control)
            form.addRow(row)
        self.frames_label = QLabel()
        form.addRow(self.frames_label, self.frames)
        self.time_label = QLabel()
        self.time_policy = QComboBox()
        self.time_policy.addItem("", "strict")
        self.time_policy.addItem("", "sort_unique")
        form.addRow(self.time_label, self.time_policy)
        layout.addLayout(form)
        self.file_button.clicked.connect(self._browse_file)
        self.folder_button.clicked.connect(self._browse_folder)
        self.reference_button.clicked.connect(self._browse_reference)
        self.run_button = QPushButton()
        self.run_button.clicked.connect(self.start)
        layout.addWidget(self.run_button)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.canvas = RecordedCanvas()
        layout.addWidget(self.canvas, 1)
        self.retranslate("es")

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "CARMEN FLASER", "", "CARMEN (*.clf *.log)")
        if path:
            self.source.setText(path)

    def _browse_folder(self):
        path = QFileDialog.getExistingDirectory(self, "CSV")
        if path:
            self.source.setText(path)

    def _browse_reference(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Referencia" if self.language == "es" else "Reference", "", "CSV (*.csv)"
        )
        if path:
            self.reference.setText(path)

    def start(self):
        if self.worker is not None or not self.can_start():
            return
        try:
            source = self.source.text().strip()
            output = self.get_output().strip()
            if not source or not output:
                raise ValueError("recording path and output folder are required")
            self.worker = ReplayWorker(
                source,
                self.get_config(),
                self.reference.text().strip(),
                self.frames.value(),
                output,
                self.time_policy.currentData(),
            )
            self.failed = False
            self.worker.completed.connect(self._completed)
            self.worker.failed.connect(self._failed)
            self.worker.finished.connect(self._finished)
            self.busy.emit(True)
            self.retranslate(self.language)
            self.worker.start()
        except Exception as exc:
            self.worker = None
            self.show_error(str(exc))

    def _completed(self, result, path):
        self.last_result, self.last_path = result, path
        self.canvas.result = result
        self.canvas.update()

    def _failed(self, detail):
        self.failed = True
        self.show_error(detail)

    def _finished(self):
        self.worker = None
        self.busy.emit(False)
        self.retranslate(self.language)

    def retranslate(self, language):
        self.language = language
        es = language == "es"
        self.hint.setText(
            "Reproduzca CARMEN FLASER o CSV medidos con el algoritmo del Experimento. Referencia opcional: "
            "timestamp_s,x_m,y_m,yaw_rad, en tiempo relativo. Sin referencia independiente no se evalúa la precisión. "
            "La odometría del registro nunca se considera verdad. Nube cian, trayectoria naranja, referencia verde."
            if es
            else "Replay CARMEN FLASER or measured CSV using the Experiment algorithm. Optional reference: "
            "timestamp_s,x_m,y_m,yaw_rad, in relative time. No accuracy evaluation without an independent reference. "
            "Log odometry is never ground truth. Cyan map, orange trajectory, green reference."
        )
        for control, label in (
            (self.file_button, "Archivo CARMEN" if es else "CARMEN file"),
            (self.folder_button, "Carpeta CSV" if es else "CSV folder"),
            (self.reference_button, "Referencia CSV (opcional)" if es else "Reference CSV (optional)"),
            (self.run_button, "Reproducir y exportar" if es else "Replay and export"),
        ):
            control.setText(label)
        self.frames_label.setText("Máximo de pasos [entero]" if es else "Maximum steps [integer]")
        self.time_label.setText("Tiempos CARMEN" if es else "CARMEN timestamps")
        self.time_policy.setItemText(0, "Estricto: rechazar desorden" if es else "Strict: reject disorder")
        self.time_policy.setItemText(
            1, "Ordenar y descartar tiempos duplicados" if es else "Sort and discard duplicate times"
        )
        self.time_policy.setToolTip(
            "No modifica el archivo. Registra inversiones y duplicados descartados en los metadatos."
            if es
            else "Does not modify the file. Logs inversions and discarded duplicate times in metadata."
        )
        if self.worker is not None:
            self.status.setText("Procesando mediciones…" if es else "Processing measurements…")
        elif self.failed:
            self.status.setText(
                "No se completó la reproducción; revise el error."
                if es
                else "Replay failed; inspect the error."
            )
        elif self.last_result is not None:
            result = self.last_result
            value = result.metrics.get("ate_rmse_m")
            accuracy = (
                (f"ATE alineado: {value:.4f} m" if es else f"Aligned ATE: {value:.4f} m")
                if value is not None
                else (
                    "Precisión no evaluada: falta referencia independiente."
                    if es
                    else "Accuracy not evaluated: independent reference unavailable."
                )
            )
            self.status.setText(
                f"{result.metadata['frames']} {'pasos' if es else 'steps'} · {accuracy}\n{self.last_path}"
            )
            if result.metadata.get("timestamp_policy") == "sort_unique":
                self.status.setText(
                    self.status.text()
                    + (
                        f"\nOrden temporal aplicado: {result.metadata['original_temporal_inversions']} inversiones, "
                        f"{result.metadata['duplicate_timestamps_discarded']} duplicados descartados."
                        if es
                        else f"\nTemporal ordering applied: {result.metadata['original_temporal_inversions']} inversions, "
                        f"{result.metadata['duplicate_timestamps_discarded']} duplicates discarded."
                    )
                )
        else:
            self.status.setText(
                "Seleccione mediciones para comenzar." if es else "Select measurements to begin."
            )
