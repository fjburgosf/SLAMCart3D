"""Bilingual step assessment with explicit units, thresholds and limitations."""

from dataclasses import replace

import numpy as np
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from slamcart3d.core.config import EvaluationConfig
from slamcart3d.metrics.live_quality import evaluate_live


class QualityPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.language = "es"
        self.config = EvaluationConfig()
        self.result = None
        self.quality = None
        self.map_time_s = None
        self.last_step = -1
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.navigation_label = QLabel()
        self.navigation_label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.addWidget(self.label)
        layout.addWidget(self.note)
        layout.addWidget(self.navigation_label)

    def reset(self) -> None:
        self.quality = None
        self.map_time_s = None
        self.last_step = -1
        self.navigation_label.clear()

    def show_result(self, result, config: EvaluationConfig, live: bool = False) -> None:
        self.result, self.config = result, config
        index = len(result.timestamps) - 1 if result is not None else 0
        if (
            live
            and result is not None
            and self.quality is not None
            and index > self.last_step
            and index % 10
            and not hasattr(result, "metrics")
        ):
            delta = result.estimated[-1] - result.ground_truth[-1]
            self.quality = replace(
                self.quality,
                position_m=float(np.linalg.norm(delta[:2])),
                heading_rad=float(abs((delta[2] + np.pi) % (2 * np.pi) - np.pi)),
                sampled=True,
            )
        else:
            self.quality = evaluate_live(result) if result is not None else None
            self.map_time_s = float(result.timestamps[-1]) if result is not None else None
        self.last_step = index
        self.retranslate(self.language)

    def retranslate(self, language: str) -> None:
        self.language = language
        es = language == "es"
        navigation = getattr(self.result, "navigation", getattr(self.result, "diagnostics", {}))
        state = navigation.get("navigation_state")
        states = {
            "initial_scan": ("tomando el primer barrido", "acquiring the first scan"),
            "following_frontier": (
                "hacia una frontera observada/desconocida",
                "heading to an observed/unknown frontier",
            ),
            "goal_reached": ("destino alcanzado; buscando otro", "goal reached; selecting another"),
            "obstacle_turn": (
                "avance detenido por obstáculo medido",
                "translation stopped by a measured obstacle",
            ),
            "sensor_unavailable": (
                "detenido: mediciones insuficientes",
                "stopped: insufficient measurements",
            ),
            "no_reachable_frontier": (
                "sin frontera alcanzable; gira para observar, no significa mapa completo",
                "no reachable frontier; rotating to observe, not a complete-map claim",
            ),
        }
        navigation_text = (
            ("Exploración: " if es else "Exploration: ")
            + states.get(state, (str(state), str(state)))[0 if es else 1]
            if state
            else ""
        )
        if state and navigation.get("navigation_goal_x_m") is not None:
            navigation_text += (
                f" · x={navigation['navigation_goal_x_m']:.2f} m, y={navigation['navigation_goal_y_m']:.2f} m"
            )
        self.navigation_label.setText(navigation_text)
        q = self.quality
        if q is None:
            self.label.setText(
                "Ejecute una corrida para evaluar el paso mostrado."
                if es
                else "Run an experiment to assess the displayed step."
            )
            self.note.setText("")
            return
        position_ok = q.position_m <= self.config.position_limit_m
        position = (
            ("dentro del límite" if position_ok else "supera el límite")
            if es
            else ("within limit" if position_ok else "exceeds limit")
        )
        title = "Error directo de posición" if es else "Direct position error"
        heading = "Giro" if es else "Heading"
        self.label.setText(
            f"{title}: {q.position_m:.4f} m ({position}; ≤ {self.config.position_limit_m:g} m)"
            f" · {heading}: {q.heading_rad:.4f} rad"
        )
        self.label.setStyleSheet(f"color: {'#166534' if position_ok else '#b91c1c'};")
        if q.wall_distance_m is None:
            detail = (
                "Mapa todavía no evaluable; no significa error cero."
                if es
                else ("Map not evaluable yet; this does not mean zero error.")
            )
        else:
            map_ok = q.wall_distance_m <= self.config.wall_distance_limit_m
            quality = (
                ("dentro del límite" if map_ok else "supera el límite")
                if es
                else ("within limit" if map_ok else "exceeds limit")
            )
            coverage_ok = q.wall_coverage >= self.config.coverage_target
            coverage = (
                ("objetivo alcanzado" if coverage_ok else "cobertura incompleta")
                if es
                else ("target reached" if coverage_ok else "incomplete coverage")
            )
            detail = (
                f"Distancia a paredes: {q.wall_distance_m:.4f} m ({quality}; "
                f"≤ {self.config.wall_distance_limit_m:g} m). Cobertura: {q.wall_coverage:.1%} "
                f"({coverage}; objetivo {self.config.coverage_target:.0%})."
                if es
                else f"Wall distance: {q.wall_distance_m:.4f} m ({quality}; "
                f"≤ {self.config.wall_distance_limit_m:g} m). Coverage: {q.wall_coverage:.1%} "
                f"({coverage}; target {self.config.coverage_target:.0%})."
            )
        note = (
            "Evaluación con referencia, no entrada del SLAM. Cobertura de paredes con tolerancia 0.25 m; "
            "vista muestreada, consulte métricas finales. Precisión y cobertura son distintas."
            if es
            else "Reference-based evaluation, not SLAM input. Wall coverage at 0.25 m tolerance; "
            "sampled display, consult final metrics. Accuracy and coverage are different."
        )
        if not q.sampled:
            note = note.replace(
                "vista muestreada, consulte métricas finales", "datos completos del paso mostrado"
            )
            note = note.replace("sampled display, consult final metrics", "full data of the displayed step")
        if self.result is not None and getattr(self.result, "online_estimated", None) is not None:
            note += " Resultado optimizado al finalizar." if es else " Result optimized at completion."
        if (
            self.result is not None
            and self.map_time_s is not None
            and self.map_time_s < self.result.timestamps[-1]
        ):
            note += (
                f" Mapa evaluado en t={self.map_time_s:.2f} s; posición en el paso mostrado."
                if es
                else f" Map assessed at t={self.map_time_s:.2f} s; position at the displayed step."
            )
        self.note.setText(detail + "\n" + note)
