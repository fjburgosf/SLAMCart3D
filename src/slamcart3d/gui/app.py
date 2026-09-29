"""Bilingual Qt experiment runner with real metrics and trajectory display."""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from slamcart3d.core.config import ExperimentConfig, WallConfig
from slamcart3d.core.provenance import source_sha256
from slamcart3d.datasets.generator import generate_dataset
from slamcart3d.experiments.batch import export_batch, run_batch
from slamcart3d.experiments.sweep import SWEEP_PARAMETERS
from slamcart3d.gui.paths import (
    bundled_example_path,
    configure_local_runtime,
    default_datasets_path,
    default_results_path,
    export_to_requested_path,
)
from slamcart3d.gui.robustness import FrontierWorker, RobustnessCanvas
from slamcart3d.gui.sweep import SweepCanvas, SweepWorker
from slamcart3d.gui.viewer3d import PointCloudCanvas
from slamcart3d.io.export import IncrementalRunWriter, export_result
from slamcart3d.robustness.study import run_frontier_study
from slamcart3d.simulation.runner import RunPreview, RunResult, RunStarted, RunStep, run_simulation

TEXT = {
    "es": {
        "setup": "Experimento",
        "scene": "Trayectoria y mapa",
        "occupancy": "Ocupación 2D",
        "empty_occupancy": "Ejecute un experimento con el mapa de ocupación activado para ver celdas libres, ocupadas y desconocidas.",
        "occupancy_free": "Libre observado",
        "occupancy_occupied": "Ocupado estimado",
        "occupancy_unknown": "Pared sin observar",
        "cloud3d": "Nube 3D",
        "dataset": "Dataset",
        "dataset_hint": "Genere observaciones ideales y medidas sin ejecutar un algoritmo SLAM. La vista previa muestra un cuadro RGB medido cuando la cámara está activa.",
        "dataset_output": "Carpeta de datasets",
        "dataset_run": "Generar dataset",
        "dataset_running": "Generando dataset…",
        "dataset_empty": "Genere un dataset para consultar sus archivos.",
        "dataset_completed": "Dataset guardado en {path}",
        "dataset_no_camera": "Este dataset no tiene imágenes de cámara medidas.",
        "cloud3d_hint": "El piso cuadriculado está en z=0; las superficies azules son paredes verdaderas, la nube cian es el mapa estimado. Arrastre para girar y use la rueda para acercar. No es SLAM 6DoF.",
        "step_previous": "◀ Anterior",
        "step_next": "Siguiente ▶",
        "step_label": "Paso {index}/{total} · t = {time:.2f} s",
        "step_empty": "Ejecute un experimento para recorrer sus pasos.",
        "step_saved": "Guardado incremental: {path}",
        "show_reference": "Mostrar realidad simulada",
        "sweep": "Estudio controlado",
        "sweep_hint": "Cambie una sola condición y repita las mismas semillas en cada nivel. La curva muestra la media y un intervalo descriptivo del 95 %; la línea roja es el umbral de fallo.",
        "sweep_parameter": "Variable experimental",
        "sweep_min": "Mínimo",
        "sweep_max": "Máximo",
        "sweep_levels": "Niveles",
        "sweep_seeds": "Semillas pareadas",
        "sweep_metric": "Métrica de respuesta",
        "sweep_threshold": "Umbral de fallo",
        "sweep_run": "Ejecutar estudio",
        "sweep_running": "Ejecutando estudio…",
        "sweep_empty": "Defina una variable, al menos dos niveles y varias semillas para evaluar una hipótesis.",
        "sweep_completed": "Estudio guardado en {path}",
        "sweep_failure": "Fracción de fallos",
        "empty_cloud3d": "Ejecute un experimento con lidar3d.enabled=true para ver la nube 3D.",
        "metrics": "Métricas",
        "batch": "Lotes",
        "batch_seeds": "Semillas [enteros separados por coma]",
        "batch_worlds": "Entornos",
        "batch_algorithms": "Algoritmos",
        "batch_run": "Ejecutar lote (1 proceso)",
        "batch_running": "Ejecutando lote…",
        "batch_hint": "Seleccione entornos y algoritmos. Se exportan todas las corridas y su resumen estadístico.",
        "batch_empty": "Ejecute un lote para consultar medias por entorno, algoritmo y métrica.",
        "batch_completed": "Lote guardado en {path}",
        "batch_mean": "Media",
        "frontier": "Robustez ASRF",
        "frontier_hint": "Mapa exploratorio de P(éxito); no es una probabilidad calibrada. Cada muestra evalúa todas las semillas.",
        "frontier_x": "Variable horizontal",
        "frontier_y": "Variable vertical",
        "frontier_min": "Límite inferior",
        "frontier_max": "Límite superior",
        "frontier_metric": "Métrica de fallo",
        "frontier_threshold": "Umbral de fallo",
        "frontier_seeds": "Semillas [enteros separados por coma]",
        "frontier_budget": "Presupuesto [condiciones]",
        "frontier_run": "Estimar frontera ASRF",
        "frontier_running": "Estimando frontera…",
        "frontier_empty": "Ejecute un estudio ASRF para ver el mapa de robustez.",
        "frontier_completed": "Estudio ASRF guardado en {path}",
        "frontier_success": "Éxito",
        "frontier_failure": "Fallo",
        "frontier_requires_3d": "La métrica de paredes 3D requiere activar el LiDAR multicapa.",
        "frontier_requires_srs": "La métrica SRS requiere activar el índice SRS en el experimento.",
        "frontier_requires_occupancy": "La métrica de ocupación requiere activar el mapa de ocupación.",
        "param_wheel_rate_noise_std_rad_s": "Ruido de rueda σ [rad/s]",
        "param_odometry_dropout_probability": "Pérdida de odometría [0–1]",
        "param_lidar_noise_std_m": "Ruido LiDAR σ [m]",
        "param_lidar_dropout_probability": "Pérdida LiDAR [0–1]",
        "param_imu_gyro_bias_rad_s": "Sesgo giroscópico [rad/s]",
        "param_simulation.linear_velocity_m_s": "Velocidad longitudinal [m/s]",
        "param_simulation.yaw_rate_rad_s": "Velocidad de giro [rad/s]",
        "language": "Idioma",
        "config_path": "Archivo de configuración",
        "browse": "Buscar…",
        "load": "Cargar",
        "save": "Guardar configuración…",
        "run": "Ejecutar experimento",
        "running": "Ejecutando…",
        "ready": "Listo. Configure o cargue un experimento.",
        "completed": "Resultado guardado en {path}",
        "error_title": "Error de experimento",
        "error": "No se pudo completar la operación. Revise la configuración.\n{detail}",
        "wait_for_run": "Espere a que termine el experimento antes de cerrar la ventana.",
        "empty_scene": "Ejecute un experimento para ver la trayectoria y el mapa.",
        "empty_metrics": "Ejecute un experimento para ver métricas.",
        "metric_name": "Métrica",
        "metric_value": "Valor",
        "truth": "Trayectoria verdadera",
        "estimate": "Trayectoria estimada",
        "map_points": "Mapa estimado",
        "world_wall": "Pared verdadera",
        "moving_object": "Objeto móvil (posición final)",
        "name": "Nombre del experimento",
        "seed": "Semilla [entero]",
        "world": "Entorno",
        "algorithm": "Algoritmo",
        "motion": "Perfil de movimiento",
        "duration": "Duración [s]",
        "timestep": "Paso de simulación [s]",
        "speed": "Velocidad longitudinal [m/s]",
        "yaw_rate": "Velocidad de giro [rad/s]",
        "lidar_noise": "Ruido LiDAR σ [m]",
        "lidar_dropout": "Pérdida LiDAR [0–1]",
        "lidar3d_enabled": "Activar LiDAR multicapa 3D",
        "camera_enabled": "Activar cámara RGB-D sintética",
        "score_enabled": "Calcular índice SRS opcional",
        "occupancy_enabled": "Evaluar mapa de ocupación 2D",
        "dropout_region_enabled": "Pérdida localizada por zona",
        "dropout_region_x_min_m": "Zona: x mínimo [m]",
        "dropout_region_x_max_m": "Zona: x máximo [m]",
        "dropout_region_y_min_m": "Zona: y mínimo [m]",
        "dropout_region_y_max_m": "Zona: y máximo [m]",
        "imu_bias": "Sesgo giroscópico [rad/s]",
        "imu_dropout": "Pérdida IMU [0–1]",
        "output": "Carpeta de resultados",
        "world_indoor_simple": "Interior simple",
        "world_indoor_corridor": "Corredor interior",
        "world_indoor_complex": "Interior de varias salas",
        "world_warehouse_shelves": "Almacén con estanterías",
        "world_indoor_dynamic": "Interior dinámico",
        "world_outdoor_blocks": "Exterior con edificios",
        "world_custom": "Entorno personalizado",
        "world_segments": "Paredes 3D (x1, y1, x2, y2, altura; una por línea)",
        "stop_at_walls": "Detener carrito al chocar con paredes",
        "algorithm_dead_reckoning": "Odometría de ruedas",
        "algorithm_icp_odometry": "Odometría LiDAR ICP",
        "algorithm_wheel_imu_fusion": "Fusión ruedas/IMU",
        "algorithm_adaptive_hybrid": "Híbrido adaptativo propuesto",
        "motion_constant": "Constante",
        "motion_sinusoidal_yaw": "Giro sinusoidal",
        "motion_stop_turn": "Avance y giro alternados",
        "motion_scripted": "Trayectoria por tramos (YAML)",
    },
    "en": {
        "setup": "Experiment",
        "scene": "Trajectory and map",
        "occupancy": "2D occupancy",
        "empty_occupancy": "Run an experiment with occupancy mapping enabled to view free, occupied, and unknown cells.",
        "occupancy_free": "Observed free",
        "occupancy_occupied": "Estimated occupied",
        "occupancy_unknown": "Unobserved wall",
        "cloud3d": "3D point cloud",
        "dataset": "Dataset",
        "dataset_hint": "Generate ideal and measured observations without running a SLAM algorithm. The preview shows a measured RGB frame when the camera is enabled.",
        "dataset_output": "Dataset folder",
        "dataset_run": "Generate dataset",
        "dataset_running": "Generating dataset…",
        "dataset_empty": "Generate a dataset to inspect its files.",
        "dataset_completed": "Dataset saved to {path}",
        "dataset_no_camera": "This dataset has no measured camera images.",
        "cloud3d_hint": "The gridded floor is z=0; blue surfaces are true walls and cyan points are the estimated map. Drag to rotate and use the wheel to zoom. This is not 6DoF SLAM.",
        "step_previous": "◀ Previous",
        "step_next": "Next ▶",
        "step_label": "Step {index}/{total} · t = {time:.2f} s",
        "step_empty": "Run an experiment to inspect its steps.",
        "step_saved": "Incremental files: {path}",
        "show_reference": "Show simulated ground truth",
        "sweep": "Controlled study",
        "sweep_hint": "Vary one condition and repeat the same seeds at every level. The curve shows the mean and a descriptive 95% interval; the red line is the failure threshold.",
        "sweep_parameter": "Experimental variable",
        "sweep_min": "Minimum",
        "sweep_max": "Maximum",
        "sweep_levels": "Levels",
        "sweep_seeds": "Paired seeds",
        "sweep_metric": "Response metric",
        "sweep_threshold": "Failure threshold",
        "sweep_run": "Run study",
        "sweep_running": "Running study…",
        "sweep_empty": "Choose one variable, at least two levels and repeated seeds to test a hypothesis.",
        "sweep_completed": "Study saved to {path}",
        "sweep_failure": "Failure fraction",
        "empty_cloud3d": "Run an experiment with lidar3d.enabled=true to view the 3D cloud.",
        "metrics": "Metrics",
        "batch": "Batch",
        "batch_seeds": "Seeds [comma-separated integers]",
        "batch_worlds": "Environments",
        "batch_algorithms": "Algorithms",
        "batch_run": "Run batch (1 process)",
        "batch_running": "Running batch…",
        "batch_hint": "Select environments and algorithms. All runs and a statistical summary are exported.",
        "batch_empty": "Run a batch to inspect means by environment, algorithm, and metric.",
        "batch_completed": "Batch saved to {path}",
        "batch_mean": "Mean",
        "frontier": "ASRF robustness",
        "frontier_hint": "Exploratory P(success) map; it is not a calibrated probability. Each sample evaluates all seeds.",
        "frontier_x": "Horizontal variable",
        "frontier_y": "Vertical variable",
        "frontier_min": "Lower bound",
        "frontier_max": "Upper bound",
        "frontier_metric": "Failure metric",
        "frontier_threshold": "Failure threshold",
        "frontier_seeds": "Seeds [comma-separated integers]",
        "frontier_budget": "Budget [conditions]",
        "frontier_run": "Estimate ASRF frontier",
        "frontier_running": "Estimating frontier…",
        "frontier_empty": "Run an ASRF study to view the robustness map.",
        "frontier_completed": "ASRF study saved to {path}",
        "frontier_success": "Success",
        "frontier_failure": "Failure",
        "frontier_requires_3d": "The 3D wall metric requires enabling the multilayer LiDAR.",
        "frontier_requires_srs": "The SRS metric requires enabling the SRS index in the experiment.",
        "frontier_requires_occupancy": "The occupancy metric requires enabling the occupancy map.",
        "param_wheel_rate_noise_std_rad_s": "Wheel noise σ [rad/s]",
        "param_odometry_dropout_probability": "Odometry dropout [0–1]",
        "param_lidar_noise_std_m": "LiDAR noise σ [m]",
        "param_lidar_dropout_probability": "LiDAR dropout [0–1]",
        "param_imu_gyro_bias_rad_s": "Gyroscope bias [rad/s]",
        "param_simulation.linear_velocity_m_s": "Forward speed [m/s]",
        "param_simulation.yaw_rate_rad_s": "Yaw rate [rad/s]",
        "language": "Language",
        "config_path": "Configuration file",
        "browse": "Browse…",
        "load": "Load",
        "save": "Save configuration…",
        "run": "Run experiment",
        "running": "Running…",
        "ready": "Ready. Configure or load an experiment.",
        "completed": "Result saved to {path}",
        "error_title": "Experiment error",
        "error": "The operation could not be completed. Check the configuration.\n{detail}",
        "wait_for_run": "Wait for the experiment to finish before closing the window.",
        "empty_scene": "Run an experiment to view the trajectory and map.",
        "empty_metrics": "Run an experiment to view metrics.",
        "metric_name": "Metric",
        "metric_value": "Value",
        "truth": "Ground-truth trajectory",
        "estimate": "Estimated trajectory",
        "map_points": "Estimated map",
        "world_wall": "True wall",
        "moving_object": "Moving object (final position)",
        "name": "Experiment name",
        "seed": "Seed [integer]",
        "world": "Environment",
        "algorithm": "Algorithm",
        "motion": "Motion profile",
        "duration": "Duration [s]",
        "timestep": "Simulation step [s]",
        "speed": "Forward speed [m/s]",
        "yaw_rate": "Yaw rate [rad/s]",
        "lidar_noise": "LiDAR noise σ [m]",
        "lidar_dropout": "LiDAR dropout [0–1]",
        "lidar3d_enabled": "Enable multilayer 3D LiDAR",
        "camera_enabled": "Enable synthetic RGB-D camera",
        "score_enabled": "Calculate optional SRS index",
        "occupancy_enabled": "Evaluate 2D occupancy map",
        "dropout_region_enabled": "Enable spatial dropout zone",
        "dropout_region_x_min_m": "Zone: minimum x [m]",
        "dropout_region_x_max_m": "Zone: maximum x [m]",
        "dropout_region_y_min_m": "Zone: minimum y [m]",
        "dropout_region_y_max_m": "Zone: maximum y [m]",
        "imu_bias": "Gyroscope bias [rad/s]",
        "imu_dropout": "IMU dropout [0–1]",
        "output": "Results folder",
        "world_indoor_simple": "Simple indoor",
        "world_indoor_corridor": "Indoor corridor",
        "world_indoor_complex": "Multi-room indoor",
        "world_warehouse_shelves": "Warehouse shelves",
        "world_indoor_dynamic": "Dynamic indoor",
        "world_outdoor_blocks": "Outdoor buildings",
        "world_custom": "Custom environment",
        "world_segments": "3D walls (x1, y1, x2, y2, height; one per line)",
        "stop_at_walls": "Stop cart at walls",
        "algorithm_dead_reckoning": "Wheel odometry",
        "algorithm_icp_odometry": "LiDAR ICP odometry",
        "algorithm_wheel_imu_fusion": "Wheel/IMU fusion",
        "algorithm_adaptive_hybrid": "Proposed adaptive hybrid",
        "motion_constant": "Constant",
        "motion_sinusoidal_yaw": "Sinusoidal yaw",
        "motion_stop_turn": "Alternating drive and turn",
        "motion_scripted": "Segmented trajectory (YAML)",
    },
}

TOOLTIPS = {
    "stop_at_walls": (
        "Bloquea el avance del carrito real si su trayectoria se acerca a una pared a menos de medio ancho de ejes.",
        "Blocks the true cart when its path comes within half the wheelbase of a wall.",
    ),
    "world_segments": (
        "En entorno personalizado, escriba una pared por línea en metros: x1, y1, x2, y2, altura.",
        "For a custom environment, enter one wall per line in metres: x1, y1, x2, y2, height.",
    ),
    "name": (
        "Identificador legible que se guarda con cada corrida.",
        "Readable identifier stored with every run.",
    ),
    "seed": (
        "Entero no negativo para repetir mundo y sensores.",
        "Nonnegative integer for replaying world and sensors.",
    ),
    "world": (
        "Geometría 2.5D reproducible elegida para la corrida.",
        "Reproducible 2.5D geometry for the run.",
    ),
    "algorithm": (
        "Baseline de estimación; ninguno cierra bucles.",
        "Estimation baseline; none closes loops.",
    ),
    "motion": (
        "Perfil cinemático. El modo por tramos requiere motion_segments en el YAML; la duración total debe coincidir.",
        "Kinematic profile. Segmented mode requires motion_segments in YAML; total duration must match.",
    ),
    "duration": (
        "Tiempo total simulado; debe ser múltiplo del paso.",
        "Total simulated time; must be a multiple of the step.",
    ),
    "timestep": ("Intervalo de integración y observación.", "Integration and observation interval."),
    "speed": (
        "Velocidad longitudinal ordenada; puede ser negativa.",
        "Commanded forward speed; may be negative.",
    ),
    "yaw_rate": (
        "Velocidad angular máxima o constante de guiñada.",
        "Maximum or constant commanded yaw rate.",
    ),
    "lidar_noise": (
        "Desviación estándar del ruido gaussiano por haz válido.",
        "Gaussian standard deviation per valid LiDAR beam.",
    ),
    "lidar_dropout": (
        "Probabilidad independiente de perder cada haz LiDAR.",
        "Independent probability of losing each LiDAR beam.",
    ),
    "lidar3d_enabled": (
        "Activa barridos 3D; número de capas, haces y campo visual se conservan del YAML.",
        "Enable 3D scans; layer count, beam count, and field of view are preserved from YAML.",
    ),
    "camera_enabled": (
        "Activa imágenes RGB-D esquemáticas para el dataset; los baselines SLAM actuales no las utilizan.",
        "Enable schematic RGB-D dataset images; current SLAM baselines do not use them.",
    ),
    "score_enabled": (
        "Índice compuesto opcional; escalas y pesos se conservan del YAML. No sustituye las métricas individuales.",
        "Optional composite index; scales and weights are preserved from YAML. It does not replace individual metrics.",
    ),
    "occupancy_enabled": (
        "Genera celdas ocupadas/libres desde rayos LiDAR válidos; resolución y salto de haces se conservan del YAML.",
        "Build occupied/free cells from valid LiDAR rays; resolution and beam stride are preserved from YAML.",
    ),
    "dropout_region_enabled": (
        "Limita la pérdida LiDAR, odometría e IMU a una zona rectangular del mundo; usa la posición real.",
        "Limit LiDAR, odometry, and IMU dropout to a world-frame rectangle using ground-truth position.",
    ),
    "dropout_region_x_min_m": (
        "Borde izquierdo inclusivo de la zona de pérdida [m]. Debe ser menor que x máximo.",
        "Inclusive left edge of the dropout zone [m]. Must be below maximum x.",
    ),
    "dropout_region_x_max_m": (
        "Borde derecho inclusivo de la zona de pérdida [m]. Debe superar x mínimo.",
        "Inclusive right edge of the dropout zone [m]. Must exceed minimum x.",
    ),
    "dropout_region_y_min_m": (
        "Borde inferior inclusivo de la zona de pérdida [m]. Debe ser menor que y máximo.",
        "Inclusive lower edge of the dropout zone [m]. Must be below maximum y.",
    ),
    "dropout_region_y_max_m": (
        "Borde superior inclusivo de la zona de pérdida [m]. Debe superar y mínimo.",
        "Inclusive upper edge of the dropout zone [m]. Must exceed minimum y.",
    ),
    "imu_bias": (
        "Sesgo inicial del giroscopio planar; se suma a la guiñada.",
        "Initial planar gyro bias added to yaw rate.",
    ),
    "imu_dropout": (
        "Probabilidad de perder ambos canales de un paquete IMU.",
        "Probability of losing both channels of an IMU packet.",
    ),
    "output": (
        "Carpeta donde se crea una subcarpeta EXP por corrida.",
        "Folder where an EXP subfolder is created per run.",
    ),
    "dataset_output": (
        "Carpeta donde se crea una subcarpeta DATASET por generación.",
        "Folder where a DATASET subfolder is created per generation.",
    ),
    "config_path": ("Archivo YAML o JSON del experimento.", "Experiment YAML or JSON file."),
    "batch_seeds": (
        "Semillas únicas, enteras y no negativas; se ejecuta cada combinación seleccionada.",
        "Unique nonnegative integer seeds; every selected combination is run.",
    ),
    "batch_worlds": (
        "Cada entorno se evalúa con todos los algoritmos y semillas seleccionados.",
        "Each environment is evaluated with all selected algorithms and seeds.",
    ),
    "batch_algorithms": (
        "Cada baseline se evalúa con todos los entornos y semillas seleccionados.",
        "Each baseline is evaluated with all selected environments and seeds.",
    ),
    "frontier_x": (
        "Condición experimental horizontal; escoja límites físicamente válidos.",
        "Horizontal experimental condition; choose physically valid bounds.",
    ),
    "frontier_y": (
        "Condición experimental vertical, diferente de la horizontal.",
        "Vertical experimental condition, distinct from the horizontal one.",
    ),
    "frontier_metric": (
        "El fallo se decide por el umbral, respetando si la métrica es mejor alta o baja.",
        "Failure follows the threshold, respecting whether high or low values are better.",
    ),
    "frontier_budget": (
        "Número máximo de condiciones adaptativas; cada una se repite con todas las semillas.",
        "Maximum adaptive conditions; each is repeated for every seed.",
    ),
}

FRONTIER_PARAMETERS = (
    "wheel_rate_noise_std_rad_s",
    "odometry_dropout_probability",
    "lidar_noise_std_m",
    "lidar_dropout_probability",
    "imu_gyro_bias_rad_s",
    "simulation.linear_velocity_m_s",
    "simulation.yaw_rate_rad_s",
)
FRONTIER_METRICS = (
    "ate_rmse_m",
    "map_f1_0p25",
    "map3d_wall_f1_0p25",
    "occupancy_occupied_iou",
    "srs",
    "tracking_availability",
)

METRIC_LABELS = {
    "ate_rmse_m": ("ATE RMSE [m]", "ATE RMSE [m]"),
    "rpe_translation_rmse_m": ("RPE traslacional RMSE [m]", "Translational RPE RMSE [m]"),
    "rpe_rotation_rmse_rad": ("RPE angular RMSE [rad]", "Rotational RPE RMSE [rad]"),
    "final_position_error_m": ("Error final de posición [m]", "Final position error [m]"),
    "drift_per_meter": ("Deriva por metro [1]", "Drift per metre [1]"),
    "tracking_availability": ("Disponibilidad de poses finitas [1]", "Finite-pose availability [1]"),
    "ground_truth_static_wall_contact_steps": (
        "Pasos con contacto pared estática [n]",
        "Static-wall contact steps [n]",
    ),
    "wall_blocked_steps": ("Pasos bloqueados por pared [n]", "Wall-blocked steps [n]"),
    "icp_registration_availability": (
        "Disponibilidad de registro ICP [1]",
        "ICP registration availability [1]",
    ),
    "map_accuracy_mean_m": ("Distancia mapa→pared [m]", "Map-to-wall distance [m]"),
    "map_completeness_mean_m": ("Distancia pared→mapa [m]", "Wall-to-map distance [m]"),
    "map_chamfer_mean_m": ("Distancia Chamfer [m]", "Chamfer distance [m]"),
    "map_precision_0p25": ("Precisión de mapa a 0.25 m [1]", "Map precision at 0.25 m [1]"),
    "map_recall_0p25": ("Cobertura de mapa a 0.25 m [1]", "Map recall at 0.25 m [1]"),
    "map_f1_0p25": ("F1 de mapa a 0.25 m [1]", "Map F1 at 0.25 m [1]"),
    "map3d_wall_accuracy_mean_m": ("Distancia nube 3D→pared [m]", "3D cloud-to-wall distance [m]"),
    "map3d_wall_completeness_mean_m": (
        "Distancia pared 3D→nube [m]",
        "3D wall-to-cloud distance [m]",
    ),
    "map3d_wall_chamfer_mean_m": ("Chamfer 3D de paredes [m]", "3D wall Chamfer distance [m]"),
    "map3d_wall_precision_0p25": (
        "Precisión de pared 3D a 0.25 m [1]",
        "3D wall precision at 0.25 m [1]",
    ),
    "map3d_wall_recall_0p25": (
        "Cobertura de pared 3D a 0.25 m [1]",
        "3D wall recall at 0.25 m [1]",
    ),
    "map3d_wall_f1_0p25": ("F1 de pared 3D a 0.25 m [1]", "3D wall F1 at 0.25 m [1]"),
    "occupancy_occupied_precision": ("Precisión de celdas ocupadas [1]", "Occupied-cell precision [1]"),
    "occupancy_occupied_recall": ("Cobertura de celdas ocupadas [1]", "Occupied-cell recall [1]"),
    "occupancy_occupied_f1": ("F1 de celdas ocupadas [1]", "Occupied-cell F1 [1]"),
    "occupancy_occupied_iou": ("IoU de celdas ocupadas [1]", "Occupied-cell IoU [1]"),
    "occupancy_observed_free_accuracy": (
        "Exactitud de celdas libres observadas [1]",
        "Observed-free-cell accuracy [1]",
    ),
    "occupancy_occupied_cells": ("Celdas ocupadas [n]", "Occupied cells [n]"),
    "occupancy_observed_free_cells": ("Celdas libres observadas [n]", "Observed free cells [n]"),
    "occupancy_reference_occupied_cells": (
        "Celdas ocupadas de referencia [n]",
        "Reference occupied cells [n]",
    ),
    "srs": ("Índice de robustez SRS [0–1]", "SRS robustness index [0–1]"),
    "srs_ate_quality": ("SRS: calidad ATE [0–1]", "SRS: ATE quality [0–1]"),
    "srs_rpe_translation_quality": ("SRS: calidad RPE [0–1]", "SRS: RPE quality [0–1]"),
    "srs_map_f1_quality": ("SRS: calidad mapa F1 [0–1]", "SRS: map F1 quality [0–1]"),
    "srs_tracking_availability_quality": (
        "SRS: calidad disponibilidad [0–1]",
        "SRS: availability quality [0–1]",
    ),
    "srs_weight_sensitivity_ate": (
        "Sensibilidad SRS: peso ATE [1]",
        "SRS sensitivity: ATE weight [1]",
    ),
    "srs_weight_sensitivity_rpe_translation": (
        "Sensibilidad SRS: peso RPE [1]",
        "SRS sensitivity: RPE weight [1]",
    ),
    "srs_weight_sensitivity_map_f1": (
        "Sensibilidad SRS: peso mapa [1]",
        "SRS sensitivity: map weight [1]",
    ),
    "srs_weight_sensitivity_tracking_availability": (
        "Sensibilidad SRS: peso disponibilidad [1]",
        "SRS sensitivity: availability weight [1]",
    ),
    "runtime_s": ("Tiempo de cómputo [s]", "Computation time [s]"),
}


class ExperimentWorker(QThread):
    """Execute core simulation without blocking the Qt event loop."""

    completed = Signal(object, str)
    failed = Signal(str)
    started_run = Signal(object, str)
    step_ready = Signal(object)

    def __init__(self, config: ExperimentConfig, output_root: Path) -> None:
        super().__init__()
        self.config = config
        self.output_root = output_root

    def run(self) -> None:
        writer: IncrementalRunWriter | None = None

        def on_start(started: RunStarted) -> None:
            nonlocal writer
            writer = IncrementalRunWriter(started, self.output_root)
            self.started_run.emit(started, str(writer.target))

        def on_step(step: RunStep) -> None:
            assert writer is not None
            writer.append(step)
            self.step_ready.emit(step)
            self.msleep(70)

        try:
            result = run_simulation(self.config, on_start=on_start, on_step=on_step)
            assert writer is not None
            writer.close()
            target = export_result(result, self.output_root, reuse_existing=True)
            writer.complete()
            self.completed.emit(result, str(target))
        except Exception as exc:
            if writer is not None:
                writer.close()
                writer.fail(str(exc))
            self.failed.emit(str(exc))


class BatchWorker(QThread):
    """Run and export a batch without blocking GUI interaction."""

    completed = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        config: ExperimentConfig,
        seeds: tuple[int, ...],
        algorithms: tuple[str, ...],
        worlds: tuple[str, ...],
        output_root: Path,
    ) -> None:
        super().__init__()
        self.config = config
        self.seeds = seeds
        self.algorithms = algorithms
        self.worlds = worlds
        self.output_root = output_root

    def run(self) -> None:
        try:
            result = run_batch(self.config, self.seeds, self.algorithms, self.worlds, workers=1)
            target = export_to_requested_path(self.output_root, lambda root: export_batch(result, root))
            self.completed.emit(str(target))
        except Exception as exc:
            self.failed.emit(str(exc))


class DatasetWorker(QThread):
    """Generate and export synthetic observations without blocking the GUI."""

    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, config: ExperimentConfig, output_root: Path) -> None:
        super().__init__()
        self.config = config
        self.output_root = output_root

    def run(self) -> None:
        try:
            target = export_to_requested_path(
                self.output_root,
                lambda root: generate_dataset(self.config, root),
                kind="datasets",
            )
            self.completed.emit(str(target))
        except Exception as exc:
            self.failed.emit(str(exc))


class WorldCanvas(QWidget):
    """Scale a 2D world, point map, and two trajectories to the widget."""

    def __init__(self, occupancy_mode: bool = False) -> None:
        super().__init__()
        self.result: RunResult | None = None
        self.language = "es"
        self.occupancy_mode = occupancy_mode
        self.show_reference = True
        self.setMinimumSize(560, 420)

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#111827"))
        if self.result is None:
            painter.setPen(QColor("#d1d5db"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, TEXT[self.language]["empty_scene"])
            return
        result = self.result
        if self.occupancy_mode and result.occupancy_grid is None:
            painter.setPen(QColor("#d1d5db"))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, TEXT[self.language]["empty_occupancy"]
            )
            return
        walls = result.world.segments_at(float(result.timestamps[-1]))
        values_x = [coordinate for wall in walls for coordinate in (wall.x1, wall.x2)]
        values_y = [coordinate for wall in walls for coordinate in (wall.y1, wall.y2)]
        values_x.extend(result.ground_truth[:, 0])
        values_y.extend(result.ground_truth[:, 1])
        values_x.extend(result.estimated[:, 0])
        values_y.extend(result.estimated[:, 1])
        min_x, max_x = min(values_x), max(values_x)
        min_y, max_y = min(values_y), max(values_y)
        span = max(max_x - min_x, max_y - min_y, 1.0)
        scale = min((self.width() - 56) / span, (self.height() - 80) / span)
        center_x, center_y = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0

        def point(x: float, y: float) -> QPointF:
            return QPointF(
                self.width() / 2 + (x - center_x) * scale, self.height() / 2 - (y - center_y) * scale
            )

        if self.occupancy_mode:
            grid = result.occupancy_grid
            assert grid is not None
            colors = {"free": "#1d4ed8", "unknown": "#64748b", "occupied": "#f97316"}
            cell_size = max(1.0, grid.resolution_m * scale)
            for _, _, center_x_m, center_y_m, state, _ in grid.rows(result.world):
                center = point(center_x_m, center_y_m)
                painter.fillRect(
                    QRectF(
                        center.x() - cell_size / 2,
                        center.y() - cell_size / 2,
                        cell_size,
                        cell_size,
                    ),
                    QColor(colors[state]),
                )
        if self.show_reference:
            painter.setPen(QPen(QColor("#94a3b8"), 2.0))
            for wall in result.world.segments:
                painter.drawLine(point(wall.x1, wall.y1), point(wall.x2, wall.y2))
            painter.setPen(QPen(QColor("#a855f7"), 2.0))
            for wall in walls[len(result.world.segments) :]:
                painter.drawLine(point(wall.x1, wall.y1), point(wall.x2, wall.y2))
        if not self.occupancy_mode:
            painter.setPen(QPen(QColor("#38bdf8"), 2.0))
            for x, y in result.map_points[:: max(1, len(result.map_points) // 2500)]:
                p = point(float(x), float(y))
                painter.drawPoint(p)
        if self.show_reference:
            painter.setPen(QPen(QColor("#22c55e"), 2.4))
            painter.drawPolyline(
                QPolygonF([point(float(x), float(y)) for x, y in result.ground_truth[:, :2]])
            )
        estimate_color = "#facc15" if self.occupancy_mode else "#f97316"
        painter.setPen(QPen(QColor(estimate_color), 2.4))
        painter.drawPolyline(QPolygonF([point(float(x), float(y)) for x, y in result.estimated[:, :2]]))
        painter.setPen(QColor("#e5e7eb"))
        labels = TEXT[self.language]
        painter.drawText(18, 25, f"x [{min_x:.1f}, {max_x:.1f}] m    y [{min_y:.1f}, {max_y:.1f}] m")
        legend = [("#94a3b8", labels["world_wall"])] if self.show_reference else []
        if result.world.moving_boxes and self.show_reference:
            legend.append(("#a855f7", labels["moving_object"]))
        if self.occupancy_mode:
            legend.extend(
                [
                    ("#1d4ed8", labels["occupancy_free"]),
                    ("#f97316", labels["occupancy_occupied"]),
                    ("#64748b", labels["occupancy_unknown"]),
                ]
            )
        else:
            legend.append(("#38bdf8", labels["map_points"]))
        if self.show_reference:
            legend.append(("#22c55e", labels["truth"]))
        legend.append((estimate_color, labels["estimate"]))
        legend_x = 18
        legend_y = self.height() - 15
        for color, label in legend:
            width = 32 + painter.fontMetrics().horizontalAdvance(label)
            if legend_x + width > self.width() - 18:
                legend_x = 18
                legend_y -= 22
            painter.setPen(QPen(QColor(color), 3.0))
            painter.drawLine(legend_x, legend_y - 4, legend_x + 14, legend_y - 4)
            painter.setPen(QColor("#e5e7eb"))
            painter.drawText(legend_x + 20, legend_y, label)
            legend_x += width


class CompactDoubleSpinBox(QDoubleSpinBox):
    """Keep high stored precision without displaying trailing zeroes."""

    def textFromValue(self, value: float) -> str:
        return f"{value:.9f}".rstrip("0").rstrip(".") or "0"


class MainWindow(QMainWindow):
    """A connected desktop workflow for single reproducible experiments."""

    def __init__(self) -> None:
        super().__init__()
        self.language = "es"
        self._base_config = ExperimentConfig()
        self._last_result: RunResult | None = None
        self._live_start: RunStarted | None = None
        self._live_steps: list[RunStep] = []
        self._live_follow = True
        self._live_path = ""
        self._last_batch_path: Path | None = None
        self._worker: ExperimentWorker | None = None
        self._batch_worker: BatchWorker | None = None
        self._frontier_worker: FrontierWorker | None = None
        self._dataset_worker: DatasetWorker | None = None
        self._last_frontier_path: Path | None = None
        self._last_dataset_path: Path | None = None
        self._sweep_worker: SweepWorker | None = None
        self._last_sweep_path: Path | None = None
        self._labels: dict[str, QLabel] = {}
        self._inputs: dict[str, QWidget] = {}
        self._build_ui()
        self._set_from_config(self._base_config)
        self._retranslate()
        self.statusBar().showMessage(self._text("ready"))

    def _spin(self, minimum: float, maximum: float, decimals: int, step: float) -> QDoubleSpinBox:
        control = CompactDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setDecimals(max(decimals, 9))
        control.setSingleStep(step)
        return control

    def _add_field(self, form: QFormLayout, key: str, control: QWidget) -> None:
        label = QLabel()
        form.addRow(label, control)
        self._labels[key] = label
        self._inputs[key] = control

    def _build_ui(self) -> None:
        self.setWindowTitle("SLAMCart3D")
        self.resize(1160, 760)
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        toolbar = QHBoxLayout()
        toolbar.addStretch()
        self.language_label = QLabel()
        self.language_combo = QComboBox()
        self.language_combo.addItem("Español", "es")
        self.language_combo.addItem("English", "en")
        self.language_combo.currentIndexChanged.connect(self._change_language)
        toolbar.addWidget(self.language_label)
        toolbar.addWidget(self.language_combo)
        outer.addLayout(toolbar)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)
        timeline = QHBoxLayout()
        self.previous_button = QPushButton()
        self.next_button = QPushButton()
        self.step_slider = QSlider(Qt.Orientation.Horizontal)
        self.step_slider.setRange(0, 0)
        self.step_label = QLabel()
        self.previous_button.clicked.connect(lambda: self.step_slider.setValue(self.step_slider.value() - 1))
        self.next_button.clicked.connect(lambda: self.step_slider.setValue(self.step_slider.value() + 1))
        self.step_slider.valueChanged.connect(self._show_step)
        self.reference_checkbox = QCheckBox()
        self.reference_checkbox.setChecked(True)
        self.reference_checkbox.toggled.connect(self._toggle_reference)
        timeline.addWidget(self.reference_checkbox)
        timeline.addWidget(self.previous_button)
        timeline.addWidget(self.step_slider, 1)
        timeline.addWidget(self.next_button)
        timeline.addWidget(self.step_label)
        outer.addLayout(timeline)

        setup = QWidget()
        setup_outer = QVBoxLayout(setup)
        path_row = QHBoxLayout()
        self.path_label = QLabel()
        self.config_path = QLineEdit(bundled_example_path())
        self.browse_button = QPushButton()
        self.load_button = QPushButton()
        self.browse_button.clicked.connect(self._browse_config)
        self.load_button.clicked.connect(self._load_config)
        path_row.addWidget(self.path_label)
        path_row.addWidget(self.config_path, 1)
        path_row.addWidget(self.browse_button)
        path_row.addWidget(self.load_button)
        setup_outer.addLayout(path_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        form_body = QWidget()
        form = QFormLayout(form_body)
        form.setSpacing(12)
        self._add_field(form, "name", QLineEdit())
        seed = QSpinBox()
        seed.setRange(0, 2_147_483_647)
        self._add_field(form, "seed", seed)
        for key, values in (
            (
                "world",
                (
                    "indoor_simple",
                    "indoor_corridor",
                    "indoor_complex",
                    "warehouse_shelves",
                    "indoor_dynamic",
                    "outdoor_blocks",
                    "custom",
                ),
            ),
            ("algorithm", ("dead_reckoning", "icp_odometry", "wheel_imu_fusion", "adaptive_hybrid")),
            ("motion", ("constant", "sinusoidal_yaw", "stop_turn", "scripted")),
        ):
            combo = QComboBox()
            for value in values:
                combo.addItem(value, value)
            self._add_field(form, key, combo)
        self._add_field(form, "duration", self._spin(0.1, 3600.0, 2, 1.0))
        walls_editor = QPlainTextEdit()
        walls_editor.setPlaceholderText("0, 0, 10, 0, 2.5\n10, 0, 10, 10, 2.5")
        walls_editor.setMaximumHeight(105)
        self._add_field(form, "world_segments", walls_editor)
        self._add_field(form, "stop_at_walls", QCheckBox())
        self._add_field(form, "timestep", self._spin(0.001, 1.0, 3, 0.01))
        self._add_field(form, "speed", self._spin(-3.0, 3.0, 3, 0.05))
        self._add_field(form, "yaw_rate", self._spin(-6.0, 6.0, 3, 0.05))
        self._add_field(form, "lidar_noise", self._spin(0.0, 2.0, 4, 0.01))
        self._add_field(form, "lidar_dropout", self._spin(0.0, 1.0, 3, 0.05))
        self._add_field(form, "lidar3d_enabled", QCheckBox())
        self._add_field(form, "camera_enabled", QCheckBox())
        self._add_field(form, "score_enabled", QCheckBox())
        self._add_field(form, "occupancy_enabled", QCheckBox())
        self._add_field(form, "dropout_region_enabled", QCheckBox())
        for key in (
            "dropout_region_x_min_m",
            "dropout_region_x_max_m",
            "dropout_region_y_min_m",
            "dropout_region_y_max_m",
        ):
            self._add_field(form, key, self._spin(-1000.0, 1000.0, 2, 0.1))
        self._add_field(form, "imu_bias", self._spin(-1.0, 1.0, 4, 0.01))
        self._add_field(form, "imu_dropout", self._spin(0.0, 1.0, 3, 0.05))
        self._add_field(form, "output", QLineEdit(default_results_path()))
        scroll.setWidget(form_body)
        setup_outer.addWidget(scroll, 1)
        action_row = QHBoxLayout()
        self.save_button = QPushButton()
        self.run_button = QPushButton()
        self.save_button.clicked.connect(self._save_config)
        self.run_button.clicked.connect(self._run_experiment)
        action_row.addWidget(self.save_button)
        action_row.addStretch()
        action_row.addWidget(self.run_button)
        setup_outer.addLayout(action_row)
        self.tabs.addTab(setup, "")

        self.canvas = WorldCanvas()
        self.tabs.addTab(self.canvas, "")
        metrics_page = QWidget()
        metrics_layout = QVBoxLayout(metrics_page)
        self.metrics_hint = QLabel()
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setColumnWidth(0, 360)
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        metrics_layout.addWidget(self.metrics_hint)
        metrics_layout.addWidget(self.metrics_table)
        self.tabs.addTab(metrics_page, "")

        batch_page = QWidget()
        batch_layout = QVBoxLayout(batch_page)
        self.batch_hint = QLabel()
        batch_layout.addWidget(self.batch_hint)
        batch_form = QFormLayout()
        self.batch_seed_label = QLabel()
        self.batch_seeds = QLineEdit("0,1,2,3,4")
        batch_form.addRow(self.batch_seed_label, self.batch_seeds)
        self.batch_world_label = QLabel()
        world_row = QWidget()
        world_layout = QGridLayout(world_row)
        self.batch_worlds = {}
        for index, value in enumerate(
            (
                "indoor_simple",
                "indoor_corridor",
                "indoor_complex",
                "warehouse_shelves",
                "indoor_dynamic",
                "outdoor_blocks",
                "custom",
            )
        ):
            checkbox = QCheckBox()
            checkbox.setChecked(value == "indoor_simple")
            self.batch_worlds[value] = checkbox
            world_layout.addWidget(checkbox, index // 3, index % 3)
        batch_form.addRow(self.batch_world_label, world_row)
        self.batch_algorithm_label = QLabel()
        algorithm_row = QWidget()
        algorithm_layout = QHBoxLayout(algorithm_row)
        self.batch_algorithms = {}
        for value in ("dead_reckoning", "icp_odometry", "wheel_imu_fusion", "adaptive_hybrid"):
            checkbox = QCheckBox()
            checkbox.setChecked(value == "dead_reckoning")
            self.batch_algorithms[value] = checkbox
            algorithm_layout.addWidget(checkbox)
        batch_form.addRow(self.batch_algorithm_label, algorithm_row)
        batch_layout.addLayout(batch_form)
        self.batch_button = QPushButton()
        self.batch_button.clicked.connect(self._run_batch)
        batch_layout.addWidget(self.batch_button)
        self.batch_status = QLabel()
        batch_layout.addWidget(self.batch_status)
        self.batch_table = QTableWidget(0, 4)
        self.batch_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.batch_table.verticalHeader().setVisible(False)
        self.batch_table.horizontalHeader().setStretchLastSection(True)
        batch_layout.addWidget(self.batch_table, 1)
        self.tabs.addTab(batch_page, "")

        frontier_page = QWidget()
        frontier_layout = QVBoxLayout(frontier_page)
        self.frontier_hint = QLabel()
        frontier_layout.addWidget(self.frontier_hint)
        frontier_form = QFormLayout()
        self.frontier_x_label = QLabel()
        self.frontier_x_combo = QComboBox()
        self.frontier_y_label = QLabel()
        self.frontier_y_combo = QComboBox()
        for value in FRONTIER_PARAMETERS:
            self.frontier_x_combo.addItem(value, value)
            self.frontier_y_combo.addItem(value, value)
        self.frontier_y_combo.setCurrentIndex(1)
        for axis in ("x", "y"):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            minimum_label = QLabel()
            maximum_label = QLabel()
            minimum = self._spin(-10.0, 10.0, 4, 0.05)
            maximum = self._spin(-10.0, 10.0, 4, 0.05)
            maximum.setValue(0.8 if axis == "x" else 0.5)
            setattr(self, f"frontier_{axis}_min_label", minimum_label)
            setattr(self, f"frontier_{axis}_max_label", maximum_label)
            setattr(self, f"frontier_{axis}_min", minimum)
            setattr(self, f"frontier_{axis}_max", maximum)
            row_layout.addWidget(getattr(self, f"frontier_{axis}_combo"), 1)
            row_layout.addWidget(minimum_label)
            row_layout.addWidget(minimum)
            row_layout.addWidget(maximum_label)
            row_layout.addWidget(maximum)
            frontier_form.addRow(getattr(self, f"frontier_{axis}_label"), row)
        self.frontier_metric_label = QLabel()
        metric_row = QWidget()
        metric_layout = QHBoxLayout(metric_row)
        metric_layout.setContentsMargins(0, 0, 0, 0)
        self.frontier_metric_combo = QComboBox()
        for value in FRONTIER_METRICS:
            self.frontier_metric_combo.addItem(value, value)
        self.frontier_threshold_label = QLabel()
        self.frontier_threshold = self._spin(-100.0, 100.0, 4, 0.01)
        self.frontier_threshold.setValue(0.05)
        metric_layout.addWidget(self.frontier_metric_combo, 1)
        metric_layout.addWidget(self.frontier_threshold_label)
        metric_layout.addWidget(self.frontier_threshold)
        frontier_form.addRow(self.frontier_metric_label, metric_row)
        self.frontier_seed_label = QLabel()
        seeds_row = QWidget()
        seeds_layout = QHBoxLayout(seeds_row)
        seeds_layout.setContentsMargins(0, 0, 0, 0)
        self.frontier_seeds = QLineEdit("0,1,2,3,4")
        self.frontier_budget_label = QLabel()
        self.frontier_budget = QSpinBox()
        self.frontier_budget.setRange(9, 500)
        self.frontier_budget.setValue(12)
        seeds_layout.addWidget(self.frontier_seeds, 1)
        seeds_layout.addWidget(self.frontier_budget_label)
        seeds_layout.addWidget(self.frontier_budget)
        frontier_form.addRow(self.frontier_seed_label, seeds_row)
        frontier_layout.addLayout(frontier_form)
        self.frontier_button = QPushButton()
        self.frontier_button.clicked.connect(self._run_frontier)
        frontier_layout.addWidget(self.frontier_button)
        self.frontier_status = QLabel()
        frontier_layout.addWidget(self.frontier_status)
        self.frontier_canvas = RobustnessCanvas()
        frontier_layout.addWidget(self.frontier_canvas, 1)
        self.tabs.addTab(frontier_page, "")
        cloud_page = QWidget()
        cloud_layout = QVBoxLayout(cloud_page)
        self.cloud_hint = QLabel()
        self.cloud_hint.setWordWrap(True)
        cloud_layout.addWidget(self.cloud_hint)
        self.cloud_canvas = PointCloudCanvas()
        cloud_layout.addWidget(self.cloud_canvas, 1)
        self.tabs.addTab(cloud_page, "")
        dataset_page = QWidget()
        dataset_layout = QVBoxLayout(dataset_page)
        self.dataset_hint = QLabel()
        self.dataset_hint.setWordWrap(True)
        dataset_layout.addWidget(self.dataset_hint)
        dataset_form = QFormLayout()
        self.dataset_output_label = QLabel()
        self.dataset_output = QLineEdit(default_datasets_path())
        dataset_form.addRow(self.dataset_output_label, self.dataset_output)
        dataset_layout.addLayout(dataset_form)
        self.dataset_button = QPushButton()
        self.dataset_button.clicked.connect(self._generate_dataset)
        dataset_layout.addWidget(self.dataset_button)
        self.dataset_status = QLabel()
        dataset_layout.addWidget(self.dataset_status)
        self.camera_preview = QLabel()
        self.camera_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_preview.setMinimumHeight(280)
        dataset_layout.addWidget(self.camera_preview, 1)
        self.tabs.addTab(dataset_page, "")
        self.occupancy_canvas = WorldCanvas(occupancy_mode=True)
        self.tabs.addTab(self.occupancy_canvas, "")
        sweep_page = QWidget()
        sweep_layout = QVBoxLayout(sweep_page)
        self.sweep_hint = QLabel()
        self.sweep_hint.setWordWrap(True)
        sweep_layout.addWidget(self.sweep_hint)
        sweep_form = QFormLayout()
        self.sweep_parameter_label = QLabel()
        self.sweep_parameter = QComboBox()
        for name in SWEEP_PARAMETERS:
            self.sweep_parameter.addItem(name, name)
        self.sweep_parameter.currentIndexChanged.connect(self._update_sweep_axis_labels)
        sweep_form.addRow(self.sweep_parameter_label, self.sweep_parameter)
        bounds = QWidget()
        bounds_layout = QHBoxLayout(bounds)
        bounds_layout.setContentsMargins(0, 0, 0, 0)
        self.sweep_min_label = QLabel()
        self.sweep_min = self._spin(-3.0, 3.0, 3, 0.05)
        self.sweep_min.setValue(0.0)
        self.sweep_max_label = QLabel()
        self.sweep_max = self._spin(-3.0, 3.0, 3, 0.05)
        self.sweep_max.setValue(0.4)
        self.sweep_levels_label = QLabel()
        self.sweep_levels = QSpinBox()
        self.sweep_levels.setRange(2, 20)
        self.sweep_levels.setValue(5)
        for control in (
            self.sweep_min_label,
            self.sweep_min,
            self.sweep_max_label,
            self.sweep_max,
            self.sweep_levels_label,
            self.sweep_levels,
        ):
            bounds_layout.addWidget(control)
        sweep_form.addRow(bounds)
        self.sweep_seeds_label = QLabel()
        self.sweep_seeds = QLineEdit("0,1,2,3,4")
        sweep_form.addRow(self.sweep_seeds_label, self.sweep_seeds)
        self.sweep_metric_label = QLabel()
        self.sweep_metric = QComboBox()
        for metric in ("ate_rmse_m", "rpe_translation_rmse_m", "map_f1_0p25"):
            self.sweep_metric.addItem(metric, metric)
        self.sweep_metric.currentIndexChanged.connect(self._update_sweep_axis_labels)
        sweep_form.addRow(self.sweep_metric_label, self.sweep_metric)
        self.sweep_threshold_label = QLabel()
        self.sweep_threshold = self._spin(-100.0, 100.0, 3, 0.05)
        self.sweep_threshold.setValue(0.5)
        sweep_form.addRow(self.sweep_threshold_label, self.sweep_threshold)
        sweep_layout.addLayout(sweep_form)
        self.sweep_button = QPushButton()
        self.sweep_button.clicked.connect(self._run_sweep)
        sweep_layout.addWidget(self.sweep_button)
        self.sweep_status = QLabel()
        sweep_layout.addWidget(self.sweep_status)
        self.sweep_canvas = SweepCanvas()
        sweep_layout.addWidget(self.sweep_canvas, 1)
        self.sweep_table = QTableWidget(0, 4)
        self.sweep_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.sweep_table.verticalHeader().setVisible(False)
        self.sweep_table.setMaximumHeight(180)
        self.sweep_table.setColumnWidth(0, 310)
        self.sweep_table.setColumnWidth(1, 180)
        self.sweep_table.setColumnWidth(2, 210)
        self.sweep_table.horizontalHeader().setStretchLastSection(True)
        sweep_layout.addWidget(self.sweep_table)
        self.tabs.addTab(sweep_page, "")
        self.frontier_x_combo.currentIndexChanged.connect(self._update_frontier_axis_labels)
        self.frontier_y_combo.currentIndexChanged.connect(self._update_frontier_axis_labels)
        self.setStyleSheet(
            "QMainWindow, QWidget { background: #f8fafc; color: #172033; }"
            "QPushButton { background: #1d4ed8; color: white; border: 0; padding: 8px 14px; border-radius: 5px; }"
            "QPushButton:disabled { background: #94a3b8; }"
            "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: white; padding: 5px; }"
            "QTabWidget::pane { border: 1px solid #cbd5e1; }"
        )

    def _text(self, key: str) -> str:
        return TEXT[self.language][key]

    def _retranslate(self) -> None:
        self.language_label.setText(self._text("language"))
        self.path_label.setText(self._text("config_path"))
        self.browse_button.setText(self._text("browse"))
        self.load_button.setText(self._text("load"))
        self.save_button.setText(self._text("save"))
        self.run_button.setText(self._text("running" if self._worker and self._worker.isRunning() else "run"))
        self.batch_button.setText(
            self._text(
                "batch_running" if self._batch_worker and self._batch_worker.isRunning() else "batch_run"
            )
        )
        self.frontier_button.setText(
            self._text(
                "frontier_running"
                if self._frontier_worker and self._frontier_worker.isRunning()
                else "frontier_run"
            )
        )
        self.tabs.setTabText(0, self._text("setup"))
        self.tabs.setTabText(1, self._text("scene"))
        self.tabs.setTabText(2, self._text("metrics"))
        self.tabs.setTabText(3, self._text("batch"))
        self.tabs.setTabText(4, self._text("frontier"))
        self.tabs.setTabText(5, self._text("cloud3d"))
        self.tabs.setTabText(6, self._text("dataset"))
        self.tabs.setTabText(7, self._text("occupancy"))
        self.tabs.setTabText(8, self._text("sweep"))
        self.sweep_hint.setText(self._text("sweep_hint"))
        self.sweep_parameter_label.setText(self._text("sweep_parameter"))
        self.sweep_min_label.setText(self._text("sweep_min"))
        self.sweep_max_label.setText(self._text("sweep_max"))
        self.sweep_levels_label.setText(self._text("sweep_levels"))
        self.sweep_seeds_label.setText(self._text("sweep_seeds"))
        self.sweep_metric_label.setText(self._text("sweep_metric"))
        self.sweep_threshold_label.setText(self._text("sweep_threshold"))
        self.sweep_button.setText(
            self._text(
                "sweep_running" if self._sweep_worker and self._sweep_worker.isRunning() else "sweep_run"
            )
        )
        self.sweep_canvas.empty_label = self._text("sweep_empty")
        self.sweep_canvas.x_label = self._text(f"param_{self.sweep_parameter.currentData()}")
        self.sweep_canvas.y_label = METRIC_LABELS[self.sweep_metric.currentData()][
            0 if self.language == "es" else 1
        ]
        for index in range(self.sweep_parameter.count()):
            self.sweep_parameter.setItemText(
                index, self._text(f"param_{self.sweep_parameter.itemData(index)}")
            )
        for index in range(self.sweep_metric.count()):
            metric = self.sweep_metric.itemData(index)
            self.sweep_metric.setItemText(index, METRIC_LABELS[metric][0 if self.language == "es" else 1])
        self.sweep_table.setHorizontalHeaderLabels(
            [self._text("sweep_parameter"), self._text("batch_mean"), self._text("sweep_failure"), "n"]
        )
        self.sweep_canvas.update()
        if self._last_sweep_path is not None:
            self.sweep_status.setText(self._text("sweep_completed").format(path=self._last_sweep_path))
        elif self._sweep_worker is None:
            self.sweep_status.setText(self._text("sweep_empty"))
        self.cloud_hint.setText(self._text("cloud3d_hint"))
        self.cloud_canvas.empty_label = self._text("empty_cloud3d")
        self.cloud_canvas.count_label = "puntos" if self.language == "es" else "points"
        self.cloud_canvas.floor_label = "Piso z = 0 m" if self.language == "es" else "Floor z = 0 m"
        self.cloud_canvas.no_points_label = (
            "Sin nube 3D: active LiDAR multicapa"
            if self.language == "es"
            else "No 3D cloud: enable multilayer LiDAR"
        )
        self.cloud_canvas.legend_labels = (
            ("Entorno real", "Nube estimada", "Carrito real", "Carrito estimado")
            if self.language == "es"
            else ("True world", "Estimated cloud", "True cart", "Estimated cart")
        )
        self.previous_button.setText(self._text("step_previous"))
        self.next_button.setText(self._text("step_next"))
        self.reference_checkbox.setText(self._text("show_reference"))
        self._update_step_label()
        self.cloud_canvas.update()
        self.dataset_hint.setText(self._text("dataset_hint"))
        self.dataset_output_label.setText(self._text("dataset_output"))
        self.dataset_output.setToolTip(TOOLTIPS["dataset_output"][0 if self.language == "es" else 1])
        self.dataset_button.setText(
            self._text(
                "dataset_running"
                if self._dataset_worker and self._dataset_worker.isRunning()
                else "dataset_run"
            )
        )
        if self._dataset_worker is not None and self._dataset_worker.isRunning():
            self.dataset_status.setText(self._text("dataset_running"))
        elif self._last_dataset_path is not None:
            self.dataset_status.setText(self._text("dataset_completed").format(path=self._last_dataset_path))
            if self.camera_preview.pixmap() is None:
                self.camera_preview.setText(self._text("dataset_no_camera"))
        elif self._dataset_worker is None:
            self.dataset_status.setText(self._text("dataset_empty"))
        self.batch_hint.setText(self._text("batch_hint"))
        self.batch_seed_label.setText(self._text("batch_seeds"))
        self.batch_world_label.setText(self._text("batch_worlds"))
        self.batch_algorithm_label.setText(self._text("batch_algorithms"))
        self.batch_seeds.setToolTip(TOOLTIPS["batch_seeds"][0 if self.language == "es" else 1])
        for key, checkbox in self.batch_worlds.items():
            checkbox.setText(self._text(f"world_{key}"))
            checkbox.setToolTip(TOOLTIPS["batch_worlds"][0 if self.language == "es" else 1])
        for key, checkbox in self.batch_algorithms.items():
            checkbox.setText(self._text(f"algorithm_{key}"))
            checkbox.setToolTip(TOOLTIPS["batch_algorithms"][0 if self.language == "es" else 1])
        if self._last_batch_path is not None:
            self._show_batch_summary(self._last_batch_path)
            self.batch_status.setText(self._text("batch_completed").format(path=self._last_batch_path))
        elif self.batch_table.rowCount() == 0:
            self.batch_status.setText(self._text("batch_empty"))
        self.batch_table.setHorizontalHeaderLabels(
            [
                self._text("world"),
                self._text("algorithm"),
                self._text("metric_name"),
                self._text("batch_mean"),
            ]
        )
        self.frontier_hint.setText(self._text("frontier_hint"))
        self.frontier_x_label.setText(self._text("frontier_x"))
        self.frontier_y_label.setText(self._text("frontier_y"))
        self.frontier_metric_label.setText(self._text("frontier_metric"))
        self.frontier_threshold_label.setText(self._text("frontier_threshold"))
        self.frontier_seed_label.setText(self._text("frontier_seeds"))
        self.frontier_budget_label.setText(self._text("frontier_budget"))
        for axis in ("x", "y"):
            getattr(self, f"frontier_{axis}_min_label").setText(self._text("frontier_min"))
            getattr(self, f"frontier_{axis}_max_label").setText(self._text("frontier_max"))
            combo = getattr(self, f"frontier_{axis}_combo")
            for index in range(combo.count()):
                combo.setItemText(index, self._text(f"param_{combo.itemData(index)}"))
            tooltip = TOOLTIPS[f"frontier_{axis}"][0 if self.language == "es" else 1]
            for control in (
                combo,
                getattr(self, f"frontier_{axis}_min"),
                getattr(self, f"frontier_{axis}_max"),
            ):
                control.setToolTip(tooltip)
        for index in range(self.frontier_metric_combo.count()):
            metric = self.frontier_metric_combo.itemData(index)
            self.frontier_metric_combo.setItemText(
                index, METRIC_LABELS[metric][0 if self.language == "es" else 1]
            )
        self.frontier_metric_combo.setToolTip(TOOLTIPS["frontier_metric"][0 if self.language == "es" else 1])
        self.frontier_threshold.setToolTip(TOOLTIPS["frontier_metric"][0 if self.language == "es" else 1])
        self.frontier_seeds.setToolTip(TOOLTIPS["batch_seeds"][0 if self.language == "es" else 1])
        self.frontier_budget.setToolTip(TOOLTIPS["frontier_budget"][0 if self.language == "es" else 1])
        self.frontier_canvas.empty_label = self._text("frontier_empty")
        self.frontier_canvas.success_label = self._text("frontier_success")
        self.frontier_canvas.failure_label = self._text("frontier_failure")
        self._update_frontier_axis_labels()
        if self._last_frontier_path is not None:
            self.frontier_status.setText(
                self._text("frontier_completed").format(path=self._last_frontier_path)
            )
        elif self._frontier_worker is None:
            self.frontier_status.setText(self._text("frontier_empty"))
        self.config_path.setToolTip(TOOLTIPS["config_path"][0 if self.language == "es" else 1])
        for key, label in self._labels.items():
            label.setText(self._text(key))
            tooltip = TOOLTIPS[key][0 if self.language == "es" else 1]
            label.setToolTip(tooltip)
            self._inputs[key].setToolTip(tooltip)
        for key in ("world", "algorithm", "motion"):
            combo = self._inputs[key]
            assert isinstance(combo, QComboBox)
            for index in range(combo.count()):
                combo.setItemText(index, self._text(f"{key}_{combo.itemData(index)}"))
        self.metrics_hint.setText("" if self._last_result else self._text("empty_metrics"))
        self.metrics_table.setHorizontalHeaderLabels([self._text("metric_name"), self._text("metric_value")])
        self.canvas.language = self.language
        self.canvas.update()
        self.occupancy_canvas.language = self.language
        self.occupancy_canvas.update()
        if self._last_result:
            self._show_metrics(self._last_result)

    def _change_language(self) -> None:
        self.language = self.language_combo.currentData()
        self._retranslate()

    def _update_frontier_axis_labels(self) -> None:
        names = (
            self.frontier_canvas.study.parameter_names
            if self.frontier_canvas.study is not None
            else (self.frontier_x_combo.currentData(), self.frontier_y_combo.currentData())
        )
        self.frontier_canvas.x_label = self._text(f"param_{names[0]}")
        self.frontier_canvas.y_label = self._text(f"param_{names[1]}")
        self.frontier_canvas.update()

    def _update_sweep_axis_labels(self) -> None:
        if not hasattr(self, "sweep_canvas"):
            return
        self.sweep_canvas.x_label = self._text(f"param_{self.sweep_parameter.currentData()}")
        self.sweep_canvas.y_label = METRIC_LABELS[self.sweep_metric.currentData()][
            0 if self.language == "es" else 1
        ]
        self.sweep_canvas.update()

    def _set_from_config(self, config: ExperimentConfig) -> None:
        self._base_config = config
        self._inputs["lidar3d_enabled"].setChecked(config.lidar3d.enabled)
        self._inputs["camera_enabled"].setChecked(config.camera.enabled)
        self._inputs["score_enabled"].setChecked(config.robustness_score.enabled)
        self._inputs["occupancy_enabled"].setChecked(config.occupancy_grid.enabled)
        self._inputs["dropout_region_enabled"].setChecked(config.degradation.dropout_region_enabled)
        self._inputs["name"].setText(config.name)
        self._inputs["world_segments"].setPlainText(
            "\n".join(
                f"{wall.x1:g}, {wall.y1:g}, {wall.x2:g}, {wall.y2:g}, {wall.height_m:g}"
                for wall in config.simulation.world_segments
            )
        )
        self._inputs["stop_at_walls"].setChecked(config.simulation.stop_at_walls)
        self._inputs["seed"].setValue(config.seed)
        for key, value in (
            ("world", config.simulation.world),
            ("algorithm", config.slam.algorithm),
            ("motion", config.simulation.motion_profile),
        ):
            index = self._inputs[key].findData(value)
            if index < 0:
                raise ValueError(f"unsupported GUI selection for {key}: {value}")
            self._inputs[key].setCurrentIndex(index)
        for key, value in (
            ("duration", config.simulation.duration_s),
            ("timestep", config.simulation.timestep_s),
            ("speed", config.simulation.linear_velocity_m_s),
            ("yaw_rate", config.simulation.yaw_rate_rad_s),
            ("lidar_noise", config.degradation.lidar_noise_std_m),
            ("lidar_dropout", config.degradation.lidar_dropout_probability),
            ("imu_bias", config.degradation.imu_gyro_bias_rad_s),
            ("imu_dropout", config.degradation.imu_dropout_probability),
            ("dropout_region_x_min_m", config.degradation.dropout_region_x_min_m),
            ("dropout_region_x_max_m", config.degradation.dropout_region_x_max_m),
            ("dropout_region_y_min_m", config.degradation.dropout_region_y_min_m),
            ("dropout_region_y_max_m", config.degradation.dropout_region_y_max_m),
        ):
            control = self._inputs[key]
            if value < control.minimum():
                control.setMinimum(value)
            if value > control.maximum():
                control.setMaximum(value)
            control.setValue(value)

    def _parse_walls(self) -> tuple[WallConfig, ...]:
        walls: list[WallConfig] = []
        for line_number, line in enumerate(
            self._inputs["world_segments"].toPlainText().splitlines(), start=1
        ):
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 5:
                raise ValueError(f"wall line {line_number}: expected x1, y1, x2, y2, height")
            try:
                walls.append(WallConfig(*(float(field) for field in fields)))
            except ValueError as exc:
                raise ValueError(f"wall line {line_number}: coordinates must be numbers") from exc
        return tuple(walls)

    def current_config(self) -> ExperimentConfig:
        """Preserve advanced YAML fields while applying visible edits."""

        base = self._base_config
        simulation = replace(
            base.simulation,
            world=self._inputs["world"].currentData(),
            world_segments=self._parse_walls(),
            stop_at_walls=self._inputs["stop_at_walls"].isChecked(),
            motion_profile=self._inputs["motion"].currentData(),
            duration_s=self._inputs["duration"].value(),
            timestep_s=self._inputs["timestep"].value(),
            linear_velocity_m_s=self._inputs["speed"].value(),
            yaw_rate_rad_s=self._inputs["yaw_rate"].value(),
        )
        degradation = replace(
            base.degradation,
            lidar_noise_std_m=self._inputs["lidar_noise"].value(),
            lidar_dropout_probability=self._inputs["lidar_dropout"].value(),
            imu_gyro_bias_rad_s=self._inputs["imu_bias"].value(),
            imu_dropout_probability=self._inputs["imu_dropout"].value(),
            dropout_region_enabled=self._inputs["dropout_region_enabled"].isChecked(),
            dropout_region_x_min_m=self._inputs["dropout_region_x_min_m"].value(),
            dropout_region_x_max_m=self._inputs["dropout_region_x_max_m"].value(),
            dropout_region_y_min_m=self._inputs["dropout_region_y_min_m"].value(),
            dropout_region_y_max_m=self._inputs["dropout_region_y_max_m"].value(),
        )
        return replace(
            base,
            name=self._inputs["name"].text().strip(),
            seed=self._inputs["seed"].value(),
            simulation=simulation,
            degradation=degradation,
            lidar3d=replace(base.lidar3d, enabled=self._inputs["lidar3d_enabled"].isChecked()),
            camera=replace(base.camera, enabled=self._inputs["camera_enabled"].isChecked()),
            robustness_score=replace(
                base.robustness_score, enabled=self._inputs["score_enabled"].isChecked()
            ),
            occupancy_grid=replace(
                base.occupancy_grid, enabled=self._inputs["occupancy_enabled"].isChecked()
            ),
            slam=replace(base.slam, algorithm=self._inputs["algorithm"].currentData()),
        )

    def _browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, self._text("config_path"), "configs", "YAML/JSON (*.yaml *.yml *.json)"
        )
        if path:
            self.config_path.setText(path)

    def _load_config(self) -> None:
        try:
            config = ExperimentConfig.load(self.config_path.text())
            self._set_from_config(config)
            self.statusBar().showMessage(self._text("ready"))
        except Exception as exc:
            self._show_error(str(exc))

    def _save_config(self) -> None:
        try:
            config = self.current_config()
            path, _ = QFileDialog.getSaveFileName(
                self, self._text("save"), self.config_path.text(), "YAML (*.yaml)"
            )
            if path:
                Path(path).write_text(
                    yaml.safe_dump({"experiment": config.to_dict()}, sort_keys=False, allow_unicode=True),
                    encoding="utf-8",
                )
                self.config_path.setText(path)
        except Exception as exc:
            self._show_error(str(exc))

    def _run_experiment(self) -> None:
        try:
            config = self.current_config()
            output_text = self._inputs["output"].text().strip()
            if not output_text:
                raise ValueError("empty output folder")
            output_root = Path(output_text)
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.run_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.batch_button.setEnabled(False)
        self.frontier_button.setEnabled(False)
        self.dataset_button.setEnabled(False)
        self.sweep_button.setEnabled(False)
        self.run_button.setText(self._text("running"))
        self.statusBar().showMessage(self._text("running"))
        self._worker = ExperimentWorker(config, output_root)
        self._worker.started_run.connect(self._on_started_run)
        self._worker.step_ready.connect(self._on_step_ready)
        self._worker.completed.connect(self._on_completed)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_started_run(self, started: RunStarted, path: str) -> None:
        self._live_start = started
        self._live_steps = []
        self._live_follow = True
        self._live_path = path
        self._last_result = None
        self.step_slider.setRange(0, 0)
        self._show_step(0)
        self.tabs.setCurrentIndex(5)
        self.statusBar().showMessage(self._text("step_saved").format(path=path))

    def _on_step_ready(self, step: RunStep) -> None:
        if self._live_start is None:
            return
        self._live_steps.append(step)
        follow = self._live_follow
        self.step_slider.setMaximum(step.index)
        if follow:
            self.step_slider.setValue(step.index)

    def _toggle_reference(self, checked: bool) -> None:
        for canvas in (self.canvas, self.occupancy_canvas, self.cloud_canvas):
            canvas.show_reference = checked
            canvas.update()

    def _update_step_label(self) -> None:
        if self._live_start is None:
            self.step_label.setText(self._text("step_empty"))
        else:
            index = self.step_slider.value()
            time_s = self._live_steps[index - 1].timestamp_s if index else 0.0
            self.step_label.setText(
                self._text("step_label").format(index=index, total=self._live_start.total_steps, time=time_s)
            )
        self.previous_button.setEnabled(self.step_slider.value() > 0)
        self.next_button.setEnabled(self.step_slider.value() < self.step_slider.maximum())

    def _show_step(self, index: int) -> None:
        if self._live_start is None:
            self._update_step_label()
            return
        self._live_follow = index == self.step_slider.maximum()
        if self._last_result is not None and index == len(self._live_steps):
            preview: RunPreview | RunResult = self._last_result
        else:
            steps = self._live_steps[:index]
            initial = self._live_start.initial_pose
            timestamps = np.array([0.0, *(step.timestamp_s for step in steps)])
            truth = np.vstack([initial, *(step.ground_truth for step in steps)])
            estimate = np.vstack([initial, *(step.estimated for step in steps)])
            points = [step.map_points_added for step in steps if len(step.map_points_added)]
            points_3d = [step.map_points_3d_added for step in steps if len(step.map_points_3d_added)]
            preview = RunPreview(
                self._live_start.world,
                timestamps,
                truth,
                estimate,
                np.vstack(points) if points else np.empty((0, 2)),
                np.vstack(points_3d) if points_3d else np.empty((0, 3)),
                steps[-1].occupancy_grid if steps else None,
            )
        for canvas in (self.canvas, self.occupancy_canvas, self.cloud_canvas):
            canvas.result = preview
            canvas.update()
        self._update_step_label()

    def _on_completed(self, result: RunResult, path: str) -> None:
        self._last_result = result
        if self._live_start is None:
            for canvas in (self.canvas, self.occupancy_canvas, self.cloud_canvas):
                canvas.result = result
                canvas.update()
        else:
            self._show_step(self.step_slider.value())
        self._show_metrics(result)
        self.metrics_hint.setText("")
        self.tabs.setCurrentIndex(1)
        self.statusBar().showMessage(self._text("completed").format(path=path))

    def _on_failed(self, detail: str) -> None:
        self._show_error(detail)

    def _on_worker_finished(self) -> None:
        self.run_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.batch_button.setEnabled(True)
        self.frontier_button.setEnabled(True)
        self.dataset_button.setEnabled(True)
        self.sweep_button.setEnabled(True)
        self.run_button.setText(self._text("run"))
        self._worker = None

    def _run_batch(self) -> None:
        try:
            config = self.current_config()
            seeds = tuple(int(value.strip()) for value in self.batch_seeds.text().split(","))
            if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
                raise ValueError("batch seeds must be unique nonnegative integers")
            worlds = tuple(key for key, checkbox in self.batch_worlds.items() if checkbox.isChecked())
            algorithms = tuple(key for key, checkbox in self.batch_algorithms.items() if checkbox.isChecked())
            if not worlds or not algorithms:
                raise ValueError("select at least one environment and one algorithm")
            output_text = self._inputs["output"].text().strip()
            if not output_text:
                raise ValueError("empty output folder")
            output_root = Path(output_text)
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.run_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.batch_button.setEnabled(False)
        self.frontier_button.setEnabled(False)
        self.dataset_button.setEnabled(False)
        self.sweep_button.setEnabled(False)
        self.batch_button.setText(self._text("batch_running"))
        self.batch_status.setText(self._text("batch_running"))
        self._batch_worker = BatchWorker(config, seeds, algorithms, worlds, output_root)
        self._batch_worker.completed.connect(self._on_batch_completed)
        self._batch_worker.failed.connect(self._on_failed)
        self._batch_worker.finished.connect(self._on_batch_finished)
        self._batch_worker.start()

    def _on_batch_completed(self, path: str) -> None:
        self._last_batch_path = Path(path)
        self._show_batch_summary(self._last_batch_path)
        message = self._text("batch_completed").format(path=path)
        self.batch_status.setText(message)
        self.statusBar().showMessage(message)
        self.tabs.setCurrentIndex(3)

    def _show_batch_summary(self, path: Path) -> None:
        with (path / "summary.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.batch_table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            for column, key in enumerate(("world", "algorithm", "metric", "mean")):
                value = row[key]
                if key == "world":
                    value = self._text(f"world_{value}")
                elif key == "algorithm":
                    value = self._text(f"algorithm_{value}")
                elif key == "metric":
                    value = METRIC_LABELS.get(value, (value, value))[0 if self.language == "es" else 1]
                self.batch_table.setItem(index, column, QTableWidgetItem(value))
        self.batch_table.resizeColumnsToContents()

    def _on_batch_finished(self) -> None:
        self.run_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.batch_button.setEnabled(True)
        self.frontier_button.setEnabled(True)
        self.dataset_button.setEnabled(True)
        self.sweep_button.setEnabled(True)
        self.batch_button.setText(self._text("batch_run"))
        self._batch_worker = None

    def _run_frontier(self) -> None:
        try:
            config = self.current_config()
            if (
                self.frontier_metric_combo.currentData() == "map3d_wall_f1_0p25"
                and not config.lidar3d.enabled
            ):
                raise ValueError(self._text("frontier_requires_3d"))
            if self.frontier_metric_combo.currentData() == "srs" and not config.robustness_score.enabled:
                raise ValueError(self._text("frontier_requires_srs"))
            if (
                self.frontier_metric_combo.currentData() == "occupancy_occupied_iou"
                and not config.occupancy_grid.enabled
            ):
                raise ValueError(self._text("frontier_requires_occupancy"))
            names = (self.frontier_x_combo.currentData(), self.frontier_y_combo.currentData())
            if names[0] == names[1]:
                raise ValueError("choose two distinct ASRF parameters")
            lower = (self.frontier_x_min.value(), self.frontier_y_min.value())
            upper = (self.frontier_x_max.value(), self.frontier_y_max.value())
            if any(low >= high for low, high in zip(lower, upper, strict=True)):
                raise ValueError("each upper ASRF bound must exceed its lower bound")
            for name, low, high in zip(names, lower, upper, strict=True):
                if name.endswith("probability") and not 0.0 <= low < high <= 1.0:
                    raise ValueError(f"{name} must stay within [0, 1]")
                if "noise_std" in name and low < 0.0:
                    raise ValueError(f"{name} must be nonnegative")
            seeds = tuple(int(value.strip()) for value in self.frontier_seeds.text().split(","))
            if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
                raise ValueError("ASRF seeds must be unique nonnegative integers")
            output_text = self._inputs["output"].text().strip()
            if not output_text:
                raise ValueError("empty output folder")
            output_root = Path(output_text)
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.run_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.batch_button.setEnabled(False)
        self.frontier_button.setEnabled(False)
        self.dataset_button.setEnabled(False)
        self.sweep_button.setEnabled(False)
        self.frontier_button.setText(self._text("frontier_running"))
        self.frontier_status.setText(self._text("frontier_running"))
        self._frontier_worker = FrontierWorker(
            config,
            names,
            lower,
            upper,
            self.frontier_metric_combo.currentData(),
            self.frontier_threshold.value(),
            seeds,
            self.frontier_budget.value(),
            output_root,
        )
        self._frontier_worker.completed.connect(self._on_frontier_completed)
        self._frontier_worker.failed.connect(self._on_frontier_failed)
        self._frontier_worker.finished.connect(self._on_frontier_finished)
        self._frontier_worker.start()

    def _on_frontier_completed(self, study, path: str) -> None:
        self._last_frontier_path = Path(path)
        self.frontier_canvas.study = study
        self._update_frontier_axis_labels()
        self.tabs.setCurrentIndex(4)
        if not self.frontier_canvas.grab().save(str(self._last_frontier_path / "robustness_map_gui.png")):
            self._show_error("could not save ASRF map image")
        message = self._text("frontier_completed").format(path=path)
        self.frontier_status.setText(message)
        self.statusBar().showMessage(message)

    def _on_frontier_failed(self, detail: str) -> None:
        self.frontier_status.setText(detail)
        self._show_error(detail)

    def _on_frontier_finished(self) -> None:
        self.run_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.batch_button.setEnabled(True)
        self.frontier_button.setEnabled(True)
        self.dataset_button.setEnabled(True)
        self.sweep_button.setEnabled(True)
        self.frontier_button.setText(self._text("frontier_run"))
        self._frontier_worker = None

    def _generate_dataset(self) -> None:
        try:
            config = self.current_config()
            output_text = self.dataset_output.text().strip()
            if not output_text:
                raise ValueError("empty dataset folder")
            output_root = Path(output_text)
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.run_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.batch_button.setEnabled(False)
        self.frontier_button.setEnabled(False)
        self.dataset_button.setEnabled(False)
        self.sweep_button.setEnabled(False)
        self.dataset_button.setText(self._text("dataset_running"))
        self.dataset_status.setText(self._text("dataset_running"))
        self._dataset_worker = DatasetWorker(config, output_root)
        self._dataset_worker.completed.connect(self._on_dataset_completed)
        self._dataset_worker.failed.connect(self._on_dataset_failed)
        self._dataset_worker.finished.connect(self._on_dataset_finished)
        self._dataset_worker.start()

    def _on_dataset_completed(self, path: str) -> None:
        self._last_dataset_path = Path(path)
        self.camera_preview.clear()
        try:
            index_path = self._last_dataset_path / "camera_index.csv"
            if index_path.is_file():
                with index_path.open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                image_path = next((row["measured_rgb_png"] for row in rows if row["measured_rgb_png"]), None)
                if image_path is not None:
                    pixmap = QPixmap(str(self._last_dataset_path / image_path))
                    if pixmap.isNull():
                        raise ValueError("could not read measured camera image")
                    self.camera_preview.setPixmap(
                        pixmap.scaledToWidth(560, Qt.TransformationMode.SmoothTransformation)
                    )
            if self.camera_preview.pixmap() is None:
                self.camera_preview.setText(self._text("dataset_no_camera"))
        except Exception as exc:
            self._show_error(str(exc))
        message = self._text("dataset_completed").format(path=path)
        self.dataset_status.setText(message)
        self.statusBar().showMessage(message)
        self.tabs.setCurrentIndex(6)

    def _on_dataset_finished(self) -> None:
        self.run_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.batch_button.setEnabled(True)
        self.frontier_button.setEnabled(True)
        self.dataset_button.setEnabled(True)
        self.sweep_button.setEnabled(True)
        self.dataset_button.setText(self._text("dataset_run"))
        self._dataset_worker = None

    def _run_sweep(self) -> None:
        try:
            config = self.current_config()
            low, high = self.sweep_min.value(), self.sweep_max.value()
            if low >= high:
                raise ValueError("study maximum must exceed its minimum")
            parameter = self.sweep_parameter.currentData()
            if parameter.endswith("probability") and not 0.0 <= low < high <= 1.0:
                raise ValueError("a probability study must stay within [0, 1]")
            if "noise_std" in parameter and low < 0.0:
                raise ValueError("noise standard deviations must be nonnegative")
            levels = tuple(float(value) for value in np.linspace(low, high, self.sweep_levels.value()))
            seeds = tuple(int(value.strip()) for value in self.sweep_seeds.text().split(","))
            if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
                raise ValueError("study requires at least two unique nonnegative seeds")
            output_text = self._inputs["output"].text().strip()
            if not output_text:
                raise ValueError("empty output folder")
            output_root = Path(output_text)
        except Exception as exc:
            self._show_error(str(exc))
            return
        for button in (
            self.run_button,
            self.save_button,
            self.batch_button,
            self.frontier_button,
            self.dataset_button,
            self.sweep_button,
        ):
            button.setEnabled(False)
        self.sweep_button.setText(self._text("sweep_running"))
        self.sweep_status.setText(self._text("sweep_running"))
        self._sweep_worker = SweepWorker(
            config,
            parameter,
            levels,
            seeds,
            self.sweep_metric.currentData(),
            self.sweep_threshold.value(),
            output_root,
        )
        self._sweep_worker.completed.connect(self._on_sweep_completed)
        self._sweep_worker.failed.connect(self._on_failed)
        self._sweep_worker.finished.connect(self._on_sweep_finished)
        self._sweep_worker.start()

    def _on_sweep_completed(self, study, path: str) -> None:
        self._last_sweep_path = Path(path)
        self.sweep_canvas.study = study
        self.sweep_canvas.x_label = self._text(f"param_{study.parameter_name}")
        self.sweep_canvas.y_label = METRIC_LABELS[study.metric_name][0 if self.language == "es" else 1]
        self.sweep_canvas.update()
        self.sweep_table.setRowCount(len(study.summary))
        for index, row in enumerate(study.summary):
            for column, value in enumerate(
                (
                    f"{row['level']:.4g}",
                    f"{row['mean']:.5g}",
                    f"{row['failure_fraction']:.2f}",
                    str(row["count"]),
                )
            ):
                self.sweep_table.setItem(index, column, QTableWidgetItem(value))
        self.tabs.setCurrentIndex(8)
        if not self.sweep_canvas.grab().save(str(self._last_sweep_path / "study_curve_gui.png")):
            self._show_error("could not save study curve image")
        message = self._text("sweep_completed").format(path=path)
        self.sweep_status.setText(message)
        self.statusBar().showMessage(message)

    def _on_sweep_finished(self) -> None:
        for button in (
            self.run_button,
            self.save_button,
            self.batch_button,
            self.frontier_button,
            self.dataset_button,
            self.sweep_button,
        ):
            button.setEnabled(True)
        self.sweep_button.setText(self._text("sweep_run"))
        self._sweep_worker = None

    def _on_dataset_failed(self, detail: str) -> None:
        self.dataset_status.setText(detail)
        self._show_error(detail)

    def _show_metrics(self, result: RunResult) -> None:
        self.metrics_table.setRowCount(len(result.metrics))
        language_index = 0 if self.language == "es" else 1
        for row, (metric, value) in enumerate(result.metrics.items()):
            label = METRIC_LABELS.get(metric, (metric, metric))[language_index]
            self.metrics_table.setItem(row, 0, QTableWidgetItem(label))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{value:.6g}"))
        self.metrics_table.setColumnWidth(0, 360)

    def _show_error(self, detail: str) -> None:
        QMessageBox.critical(self, self._text("error_title"), self._text("error").format(detail=detail))

    def closeEvent(self, event) -> None:
        if (
            (self._worker is not None and self._worker.isRunning())
            or (self._batch_worker is not None and self._batch_worker.isRunning())
            or (self._frontier_worker is not None and self._frontier_worker.isRunning())
            or (self._dataset_worker is not None and self._dataset_worker.isRunning())
            or (self._sweep_worker is not None and self._sweep_worker.isRunning())
        ):
            self.statusBar().showMessage(self._text("wait_for_run"))
            event.ignore()
        else:
            event.accept()


def main() -> int:
    """Start the desktop interface without changing the headless API."""

    configure_local_runtime()
    application = QApplication(sys.argv)
    application.setStyle("Fusion")
    application.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    if "--self-test" in sys.argv:
        from io import BytesIO

        from PIL import Image

        from slamcart3d.experiments.sweep import run_parameter_sweep
        from slamcart3d.simulation.observations import ObservationStream

        config = ExperimentConfig.load(bundled_example_path())
        config = replace(
            config,
            simulation=replace(config.simulation, duration_s=0.2, timestep_s=0.1),
        )
        result = run_simulation(config)
        hybrid_result = run_simulation(
            replace(config, slam=replace(config.slam, algorithm="adaptive_hybrid"))
        )
        sweep = run_parameter_sweep(config, "wheel_rate_noise_std_rad_s", (0.0, 0.1), (0, 1))
        cloud_config = replace(
            config,
            lidar3d=replace(config.lidar3d, enabled=True, azimuth_beams=3, elevation_beams=3),
        )
        cloud_result = run_simulation(cloud_config)
        camera_config = replace(config, camera=replace(config.camera, enabled=True, width_px=4, height_px=3))
        camera_frame = next(ObservationStream(camera_config).frames()).measured_camera
        if camera_frame is None:
            return 1
        png_buffer = BytesIO()
        Image.fromarray(camera_frame.rgb).save(png_buffer, format="PNG")
        batch = run_batch(config, (0,), ("dead_reckoning",), ("indoor_simple",), workers=1)
        frontier = run_frontier_study(
            config,
            ("wheel_rate_noise_std_rad_s", "odometry_dropout_probability"),
            np.array([0.0, 0.0]),
            np.array([0.1, 0.1]),
            "ate_rmse_m",
            0.05,
            (0,),
            budget=8,
            grid_resolution=9,
        )
        window._on_completed(result, "in-memory self-test")
        window.language_combo.setCurrentIndex(1)
        if not math.isfinite(result.metrics["ate_rmse_m"]):
            return 1
        if (
            len(source_sha256()) != 64
            or window.metrics_table.rowCount() == 0
            or window.tabs.count() != 9
            or not math.isfinite(hybrid_result.metrics["ate_rmse_m"])
            or len(hybrid_result.map_points) == 0
            or len(sweep.runs) != 4
            or len(cloud_result.map_points_3d) == 0
            or not png_buffer.getvalue().startswith(b"\x89PNG")
            or len(batch.runs) != 1
            or len(frontier.records) != 8
        ):
            return 1
        return 0
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
