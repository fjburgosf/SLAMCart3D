"""Readable bilingual names for bundled examples and desktop workflows."""

from slamcart3d.gui.paths import bundled_configs_path

# File names are stable identifiers; users choose the translated titles instead.
EXAMPLE_TITLES = {
    "example_hybrid_3d.yaml": (
        "Reconstrucción 3D y localización XYZ",
        "3D reconstruction and XYZ localization",
    ),
    "example_01_clean_environment.yaml": ("Entorno sin degradación", "Environment without degradation"),
    "example_02_lidar_noise.yaml": ("Ruido en el LiDAR", "LiDAR noise"),
    "example_03_imu_drift.yaml": ("Deriva de la IMU", "IMU drift"),
    "example_04_dynamic_environment.yaml": (
        "Entorno con obstáculo móvil",
        "Environment with moving obstacle",
    ),
    "example_05_algorithm_comparison_icp.yaml": ("Localización mediante ICP", "ICP localization"),
    "example_07_asrf.yaml": ("Robustez de ruedas y odometría", "Wheel and odometry robustness"),
    "example_08_icp_lidar_frontier.yaml": ("Robustez del LiDAR con ICP", "LiDAR robustness with ICP"),
    "example_adaptive_hybrid.yaml": ("SLAM híbrido adaptativo", "Adaptive hybrid SLAM"),
    "example_autonomous_exploration.yaml": (
        "Exploración autónoma del entorno",
        "Autonomous environment exploration",
    ),
    "example_camera_rgbd.yaml": (
        "Cámara RGB-D: generar imágenes y profundidad",
        "RGB-D camera: images and depth",
    ),
    "example_custom_room.yaml": ("Habitación personalizada editable", "Editable custom room"),
    "example_hybrid_wheel_degradation.yaml": ("Híbrido con ruedas degradadas", "Hybrid with degraded wheels"),
    "example_indoor_complex.yaml": (
        "Interior con habitaciones y pasillos",
        "Interior with rooms and corridors",
    ),
    "example_indoor_maze.yaml": (
        "Laberinto: exploración autónoma y mapa 3D",
        "Maze: autonomous exploration and 3D map",
    ),
    "example_lidar_occlusion.yaml": ("LiDAR con oclusiones", "LiDAR occlusions"),
    "example_loop_closure.yaml": ("Recorrido con cierre de bucles", "Route with loop closure"),
    "example_motion_stop_turn.yaml": ("Movimiento: avanzar y girar", "Motion: move and turn"),
    "example_multilayer_lidar.yaml": ("LiDAR multicapa y nube 3D", "Multilayer LiDAR and 3D cloud"),
    "example_occupancy_grid.yaml": ("Mapa de ocupación 2D", "2D occupancy map"),
    "example_outdoor_blocks.yaml": ("Entorno exterior con bloques", "Outdoor environment with blocks"),
    "example_recorded_lidar.yaml": (
        "Configuración para mediciones guardadas",
        "Configuration for recorded measurements",
    ),
    "example_scripted_trajectory.yaml": ("Trayectoria definida por tramos", "Segmented trajectory"),
    "example_spatial_dropout.yaml": ("Pérdida de sensores en una zona", "Sensor dropout in a region"),
    "example_srs.yaml": ("Evaluación con índice SRS", "Evaluation with SRS index"),
    "example_temporal_dropout.yaml": ("Pérdida de sensores durante la corrida", "Sensor dropout over time"),
}

WORKFLOWS = (
    ("run", 0, "Simulación individual", "Single simulation"),
    ("batch", 3, "Comparación por lotes", "Batch comparison"),
    ("frontier", 4, "Robustez ASRF (dos variables)", "ASRF robustness (two variables)"),
    ("sweep", 8, "Estudio controlado (una variable)", "Controlled study (one variable)"),
    ("dataset", 6, "Generación de dataset", "Dataset generation"),
    ("replay", 9, "Reproducción de mediciones", "Recorded measurement replay"),
)


def available_examples():
    """Only offer readable YAML files actually present in the distribution."""
    return tuple(sorted(bundled_configs_path().glob("example_*.yaml")))
