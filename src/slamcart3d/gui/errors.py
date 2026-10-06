"""Translate expected input errors while preserving unknown diagnostic details."""

import re

SPANISH_ERRORS = {
    "hybrid_3d requires enabled multilayer LiDAR": "Active el LiDAR multicapa para usar el híbrido XYZ con pose 3D.",
    "reconstruction resolution must be in [0.05, 1.0] m and min_hits positive": "El tamaño de vóxel debe estar entre 0.05 y 1.0 m y las observaciones mínimas deben ser positivas.",
    "hybrid_3d requires synchronized multilayer LiDAR measurements; this replay format contains planar LiDAR only": "El híbrido XYZ necesita mediciones multicapa sincronizadas. Este formato de reproducción solo contiene LiDAR planar; elija otro estimador.",
    "recording path and output folder are required": "Indique el archivo o carpeta de mediciones y la carpeta de resultados.",
    "recorded timestamps must be finite and strictly increasing": "Los tiempos medidos deben ser finitos y estrictamente crecientes.",
    "recording requires at least two FLASER scans": "El registro necesita al menos dos barridos FLASER.",
    "reference poses must be finite and cover the entire recording; no extrapolation": "La referencia debe contener poses finitas y cubrir todos los tiempos del registro; no se extrapola.",
    "recorded LiDAR/wheel timestamps do not match": "Los tiempos del LiDAR y de las ruedas no coinciden.",
    "recorded IMU timestamps do not match": "Los tiempos de la IMU no coinciden con el LiDAR.",
    "recorded measured values are invalid": "Hay mediciones inválidas en el registro.",
    "first recorded timestamp must be greater than zero": "El primer tiempo medido debe ser mayor que cero y relativo al inicio.",
    "CARMEN nonzero laser offset requires an explicit calibration adapter": "Este registro CARMEN necesita declarar la calibración del desplazamiento del láser; no se asumirá cero.",
    "frontier exploration requires wall protection and positive forward speed": "La exploración autónoma requiere protección de paredes y velocidad de avance positiva.",
    "navigation resolution must be between 0.05 and 1.0 m": "La resolución de navegación debe estar entre 0.05 y 1.0 m.",
    "navigation turn rate and replan steps must be positive; margin nonnegative": "El giro y el intervalo de planificación deben ser positivos; el margen de seguridad no puede ser negativo.",
    "evaluation distance limits must be positive and coverage in (0, 1]": "Los límites de distancia deben ser positivos y la cobertura objetivo debe estar entre 0 y 1, sin incluir cero.",
    "paired benchmark requires programmed motion; use recorded replay for autonomous data": "La comparación pareada requiere movimiento programado; use reproducción de mediciones para comparar sobre datos autónomos idénticos.",
    "empty output folder": "Indique una carpeta para guardar los resultados.",
    "empty dataset folder": "Indique una carpeta para guardar el dataset.",
    "batch seeds must be unique nonnegative integers": "Las semillas del lote deben ser enteros distintos y no negativos.",
    "ASRF seeds must be unique nonnegative integers": "Las semillas ASRF deben ser enteros distintos y no negativos.",
    "select at least one environment and one algorithm": "Seleccione al menos un entorno y un algoritmo.",
    "choose two distinct ASRF parameters": "Elija dos parámetros distintos para el estudio ASRF.",
    "each upper ASRF bound must exceed its lower bound": "En cada eje ASRF, el máximo debe ser mayor que el mínimo.",
    "study maximum must exceed its minimum": "El máximo del estudio debe ser mayor que el mínimo.",
    "a probability study must stay within [0, 1]": "Las probabilidades del estudio deben estar entre 0 y 1.",
    "noise standard deviations must be nonnegative": "Las desviaciones estándar del ruido no pueden ser negativas.",
    "noise standard deviations must be non-negative": "Las desviaciones estándar del ruido no pueden ser negativas.",
    "study requires at least two unique nonnegative seeds": "El estudio necesita al menos dos semillas enteras, distintas y no negativas.",
    "could not save ASRF map image": "No se pudo guardar la imagen del mapa ASRF.",
    "could not save study curve image": "No se pudo guardar la imagen de la curva del estudio.",
    "could not read measured camera image": "No se pudo leer la imagen medida de la cámara.",
    "experiment configuration must be a mapping": "La configuración debe contener campos con sus valores, no una lista ni un texto suelto.",
    "configuration wrapper must contain only experiment": "El archivo debe contener únicamente el bloque experiment en el nivel principal.",
    "experiment name must not be empty": "Escriba un nombre para el experimento.",
    "configuration values must be finite": "Los valores numéricos deben ser finitos; no se admiten NaN ni infinito.",
    "duration_s and timestep_s must be positive": "La duración y el paso de tiempo deben ser positivos.",
    "duration_s must be an integer multiple of timestep_s": "La duración debe ser un múltiplo entero del paso de tiempo.",
    "custom world requires at least one wall segment": "El entorno personalizado necesita al menos un segmento de pared.",
    "wall segments require positive height and distinct endpoints": "Cada pared debe tener altura positiva y extremos diferentes.",
    "scripted motion segments must exactly cover duration_s": "Los tramos de movimiento deben cubrir exactamente la duración del experimento.",
    "seed must be non-negative": "La semilla no puede ser negativa.",
    "probabilities must be between 0 and 1": "Las probabilidades deben estar entre 0 y 1.",
    "wheel_imu_fusion requires an enabled IMU": "Active la IMU para usar la fusión de ruedas e IMU.",
}


def localize_error(detail: str, language: str) -> str:
    """Localize known validation failures, without guessing unknown error causes."""

    if language != "es":
        return detail
    if detail in SPANISH_ERRORS:
        return SPANISH_ERRORS[detail]
    match = re.fullmatch(r"(.+) must have type (int|float|bool|str)", detail)
    if match:
        field, kind = match.groups()
        expected = {
            "int": "un entero",
            "float": "un número",
            "bool": "true o false sin comillas",
            "str": "texto",
        }[kind]
        return f"El campo {field} debe contener {expected}."
    match = re.fullmatch(r"wall line (\d+): (.+)", detail)
    if match:
        return f"Pared de la línea {match[1]}: escriba cinco números separados por comas: x1, y1, x2, y2, altura."
    if detail.startswith("unknown experiment fields:"):
        return "Campos de experimento no reconocidos:" + detail.split(":", 1)[1]
    if detail.startswith("invalid literal for int()"):
        return "Escriba semillas enteras separadas por comas, sin valores vacíos ni decimales."
    return detail
