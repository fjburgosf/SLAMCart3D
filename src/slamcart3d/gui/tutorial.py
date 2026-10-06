"""Optional, non-destructive guide connected to the desktop experiment workflow."""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

# Each destination is an existing page, not a separate simulated interface.
STEPS = (
    (
        0,
        "Qué vamos a observar",
        "What we will observe",
        "El carrito mide con ruedas, IMU y LiDAR. El estimador usa esas mediciones para calcular "
        "su posición y construir un mapa. La realidad conocida del simulador se reserva para evaluar errores. "
        "El carrito se mueve en suelo plano. El híbrido XYZ corrige una pose de seis componentes con retornos 3D; "
        "los otros estimadores localizan en el plano. No se simulan pendientes ni vuelo libre.",
        "The cart measures wheel motion, IMU and LiDAR. The estimator uses those measurements to estimate "
        "its position and build a map. The simulator's known ground truth is reserved for error evaluation. "
        "The cart moves on level ground. XYZ hybrid corrects six pose components from 3D returns; "
        "other estimators localize in the plane. Slopes and free flight are not simulated.",
    ),
    (
        0,
        "Preparar una práctica",
        "Prepare a practice run",
        "Pulse Preparar práctica para cargar una exploración autónoma de 30 segundos, semilla 42, "
        "LiDAR multicapa y el híbrido adaptativo. Se pedirá confirmar el reemplazo del formulario. "
        "Su carpeta de resultados se conserva. No comienza ninguna corrida; también puede seguir con su configuración. "
        "En Experimento, el desplegable Entorno cambia el ambiente. Para probar un laberinto con más tiempo, "
        "elija su ejemplo en Configuración de ejemplo y pulse Cargar ejemplo. Cambiar la semilla cambia sus pasillos.",
        "Press Prepare practice to load a 30-second autonomous exploration with seed 42, multilayer LiDAR "
        "and the adaptive hybrid. Replacing the form requires confirmation. Your output folder is "
        "preserved. No run starts; you can also continue with your own configuration. In Experiment, "
        "the Environment drop-down changes the scene. To try a longer maze run, choose its Example configuration "
        "and press Load example. Changing the seed changes its passages.",
    ),
    (
        0,
        "Ejecutar y observar",
        "Run and observe",
        "Abra Experimento y pulse Ejecutar experimento. El tiempo indicado es tiempo simulado, no una espera "
        "de reloj de 30 segundos. Durante el cálculo puede abrir Nube 3D. Se guardan pasos parciales; si aparece "
        "un error, revise el mensaje y la carpeta de salida antes de reintentar.",
        "Open Experiment and press Run experiment. The displayed time is simulated time, not a 30-second "
        "wall-clock wait. You can open 3D point cloud during computation. Partial steps are saved; if an "
        "error appears, check the message and output folder before retrying.",
    ),
    (
        5,
        "Leer el mundo 3D",
        "Read the 3D world",
        "La cuadrícula es el piso en z=0. Azul: paredes reales; verde: carrito real; naranja: carrito y ruta "
        "estimados; cian: puntos del mapa. Arrastre para girar y use la rueda para acercar. Corte de paredes "
        "solo facilita ver el interior. Los arcos de puntos sobre el piso son retornos del LiDAR, no la ruta. "
        "Mostrar realidad simulada permite contrastar la estimación sin darle información extra al algoritmo. "
        "Superficie medida muestra los vóxeles respaldados por observaciones; Solo reconstrucción oculta paredes y carrito reales. "
        "Los huecos no se rellenan: aún pueden ser zonas desconocidas. Para usar el estimador XYZ, cargue su ejemplo 3D.",
        "The grid is the floor at z=0. Blue: true walls; green: true cart; orange: estimated cart and route; "
        "cyan: map points. Drag to rotate and use the wheel to zoom. Cutaway only reveals the interior. "
        "Point arcs on the floor are LiDAR returns, not the route. Show simulated ground truth compares "
        "the estimate without giving the algorithm extra information. Measured surface shows voxels supported by observations; "
        "Reconstruction only hides true walls and cart. Holes are not filled: they may remain unknown. Load the XYZ example to use 3D pose correction.",
    ),
    (
        5,
        "Recorrer los pasos",
        "Inspect time steps",
        "Use Anterior, Siguiente o la barra inferior para revisar un instante ya calculado. La nube muestra "
        "lo acumulado hasta ese paso, no todo el mapa futuro. Al finalizar, la variante con grafo optimiza "
        "la trayectoria: el último paso puede cambiar; los anteriores conservan la estimación en línea. "
        "La exploración elige fronteras observadas: no necesita chocar y puede detenerse sin completar el mapa.",
        "Use Previous, Next or the bottom slider to inspect a computed time step. The cloud contains points "
        "accumulated by that step, not the future map. The graph variant optimizes the trajectory at completion: "
        "the final step may change, while earlier steps retain the online estimate. Exploration follows observed "
        "frontiers: it does not need to collide and can stop without completing the map.",
    ),
    (
        2,
        "Comprobar la estimación",
        "Check the estimate",
        "En Métricas, un RMS directo menor significa menor error de posición en el marco original. ATE "
        "compara la forma de la trayectoria tras alinearla. Distancia a paredes evalúa cercanía del mapa; "
        "cobertura indica cuánto se observó. Son medidas distintas: poco error no garantiza un mapa completo. "
        "En Nube 3D, Colorear error de pared contrasta puntos con el límite elegido usando la referencia.",
        "In Metrics, a smaller direct RMS means less position error in the original frame. ATE compares "
        "trajectory shape after alignment. Wall distance measures map proximity; coverage measures how much "
        "was observed. These are different: low error does not guarantee a complete map. In 3D point cloud, "
        "Color wall error compares points with your chosen threshold using ground truth.",
    ),
    (
        7,
        "Distinguir libre y desconocido",
        "Distinguish free and unknown",
        "En Ocupación 2D, azul significa libre observado, naranja ocupado estimado y el fondo oscuro "
        "desconocido. Un sensor sin retorno válido no demuestra espacio libre. Las paredes de referencia "
        "sirven para evaluar y pueden quedar sin observar. En Nube 3D, Zonas observadas usa otra leyenda: "
        "verde libre, rojo ocupado y gris desconocido; lea siempre la leyenda de la vista.",
        "In 2D occupancy, blue means observed free, orange estimated occupied and the dark background "
        "unknown. A missing valid return does not prove free space. Reference walls support evaluation and "
        "may remain unobserved. In 3D point cloud, Observed areas uses a different legend: green free, red "
        "occupied and grey unknown. Always read the current view's legend.",
    ),
    (
        3,
        "Comparar y conservar resultados",
        "Compare and keep results",
        "En Lotes compare algoritmos con las mismas semillas y condiciones. Estudio controlado cambia una "
        "variable; Robustez ASRF explora dos. Dataset genera observaciones sin aplicar SLAM; reproducción "
        "necesita sus mediciones y una referencia independiente para evaluar precisión. Conserve la carpeta "
        "EXP completa: config.yaml, metadata.json, trajectory.csv, metrics.csv y map. Guardar configuración "
        "guarda parámetros, no la sesión. El tutorial se puede cerrar y reabrir con su botón o F1.",
        "Use Batches to compare algorithms with the same seeds and conditions. Controlled study varies one "
        "condition; ASRF robustness explores two. Dataset generates observations without SLAM; replay needs "
        "your measurements and independent ground truth to evaluate accuracy. Keep the complete EXP folder: "
        "config.yaml, metadata.json, trajectory.csv, metrics.csv and map. Save configuration stores parameters, "
        "not a session. Close and reopen this guide using its button or F1.",
    ),
)


class TutorialDock(QDockWidget):
    """A reopenable guide; only an explicit confirmed action can change the form."""

    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self.setObjectName("integratedTutorial")
        self.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea)
        body = QWidget()
        layout = QVBoxLayout(body)
        self.step_combo = QComboBox()
        layout.addWidget(self.step_combo)
        self.content = QLabel()
        self.content.setWordWrap(True)
        self.content.setTextFormat(Qt.TextFormat.PlainText)
        self.content.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.content.setMargin(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.content)
        layout.addWidget(scroll, 1)
        self.state_label = QLabel()
        self.state_label.setWordWrap(True)
        layout.addWidget(self.state_label)
        self.prepare_button = QPushButton()
        self.prepare_button.clicked.connect(self.prepare)
        layout.addWidget(self.prepare_button)
        self.visit_button = QPushButton()
        self.visit_button.clicked.connect(self.visit)
        layout.addWidget(self.visit_button)
        row = QHBoxLayout()
        self.previous_button = QPushButton()
        self.next_button = QPushButton()
        self.previous_button.clicked.connect(lambda: self.step_combo.setCurrentIndex(self.index - 1))
        self.next_button.clicked.connect(self.advance)
        row.addWidget(self.previous_button)
        row.addWidget(self.next_button)
        layout.addLayout(row)
        self.setWidget(body)
        self.setMinimumWidth(300)
        self.step_combo.currentIndexChanged.connect(self.refresh)
        self.retranslate()
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.refresh_state)
        self.visibilityChanged.connect(lambda visible: self.timer.start() if visible else self.timer.stop())

    @property
    def index(self):
        return max(0, self.step_combo.currentIndex())

    def retranslate(self):
        index = self.index
        self.step_combo.blockSignals(True)
        self.step_combo.clear()
        language = 1 if self.host.language == "es" else 2
        for number, step in enumerate(STEPS, 1):
            self.step_combo.addItem(f"{number}. {step[language]}")
        self.step_combo.setCurrentIndex(index)
        self.step_combo.blockSignals(False)
        self.refresh()

    def refresh(self):
        es = self.host.language == "es"
        self.setWindowTitle(f"Tutorial — {self.index + 1}/{len(STEPS)}")
        self.content.setText(STEPS[self.index][3 if es else 4])
        self.previous_button.setText("Anterior" if es else "Previous")
        self.previous_button.setEnabled(self.index > 0)
        self.next_button.setText(
            ("Finalizar" if es else "Finish")
            if self.index == len(STEPS) - 1
            else ("Siguiente" if es else "Next")
        )
        self.prepare_button.setText("Preparar práctica de 30 s…" if es else "Prepare 30 s practice…")
        self.prepare_button.setVisible(self.index == 1)
        tab = STEPS[self.index][0]
        self.visit_button.setText(("Abrir: " if es else "Open: ") + self.host.tabs.tabText(tab))
        self.refresh_state()

    def refresh_state(self):
        es = self.host.language == "es"
        idle = self.host._can_replay() and self.host.recorded_page.worker is None
        self.prepare_button.setEnabled(idle)
        if not idle:
            text = (
                "Cálculo en curso. Puede recorrer el tutorial sin modificar la corrida."
                if es
                else "Computation in progress. You can read the guide without changing the run."
            )
        elif self.host._last_result is not None:
            text = (
                "Hay una corrida terminada. Revise sus métricas; el tutorial no certifica su precisión."
                if es
                else "A completed run is available. Check its metrics; this guide does not certify accuracy."
            )
        else:
            text = (
                "Aún no hay una corrida terminada. Leer el tutorial no ejecuta el experimento."
                if es
                else "No completed run yet. Reading the guide does not start an experiment."
            )
        self.state_label.setText(text)

    def visit(self):
        self.host.tabs.setCurrentIndex(STEPS[self.index][0])
        if self.index == 2:
            self.host.run_button.setFocus()
        elif self.index == 4:
            self.host.step_slider.setFocus()

    def prepare(self):
        if self.host._prepare_tutorial_practice():
            self.step_combo.setCurrentIndex(2)
            self.visit()

    def advance(self):
        if self.index == len(STEPS) - 1:
            self.hide()
        else:
            self.step_combo.setCurrentIndex(self.index + 1)
