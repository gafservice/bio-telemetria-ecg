# ============================================================
# Proyecto:
# Sistema de Bio-Telemetría ECG
# Instituto Tecnológico de Costa Rica (ITCR)
#
# Autor:
# Gerardo Araya
#
# Descripción:
# Herramienta de optimización conceptual para ubicación
# geométrica de electrodos ECG utilizando un modelo
# dipolar simplificado.
#
# Nota:
# Durante el desarrollo de este código se utilizaron
# herramientas de asistencia basadas en inteligencia
# artificial como apoyo para:
#
#   - documentación técnica;
#   - organización estructural del código;
#   - depuración;
#   - mejora de legibilidad.
#
# La integración conceptual, validación matemática,
# adaptación experimental y criterios de diseño
# corresponden al autor del proyecto.
# ============================================================




"""
Optimizador_Ubicacion_2S_Eectrodos.py

Optimizador conceptual de ubicación de electrodos ECG para dos sistemas
diferenciales independientes con electrodo inferior compartido.

Este archivo fue preparado para acompañar el Trabajo Final de Graduación (TFG)
del sistema de bio-telemetría ECG. El propósito del script es apoyar, de forma
computacional y reproducible, la selección conceptual de ubicaciones de
electrodos sobre una referencia corporal bidimensional.

El modelo utilizado NO representa una ubicación clínica definitiva. Se trata de
una herramienta de apoyo al diseño experimental. La silueta humana usada en la
figura es solamente una referencia visual simplificada.

Resumen del modelo:
    V_e(t) = K * (p(t) · r_hat) / r^2

donde:
    V_e(t)  : potencial estimado en un electrodo.
    K       : constante de escala del modelo.
    p(t)    : vector cardíaco sintético variable en el tiempo.
    r_hat   : vector unitario desde el centro cardíaco hacia el electrodo.
    r^2     : atenuación aproximada por distancia.

Sistemas evaluados:
    Sistema 1: RA01, LA01, LL01
    Sistema 2: RA02, LA02, LL02

Condición de diseño:
    LL02 = LL01

Esto significa que ambos sistemas comparten el electrodo inferior.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path

# ============================================================
# CONFIGURACIÓN GENERAL DE SIMULACIÓN
# ============================================================

# Frecuencia de muestreo sintética usada para generar las señales ECG.
# Se utiliza 500 Hz porque es una frecuencia habitual para análisis temporal
# de señales ECG y coincide con el criterio de muestreo usado en el proyecto.
fs = 500

# Duración de la señal sintética generada, en segundos.
duracion = 1.2

# Vector temporal de simulación. Contiene fs * duración muestras.
t = np.linspace(0, duracion, int(duracion * fs))

# Valor pequeño para evitar divisiones entre cero en cálculos de distancia.
EPS = 1e-6

COLORES = {
    "RA01": "#d62728",   # rojo
    "LA01": "#f2c230",   # amarillo
    "LL01": "#2ca02c",   # verde
    "RA02": "#d62728",   # rojo oscuro
    "LA02": "#f2c230",   # amarillo oscuro
    "LL02": "#2ca02c",   # mismo verde porque comparte LL01
    "SRC":  "#000000"
}


# ============================================================
# Modelo ECG sintético
# ============================================================

def gaussian(tau, mu, sigma, amp):
    """
    Genera un pulso gaussiano.

    Esta función se usa para aproximar matemáticamente las ondas P, Q, R, S y T
    que componen un ciclo ECG sintético.

    Parámetros
    ----------
    tau : ndarray
        Tiempo relativo dentro de cada ciclo cardíaco.
    mu : float
        Instante central del pulso.
    sigma : float
        Ancho temporal del pulso.
    amp : float
        Amplitud del pulso.

    Retorna
    -------
    ndarray
        Pulso gaussiano evaluado en el vector tau.
    """
    return amp * np.exp(-0.5 * ((tau - mu) / sigma) ** 2)


def synthetic_vector_ecg(t, hr=70, qrs_amp=1.0):
    """
    Genera un vector cardíaco sintético bidimensional p(t).

    El modelo aproxima un ciclo ECG usando cinco componentes gaussianas
    asociadas a las ondas P, Q, R, S y T. Cada componente tiene una amplitud,
    una posición temporal, un ancho y una dirección angular en el plano.

    Parámetros
    ----------
    t : ndarray
        Vector de tiempo de simulación.
    hr : float
        Frecuencia cardíaca simulada en latidos por minuto.
    qrs_amp : float
        Factor de escala aplicado principalmente al complejo QRS.

    Retorna
    -------
    ndarray
        Arreglo de tamaño (N, 2), donde cada fila representa el vector
        cardíaco sintético p(t) = [px(t), py(t)].
    """
    # Periodo cardíaco aproximado asociado a la frecuencia cardíaca.
    period = 60.0 / hr
    # Tiempo relativo dentro de cada ciclo. Permite repetir el mismo patrón
    # cardíaco durante toda la duración simulada.
    tau = np.mod(t, period)

    waves = [
        (0.18,           0.18 * period, 0.035 * period,  55),
        (-0.15*qrs_amp,  0.37 * period, 0.012 * period,  20),
        (1.00*qrs_amp,   0.40 * period, 0.010 * period,  65),
        (-0.35*qrs_amp,  0.43 * period, 0.014 * period, 105),
        (0.32,           0.68 * period, 0.060 * period,  35),
    ]

    # Inicializa el vector cardíaco sintético en dos dimensiones.
    p_t = np.zeros((len(t), 2))

    for amp, mu, sigma, angle_deg in waves:
        # Genera la componente temporal de la onda ECG.
        g = gaussian(tau, mu, sigma, amp)

        # Convierte la dirección angular de grados a radianes.
        th = np.deg2rad(angle_deg)

        # Dirección espacial de la componente dentro del plano corporal.
        direction = np.array([np.cos(th), -np.sin(th)])

        # Suma la contribución vectorial de esta onda al vector cardíaco total.
        p_t += g[:, None] * direction[None, :]

    return p_t


def rotate_vectors(vectors, angle_deg):
    """
    Rota un conjunto de vectores bidimensionales.

    Se utiliza para evaluar cómo cambia la señal observada cuando el vector
    cardíaco resultante se orienta con distinto ángulo en el plano frontal.

    Parámetros
    ----------
    vectors : ndarray
        Arreglo de vectores de entrada de tamaño (N, 2).
    angle_deg : float
        Ángulo de rotación en grados.

    Retorna
    -------
    ndarray
        Vectores rotados.
    """
    # Convierte el ángulo de grados a radianes.
    th = np.deg2rad(angle_deg)
    # Matriz clásica de rotación 2D.
    R = np.array([
        [np.cos(th), -np.sin(th)],
        [np.sin(th),  np.cos(th)]
    ])

    # Aplica la rotación a todos los vectores.
    return vectors @ R.T


def electrode_potential(p_t, electrode_pos, source_pos, K=1.0):
    """
    Calcula el potencial eléctrico aproximado en un electrodo.

    El cálculo se basa en un modelo dipolar simplificado:

        V_e(t) = K * (p(t) · r_hat) / r^2

    donde p(t) es el vector cardíaco sintético, r_hat es la dirección unitaria
    desde el centro cardíaco hacia el electrodo y r es la distancia entre ambos.

    Parámetros
    ----------
    p_t : ndarray
        Vector cardíaco sintético de tamaño (N, 2).
    electrode_pos : ndarray
        Coordenadas del electrodo [x, y] en centímetros.
    source_pos : ndarray
        Coordenadas aproximadas del centro cardíaco [x, y] en centímetros.
    K : float
        Constante de escala del modelo.

    Retorna
    -------
    ndarray
        Potencial eléctrico estimado en el electrodo.
    """
    # Vector desde el centro cardíaco aproximado hacia el electrodo.
    r = electrode_pos - source_pos
    # Distancia euclidiana entre el centro cardíaco y el electrodo.
    r_norm = np.linalg.norm(r)

    # Evita división entre cero si el electrodo cae demasiado cerca del centro.
    if r_norm < EPS:
        r_norm = EPS

    # Vector unitario que indica la dirección del electrodo.
    r_hat = r / r_norm

    # Producto punto p(t) · r_hat y atenuación por distancia al cuadrado.
    return K * (p_t @ r_hat) / (r_norm ** 2)


def calcular_derivaciones_raw(RA, LA, LL, SRC, angle_deg=0, hr=70, qrs_amp=1.0, gain=1.0):
    """
    Calcula derivaciones ECG crudas para una configuración de electrodos.

    A partir del modelo dipolar, primero estima los potenciales en RA, LA y LL.
    Luego calcula las derivaciones clásicas:

        DI   = V_LA - V_RA
        DII  = V_LL - V_RA
        DIII = V_LL - V_LA

    Parámetros
    ----------
    RA, LA, LL : ndarray
        Coordenadas de los electrodos.
    SRC : ndarray
        Centro cardíaco aproximado.
    angle_deg : float
        Rotación aplicada al vector cardíaco.
    hr : float
        Frecuencia cardíaca sintética.
    qrs_amp : float
        Escala de amplitud QRS.
    gain : float
        Ganancia o constante de escala K.

    Retorna
    -------
    tuple
        Señales DI, DII y DIII sin normalizar.
    """
    # Genera el vector cardíaco sintético.
    p_t = synthetic_vector_ecg(t, hr=hr, qrs_amp=qrs_amp)

    # Aplica una rotación global al vector cardíaco para evaluar orientación.
    p_t = rotate_vectors(p_t, angle_deg)

    # Estima el potencial en cada electrodo usando el modelo dipolar.
    V_RA = electrode_potential(p_t, RA, source_pos=SRC, K=gain)
    V_LA = electrode_potential(p_t, LA, source_pos=SRC, K=gain)
    V_LL = electrode_potential(p_t, LL, source_pos=SRC, K=gain)

    # Cálculo diferencial de derivaciones.
    DI   = V_LA - V_RA
    DII  = V_LL - V_RA
    DIII = V_LL - V_LA

    return DI, DII, DIII


def amplitud_pp(x):
    """
    Calcula la amplitud pico a pico de una señal.

    Parámetros
    ----------
    x : ndarray
        Señal de entrada.

    Retorna
    -------
    float
        Diferencia entre el valor máximo y el valor mínimo.
    """
    return float(np.max(x) - np.min(x))


def calcular_metricas(RA, LA, LL, SRC, angle_deg=0, hr=70, qrs_amp=1.0, gain=1.0):
    """
    Calcula señales diferenciales y métricas de amplitud para una configuración.

    La métrica principal es la amplitud pico a pico de cada derivación. También
    se calcula una métrica global, definida como la mayor amplitud pico a pico
    entre DI, DII y DIII.

    Retorna
    -------
    dict
        Diccionario con amplitudes pico a pico y señales crudas.
    """
    DI, DII, DIII = calcular_derivaciones_raw(
        RA, LA, LL, SRC,
        angle_deg=angle_deg,
        hr=hr,
        qrs_amp=qrs_amp,
        gain=gain
    )

    return {
        "DI_pp": amplitud_pp(DI),
        "DII_pp": amplitud_pp(DII),
        "DIII_pp": amplitud_pp(DIII),
        "global_pp": max(amplitud_pp(DI), amplitud_pp(DII), amplitud_pp(DIII)),
        "DI": DI,
        "DII": DII,
        "DIII": DIII,
    }


# ============================================================
# Restricciones geométricas
# ============================================================

def distancia(a, b):
    """
    Calcula la distancia euclidiana entre dos puntos del plano.

    Parámetros
    ----------
    a, b : ndarray
        Coordenadas [x, y] de los puntos.

    Retorna
    -------
    float
        Distancia entre los puntos.
    """
    return float(np.linalg.norm(a - b))


def cumple_restricciones_sistema(
    RA, LA, LL, SRC,
    d_min_corazon=8.0,
    d_min_electrodos=10.0,
    d_max_electrodos=55.0
):
    """
    Verifica restricciones geométricas básicas para un sistema de electrodos.

    Las restricciones evitan configuraciones poco razonables, por ejemplo:
    electrodos demasiado cerca del centro cardíaco, electrodos muy cercanos
    entre sí o distancias excesivas dentro de la silueta corporal simplificada.

    Retorna
    -------
    bool
        True si la configuración cumple las restricciones, False si no.
    """
    electrodos = [RA, LA, LL]

    for e in electrodos:
        if distancia(e, SRC) < d_min_corazon:
            return False

    for a, b in [(RA, LA), (RA, LL), (LA, LL)]:
        d = distancia(a, b)
        if d < d_min_electrodos:
            return False
        if d > d_max_electrodos:
            return False

    return True


def cumple_restricciones_sistema_2(
    RA02, LA02, LL_compartido, SRC,
    RA01, LA01,
    d_min_corazon=8.0,
    d_min_electrodos=10.0,
    d_max_electrodos=55.0,
    d_min_entre_sistemas=6.0
):
    """
    Restricciones del segundo sistema.

    Además de las restricciones geométricas básicas, se evita que RA02 y LA02
    caigan demasiado cerca de RA01 y LA01.
    """

    if not cumple_restricciones_sistema(
        RA02, LA02, LL_compartido, SRC,
        d_min_corazon=d_min_corazon,
        d_min_electrodos=d_min_electrodos,
        d_max_electrodos=d_max_electrodos
    ):
        return False

    # Evita superposición con electrodos superiores del sistema 1.
    if distancia(RA02, RA01) < d_min_entre_sistemas:
        return False
    if distancia(RA02, LA01) < d_min_entre_sistemas:
        return False
    if distancia(LA02, RA01) < d_min_entre_sistemas:
        return False
    if distancia(LA02, LA01) < d_min_entre_sistemas:
        return False

    return True


# ============================================================
# Búsqueda del sistema 1
# ============================================================

def buscar_sistema_1(
    SRC=np.array([2.0, 0.0]),
    angle_deg=0,
    hr=70,
    qrs_amp=1.0,
    gain=1.0,
    objetivo="global_pp",
    n_muestras=40000,
    semilla=7
):
    """
    Busca una configuración candidata para el primer sistema ECG.

    El algoritmo genera posiciones aleatorias para RA01, LA01 y LL01 dentro de
    regiones del torso definidas manualmente. Cada configuración se evalúa con
    el modelo dipolar y se conserva la que maximiza la métrica seleccionada.

    Parámetros
    ----------
    SRC : ndarray
        Centro cardíaco aproximado.
    objetivo : str
        Métrica usada para seleccionar la mejor configuración. Por defecto se
        usa 'global_pp'.
    n_muestras : int
        Cantidad de configuraciones aleatorias evaluadas.
    semilla : int
        Semilla del generador aleatorio para reproducibilidad.

    Retorna
    -------
    dict
        Mejor configuración encontrada para el sistema 1.
    """
    # Generador aleatorio reproducible.
    rng = np.random.default_rng(semilla)

    # Regiones admisibles de búsqueda para RA02 y LA02.
    limites = {
        "RA01": {"x": (-24.0,  0.0), "y": (-8.0, 24.0)},
        "LA01": {"x": (  0.0, 24.0), "y": (-8.0, 24.0)},
        "LL01": {"x": (-16.0, 16.0), "y": (-36.0, -8.0)},
    }

    mejor = None

    # Evalúa configuraciones candidatas para el segundo sistema.
    for _ in range(n_muestras):
        RA01 = np.array([
            rng.uniform(*limites["RA01"]["x"]),
            rng.uniform(*limites["RA01"]["y"])
        ])
        LA01 = np.array([
            rng.uniform(*limites["LA01"]["x"]),
            rng.uniform(*limites["LA01"]["y"])
        ])
        LL01 = np.array([
            rng.uniform(*limites["LL01"]["x"]),
            rng.uniform(*limites["LL01"]["y"])
        ])

        # Descarta configuraciones que no cumplen las restricciones geométricas.
        if not cumple_restricciones_sistema(RA01, LA01, LL01, SRC):
            continue

        metricas = calcular_metricas(
            RA01, LA01, LL01, SRC,
            angle_deg=angle_deg,
            hr=hr,
            qrs_amp=qrs_amp,
            gain=gain
        )

        # Valor de la métrica usada como criterio de optimización.
        valor = metricas[objetivo]

        # Actualiza la mejor configuración si la nueva tiene mayor métrica.
        if mejor is None or valor > mejor["valor"]:
            mejor = {
                "RA01": RA01,
                "LA01": LA01,
                "LL01": LL01,
                "SRC": SRC,
                "valor": valor,
                "metricas": metricas
            }

    return mejor


# ============================================================
# Búsqueda del sistema 2 con LL compartido
# ============================================================

def buscar_sistema_2_con_LL_compartido(
    sistema_1,
    angle_deg=0,
    hr=70,
    qrs_amp=1.0,
    gain=1.0,
    objetivo="global_pp",
    n_muestras=40000,
    semilla=21
):
    """
    Busca una configuración candidata para el primer sistema ECG.

    El algoritmo genera posiciones aleatorias para RA01, LA01 y LL01 dentro de
    regiones del torso definidas manualmente. Cada configuración se evalúa con
    el modelo dipolar y se conserva la que maximiza la métrica seleccionada.

    Parámetros
    ----------
    SRC : ndarray
        Centro cardíaco aproximado.
    objetivo : str
        Métrica usada para seleccionar la mejor configuración. Por defecto se
        usa 'global_pp'.
    n_muestras : int
        Cantidad de configuraciones aleatorias evaluadas.
    semilla : int
        Semilla del generador aleatorio para reproducibilidad.

    Retorna
    -------
    dict
        Mejor configuración encontrada para el sistema 1.
    """
    # Generador aleatorio reproducible.
    # Generador aleatorio reproducible.
    rng = np.random.default_rng(semilla)

    # Recupera las posiciones optimizadas del sistema 1.
    RA01 = sistema_1["RA01"]
    LA01 = sistema_1["LA01"]
    LL01 = sistema_1["LL01"]
    SRC  = sistema_1["SRC"]

    # Condición principal del diseño: el sistema 2 comparte el mismo LL.
    LL02 = LL01.copy()

    # Regiones admisibles de búsqueda para RA02 y LA02.
    limites = {
        "RA02": {"x": (-24.0,  0.0), "y": (-8.0, 24.0)},
        "LA02": {"x": (  0.0, 24.0), "y": (-8.0, 24.0)},
    }

    mejor = None

    # Evalúa configuraciones candidatas para el segundo sistema.
    for _ in range(n_muestras):
        RA02 = np.array([
            rng.uniform(*limites["RA02"]["x"]),
            rng.uniform(*limites["RA02"]["y"])
        ])

        LA02 = np.array([
            rng.uniform(*limites["LA02"]["x"]),
            rng.uniform(*limites["LA02"]["y"])
        ])

        # Verifica restricciones propias del sistema 2, incluyendo separación
        # respecto a los electrodos superiores del sistema 1.
        if not cumple_restricciones_sistema_2(
            RA02, LA02, LL02, SRC,
            RA01=RA01,
            LA01=LA01
        ):
            continue

        metricas = calcular_metricas(
            RA02, LA02, LL02, SRC,
            angle_deg=angle_deg,
            hr=hr,
            qrs_amp=qrs_amp,
            gain=gain
        )

        # Valor de la métrica usada como criterio de optimización.
        valor = metricas[objetivo]

        # Actualiza la mejor configuración si la nueva tiene mayor métrica.
        if mejor is None or valor > mejor["valor"]:
            mejor = {
                "RA02": RA02,
                "LA02": LA02,
                "LL02": LL02,
                "SRC": SRC,
                "valor": valor,
                "metricas": metricas
            }

    return mejor


# ============================================================
# Silueta humana de referencia
# ============================================================

def dibujar_silueta_torso(ax):
    """
    Dibuja una silueta corporal simplificada sobre el eje suministrado.

    La silueta no representa anatomía clínica. Se usa solamente como referencia
    visual para ubicar electrodos dentro de un plano corporal 2D.
    """
    # Cabeza esquemática.
    cabeza = Circle((0, 35), 5.0, fill=False, linewidth=1.2, alpha=0.45)
    ax.add_patch(cabeza)

    ax.plot([-3.2, -3.2], [30, 25], linewidth=1.0, alpha=0.35)
    ax.plot([ 3.2,  3.2], [30, 25], linewidth=1.0, alpha=0.35)

    verts = [
        (-18, 24), (-23, 17), (-21,  5), (-18, -8),
        (-15, -25), (-11, -38), (11, -38), (15, -25),
        (18, -8), (21, 5), (23, 17), (18, 24),
        (10, 25), (3, 24), (0, 23), (-3, 24),
        (-10, 25), (-18, 24),
    ]

    codes = [Path.MOVETO] + [Path.CURVE3]*(len(verts)-2) + [Path.CLOSEPOLY]
    patch = PathPatch(Path(verts, codes), fill=False, linewidth=1.4, alpha=0.45)
    ax.add_patch(patch)

    ax.plot([0, 0], [24, -36], linestyle=":", linewidth=1.0, alpha=0.35)
    ax.plot([-18, 18], [22, 22], linestyle=":", linewidth=0.9, alpha=0.25)
    ax.plot([-15, 15], [-8, -8], linestyle=":", linewidth=0.9, alpha=0.25)

    ax.text(0, 41.5, "Referencia corporal", ha="center", fontsize=9, alpha=0.65)


# ============================================================
# Formatos de información
# ============================================================

def formato_coordenadas_doble(s1, s2):
    """
    Genera texto con las coordenadas optimizadas de ambos sistemas.

    Se usa para imprimir en consola y para colocar un cuadro informativo dentro
    de la figura generada.
    """
    RA01, LA01, LL01, SRC = s1["RA01"], s1["LA01"], s1["LL01"], s1["SRC"]
    RA02, LA02, LL02 = s2["RA02"], s2["LA02"], s2["LL02"]

    return (
        "Coordenadas optimizadas [cm]\n"
        "Sistema 1\n"
        f"RA01 = ({RA01[0]:6.2f}, {RA01[1]:6.2f})\n"
        f"LA01 = ({LA01[0]:6.2f}, {LA01[1]:6.2f})\n"
        f"LL01 = ({LL01[0]:6.2f}, {LL01[1]:6.2f})\n\n"
        "Sistema 2\n"
        f"RA02 = ({RA02[0]:6.2f}, {RA02[1]:6.2f})\n"
        f"LA02 = ({LA02[0]:6.2f}, {LA02[1]:6.2f})\n"
        f"LL02 = ({LL02[0]:6.2f}, {LL02[1]:6.2f})\n\n"
        f"SRC  = ({SRC[0]:6.2f}, {SRC[1]:6.2f})"
    )


def formato_distancias_doble(s1, s2):
    """
    Genera texto con distancias principales entre electrodos.

    Incluye distancias internas de cada sistema y separación entre los puntos
    superiores de ambos sistemas. También verifica que LL02 y LL01 coincidan.
    """
    RA01, LA01, LL01, SRC = s1["RA01"], s1["LA01"], s1["LL01"], s1["SRC"]
    RA02, LA02, LL02 = s2["RA02"], s2["LA02"], s2["LL02"]

    return (
        "Distancias principales [cm]\n"
        "Sistema 1\n"
        f"LA01 - RA01 = {distancia(LA01, RA01):6.2f}\n"
        f"LL01 - RA01 = {distancia(LL01, RA01):6.2f}\n"
        f"LL01 - LA01 = {distancia(LL01, LA01):6.2f}\n\n"
        "Sistema 2\n"
        f"LA02 - RA02 = {distancia(LA02, RA02):6.2f}\n"
        f"LL02 - RA02 = {distancia(LL02, RA02):6.2f}\n"
        f"LL02 - LA02 = {distancia(LL02, LA02):6.2f}\n\n"
        "Separación entre sistemas\n"
        f"RA02 - RA01 = {distancia(RA02, RA01):6.2f}\n"
        f"LA02 - LA01 = {distancia(LA02, LA01):6.2f}\n"
        f"LL02 - LL01 = {distancia(LL02, LL01):6.2f}"
    )


def imprimir_resultado_doble(s1, s2):
    """
    Imprime en consola el resultado de la optimización.

    Muestra coordenadas, distancias y amplitudes pico a pico para ambos
    sistemas ECG. Esta salida facilita documentar y verificar los resultados
    obtenidos por el script.
    """
    print("\n================ SISTEMA 1 =================\n")
    print(formato_coordenadas_doble(s1, s2))
    print()
    print(formato_distancias_doble(s1, s2))

    print("\nAmplitudes pico a pico crudas:")
    print("Sistema 1")
    print(f"  DI   = {s1['metricas']['DI_pp']:.6f}")
    print(f"  DII  = {s1['metricas']['DII_pp']:.6f}")
    print(f"  DIII = {s1['metricas']['DIII_pp']:.6f}")
    print(f"  Global = {s1['metricas']['global_pp']:.6f}")

    print("Sistema 2")
    print(f"  DI   = {s2['metricas']['DI_pp']:.6f}")
    print(f"  DII  = {s2['metricas']['DII_pp']:.6f}")
    print(f"  DIII = {s2['metricas']['DIII_pp']:.6f}")
    print(f"  Global = {s2['metricas']['global_pp']:.6f}")

    print("\nNota:")
    print("  LL02 comparte exactamente la misma posición que LL01.")
    print("  La optimización es conceptual y depende del modelo dipolar simplificado.\n")


# ============================================================
# Gráfica
# ============================================================

def graficar_resultado_doble(s1, s2, nombre_base="resultado_optimizacion_dos_sistemas_LL_compartido"):
    """
    Genera la figura resumen de la optimización.

    La figura incluye:
    - silueta corporal de referencia;
    - electrodos del sistema 1;
    - electrodos del sistema 2;
    - electrodo LL compartido;
    - centro cardíaco aproximado;
    - señales diferenciales simuladas;
    - coordenadas;
    - distancias principales;
    - leyenda.

    También guarda la figura en PNG y PDF con marca temporal.
    """
    RA01, LA01, LL01, SRC = s1["RA01"], s1["LA01"], s1["LL01"], s1["SRC"]
    RA02, LA02, LL02 = s2["RA02"], s2["LA02"], s2["LL02"]
    m1, m2 = s1["metricas"], s2["metricas"]

    fig = plt.figure(figsize=(15.2, 8.0))

    # Croquis corporal
    ax0 = fig.add_axes([0.04, 0.12, 0.36, 0.78])
    ax0.set_title("Dos sistemas de electrodos con LL compartido")
    ax0.set_aspect("equal")
    dibujar_silueta_torso(ax0)

    # Sistema 1 - líneas sólidas
    ax0.plot([RA01[0], LA01[0]], [RA01[1], LA01[1]],
             linewidth=1.9, color="#777777", alpha=0.75)
    ax0.plot([RA01[0], LL01[0]], [RA01[1], LL01[1]],
             linewidth=1.9, color="#777777", alpha=0.55)
    ax0.plot([LA01[0], LL01[0]], [LA01[1], LL01[1]],
             linewidth=1.9, color="#777777", alpha=0.55)

    # Sistema 2 - líneas punteadas
    ax0.plot([RA02[0], LA02[0]], [RA02[1], LA02[1]],
             linewidth=1.8, color="#333333", alpha=0.85, linestyle="--")
    ax0.plot([RA02[0], LL02[0]], [RA02[1], LL02[1]],
             linewidth=1.8, color="#333333", alpha=0.60, linestyle="--")
    ax0.plot([LA02[0], LL02[0]], [LA02[1], LL02[1]],
             linewidth=1.8, color="#333333", alpha=0.60, linestyle="--")

    # Electrodos sistema 1
    ax0.scatter(RA01[0], RA01[1], s=175, color=COLORES["RA01"],
                edgecolor="black", linewidth=0.8, zorder=5)
    ax0.scatter(LA01[0], LA01[1], s=175, color=COLORES["LA01"],
                edgecolor="black", linewidth=0.8, zorder=5)
    ax0.scatter(LL01[0], LL01[1], s=210, color=COLORES["LL01"],
                edgecolor="black", linewidth=0.8, zorder=6)

    # Electrodos sistema 2
    ax0.scatter(RA02[0], RA02[1], s=160, color=COLORES["RA02"],
                edgecolor="black", linewidth=0.8, marker="s", zorder=5)
    ax0.scatter(LA02[0], LA02[1], s=160, color=COLORES["LA02"],
                edgecolor="black", linewidth=0.8, marker="s", zorder=5)

    # Centro cardíaco representado con icono de corazón
    ax0.text(
        SRC[0],
        SRC[1],
        "❤",
        fontsize=22,
        ha="center",
        va="center",
        color="crimson",
        zorder=7
    )

    # Etiquetas
    ax0.text(RA01[0]-3.0, RA01[1]+2.3, "RA01", fontsize=9, weight="bold", color=COLORES["RA01"])
    ax0.text(LA01[0]+1.2, LA01[1]+2.3, "LA01", fontsize=9, weight="bold", color="#9a7a00")
    ax0.text(LL01[0]+1.2, LL01[1]-3.0, "LL01/LL02", fontsize=9, weight="bold", color=COLORES["LL01"])
    ax0.text(RA02[0]-3.0, RA02[1]-3.0, "RA02", fontsize=9, weight="bold", color=COLORES["RA02"])
    ax0.text(LA02[0]+1.2, LA02[1]-3.0, "LA02", fontsize=9, weight="bold", color="#8a6500")
    ax0.text(SRC[0]+1.4, SRC[1]+1.8, "Corazón", fontsize=9, weight="bold", color="crimson")

    ax0.set_xlim(-30, 30)
    ax0.set_ylim(-42, 45)
    ax0.set_xlabel("x [cm]")
    ax0.set_ylabel("y [cm]")
    ax0.grid(True, alpha=0.25)

    

    # Señales sistema 1 (columna izquierda)
    ax1 = fig.add_axes([0.44, 0.72, 0.18, 0.14])
    ax2 = fig.add_axes([0.44, 0.53, 0.18, 0.14])
    ax3 = fig.add_axes([0.44, 0.34, 0.18, 0.14])

    ax1.plot(t, m1["DI"], linewidth=1.2, color="#444444")
    ax1.set_title(f"S1 DI pp={m1['DI_pp']:.4f}", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, m1["DII"], linewidth=1.2, color="#444444")
    ax2.set_title(f"S1 DII pp={m1['DII_pp']:.4f}", fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3.plot(t, m1["DIII"], linewidth=1.2, color="#444444")
    ax3.set_title(f"S1 DIII pp={m1['DIII_pp']:.4f}", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Señales sistema 2 (columna derecha)
    ax4 = fig.add_axes([0.64, 0.72, 0.18, 0.14])
    ax5 = fig.add_axes([0.64, 0.53, 0.18, 0.14])
    ax6 = fig.add_axes([0.64, 0.34, 0.18, 0.14])

    ax4.plot(t, m2["DI"], linewidth=1.2, color="#111111", linestyle="--")
    ax4.set_title(f"S2 DI pp={m2['DI_pp']:.4f}", fontsize=9)
    ax4.grid(True, alpha=0.3)

    ax5.plot(t, m2["DII"], linewidth=1.2, color="#111111", linestyle="--")
    ax5.set_title(f"S2 DII pp={m2['DII_pp']:.4f}", fontsize=9)
    ax5.grid(True, alpha=0.3)

    ax6.plot(t, m2["DIII"], linewidth=1.2, color="#111111", linestyle="--")
    ax6.set_title(f"S2 DIII pp={m2['DIII_pp']:.4f}", fontsize=9)
    ax6.grid(True, alpha=0.3)

    # Cuadros externos
    ax_info1 = fig.add_axes([0.44, 0.06, 0.18, 0.20])
    ax_info1.axis("off")
    ax_info1.text(
        0.0, 1.0,
        formato_coordenadas_doble(s1, s2),
        va="top",
        ha="left",
        fontsize=8.2,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="gray", alpha=0.95)
    )

    ax_info2 = fig.add_axes([0.64, 0.06, 0.18, 0.20])
    ax_info2.axis("off")
    ax_info2.text(
        0.0, 1.0,
        formato_distancias_doble(s1, s2),
        va="top",
        ha="left",
        fontsize=8.2,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="gray", alpha=0.95)
    )

    ax_legend = fig.add_axes([0.84, 0.06, 0.14, 0.20])
    ax_legend.axis("off")

    leyenda = (
        "Simbología\n"
        "Sistema 1: círculos, líneas sólidas\n"
        "Sistema 2: cuadros, líneas punteadas\n"
        "RA01/RA02: rojo\n"
        "LA01/LA02: amarillo\n"
        "LL01=LL02: verde compartido\n"
        "❤ : centro cardíaco aproximado"
    )

    ax_legend.text(
        0.0, 1.0,
        leyenda,
        va="top",
        ha="left",
        fontsize=8.2,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="gray", alpha=0.95)
    )

    fig.suptitle(
        "Optimización conceptual de dos sistemas de electrodos ECG con LL compartido",
        fontsize=14,
        fontweight="bold"
    )

    fig.text(
        0.04,
        0.035,
        "La silueta es una referencia visual. La optimización corresponde únicamente al modelo dipolar simplificado.",
        fontsize=8
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png = f"{nombre_base}_{timestamp}.png"
    pdf = f"{nombre_base}_{timestamp}.pdf"

    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")

    print(f"Figura guardada: {png}")
    print(f"Figura guardada: {pdf}")

    plt.show()


# ============================================================
# Ejecución
# ============================================================

if __name__ == "__main__":
    # Punto de entrada del programa.
    # Primero se optimiza el sistema 1. Luego, usando el LL obtenido en ese
    # sistema, se optimiza el sistema 2 con LL compartido.
    sistema_1 = buscar_sistema_1(
        SRC=np.array([2.0, 0.0]),
        angle_deg=0,
        hr=70,
        qrs_amp=1.0,
        gain=1.0,
        objetivo="global_pp",
        n_muestras=40000,
        semilla=7
    )

    sistema_2 = buscar_sistema_2_con_LL_compartido(
        sistema_1,
        angle_deg=0,
        hr=70,
        qrs_amp=1.0,
        gain=1.0,
        objetivo="global_pp",
        n_muestras=40000,
        semilla=21
    )

    imprimir_resultado_doble(sistema_1, sistema_2)
    graficar_resultado_doble(sistema_1, sistema_2)
