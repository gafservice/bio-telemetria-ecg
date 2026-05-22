# bio-telemetria-ecg
# Optimizador conceptual de ubicación de electrodos ECG

Herramienta computacional desarrollada como parte del Trabajo Final de
Graduación (TFG) del proyecto de bio-telemetría ECG del Instituto Tecnológico
de Costa Rica (ITCR).

El propósito de este script es apoyar la evaluación conceptual de la ubicación
de electrodos ECG utilizando un modelo dipolar simplificado sobre una
referencia corporal bidimensional.

---

# Descripción general

El sistema implementa dos configuraciones ECG diferenciales independientes:

- Sistema 1:
  - RA01
  - LA01
  - LL01

- Sistema 2:
  - RA02
  - LA02
  - LL02

La condición principal del modelo es:

```text
LL02 = LL01
```

Es decir, ambos sistemas comparten el mismo electrodo inferior de referencia.

El algoritmo genera configuraciones geométricas aleatorias y evalúa la amplitud
de las derivaciones ECG obtenidas mediante un modelo dipolar simplificado.

---

# Modelo matemático utilizado

El potencial eléctrico estimado en cada electrodo se aproxima mediante:

```math
V_e(t)=K\frac{\vec{p}(t)\cdot\hat{r}}{r^2}
```

donde:

- `V_e(t)` : potencial estimado en el electrodo.
- `K` : constante de escala.
- `p(t)` : vector cardíaco sintético.
- `r_hat` : dirección unitaria hacia el electrodo.
- `r²` : atenuación aproximada por distancia.

A partir de estos potenciales se calculan derivaciones ECG diferenciales:

```math
DI = V_{LA} - V_{RA}
```

```math
DII = V_{LL} - V_{RA}
```

```math
DIII = V_{LL} - V_{LA}
```

---

# Objetivo del script

El objetivo principal es:

- explorar configuraciones espaciales de electrodos;
- analizar amplitudes diferenciales;
- mantener separación geométrica entre sistemas;
- conservar coherencia entre derivaciones;
- apoyar el diseño experimental del prototipo ECG.

El modelo NO pretende definir posiciones clínicas definitivas.

---

# Características principales

- Modelo dipolar ECG sintético.
- Simulación bidimensional.
- Dos sistemas ECG independientes.
- Electrodo inferior compartido.
- Restricciones geométricas.
- Optimización conceptual por amplitud pico-pico.
- Generación automática de figuras PNG y PDF.
- Visualización de derivaciones simuladas.
- Cálculo automático de distancias entre electrodos.

---

# Requisitos

Instalar dependencias:

```bash
pip install numpy matplotlib
```

o usando:

```bash
pip install -r requirements.txt
```

---

# Ejecución

Ejecutar:

```bash
python3 optimizador_electrodos_ecg_2s_rl_compartido.py
```

El programa:

1. Genera configuraciones aleatorias.
2. Evalúa restricciones geométricas.
3. Calcula derivaciones ECG sintéticas.
4. Selecciona configuraciones óptimas.
5. Genera visualización gráfica.
6. Exporta resultados en PNG y PDF.

---

# Resultados generados

El script genera automáticamente:

- figura PNG;
- figura PDF;
- coordenadas optimizadas;
- distancias geométricas;
- amplitudes pico-pico;
- visualización de derivaciones ECG.

---

# Archivos generados

Ejemplo:

```text
resultado_optimizacion_dos_sistemas_LL_compartido_20260521_153000.png
resultado_optimizacion_dos_sistemas_LL_compartido_20260521_153000.pdf
```

---

# Limitaciones del modelo

Este modelo corresponde únicamente a una aproximación conceptual y simplificada.

NO considera completamente:

- anatomía real tridimensional;
- impedancia de tejidos;
- variabilidad corporal;
- contacto electrodo-piel;
- ruido muscular;
- movimiento corporal;
- interferencias eléctricas reales;
- posición anatómica exacta del corazón.

Las coordenadas obtenidas deben interpretarse como una herramienta de apoyo al
diseño experimental y no como una configuración clínica estandarizada.

---

# Contexto académico

Este código fue desarrollado como apoyo al proyecto:

```text
Sistema de Bio-Telemetría ECG
Instituto Tecnológico de Costa Rica (ITCR)
```

El script forma parte del proceso de evaluación geométrica y conceptual de
ubicación de electrodos ECG dentro de una arquitectura de bio-telemetría
multicanal.

---

# Nota de desarrollo

Durante el desarrollo de este código se utilizaron herramientas de asistencia
basadas en inteligencia artificial como apoyo para:

- documentación técnica;
- organización estructural;
- depuración;
- mejora de legibilidad.

La integración conceptual, validación matemática, criterios de diseño y
adaptación experimental corresponden al autor del proyecto.

---

# Autor

Gerardo Araya

Instituto Tecnológico de Costa Rica (ITCR)

2026
