# TP N° 2 — Procesamiento de Imágenes I · TUIA

Trabajo práctico de la materia **Procesamiento de Imágenes I (IA 4.4)** de la
Tecnicatura Universitaria en Inteligencia Artificial — **UNR · FCEIA**.
Año 2026, 1° cuatrimestre.

El objetivo del TP es resolver dos problemas reales de visión por computadora
usando **únicamente técnicas clásicas de procesamiento de imágenes** (sin redes
neuronales ni modelos de machine learning): umbralización, morfología, espacios
de color, componentes conectados y análisis geométrico.

| | Problema 1 — Pastillas | Problema 2 — Patentes |
|---|---|---|
| **Entrada** | 1 foto de una cinta industrial | 12 fotos de autos |
| **Tarea** | Detectar, clasificar y contar pastillas | Detectar la patente y separar sus caracteres |
| **Resultado** | Cada pastilla rotulada con tipo e id | Patente recortada + 7 caracteres segmentados |

<p align="center">
  <img src="assets/e1_10_resultado.png" width="48%" alt="Pastillas clasificadas">
  <img src="assets/e2_08_bbox.png" width="48%" alt="Patente detectada">
</p>

---

## 📦 Problema 1 — Detección y clasificación de pastillas

Se parte de **una sola imagen** de una cinta transportadora con pastillas de
distintos colores y formas. El programa segmenta la cinta, aísla cada pastilla,
la **clasifica por tipo** (redonda blanca, redonda rosa, cuadrada, cápsula
amarilla, cápsula azul-blanca) y devuelve el conteo total junto con una imagen
rotulada.

➡️ **[Ver detalle del Problema 1](ejercicio_1/README.md)**

## 🚗 Problema 2 — Detección de placas patente

Se procesan **12 fotos de vehículos**, cada uno con su patente argentina modelo
Mercosur. Para cada imagen el programa **localiza la patente** buscando la fila
de 7 caracteres que cumple la geometría oficial, la recorta y **segmenta cada
carácter** por separado.

➡️ **[Ver detalle del Problema 2](ejercicio_2/README.md)**

---

## 🛠️ Tecnologías

- **Python 3.10+**
- **OpenCV** — procesamiento de imágenes
- **NumPy** — operaciones numéricas
- **Matplotlib** — visualización de resultados

## ▶️ Cómo ejecutarlo

El proyecto usa [**uv**](https://docs.astral.sh/uv/) como gestor de entornos
(durante el desarrollo `pip` nos daba errores de instalación en algunas PC).

```bash
# 1. Instalar dependencias
uv sync

# 2. Problema 1 — pastillas
uv run --directory ejercicio_1 python pastillas.py

# 3. Problema 2 — patentes
uv run --directory ejercicio_2 python patentes.py
```

> 💡 Cada script tiene un **modo debug** que muestra todas las etapas
> intermedias del procesamiento. Está documentado en el README de cada problema.

## 📁 Estructura del repositorio

```
tp_2/
├── ejercicio_1/          → Problema 1: pastillas
│   ├── pastillas.py
│   ├── img/pills.png
│   └── README.md
├── ejercicio_2/          → Problema 2: patentes
│   ├── patentes.py       → script principal
│   ├── placa.py          → detección de la patente
│   ├── caracteres.py     → segmentación de caracteres
│   ├── img/              → img_1.jpg … img_12.jpg
│   └── README.md
├── assets/               → imágenes de los pasos (para los README)
└── README.md             → este archivo
```

## 👥 Integrantes

- Mendez Ignacio
- Ramirez Lucas
- Lujambio Valentín
