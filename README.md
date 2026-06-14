# TP N° 2 — Procesamiento de Imágenes I - TUIA

Trabajo práctico de la materia **Procesamiento de Imágenes I (IA 4.4)** de la Tecnicatura Universitaria en Inteligencia Artificial — UNR, FCEIA.

Año 2026, 1° cuatrimestre.

## Integrantes

Mendez Ignacio
Ramirez Lucas
Lujambio Valentin

## Contenido

El TP se divide en dos problemas independientes, resueltos con técnicas que vimos en clases de procesamiento de imágenes (PDI):

- **[Problema 1](README_problema_1.md)** — Detección y clasificación de pastillas sobre una cinta transportadora industrial.
- **[Problema 2](README_problema_2.md)** — Detección de placas patente y segmentación de sus caracteres a partir de imágenes de vehículos.

### Ejecutar cada ejercicio con uv (porque pip en una pc nos estaba dando error de instalación)

```bash
uv sync

# Ejercicio 1 - pastillas
uv run --directory ejercicio_1 python pastillas.py

# Ejercicio 2 - patentes
uv run --directory ejercicio_2 python patentes.py
```
