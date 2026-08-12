"""
Puente hacia el codigo original de los ejercicios.

Los dos ejercicios se escribieron como scripts sueltos: `ejercicio_1/pastillas.py`
y `ejercicio_2/{patentes,placa,caracteres,help_show}.py`, con imports planos
entre ellos (`from caracteres import ...`). En vez de reacomodar ese codigo —que
es el que se entrego y el que resuelve el problema— este modulo agrega las dos
carpetas al `sys.path` y reexporta las funciones que necesita la API.

Tambien fija el backend `Agg` de matplotlib *antes* de importarlos: esos modulos
hacen `import matplotlib.pyplot` a nivel de modulo y en un servidor sin display
el backend interactivo por defecto falla al importar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
E1 = ROOT / "ejercicio_1"
E2 = ROOT / "ejercicio_2"

for path in (E1, E2):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# --- Ejercicio 1: pastillas -------------------------------------------------
from pastillas import (  # noqa: E402
    NOMBRES,
    clasificar_pastilla,
    detectar_pastillas,
    segmentar_cinta,
)

# --- Ejercicio 2: patentes --------------------------------------------------
from caracteres import segmentar_caracteres  # noqa: E402
from placa import detectar_patente  # noqa: E402

__all__ = [
    "ROOT",
    "E1",
    "E2",
    "NOMBRES",
    "clasificar_pastilla",
    "detectar_pastillas",
    "segmentar_cinta",
    "segmentar_caracteres",
    "detectar_patente",
]
