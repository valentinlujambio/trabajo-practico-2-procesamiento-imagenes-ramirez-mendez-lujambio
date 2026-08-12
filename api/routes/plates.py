"""
Problema 2 — lectura de patentes en un punto de acceso.

Entra la foto de un auto, sale la patente localizada y sus 7 caracteres
separados, listos para el reconocimiento. La deteccion no busca "un rectangulo
blanco" —falla con patentes lejanas, inclinadas o con sombra— sino la fila de
caracteres que cumple la geometria oficial de una patente Mercosur.

La vision es la de `ejercicio_2/{placa,caracteres}.py`; acá se orquesta y se
serializa el paso a paso.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api import bridge, imaging

router = APIRouter(prefix="/plates", tags=["patentes"])

_IMG_DIR = bridge.E2 / "img"

# Patente real de cada foto. No la usa el algoritmo (no hay OCR acá, solo
# deteccion y segmentacion): sirve para que quien mira la demo pueda contrastar.
EXPECTED = {
    "img_1": "AG678UA",
    "img_2": "AB000RT",
    "img_3": "AC164JM",
    "img_4": "AG456NR",
    "img_5": "AH482KU",
    "img_6": "AG000GA",
    "img_7": "AE001ET",
    "img_8": "AH486ML",
    "img_9": "AI003UM",
    "img_10": "AE233UG",
    "img_11": "AA017EA",
    "img_12": "AC712QM",
}


def _decode(raw: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=422, detail="No se pudo leer la imagen")
    return image


def _char_strip(plate_bgr: np.ndarray, cajas: list[tuple[int, int, int, int]]) -> np.ndarray:
    """
    Pega los caracteres recortados uno al lado del otro, a la misma altura.
    Es la forma mas directa de mostrar que la segmentacion separo los 7.
    """
    if not cajas:
        return plate_bgr

    height = 120
    gap = 10
    crops = []
    for x, y, w, h in cajas:
        crop = plate_bgr[max(0, y) : y + h, max(0, x) : x + w]
        if crop.size == 0:
            continue
        scale = height / float(crop.shape[0])
        crops.append(
            cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), height), interpolation=cv2.INTER_CUBIC)
        )

    if not crops:
        return plate_bgr

    total = sum(c.shape[1] for c in crops) + gap * (len(crops) + 1)
    strip = np.full((height + 2 * gap, total, 3), 18, dtype=np.uint8)
    offset = gap
    for crop in crops:
        strip[gap : gap + height, offset : offset + crop.shape[1]] = crop
        offset += crop.shape[1] + gap
    return strip


def _run(img_bgr: np.ndarray) -> dict:
    started = time.perf_counter()

    steps = [
        imaging.step(
            "original",
            "Foto del vehiculo",
            "Distancia, angulo, sombras y color de auto cambian en cada foto. Nada de eso se puede asumir fijo.",
            img_bgr,
        )
    ]

    patente, bbox = bridge.detectar_patente(img_bgr)

    if patente is None or patente.size == 0:
        return {
            "ms": int((time.perf_counter() - started) * 1000),
            "steps": steps,
            "result": {"detected": False, "bbox": None, "chars": 0, "charBoxes": []},
        }

    x, y, w, h = bbox
    located = img_bgr.copy()
    cv2.rectangle(located, (x, y), (x + w, y + h), (0, 255, 0), 3)
    steps.append(
        imaging.step(
            "located",
            "Patente localizada",
            "Se puntuan todas las filas de candidatos a caracter: cantidad cercana a 7, ancho total ~5.85x la altura, ancho de caracter ~0.55x, espaciado regular, fondo blanco y franja azul arriba. Gana la mejor.",
            located,
        )
    )
    steps.append(
        imaging.step(
            "crop",
            "Recorte de la chapa",
            "La caja se ajusta a partir de los caracteres ya refinados sobre el recorte, no del primer bloque encontrado.",
            patente,
        )
    )

    cajas, plate_resized = bridge.segmentar_caracteres(patente)

    if cajas:
        scale = plate_resized.shape[0] / float(patente.shape[0])
        marked = plate_resized.copy()
        for cx, cy, cw, ch in cajas:
            cv2.rectangle(
                marked,
                (int(cx * scale), int(cy * scale)),
                (int((cx + cw) * scale), int((cy + ch) * scale)),
                (0, 255, 0),
                2,
            )
        steps.append(
            imaging.step(
                "chars",
                f"{len(cajas)} caracteres segmentados",
                "Sobre el recorte se prueban varias binarizaciones y se queda la que deja la linea dominante mas cercana a 7 caracteres.",
                marked,
            )
        )
        steps.append(
            imaging.step(
                "strip",
                "Caracteres aislados",
                "Cada caracter por separado: esta es la entrada que espera un clasificador de caracteres.",
                _char_strip(patente, cajas),
            )
        )

    return {
        "ms": int((time.perf_counter() - started) * 1000),
        "steps": steps,
        "result": {
            "detected": True,
            "bbox": [int(v) for v in bbox],
            "chars": len(cajas),
            "charBoxes": [[int(v) for v in caja] for caja in cajas],
        },
    }


@router.get("/samples")
def list_samples() -> dict:
    samples = []
    for index in range(1, 13):
        sample_id = f"img_{index}"
        if not (_IMG_DIR / f"{sample_id}.jpg").exists():
            continue
        samples.append(
            {
                "id": sample_id,
                "label": f"Vehiculo {index}",
                "note": f"Patente real: {EXPECTED[sample_id]}",
                "expected": EXPECTED[sample_id],
                "url": f"/plates/samples/{sample_id}/image",
            }
        )
    return {"samples": samples}


@router.get("/samples/{sample_id}/image")
def sample_image(sample_id: str) -> FileResponse:
    return FileResponse(_sample_path(sample_id), media_type="image/jpeg")


@router.post("/samples/{sample_id}/analyze")
def analyze_sample(sample_id: str) -> dict:
    path = _sample_path(sample_id)
    payload = _guarded(_decode(path.read_bytes()))
    payload["sample"] = {
        "id": sample_id,
        "label": f"Vehiculo {sample_id.split('_')[-1]}",
        "expected": EXPECTED.get(sample_id),
    }
    return payload


@router.post("/analyze")
async def analyze_upload(image: UploadFile = File(...)) -> dict:
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=422, detail="Imagen vacia")
    payload = _guarded(_decode(raw))
    payload["sample"] = None
    return payload


def _sample_path(sample_id: str) -> Path:
    if sample_id not in EXPECTED:
        raise HTTPException(status_code=404, detail=f"Muestra desconocida: {sample_id}")
    path = _IMG_DIR / f"{sample_id}.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Falta el archivo {sample_id}.jpg")
    return path


def _guarded(img_bgr: np.ndarray) -> dict:
    try:
        return _run(img_bgr)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - se traduce a 500 con el motivo
        raise HTTPException(status_code=500, detail=f"Error en el pipeline: {exc}") from exc
