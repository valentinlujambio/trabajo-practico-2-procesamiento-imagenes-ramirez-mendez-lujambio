"""
Problema 1 — control de calidad en una cinta de pastillas.

Una foto de la cinta transportadora entra, y sale el conteo por tipo con cada
pastilla ubicada y rotulada. La vision es la de `ejercicio_1/pastillas.py`: acá
solo se orquestan sus funciones y se devuelve cada etapa como imagen, que es lo
que vuelve auditable el resultado desde afuera.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api import bridge, imaging

router = APIRouter(prefix="/pills", tags=["pastillas"])

_IMG_DIR = bridge.E1 / "img"

SAMPLES: dict[str, dict[str, str]] = {
    "pills": {
        "file": "pills.png",
        "label": "Cinta transportadora",
        "note": "Cinta con pastillas de cinco tipos distintos: redondas blancas y rosas, cuadradas y capsulas amarillas y azul-blancas.",
    },
}


def _decode(raw: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=422, detail="No se pudo leer la imagen")
    return image


def _annotate(img_bgr: np.ndarray, pastillas: list[dict], etiquetas: list[str]) -> tuple[np.ndarray, list[dict]]:
    """
    Dibuja el resultado y arma la lista de items. El id por tipo (RB1, RB2, ...)
    se asigna en el orden de deteccion, igual que en el script original.
    """
    out = img_bgr.copy()
    contadores: dict[str, int] = {}
    items: list[dict] = []

    for pastilla, sigla in zip(pastillas, etiquetas):
        contadores[sigla] = contadores.get(sigla, 0) + 1
        etiqueta = f"{sigla}{contadores[sigla]}"
        x, y, w, h = pastilla["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(
            out,
            etiqueta,
            (x, max(14, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 60, 60),
            3,
            cv2.LINE_AA,
        )
        items.append(
            {
                "label": etiqueta,
                "code": sigla,
                "name": bridge.NOMBRES.get(sigla, sigla),
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": int(pastilla["area"]),
            }
        )

    return out, items


def _run(img_bgr: np.ndarray) -> dict:
    started = time.perf_counter()

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    steps = [
        imaging.step(
            "original",
            "Foto de la cinta",
            "Una sola foto de la linea. Todo lo que sigue sale de aca: no hay modelo entrenado ni dataset, solo color, forma y morfologia.",
            img_bgr,
        )
    ]

    # A) La cinta es el objeto mas grande tras umbralar con Otsu y cerrar los
    #    huecos que dejan las propias pastillas.
    mask_cinta, bbox_cinta = bridge.segmentar_cinta(gray)
    steps.append(
        imaging.step(
            "belt",
            "Segmentacion de la cinta",
            "Otsu invertido, apertura y cierre morfologico, y se conserva el componente conectado mas grande. Un margen hacia adentro evita los bordes metalicos.",
            mask_cinta,
            fmt=".png",
        )
    )

    # B) Brillo y saturacion se combinan porque ninguno de los dos alcanza solo:
    #    V pierde el extremo blanco de las capsulas, S pierde las blancas.
    steps.append(
        imaging.step(
            "hsv-v",
            "Canal V (brillo)",
            "Separa las pastillas claras del gris de la cinta.",
            hsv[:, :, 2],
        )
    )
    steps.append(
        imaging.step(
            "hsv-s",
            "Canal S (saturacion)",
            "Rescata las partes muy coloridas, clave en la capsula azul-blanca donde el extremo blanco casi se confunde con la cinta.",
            hsv[:, :, 1],
        )
    )

    pastillas, cleaned = bridge.detectar_pastillas(gray, mask_cinta, img_hsv=hsv)
    steps.append(
        imaging.step(
            "pills-mask",
            "Mascara de pastillas",
            f"Otsu(V) OR Otsu(S), recortado a la cinta y limpiado con morfologia. Componentes conectados: {len(pastillas)} pastillas tras descartar manchas chicas y tiras finas del borde.",
            cleaned,
            fmt=".png",
        )
    )

    # C) Clasificacion por reglas sobre forma (circularidad, aspecto) y color.
    etiquetas = [bridge.clasificar_pastilla(p["mask"], hsv) for p in pastillas]

    annotated, items = _annotate(img_bgr, pastillas, etiquetas)
    steps.append(
        imaging.step(
            "result",
            "Clasificadas y contadas",
            "Cada pastilla con su tipo y su numero dentro del tipo. Este es el informe que recibiria la linea de produccion.",
            annotated,
        )
    )

    cuenta = Counter(etiquetas)
    counts = [
        {"code": code, "name": bridge.NOMBRES.get(code, code), "count": n}
        for code, n in sorted(cuenta.items())
    ]

    return {
        "ms": int((time.perf_counter() - started) * 1000),
        "steps": steps,
        "result": {
            "total": len(items),
            "counts": counts,
            "items": items,
            "belt": [int(v) for v in bbox_cinta],
        },
    }


@router.get("/samples")
def list_samples() -> dict:
    return {
        "samples": [
            {
                "id": sample_id,
                "label": meta["label"],
                "note": meta["note"],
                "url": f"/pills/samples/{sample_id}/image",
            }
            for sample_id, meta in SAMPLES.items()
            if (_IMG_DIR / meta["file"]).exists()
        ]
    }


@router.get("/samples/{sample_id}/image")
def sample_image(sample_id: str) -> FileResponse:
    path = _sample_path(sample_id)
    return FileResponse(path, media_type="image/png")


@router.post("/samples/{sample_id}/analyze")
def analyze_sample(sample_id: str) -> dict:
    path = _sample_path(sample_id)
    payload = _guarded(_decode(path.read_bytes()))
    payload["sample"] = {"id": sample_id, "label": SAMPLES[sample_id]["label"]}
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
    meta = SAMPLES.get(sample_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Muestra desconocida: {sample_id}")
    path = _IMG_DIR / meta["file"]
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Falta el archivo {meta['file']}")
    return path


def _guarded(img_bgr: np.ndarray) -> dict:
    try:
        return _run(img_bgr)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - se traduce a 500 con el motivo
        raise HTTPException(status_code=500, detail=f"Error en el pipeline: {exc}") from exc
