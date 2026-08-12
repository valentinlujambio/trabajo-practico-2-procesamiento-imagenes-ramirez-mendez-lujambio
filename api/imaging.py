"""Serializacion de imagenes para las respuestas de la API."""

from __future__ import annotations

import base64

import cv2
import numpy as np

from api import config


def shrink(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= config.MAX_WIDTH:
        return image
    scale = config.MAX_WIDTH / float(w)
    return cv2.resize(
        image, (config.MAX_WIDTH, max(1, int(h * scale))), interpolation=cv2.INTER_AREA
    )


def encode(image: np.ndarray, fmt: str = ".jpg") -> str:
    """
    Devuelve un data URL. Las mascaras binarias van en PNG: comprimen mejor que
    JPEG y no ensucian los bordes con artefactos.
    """
    small = shrink(image)
    params = [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY] if fmt == ".jpg" else []
    ok, buffer = cv2.imencode(fmt, small, params)
    if not ok:
        raise RuntimeError("No se pudo codificar la imagen")
    mime = "image/png" if fmt == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(buffer).decode('ascii')}"


def step(
    step_id: str,
    title: str,
    caption: str,
    image: np.ndarray,
    fmt: str = ".jpg",
) -> dict:
    """Una etapa del pipeline tal como la consume el front: id + copy + imagen."""
    return {
        "id": step_id,
        "title": title,
        "caption": caption,
        "image": encode(image, fmt),
    }
