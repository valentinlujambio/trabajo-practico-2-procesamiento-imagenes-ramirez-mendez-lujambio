"""Configuracion del microservicio (todo por variables de entorno)."""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv

load_dotenv()


def parse_origins(raw: str | None) -> list[str]:
    """
    CORS_ORIGINS admite lista JSON (`["https://a.com","https://b.com"]`) o
    separada por comas. Se normaliza sin barra final: el navegador manda el
    header Origin sin barra, y `https://x.com/` nunca matchearia.
    """
    if not raw or not raw.strip():
        return []

    raw = raw.strip()
    if raw.startswith("["):
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            items = []
    else:
        items = raw.split(",")

    return [str(item).strip().rstrip("/") for item in items if str(item).strip()]


# Hoy el unico consumidor es el portfolio en Railway. Cuando cambie el dominio
# se cambia la env, no el codigo.
CORS_ORIGINS: list[str] = parse_origins(
    os.getenv(
        "CORS_ORIGINS",
        "https://portfolio-personal-production-278c.up.railway.app,http://localhost:3000",
    )
)

PORT: int = int(os.getenv("PORT", "8001"))

# Ancho maximo de las imagenes que viajan al cliente y calidad JPEG. El paso a
# paso son varias imagenes en una sola respuesta, asi que conviene acotarlas.
MAX_WIDTH: int = int(os.getenv("MAX_WIDTH", "760"))
JPEG_QUALITY: int = int(os.getenv("JPEG_QUALITY", "78"))
