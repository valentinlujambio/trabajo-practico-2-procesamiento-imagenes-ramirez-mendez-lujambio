"""
Microservicio HTTP sobre las dos soluciones del repo.

Los ejercicios se escribieron como scripts que abren ventanas de matplotlib.
Esta capa los deja consumibles desde cualquier cliente: las funciones de vision
no se tocan (ver `api/bridge.py`), acá solo se exponen como endpoints, se
sirven imagenes de ejemplo y se devuelve cada etapa del procesamiento como
imagen para poder auditar el resultado.

    uv run uvicorn api.main:app --reload --port 8001
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import config
from api.routes.pills import router as pills_router
from api.routes.plates import router as plates_router

app = FastAPI(
    title="pdi-tp2-api",
    version="1.0.0",
    description="Deteccion y clasificacion de pastillas en linea + deteccion de patentes.",
)

# Lista blanca por env (CORS_ORIGINS). Vacia = ningun origen cruzado, que es
# preferible a abrir `*` sin querer en produccion.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pills_router)
app.include_router(plates_router)


@app.get("/health", tags=["infra"])
def health() -> dict:
    return {"status": "ok", "service": "pdi-tp2-api"}


def run() -> None:
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=config.PORT, reload=False)


if __name__ == "__main__":
    run()
