"""FastAPI service surface for MatDaemon.

Install with `pip install matdaemon[api]`, then run:
    matdaemon serve --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .matdaemon import BackendName, matmul

try:  # pragma: no cover - optional dependency import
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover
    FastAPI = None
    HTTPException = None
    BaseModel = object
    Field = None
    _IMPORT_ERROR = exc
else:  # pragma: no cover
    _IMPORT_ERROR = None


class MatrixPayload(BaseModel):  # type: ignore[misc]
    a: list[list[float]]
    b: list[list[float]]
    backend: Literal["auto", "numpy", "tiled", "cuda"] = "auto"
    dtype: Literal["float32", "float64"] = "float32"


class MatrixResponse(BaseModel):  # type: ignore[misc]
    job_id: str
    backend: str
    duration_seconds: float
    shape: list[int]
    result: list[list[float]]


@dataclass
class PlatformState:
    started_at: float
    jobs_completed: int = 0


def create_app() -> "FastAPI":
    if FastAPI is None:  # pragma: no cover
        raise RuntimeError("FastAPI is not installed. Install with `pip install matdaemon[api]`.") from _IMPORT_ERROR

    app = FastAPI(
        title="MatDaemon API",
        version="0.1.0",
        description="Matrix compute API for AI agents, RAG systems, simulations, and automation workers.",
    )
    state = PlatformState(started_at=time.time())

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "service": "matdaemon",
            "jobs_completed": state.jobs_completed,
            "uptime_seconds": round(time.time() - state.started_at, 3),
        }

    @app.post("/v1/matmul", response_model=MatrixResponse)
    def matrix_multiply(payload: MatrixPayload) -> MatrixResponse:
        try:
            A = np.asarray(payload.a, dtype=payload.dtype)
            B = np.asarray(payload.b, dtype=payload.dtype)
            start = time.perf_counter()
            result = matmul(A, B, backend=payload.backend)
            duration = time.perf_counter() - start
            state.jobs_completed += 1
            return MatrixResponse(
                job_id=str(uuid.uuid4()),
                backend=payload.backend,
                duration_seconds=round(duration, 6),
                shape=list(result.shape),
                result=result.tolist(),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app() if FastAPI is not None else None
