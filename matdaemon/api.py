"""FastAPI service surface for MatDaemon.

Install with `pip install matdaemon[api]`, then run:
    matdaemon serve --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import numpy as np

from .matdaemon import matmul
from .use_cases import USE_CASES

try:  # pragma: no cover - optional dependency import
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except Exception as exc:  # pragma: no cover
    FastAPI = None
    HTTPException = None
    BaseModel = object
    _IMPORT_ERROR = exc
else:  # pragma: no cover
    _IMPORT_ERROR = None

Backend = Literal["auto", "numpy", "tiled", "cuda"]
DType = Literal["float32", "float64"]
JobStatus = Literal["queued", "running", "completed", "failed"]


class MatrixPayload(BaseModel):  # type: ignore[misc]
    a: list[list[float]]
    b: list[list[float]]
    backend: Backend = "auto"
    dtype: DType = "float32"
    use_case: Optional[str] = None


class MatrixResponse(BaseModel):  # type: ignore[misc]
    job_id: str
    backend: str
    duration_seconds: float
    shape: list[int]
    result: list[list[float]]


class JobCreatedResponse(BaseModel):  # type: ignore[misc]
    job_id: str
    status: str
    status_url: str
    result_url: str


class JobStatusResponse(BaseModel):  # type: ignore[misc]
    job_id: str
    status: str
    backend: str
    shape: Optional[list[int]] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    use_case: Optional[str] = None
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class JobRecord:
    job_id: str
    backend: str
    dtype: str
    use_case: Optional[str]
    status: JobStatus = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_seconds: Optional[float] = None
    shape: Optional[list[int]] = None
    result: Optional[list[list[float]]] = None
    error: Optional[str] = None


@dataclass
class PlatformState:
    started_at: float
    jobs_completed: int = 0
    jobs_failed: int = 0
    jobs: Dict[str, JobRecord] = field(default_factory=dict)
    executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=4))


def _as_array(values: list[list[float]], dtype: str) -> np.ndarray:
    return np.asarray(values, dtype=dtype)


def _status_response(job: JobRecord) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        backend=job.backend,
        shape=job.shape,
        duration_seconds=job.duration_seconds,
        error=job.error,
        use_case=job.use_case,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


def create_app() -> "FastAPI":
    if FastAPI is None:  # pragma: no cover
        raise RuntimeError("FastAPI is not installed. Install with `pip install matdaemon[api]`.") from _IMPORT_ERROR

    app = FastAPI(
        title="MatDaemon API",
        version="0.3.0",
        description="Matrix compute platform for AI agents, RAG systems, simulations, and automation workers.",
    )
    state = PlatformState(started_at=time.time())

    def run_job(job_id: str, payload: MatrixPayload) -> None:
        job = state.jobs[job_id]
        job.status = "running"
        job.started_at = time.time()
        try:
            A = _as_array(payload.a, payload.dtype)
            B = _as_array(payload.b, payload.dtype)
            start = time.perf_counter()
            result = matmul(A, B, backend=payload.backend)
            job.duration_seconds = round(time.perf_counter() - start, 6)
            job.shape = list(result.shape)
            job.result = result.tolist()
            job.status = "completed"
            job.completed_at = time.time()
            state.jobs_completed += 1
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.completed_at = time.time()
            state.jobs_failed += 1

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "service": "matdaemon",
            "jobs_completed": state.jobs_completed,
            "jobs_failed": state.jobs_failed,
            "jobs_total": len(state.jobs),
            "uptime_seconds": round(time.time() - state.started_at, 3),
        }

    @app.get("/v1/use-cases")
    def list_use_cases() -> dict:
        return {"use_cases": USE_CASES}

    @app.post("/v1/matmul", response_model=MatrixResponse)
    def matrix_multiply(payload: MatrixPayload) -> MatrixResponse:
        try:
            A = _as_array(payload.a, payload.dtype)
            B = _as_array(payload.b, payload.dtype)
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

    @app.post("/v1/jobs/matmul", response_model=JobCreatedResponse)
    def submit_matrix_job(payload: MatrixPayload) -> JobCreatedResponse:
        job_id = str(uuid.uuid4())
        state.jobs[job_id] = JobRecord(job_id=job_id, backend=payload.backend, dtype=payload.dtype, use_case=payload.use_case)
        state.executor.submit(run_job, job_id, payload)
        return JobCreatedResponse(job_id=job_id, status="queued", status_url=f"/v1/jobs/{job_id}", result_url=f"/v1/jobs/{job_id}/result")

    @app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
    def get_job(job_id: str) -> JobStatusResponse:
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return _status_response(job)

    @app.get("/v1/jobs/{job_id}/result", response_model=MatrixResponse)
    def get_job_result(job_id: str) -> MatrixResponse:
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status == "failed":
            raise HTTPException(status_code=400, detail=job.error)
        if job.status != "completed" or job.result is None or job.shape is None or job.duration_seconds is None:
            raise HTTPException(status_code=202, detail="Job is not complete")
        return MatrixResponse(job_id=job.job_id, backend=job.backend, duration_seconds=job.duration_seconds, shape=job.shape, result=job.result)

    return app


app = create_app() if FastAPI is not None else None
